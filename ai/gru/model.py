"""
Concrete GRU temporal wireless-channel predictor (Phase 4).

Implements the architecture `ai/gru/base.py` specifies: a historical
channel/mobility feature sequence -> a predicted future channel state
`tau` steps ahead, with an associated `prediction_uncertainty` (consumed
by the Trust Estimator's `U_t`, per `ai/gru/base.py`'s docstring).

Scope: isolated architecture only, exactly like `ai/pointpillars/model.py`
and `ai/v2x_vit/model.py`. UNTRAINED: weights are randomly initialized
(fixed seed), never trained on real or synthetic data. No checkpoint
exists, no training pipeline exists.

REAL-DATA STATUS (Phase 4 GRU audit): none of this model's declared
`input_features` (`configs/model.yaml` -> `perception.gru.input_features`
= `["csi", "snr", "beam_index", "mobility"]`) exist as real values in the
Phase 2 dataset except "mobility" -- CSI/SNR/beam_index are `None` for
every frame in all 5 real scenes (no Sionna RT run against them yet), and
`gt_future_csi`/`gt_future_beam` (the ground truth this model would be
trained/evaluated against) are equally unpopulated. This module is
therefore validated ONLY against synthetic, code-correctness-only tensors
-- never real data -- and must not be reported as real-data-validated,
trained, or accurate.

UNCERTAINTY ESTIMATION: the project's mathematical specification for how
`prediction_uncertainty` should be computed does not exist anywhere
accessible (`03_Mathematical_Formulation`/`04_Novel_Algorithm_Design` are
unrecoverable -- see the Phase 4 recovery audit). The uncertainty head
below is the smallest architecturally-sound choice that satisfies the
software contract (a non-negative per-output-dimension scalar produced
alongside the point prediction) -- an explicit PLACEHOLDER, not a
calibrated or research-validated uncertainty formulation. Do not treat its
output as a real confidence estimate.

INPUT FEATURE COMPOSITION: `configs/model.yaml` names four input features
(csi/snr/beam_index/mobility) but specifies no per-feature dimensionality
or how they are concatenated into one numeric vector -- that breakdown is
undocumented. This module therefore accepts a single flat numeric
`input_dim` per timestep (a constructor parameter, not a hardcoded
project-specified split) and does not attempt to reconstruct or assume a
csi/snr/beam_index/mobility split internally.

HORIZON: `ai/gru/base.py` documents the discretized horizon set
`{1, 2, 3, 5, 8, 10}` (selected by TAHS, Phase 5). This module conditions
its prediction on the requested horizon via a small learned embedding
looked up from that exact set -- it does not invent additional horizon
semantics (e.g. continuous horizons, multi-horizon batched output) beyond
what `base.py` documents.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import nn

from ai.gru.base import BaseGRUPredictor
from ai.utils.config_loader import load_config

#: The exact horizon set `ai/gru/base.py` documents (driven by TAHS,
#: Phase 5) -- not invented here.
VALID_HORIZONS = (1, 2, 3, 5, 8, 10)


class InvalidSequenceError(ValueError):
    """Raised when `predict` has no usable historical sequence, or the
    requested horizon is outside the documented set.

    Mirrors `ai/pointpillars/model.py`'s `MissingPointCloudError` and
    `ai/v2x_vit/model.py`'s `MissingVehicleFeaturesError` -- callers must
    catch this and skip/report rather than receive a fabricated
    prediction for invalid input.
    """


@dataclasses.dataclass
class GRUPrediction:
    """`BaseGRUPredictor.predict`'s concrete return type.

    Fields
    ------
    predicted_channel_state : torch.Tensor, shape (batch, output_dim)
        Y_hat_{t+tau} -- the predicted future channel state, `tau` steps
        (`horizon`) ahead, in the same flat numeric feature space as the
        input (see module docstring: no csi/snr/beam_index split is
        assumed).
    prediction_uncertainty : torch.Tensor, shape (batch, output_dim)
        Non-negative per-output-dimension uncertainty, consumed by the
        Trust Estimator's U_t. See module docstring: this is an explicit
        architectural placeholder, not a calibrated formulation.
    horizon : int
        The requested horizon this prediction was conditioned on, echoed
        back for the caller's convenience.
    """

    predicted_channel_state: torch.Tensor
    prediction_uncertainty: torch.Tensor
    horizon: int


class _PredictionHead(nn.Module):
    """GRU final hidden state -> point prediction."""

    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


class _UncertaintyHead(nn.Module):
    """GRU final hidden state -> non-negative per-dimension uncertainty.

    Placeholder architecture (see module docstring): a linear projection
    followed by softplus, which is the minimal way to guarantee the
    non-negativity a "prediction_uncertainty" quantity requires without
    asserting any specific calibrated formulation.
    """

    def __init__(self, hidden_size: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.linear(h))


class GRUPredictor(BaseGRUPredictor):
    """Concrete `BaseGRUPredictor`: historical sequence -> input projection
    -> GRU -> {prediction head, uncertainty head}.

    `predict(historical_sequence, horizon)` expects
    `historical_sequence` as a real `torch.Tensor` of shape
    `(batch, seq_len, input_dim)`, `float32`, `seq_len >= 1`, `batch >= 1`
    -- never a fabricated or placeholder sequence. `horizon` must be one
    of `VALID_HORIZONS`. Returns a `GRUPrediction`.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        input_dim: int = 4,
        output_dim: int | None = None,
        seed: int = 0,
    ):
        cfg = (config or load_config("model"))["perception"]["gru"]
        self.hidden_size: int = cfg["hidden_size"]
        self.num_layers: int = cfg["num_layers"]
        self.input_features: list[str] = list(cfg["input_features"])
        self.input_dim = input_dim
        # Placeholder default: predict in the same flat feature space as
        # the input, absent any documented output composition.
        self.output_dim = output_dim if output_dim is not None else input_dim

        torch.manual_seed(seed)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_size)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        # Small horizon-conditioning embedding, added to the GRU's final
        # hidden state before the prediction/uncertainty heads -- the
        # documented horizon set is discrete and small, so an embedding
        # lookup is the smallest architecturally-sound way to make the
        # prediction actually depend on which horizon was requested.
        self._horizon_index = {h: i for i, h in enumerate(VALID_HORIZONS)}
        self.horizon_embedding = nn.Embedding(len(VALID_HORIZONS), self.hidden_size)
        self.prediction_head = _PredictionHead(self.hidden_size, self.output_dim)
        self.uncertainty_head = _UncertaintyHead(self.hidden_size, self.output_dim)

        # Inference-only: no training loop exists yet, so these stay in
        # eval() permanently -- same rationale as PointPillars/V2X-ViT
        # (deterministic repeated calls on the same input).
        self.input_projection.eval()
        self.gru.eval()
        self.horizon_embedding.eval()
        self.prediction_head.eval()
        self.uncertainty_head.eval()

    def predict(self, historical_sequence: Any, horizon: int) -> GRUPrediction:
        if horizon not in VALID_HORIZONS:
            raise InvalidSequenceError(
                f"horizon={horizon!r} is not in the documented set {VALID_HORIZONS} "
                "(ai/gru/base.py / TAHS discretization) -- not inventing new horizon semantics."
            )
        if not torch.is_tensor(historical_sequence):
            raise InvalidSequenceError(
                f"historical_sequence is not a tensor: {type(historical_sequence)!r}"
            )
        if historical_sequence.ndim != 3:
            raise InvalidSequenceError(
                f"historical_sequence has shape {tuple(historical_sequence.shape)}; "
                "expected (batch, seq_len, input_dim)"
            )
        batch, seq_len, feat_dim = historical_sequence.shape
        if batch == 0 or seq_len == 0:
            raise InvalidSequenceError(
                f"historical_sequence has batch={batch}, seq_len={seq_len} -- "
                "an empty sequence/batch is never valid; the caller must skip "
                "this vehicle/frame rather than pass one through."
            )
        if feat_dim != self.input_dim:
            raise InvalidSequenceError(
                f"historical_sequence's last dim is {feat_dim}; expected input_dim={self.input_dim} "
                f"(configured input_features: {self.input_features})"
            )
        if not torch.isfinite(historical_sequence).all():
            raise InvalidSequenceError("historical_sequence contains non-finite (NaN/Inf) values")

        with torch.no_grad():
            x = self.input_projection(historical_sequence)  # (batch, seq_len, hidden_size)
            _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
            last_hidden = h_n[-1]  # (batch, hidden_size)

            horizon_idx = torch.tensor([self._horizon_index[horizon]] * batch, dtype=torch.long)
            conditioned = last_hidden + self.horizon_embedding(horizon_idx)  # (batch, hidden_size)

            predicted = self.prediction_head(conditioned)  # (batch, output_dim)
            uncertainty = self.uncertainty_head(conditioned)  # (batch, output_dim)

        return GRUPrediction(
            predicted_channel_state=predicted,
            prediction_uncertainty=uncertainty,
            horizon=horizon,
        )
