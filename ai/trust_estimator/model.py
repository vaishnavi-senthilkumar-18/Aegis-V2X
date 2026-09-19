"""
Concrete Calibrated Twin Trust Estimator (Phase 4).

Implements the sigmoid trust formulation `ai/trust_estimator/base.py`
specifies:

    S_t = sum_i(w_i * z_i)
    T_t = sigmoid(S_t / tau)

using the four trust evidence terms already established by the project's
existing (backend) trust implementation (`backend/app/crud/trust.py`):
prediction_error (e_t), prediction_uncertainty (u_t), sync_age_penalty
(a_t, a penalty), and comm_quality (q_t, a reward). The reward/penalty
sign convention is preserved exactly:

    raw_score = w4*comm_quality - w1*prediction_error
                - w2*prediction_uncertainty - w3*sync_age_penalty

Scope: isolated AI-layer architecture only, exactly like
`ai/pointpillars/model.py`, `ai/v2x_vit/model.py`, and `ai/gru/model.py`.
This module does NOT modify, replace, or wire into the existing backend
trust implementation, which remains the project's separately-running,
already-tested trust math.

=============================================================================
TRUST INPUT-CONTRACT GAP (Phase 4 architecture review -- read before editing)
=============================================================================
`BaseTrustEstimator.estimate(state: DigitalTwinState) -> float` is the
project's public contract, and `DigitalTwinState`'s own formal definition
(`DT_t = {CSI_t, SNR_t, B_t, M_t, E_t, U_t, Age_t}`, digital_twin/state.py)
does NOT include `prediction_error` or `comm_quality` as state components.
This is not an omission this module can fix by adding a field or a
fallback:

  - `prediction_error` requires comparing a *prediction* against a
    *later* ground-truth value (see
    `simulation/annotation/future_channel_labeler.py`, an OFFLINE batch
    labeler that looks ahead in a fully-known simulated trajectory) --
    structurally impossible to derive from one `DigitalTwinState`
    snapshot, which represents a single instant DT_t, not a
    predicted/actual pair. Real CSI is also `None` for every frame in all
    5 real scenes (Phase 4 Trust Estimator audit), so even the offline
    labeler's method can't run on real data today.
  - `comm_quality` is plausibly derivable from `state.channel.snr` (which
    IS a real DigitalTwinState field), but the only concrete formula found
    anywhere in the repo -- `clip(snr_db / 35.0, 0, 1)` -- lives in
    `backend/app/services/synthetic_data_service.py`, is unsourced/
    undocumented there, and is explicitly NOT treated as an authoritative
    project formula here.

Consequently `estimate(state)` keeps the exact required public signature
but ALWAYS raises `MissingTrustEvidenceError` today -- it never silently
manufactures `prediction_error` or `comm_quality`, never reads
`state.metadata` (explicitly documented as "not part of the mathematical
state vector"), and never applies a constructor-time fallback. The actual
sigmoid math is implemented and independently testable via
`compute_trust_from_evidence(...)`, which takes all four terms as
explicitly supplied arguments -- this is the "mathematical calculation
with explicitly supplied evidence" the architecture review called for,
kept clearly separate from the state-based path.

UNCALIBRATED: `weights=(0.35, 0.25, 0.20, 0.20)` and `tau=0.8` are
configurable ARCHITECTURE DEFAULTS, matching the shape of
`configs/model.yaml`'s `trust_estimator` section (not loaded from it).
They are explicitly NOT research-calibrated values -- there are currently
zero real trust records/outcomes in the project to calibrate against.

ALPHA SEPARATION: alpha_i belongs exclusively to the Context-Aware
Criticality Estimator. Nothing in this module introduces, accepts, or
references alpha_i.
"""

from __future__ import annotations

import math
from typing import Any

from ai.trust_estimator.base import BaseTrustEstimator
from digital_twin.state import DigitalTwinState

#: Architecture defaults matching the shape of configs/model.yaml's
#: trust_estimator.feature_weights -- NOT loaded from that file, and NOT
#: a calibrated result (see module docstring).
DEFAULT_WEIGHTS: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)

#: Architecture default matching configs/model.yaml's
#: calibration_temperature -- NOT loaded from that file, NOT calibrated.
DEFAULT_TAU: float = 0.8

#: Equal-width bin count for calibration_error's ECE computation, chosen
#: to match the shape of configs/model.yaml's calibration_bins (5 bands of
#: width 0.2) without actually loading that file.
DEFAULT_ECE_BINS: int = 5

_WEIGHT_SUM_TOLERANCE = 1e-6


class MissingTrustEvidenceError(RuntimeError):
    """Raised by `estimate(state)` when required trust evidence cannot be
    obtained from a `DigitalTwinState` alone.

    This is not a bug to be worked around -- it reflects a genuine,
    documented gap between `BaseTrustEstimator`'s state-based contract and
    what `DigitalTwinState` actually carries (see module docstring). Use
    `compute_trust_from_evidence(...)` with explicitly supplied evidence
    instead of trying to make `estimate()` succeed by guessing values.
    """


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _reduce_uncertainty(raw: Any) -> float:
    """`prediction_uncertainty` (a finite float/int, a scalar tensor, or a
    multi-element tensor) -> a finite scalar float.

    Documented deterministic reduction (architectural placeholder, not
    derived from any specification): a plain finite number is used as-is;
    a scalar tensor (`numel() == 1`) is reduced via `.item()`; a
    multi-element tensor (e.g. GRUPrediction.prediction_uncertainty's real
    `(batch, output_dim)` shape) is reduced via `.mean()` across all
    elements. Never silently assumes tensor == float -- non-tensor,
    non-numeric input raises `ValueError`, as does any non-finite or empty
    tensor.
    """
    if _is_finite_number(raw):
        return float(raw)

    if hasattr(raw, "numel") and hasattr(raw, "item"):  # torch.Tensor, without importing torch
        if not raw.numel():
            raise ValueError("prediction_uncertainty tensor is empty (numel() == 0)")
        if not bool(raw.isfinite().all()):
            raise ValueError("prediction_uncertainty tensor contains non-finite (NaN/Inf) values")
        if raw.numel() == 1:
            return float(raw.item())
        return float(raw.mean().item())

    raise ValueError(
        f"prediction_uncertainty must be a finite float or a torch.Tensor, got {type(raw)!r}"
    )


class TwinTrustEstimator(BaseTrustEstimator):
    """Concrete `BaseTrustEstimator`. See module docstring for the Trust
    input-contract gap this class deliberately does not paper over.
    """

    def __init__(
        self,
        weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
        tau: float = DEFAULT_TAU,
    ):
        if len(weights) != 4:
            raise ValueError(f"weights must have exactly 4 elements, got {len(weights)}")
        if not all(_is_finite_number(w) for w in weights):
            raise ValueError(f"all weights must be finite numbers, got {weights!r}")
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got {weights!r}")
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"weights must sum to 1.0 (within {_WEIGHT_SUM_TOLERANCE}), "
                f"got sum={weight_sum!r} for weights={weights!r}"
            )
        if not _is_finite_number(tau) or tau <= 0:
            raise ValueError(f"tau must be a finite number > 0, got {tau!r}")

        self.weights = tuple(float(w) for w in weights)
        self.tau = float(tau)

    def estimate(self, state: DigitalTwinState) -> float:
        """Required public contract: `estimate(state: DigitalTwinState) ->
        float`.

        ALWAYS raises `MissingTrustEvidenceError` today (see module
        docstring): `DigitalTwinState` provides no `prediction_error`
        (structurally requires a future-prediction-vs-ground-truth
        comparison this single-snapshot state cannot carry) and no
        authoritative `comm_quality` (the only formula found in the repo
        is an unauthoritative synthetic-generator placeholder, not adopted
        here). Never reads `state.metadata`, never defaults either value
        to 0, never falls back to a constructor-time guess, and never
        calls backend code.
        """
        raise MissingTrustEvidenceError(
            "estimate(state) cannot produce a real trust probability: "
            "DigitalTwinState does not provide prediction_error (requires "
            "a future-prediction-vs-later-ground-truth comparison via the "
            "offline future-channel labeling pipeline, not derivable from "
            "a single state snapshot; real CSI is also unavailable for "
            "all 5 real Phase 2 scenes) or an authoritative comm_quality "
            "derived from state.channel.snr (the only formula found in "
            "the repo, snr_db/35.0 in backend/app/services/"
            "synthetic_data_service.py, is an unauthoritative synthetic-"
            "generator placeholder, not treated as project-validated "
            "here). Use compute_trust_from_evidence(...) with explicitly "
            "supplied evidence instead of calling estimate(state)."
        )

    def compute_trust_from_evidence(
        self,
        prediction_error: float,
        prediction_uncertainty: Any,
        sync_age_penalty: float,
        comm_quality: float,
    ) -> float:
        """Isolated mathematical trust calculation, independent of
        `DigitalTwinState`. ALL FOUR trust evidence terms must be supplied
        explicitly by the caller -- this method does not read any project
        state, database, or config, and does not claim the values it's
        given are real Phase 2 data.

        `comm_quality` in particular: the project has not yet defined or
        validated an authoritative SNR-to-quality normalization (see
        module docstring) -- callers are responsible for supplying a
        value they can justify; this method does not compute one from SNR
        itself.

        `prediction_uncertainty` may be a finite float/int, a scalar
        tensor, or a multi-element tensor (see `_reduce_uncertainty` for
        the documented deterministic reduction).
        """
        if not _is_finite_number(prediction_error):
            raise ValueError(f"prediction_error must be a finite number, got {prediction_error!r}")
        if not _is_finite_number(comm_quality):
            raise ValueError(f"comm_quality must be a finite number, got {comm_quality!r}")
        if not _is_finite_number(sync_age_penalty):
            raise ValueError(f"sync_age_penalty must be a finite number, got {sync_age_penalty!r}")

        reduced_uncertainty = _reduce_uncertainty(prediction_uncertainty)

        w1, w2, w3, w4 = self.weights
        raw_score = (
            w4 * comm_quality
            - w1 * prediction_error
            - w2 * reduced_uncertainty
            - w3 * sync_age_penalty
        )
        trust_probability = 1.0 / (1.0 + math.exp(-raw_score / self.tau))

        if not (0.0 <= trust_probability <= 1.0):
            raise ValueError(
                f"computed trust_probability {trust_probability!r} outside [0, 1] "
                "-- calibration invariant violation"
            )
        return trust_probability

    def calibration_error(self, predictions: list[float], outcomes: list[bool]) -> float:
        """Expected Calibration Error (ECE) over `DEFAULT_ECE_BINS` equal-
        width bins spanning [0, 1].

        CODE-CORRECTNESS INFRASTRUCTURE ONLY: there are currently zero
        real trust records/outcomes in the project -- this method has
        never been run against real data, and its result must never be
        reported as a real calibration measurement.

        ECE = sum_b (n_b / N) * |acc(b) - conf(b)|, where acc(b) is the
        fraction of True outcomes in bin b and conf(b) is the mean
        predicted probability in bin b. Deterministic: bin edges are
        fixed, equal-width, and inputs are processed in the given order.
        """
        if len(predictions) != len(outcomes):
            raise ValueError(
                f"predictions and outcomes must have equal length, "
                f"got {len(predictions)} and {len(outcomes)}"
            )
        if len(predictions) == 0:
            raise ValueError("predictions/outcomes must not be empty")
        for i, p in enumerate(predictions):
            if not _is_finite_number(p):
                raise ValueError(f"predictions[{i}]={p!r} is not a finite number")
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"predictions[{i}]={p!r} is not a valid probability in [0, 1]")
        for i, o in enumerate(outcomes):
            if not isinstance(o, bool):
                raise ValueError(f"outcomes[{i}]={o!r} is not a bool (binary outcome)")

        n_bins = DEFAULT_ECE_BINS
        bin_preds: list[list[float]] = [[] for _ in range(n_bins)]
        bin_outcomes: list[list[bool]] = [[] for _ in range(n_bins)]

        for p, o in zip(predictions, outcomes):
            # Deterministic bin assignment; the top edge (1.0) belongs to
            # the last bin rather than overflowing past it.
            idx = min(int(p * n_bins), n_bins - 1)
            bin_preds[idx].append(p)
            bin_outcomes[idx].append(o)

        n_total = len(predictions)
        ece = 0.0
        for preds_in_bin, outcomes_in_bin in zip(bin_preds, bin_outcomes):
            n_b = len(preds_in_bin)
            if n_b == 0:
                continue
            confidence = sum(preds_in_bin) / n_b
            accuracy = sum(1 for o in outcomes_in_bin if o) / n_b
            ece += (n_b / n_total) * abs(accuracy - confidence)

        if not _is_finite_number(ece) or ece < 0:
            raise ValueError(f"computed ECE {ece!r} is not a finite non-negative number")
        return ece
