"""
Concrete Context-Aware Criticality Estimator (Phase 4).

Implements the weighted-sum criticality formulation `ai/criticality/base.py`
specifies (`03_Mathematical_Formulation.docx, Section 6`):

    C_t = sum_i(alpha_i * f_i),   sum(alpha_i) = 1

over five evidence terms: relative_speed (f1), blockage_probability (f2),
sync_age (f3), channel_degradation (f4), traffic_density (f5).

Scope: isolated AI-layer architecture only, exactly like
`ai/trust_estimator/model.py`, `ai/pointpillars/model.py`,
`ai/v2x_vit/model.py`, and `ai/gru/model.py`. This module does NOT
modify, replace, or wire into the existing backend criticality
implementation (`backend/app/crud/criticality.py` and friends), which
remains the project's separately-running, already-tested weighted-average
math.

=============================================================================
CRITICALITY INPUT-CONTRACT GAP (Phase 4 Criticality audit -- read before
editing)
=============================================================================
`BaseCriticalityEstimator.estimate(state: DigitalTwinState) -> float` is
the project's public contract. The Phase 4 Criticality audit found that
none of the five required evidence terms can currently be authoritatively
derived from `DigitalTwinState` together with the real Phase 2 dataset:

  - `relative_speed`: `DigitalTwinState.mobility.relative_speed` exists as
    a dataclass field, but no real `Frame` column stores it -- only the
    vehicle's own absolute `speed_mps` is real. A relative-speed value
    would require a pairwise computation against a specific communication
    peer, which this module does not perform.
  - `blockage_probability`: `DigitalTwinState.environment.
    blockage_probability` exists as a dataclass field, but real frames'
    `environmental_context` JSON (confirmed by direct query) contains no
    `blockage_probability` key at all -- only
    `is_at_traffic_light`/`traffic_light_state`/`road_id`.
  - `sync_age`: `DigitalTwinState.sync_age_seconds` exists and is
    technically populatable, but real `sync_offset_ms` is always `0.0`
    (since `wireless_timestamp` is a documented placeholder copy of
    `simulation_timestamp`), so it would be a trivial constant, not a
    meaningful real measurement.
  - `channel_degradation`: has NO corresponding `DigitalTwinState` field
    at all. It would need to come from `channel.snr`/`channel.path_loss`,
    both `None` for every real frame.
  - `traffic_density`: `DigitalTwinState.environment.traffic_density` is a
    `str` category, while the real `Frame.traffic_density` column is a
    `float` and is confirmed `null` for real frames -- a type mismatch on
    top of a real-data gap, with no documented category-to-float
    normalization anywhere in the repo.

Consequently `estimate(state)` keeps the exact required public signature
but ALWAYS raises `MissingCriticalityEvidenceError` today -- it never
silently manufactures any of the five terms, never reads
`state.metadata`, and never substitutes an unrelated state field. The
actual weighted-sum math is implemented and independently testable via
`compute_criticality_from_evidence(...)`, which takes all five terms as
explicitly supplied arguments.

UNCALIBRATED: `weights=(0.25, 0.25, 0.15, 0.20, 0.15)` are configuration-
shaped ARCHITECTURE DEFAULTS matching `configs/model.yaml`'s
`criticality_estimator.feature_weights` (not loaded from that file). They
are explicitly NOT learned, optimized, validated, or empirically derived
-- there are currently zero real criticality records/outcomes in the
project to calibrate against.

NO NORMALIZATION, NO THRESHOLDS: the Phase 4 Criticality audit found no
authoritative normalization rule for any of the five raw features, and no
low/medium/high criticality band boundaries anywhere accessible. This
module does not invent either -- callers of
`compute_criticality_from_evidence` are responsible for supplying already
in-range `[0, 1]` values, and this module exposes only the raw `C_t`
score, no qualitative banding.

ALPHA OWNERSHIP: alpha_i belongs exclusively to this Criticality
Estimator (never introduced into the Trust Estimator, which uses w_i).
"""

from __future__ import annotations

import math
from typing import Any

from ai.criticality.base import BaseCriticalityEstimator
from digital_twin.state import DigitalTwinState

#: Architecture defaults matching the shape of configs/model.yaml's
#: criticality_estimator.feature_weights -- NOT loaded from that file, and
#: NOT a calibrated/learned result (see module docstring).
DEFAULT_WEIGHTS: tuple[float, float, float, float, float] = (0.25, 0.25, 0.15, 0.20, 0.15)

_WEIGHT_SUM_TOLERANCE = 1e-6


class MissingCriticalityEvidenceError(RuntimeError):
    """Raised by `estimate(state)` when required criticality evidence
    cannot be obtained from a `DigitalTwinState` alone.

    This is not a bug to be worked around -- it reflects a genuine,
    documented gap between `BaseCriticalityEstimator`'s state-based
    contract and what `DigitalTwinState`/the real Phase 2 dataset
    actually carry (see module docstring). Use
    `compute_criticality_from_evidence(...)` with explicitly supplied
    evidence instead of trying to make `estimate()` succeed by guessing
    values.
    """


def _is_finite_scalar(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class CriticalityEstimator(BaseCriticalityEstimator):
    """Concrete `BaseCriticalityEstimator`. See module docstring for the
    Criticality input-contract gap this class deliberately does not paper
    over.
    """

    def __init__(self, weights: tuple[float, float, float, float, float] = DEFAULT_WEIGHTS):
        if len(weights) != 5:
            raise ValueError(f"weights must have exactly 5 elements, got {len(weights)}")
        if not all(_is_finite_scalar(w) for w in weights):
            raise ValueError(f"all weights must be finite numbers, got {weights!r}")
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be non-negative, got {weights!r}")
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"weights must sum to 1.0 (within {_WEIGHT_SUM_TOLERANCE}), "
                f"got sum={weight_sum!r} for weights={weights!r}"
            )

        self.weights = tuple(float(w) for w in weights)

    def estimate(self, state: DigitalTwinState) -> float:
        """Required public contract: `estimate(state: DigitalTwinState) ->
        float`.

        ALWAYS raises `MissingCriticalityEvidenceError` today (see module
        docstring): none of the five required evidence terms
        (relative_speed, blockage_probability, sync_age,
        channel_degradation, traffic_density) can currently be
        authoritatively obtained from `DigitalTwinState` together with the
        real Phase 2 dataset. Never reads `state.metadata`, never
        substitutes an unrelated state field, never defaults any term,
        and never calls backend code.
        """
        raise MissingCriticalityEvidenceError(
            "estimate(state) cannot produce a real criticality score: none "
            "of the five required evidence terms (relative_speed, "
            "blockage_probability, sync_age, channel_degradation, "
            "traffic_density) can currently be authoritatively derived "
            "from DigitalTwinState together with the real Phase 2 "
            "dataset -- relative_speed and channel_degradation have no "
            "populated real source, blockage_probability is absent from "
            "real environmental_context, sync_age is a trivial constant "
            "(wireless_timestamp is a placeholder), and traffic_density "
            "is null in the real data with no documented category-to-"
            "float normalization (see the Phase 4 Criticality audit). "
            "Use compute_criticality_from_evidence(...) with explicitly "
            "supplied evidence instead of calling estimate(state)."
        )

    def compute_criticality_from_evidence(
        self,
        relative_speed: float,
        blockage_probability: float,
        sync_age: float,
        channel_degradation: float,
        traffic_density: float,
    ) -> float:
        """Isolated mathematical criticality calculation, independent of
        `DigitalTwinState`. ALL FIVE evidence terms must be supplied
        explicitly by the caller -- this method does not read any project
        state, database, or config, does not normalize or clip its
        inputs, and does not claim the values it's given are real Phase 2
        data.

        Computes C_t = sum_i(alpha_i * f_i) using `self.weights`
        (architecture defaults unless overridden at construction --
        see module docstring: NOT calibrated/learned/validated).
        """
        for name, value in (
            ("relative_speed", relative_speed),
            ("blockage_probability", blockage_probability),
            ("sync_age", sync_age),
            ("channel_degradation", channel_degradation),
            ("traffic_density", traffic_density),
        ):
            if not _is_finite_scalar(value):
                raise ValueError(f"{name} must be a finite scalar number, got {value!r}")

        a1, a2, a3, a4, a5 = self.weights
        criticality_score = (
            a1 * relative_speed
            + a2 * blockage_probability
            + a3 * sync_age
            + a4 * channel_degradation
            + a5 * traffic_density
        )
        return criticality_score
