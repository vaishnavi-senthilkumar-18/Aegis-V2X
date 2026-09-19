"""
Graded Digital-Twin Consistency Authority (GDCA) — Phase 5.

CONCEPT
-------
GDCA answers a narrower question than the existing Trust Estimator does.
Trust (`ai/trust_estimator/`) estimates *communication-reliability*
confidence from four specific evidence terms (prediction_error,
prediction_uncertainty, sync_age_penalty, comm_quality), following a
documented sigmoid formula with documented weights
(`configs/model.yaml: trust_estimator`).

GDCA instead measures *Digital-Twin state consistency*: given a fresh
observed state and the Digital Twin's own predicted state, how much do
they agree, and therefore how much influence ("authority") should this
new observation be granted? This is the same conceptual quantity control
theory calls an "innovation" (observed - predicted) and its normalized
consistency check (e.g. Normalized Innovation Squared / NIS in Kalman
filtering) — GDCA is a bounded, continuous mapping of that same idea, not
a new invented concept.

    residual r_t = observed_state - predicted_state      (innovation)
    z_t = ||r_t|| / (sigma_t + eps)                       (normalized)
    A_t = exp(-0.5 * z_t^2)                                in (0, 1]

A_t = 1 when the observation exactly matches the prediction (perfect
consistency -> full authority); A_t decays smoothly toward 0 as the
residual grows relative to the uncertainty scale sigma_t. This is
DISTINCT from Trust: Trust never compares a predicted state against an
observed one, and GDCA never touches prediction_error/comm_quality/
sync_age_penalty or CSI/SNR/RSSI/beam_index/path_loss.

CRITICALITY IS NOT AN INPUT to GDCA. Per the intended architecture
(Digital Twin -> consistency -> GDCA -> Authority; Authority + Criticality
-> TAHS/FSDP), Criticality remains a separate, sibling output produced by
`ai/criticality/`, combined with GDCA's Authority only downstream. GDCA
itself has no dependency on Criticality.

WHY estimate() ALWAYS RAISES (mirrors ai/trust_estimator/model.py's
documented gap exactly)
------------------------------------------------------------------------
`BaseGDCAEstimator.estimate(observed_state, predicted_state)` takes a
*single pair* of DigitalTwinState snapshots. The residual `r_t` between
them is directly computable (both expose `mobility.position`, a real,
always-populated field). The uncertainty scale `sigma_t`, however, is
NOT a per-pair quantity — it is a spread over a *window* of recent
residuals (see `rolling_uncertainty` below, the same windowed-std
technique `simulation/annotation/future_channel_labeler.py`'s
`_rolling_uncertainty` already uses for CSI magnitude). A single
(observed, predicted) pair structurally cannot supply that window, so
`estimate()` cannot manufacture a principled `sigma_t` without silently
inventing one. Rather than defaulting to an arbitrary constant, it raises
`MissingGDCAEvidenceError` — the same choice `TwinTrustEstimator.estimate`
and `CriticalityEstimator.estimate` already make for their own
structural gaps. The actual math is implemented and independently
testable via `compute_authority_from_evidence(residual_norm,
uncertainty)`, which takes both terms as explicitly supplied arguments.

WHAT IS DELIBERATELY NOT INCLUDED
----------------------------------
- `sync_age_seconds` (DigitalTwinState's real, available field) is NOT
  incorporated into the authority formula. Its correct role and sign
  (does a stale twin deserve MORE authority for a disagreeing fresh
  observation, or LESS?) is not specified anywhere in this repository,
  and guessing a direction would be exactly the kind of unsupported term
  this module must not introduce. It remains available on
  `DigitalTwinState` for a future, evidence-backed extension.
- Temporal/multi-step consistency tracking (e.g. an EWMA of A_t across a
  sequence) is not implemented. This module computes a single-step
  authority from one (residual, uncertainty) pair; sequencing multiple
  steps is a documented future extension, not implemented here.
- No CSI/SNR/RSSI/path_loss/beam_index is read, referenced, or required
  anywhere in this module.

============================================================================
2026-09-19 DESIGN-REVIEW CORRECTION (see claude/project_status.md for the
full stress test) -- two confirmed defects in the original pooled
std-based formulation, fixed below without changing the exp(-0.5 z^2)
mapping itself:
============================================================================

DEFECT 1 -- UNCERTAINTY INFLATION ("Case Y"). The original
`rolling_uncertainty` (plain standard deviation of recent residual
norms) has a 0% breakdown point: ONE unrelated large residual sitting
anywhere in the recent window inflates sigma_t enough that a separate,
genuinely bad CURRENT residual is judged "within normal spread" and
scored with artificially high authority. Demonstrated numerically:
window=[0.05,0.05,6.0,0.05,0.05], current_residual=1.5 ->
std-based authority = 0.819871 (dangerously high) vs. a "clean" window
[1.4,1.5,1.6,1.5,1.4] with the SAME current_residual=1.5 ->
std-based authority = 0.000000. Identical current evidence, wildly
different authority, purely from one irrelevant past sample.

FIX: `robust_spread()` replaces `rolling_uncertainty()` as the
uncertainty source. It uses the Median Absolute Deviation (MAD, scaled
by 1.4826 to be a consistent estimator of standard deviation under a
Gaussian assumption -- the standard, textbook MAD-to-sigma conversion,
not an invented constant), which has a 50% breakdown point: a single
outlier cannot move it far. Re-running the same case:
window=[0.05,0.05,6.0,0.05,0.05] -> MAD-spread=0.0 (the outlier is
correctly recognized as unrepresentative) -> authority collapses to
~0.000000 for the same bad current_residual=1.5, matching the "clean
window" case exactly. `rolling_uncertainty()` (std-based) is KEPT,
unchanged, as a general-purpose utility with its own valid tests -- it
is simply no longer the recommended uncertainty source for GDCA.

DISTINGUISHING "TYPICAL LEVEL" FROM "SPREAD" (Case C / Case X). MAD is
computed as the median of |value - median(window)| -- i.e. deviations
from a robust TYPICAL LEVEL (the median), not from zero. This
explicitly separates two concepts the original formulation conflated:
the window's typical residual level (its median) vs. the spread AROUND
that level (the MAD). Critically, the FINAL authority computation still
normalizes the RAW CURRENT residual (not residual-minus-median) by that
spread -- it does NOT subtract out a learned bias before judging the
current sample. This is intentional and required: a predictor that is
perfectly CONSISTENT but SYSTEMATICALLY WRONG (e.g. every residual
exactly 3.0) has spread=0 (MAD of a constant window is 0) and therefore
still collapses to authority~=0 -- "consistent" is never allowed to
mean "accurate." Verified numerically: constant-bias window [3.0]*5,
current_residual=3.0 -> MAD-spread=0.0 -> authority=0.0, unchanged from
the original formulation's (correct, for the wrong underlying reason)
near-zero result on this case.

DEFECT 2 -- MIXED-UNIT DOMINANCE ("Case G"). The original
`residual_norm()` pools all state dimensions into one Euclidean norm
before any normalization. If those dimensions have different physical
units (e.g. position in meters, velocity in m/s, heading in radians),
whichever dimension happens to have the largest RAW NUMERIC magnitude
dominates the pooled norm regardless of whether it is, relative to its
OWN natural variability, actually the most anomalous dimension.

FIX: `standardize_residual_dimensions()` + `combine_standardized_residuals()`
(wired together by `compute_authority_from_dimensional_evidence()`)
normalize EACH dimension by its OWN independently-estimated spread
BEFORE combining -- z_i = residual_i / (spread_i + eps) for each
dimension i, then combined via RMS (root-mean-square) of the
per-dimension z-scores, a simple, weight-free, parameter-free
combination (no learned weights, no covariance matrix, no Mahalanobis
distance -- exactly what the design review's Part 3 evidence supports
and no more). The combined RMS z-score is then passed through the SAME
`compute_authority_from_evidence` mapping (exp(-0.5 z^2)) used
throughout this module -- only how z is constructed changed, not the
authority mapping itself.

`compute_residual()`, `residual_norm()`, and
`compute_authority_from_evidence()` are UNCHANGED and remain valid for
the case they were always correct for: dimensions that already share
real, comparable units (e.g. the position-only x/y/z residual used in
this module's real-data demonstration). They are not deprecated, only
no longer recommended for genuinely mixed-unit state vectors.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

from digital_twin.state import DigitalTwinState

#: Small constant preventing division by zero when uncertainty == 0.0
#: (a genuinely possible input: zero observed spread). Not a calibrated
#: parameter -- purely a numerical-stability guard.
_EPS = 1e-9


class MissingGDCAEvidenceError(RuntimeError):
    """Raised by `estimate(observed_state, predicted_state)` when the
    uncertainty scale required by `compute_authority_from_evidence`
    cannot be obtained from a single state pair alone.

    See module docstring ("WHY estimate() ALWAYS RAISES"). Use
    `compute_authority_from_evidence(residual_norm, uncertainty)` with an
    uncertainty explicitly derived from a window of recent residuals
    (e.g. via `rolling_uncertainty`) instead of trying to make
    `estimate()` succeed by guessing a value.
    """


class BaseGDCAEstimator(ABC):
    """Contract every concrete GDCA implementation must satisfy.

    Deliberately a 2-argument contract (`observed_state, predicted_state`)
    rather than the 1-argument `estimate(state)` shape
    `BaseTrustEstimator`/`BaseCriticalityEstimator` use — GDCA's entire
    purpose is comparing two states (an innovation/residual), which a
    single `DigitalTwinState` snapshot cannot express on its own.
    """

    @abstractmethod
    def estimate(
        self, observed_state: DigitalTwinState, predicted_state: DigitalTwinState
    ) -> float:
        """
        Compute the GDCA authority score for a fresh observation against
        the Digital Twin's predicted state.

        Parameters
        ----------
        observed_state : DigitalTwinState
            The fresh, real observation (e.g. from CARLA ground truth).
        predicted_state : DigitalTwinState
            The Digital Twin's predicted state at the same timestep.

        Returns
        -------
        float
            Authority score A_t, in the closed interval [0, 1].
        """
        raise NotImplementedError


def _require_finite_sequence(values: Sequence[float], name: str) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must have at least one dimension")
    for v in values:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError(f"{name} must contain only finite numbers, got {v!r} in {name}")


def compute_residual(observed: Sequence[float], predicted: Sequence[float]) -> tuple[float, ...]:
    """Innovation r_t = observed - predicted, dimension-wise.

    Parameters
    ----------
    observed, predicted : Sequence[float]
        Same-length real-valued state vectors (e.g. (x, y) or (x, y, z)
        position). Must be non-empty and finite.

    Returns
    -------
    tuple[float, ...]
    """
    _require_finite_sequence(observed, "observed")
    _require_finite_sequence(predicted, "predicted")
    if len(observed) != len(predicted):
        raise ValueError(
            f"observed and predicted must have the same number of dimensions, "
            f"got {len(observed)} vs {len(predicted)}"
        )
    return tuple(o - p for o, p in zip(observed, predicted))


def residual_norm(residual: Sequence[float]) -> float:
    """Euclidean (L2) norm of a residual vector, ||r_t||."""
    _require_finite_sequence(residual, "residual")
    return math.sqrt(sum(r * r for r in residual))


def rolling_uncertainty(recent_residual_norms: Sequence[float]) -> float:
    """Windowed standard deviation of recent residual norms -- the
    uncertainty scale sigma_t consumed by
    `compute_authority_from_evidence`.

    Same technique as
    `simulation/annotation/future_channel_labeler.py`'s
    `_rolling_uncertainty` (population std of a recent window),
    generalized from CSI magnitude to any residual-norm sequence.
    Requires >= 2 samples: a single sample has no defined spread.
    """
    _require_finite_sequence(recent_residual_norms, "recent_residual_norms")
    if len(recent_residual_norms) < 2:
        raise ValueError("rolling_uncertainty requires at least 2 samples")
    n = len(recent_residual_norms)
    mean = sum(recent_residual_norms) / n
    variance = sum((v - mean) ** 2 for v in recent_residual_norms) / n
    return math.sqrt(variance)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


#: Standard MAD-to-sigma conversion factor for a Gaussian distribution
#: (1 / Phi^-1(0.75)) -- a textbook constant, not tuned for this project.
_MAD_TO_SIGMA = 1.4826


def residual_level_and_spread(recent_values: Sequence[float]) -> tuple[float, float]:
    """Split a window of recent residual (norm or per-dimension) values
    into its two distinct components: a robust TYPICAL LEVEL (median) and
    a robust SPREAD around that level (Median Absolute Deviation, scaled
    to be a consistent sigma estimator). See module docstring ("2026-09-19
    DESIGN-REVIEW CORRECTION") for why this replaces plain mean/std as
    GDCA's uncertainty source: MAD's 50% breakdown point means a single
    unrelated outlier in the window cannot inflate it the way it can a
    standard deviation.

    Returns
    -------
    (level, spread) : tuple[float, float]
        `level` is median(recent_values); `spread` is
        1.4826 * median(|v - level| for v in recent_values). Both are
        >= 0 if `recent_values` are residual norms/magnitudes (>= 0 by
        construction); `spread` is always >= 0 regardless.
    """
    _require_finite_sequence(recent_values, "recent_values")
    if len(recent_values) < 2:
        raise ValueError("residual_level_and_spread requires at least 2 samples")
    level = _median(recent_values)
    abs_deviations = [abs(v - level) for v in recent_values]
    spread = _MAD_TO_SIGMA * _median(abs_deviations)
    return level, spread


def robust_spread(recent_values: Sequence[float]) -> float:
    """The spread component of `residual_level_and_spread` alone -- the
    uncertainty scale sigma_t GDCA now uses in place of
    `rolling_uncertainty`. See `residual_level_and_spread`'s docstring.
    """
    _, spread = residual_level_and_spread(recent_values)
    return spread


def standardize_residual_dimensions(
    residuals: Sequence[float], spreads: Sequence[float]
) -> tuple[float, ...]:
    """Per-dimension standardization: z_i = residual_i / (spread_i + eps).

    Each dimension is normalized by its OWN independently-estimated
    spread (e.g. from `robust_spread` applied to that dimension's own
    recent residual history) before any combination -- this is the Case G
    fix (see module docstring): a dimension with large raw numeric units
    (e.g. velocity in m/s) no longer dominates a dimension with small raw
    units (e.g. position in m) merely because of those units. It only
    dominates if it is, relative to its OWN natural spread, genuinely
    more inconsistent.

    Parameters
    ----------
    residuals : Sequence[float]
        One residual value per state dimension (observed_i - predicted_i).
    spreads : Sequence[float]
        One spread value per state dimension, same length and order as
        `residuals`, each >= 0.

    Returns
    -------
    tuple[float, ...]
        Per-dimension standardized residuals (z-scores), same length as
        the inputs.
    """
    _require_finite_sequence(residuals, "residuals")
    _require_finite_sequence(spreads, "spreads")
    if len(residuals) != len(spreads):
        raise ValueError(
            f"residuals and spreads must have the same number of dimensions, "
            f"got {len(residuals)} vs {len(spreads)}"
        )
    for s in spreads:
        if s < 0:
            raise ValueError(f"each dimension's spread must be >= 0, got {s!r}")
    return tuple(r / (s + _EPS) for r, s in zip(residuals, spreads))


def combine_standardized_residuals(standardized: Sequence[float]) -> float:
    """Root-mean-square (RMS) combination of per-dimension standardized
    residuals into one scalar z_t.

    Deliberately the simplest weight-free, parameter-free combination
    (no learned weights, no covariance matrix, no Mahalanobis distance --
    see module docstring's Case G fix and the design review's Part 3,
    which found no repository evidence to justify anything more complex).
    """
    _require_finite_sequence(standardized, "standardized")
    n = len(standardized)
    return math.sqrt(sum(z * z for z in standardized) / n)


def compute_authority_from_dimensional_evidence(
    residuals: Sequence[float], spreads: Sequence[float]
) -> float:
    """The corrected, dimension-aware GDCA authority computation (fixes
    both Case Y and Case G -- see module docstring's "2026-09-19
    DESIGN-REVIEW CORRECTION").

    residuals[i] should be observed_i - predicted_i for state dimension i;
    spreads[i] should be `robust_spread` applied to that SAME dimension's
    own recent residual history (never a pooled/cross-dimension value).

    Pipeline: standardize_residual_dimensions -> combine_standardized_residuals
    (RMS) -> compute_authority_from_evidence(z, uncertainty=1.0) (the
    combined z is already a standardized, unitless quantity, so it is
    passed through the SAME exp(-0.5 z^2) mapping with uncertainty fixed
    at 1.0 -- reusing that function rather than duplicating the mapping).

    Returns
    -------
    float
        Authority score A_t, in (0, 1] -- same range and mapping as
        `compute_authority_from_evidence`.
    """
    z_per_dimension = standardize_residual_dimensions(residuals, spreads)
    z_combined = combine_standardized_residuals(z_per_dimension)
    return compute_authority_from_evidence(z_combined, 1.0)


def compute_authority_from_evidence(residual_norm: float, uncertainty: float) -> float:
    """
    A_t = exp(-0.5 * (residual_norm / (uncertainty + eps))^2)

    Parameters
    ----------
    residual_norm : float
        ||observed - predicted||, >= 0 (see `residual_norm()` above).
    uncertainty : float
        sigma_t, the uncertainty scale the residual is judged against,
        >= 0 (see `rolling_uncertainty()` above). uncertainty == 0.0 is
        valid (a perfectly stable recent history) and handled via `_EPS`.

    Returns
    -------
    float
        Authority score A_t, in (0, 1] (within the required [0, 1]
        range; never exactly 0, since exp(x) > 0 for all finite x).

    Raises
    ------
    ValueError
        If either input is non-finite or negative.
    """
    if not isinstance(residual_norm, (int, float)) or isinstance(residual_norm, bool) or not math.isfinite(residual_norm):
        raise ValueError(f"residual_norm must be a finite number, got {residual_norm!r}")
    if not isinstance(uncertainty, (int, float)) or isinstance(uncertainty, bool) or not math.isfinite(uncertainty):
        raise ValueError(f"uncertainty must be a finite number, got {uncertainty!r}")
    if residual_norm < 0:
        raise ValueError(f"residual_norm must be >= 0 (it is a norm), got {residual_norm!r}")
    if uncertainty < 0:
        raise ValueError(f"uncertainty must be >= 0, got {uncertainty!r}")

    z = residual_norm / (uncertainty + _EPS)
    return math.exp(-0.5 * z * z)


class GDCAEstimator(BaseGDCAEstimator):
    """Concrete `BaseGDCAEstimator`. See module docstring for why
    `estimate()` always raises `MissingGDCAEvidenceError` -- the real,
    testable math lives in `compute_authority_from_evidence`."""

    def estimate(
        self, observed_state: DigitalTwinState, predicted_state: DigitalTwinState
    ) -> float:
        raise MissingGDCAEvidenceError(
            "GDCAEstimator.estimate(observed_state, predicted_state) cannot compute "
            "an authority score from a single state pair alone: the uncertainty/spread "
            "scale requires a window of recent residuals (see robust_spread), which one "
            "(observed, predicted) pair cannot supply. Use "
            "compute_authority_from_evidence(residual_norm, uncertainty) or, for "
            "mixed-unit state vectors, compute_authority_from_dimensional_evidence("
            "residuals, spreads) with explicitly supplied evidence instead of calling "
            "estimate()."
        )
