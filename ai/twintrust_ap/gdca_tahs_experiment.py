"""
GDCA Authority -> TAHS substitution EXPERIMENT (Option E), Phase 5.

Implements ONLY the near-term experiment approved by the 2026-09-19
Trust/GDCA/Criticality/TAHS/FSDP architecture design review
(`claude/project_status.md`): pass GDCA's real-data Authority score
through TAHS's existing `trust: float` parameter, as an EXPLICITLY
LOGGED SUBSTITUTION, because real Trust evidence (`ai/trust_estimator/`)
is currently structurally unavailable on every real CARLA scene (its own
`estimate(state)` always raises `MissingTrustEvidenceError` -- see that
module's docstring).

THIS IS NOT:
- A redefinition of Trust. `ai/trust_estimator/` is untouched by this
  module and is never imported here.
- A change to TAHS's interface or math. `ai.twintrust_ap.tahs.TAHS` is
  imported and called completely unmodified --
  `select_horizon(trust: float, criticality: float) -> int`, exactly as
  it already exists.
- A change to GDCA's mathematics. `ai.twintrust_ap.gdca`'s functions are
  imported and called completely unmodified.
- A claim that GDCA improves prediction, communication, or control
  performance, or that Trust and GDCA have been empirically compared.
  This experiment only establishes whether the EXISTING GDCA Authority
  can meaningfully drive the EXISTING TAHS horizon mechanism.

Every result row is explicitly labeled `authority_as_trust_substitution`
-- never `trust` -- so the substitution can never be silently mistaken
for a real Trust score downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ai.twintrust_ap.gdca import (
    compute_authority_from_evidence,
    compute_residual,
    residual_norm,
    robust_spread,
)
from ai.twintrust_ap.tahs import TAHS


@dataclass(frozen=True)
class SubstitutionExperimentRow:
    """One (frame, criticality) result of the Option E substitution
    experiment. `authority_as_trust_substitution` is deliberately named
    to make the substitution impossible to mistake for a real Trust
    score if this row is logged, printed, or serialized anywhere."""

    frame_index: int
    authority_as_trust_substitution: float
    criticality: float
    horizon: int


def compute_real_authority_sequence(
    position_sequence: Sequence[tuple[int, tuple[float, ...]]],
    window: int = 5,
) -> list[tuple[int, float]]:
    """(frame_index, authority) pairs from a real (or synthetic, for
    tests) position sequence, using the SAME naive-persistence-baseline +
    robust-spread mechanism already validated in
    `claude/project_status.md`'s GDCA real-data demonstration --
    `predicted_state(t) = observed_state(t-1)` (zero-order-hold, NOT a
    trained predictor -- no trained Digital-Twin predictor exists
    anywhere in this repository).

    Parameters
    ----------
    position_sequence : Sequence[tuple[int, tuple[float, ...]]]
        (frame_index, position_xyz) pairs, in ascending frame order.
    window : int
        Rolling window size for `robust_spread`, matching the prior GDCA
        real-data checkpoint's window=5.

    Returns
    -------
    list[tuple[int, float]]
        (frame_index, authority) for every frame after the first
        `window` samples have accumulated (mirrors the prior GDCA
        real-data demonstration's warm-up behavior exactly).
    """
    if len(position_sequence) < 2:
        raise ValueError("compute_real_authority_sequence requires at least 2 samples")

    norms: list[float] = []
    results: list[tuple[int, float]] = []
    for i in range(1, len(position_sequence)):
        frame_index, observed = position_sequence[i]
        _, predicted = position_sequence[i - 1]
        r = compute_residual(observed, predicted)
        n = residual_norm(r)
        norms.append(n)
        if len(norms) < 2:
            continue
        recent_window = norms[max(0, len(norms) - window):]
        spread = robust_spread(recent_window)
        authority = compute_authority_from_evidence(n, spread)
        results.append((frame_index, authority))
    return results


def run_gdca_tahs_substitution_experiment(
    authority_sequence: Sequence[tuple[int, float]],
    criticality_values: Sequence[float],
    tahs: TAHS | None = None,
) -> list[SubstitutionExperimentRow]:
    """
    Run the Option E substitution experiment: for every (frame, authority)
    pair and every controlled criticality value, call the EXISTING,
    UNMODIFIED `TAHS.select_horizon(trust=authority, criticality=c)`.

    `criticality_values` must be explicitly supplied, controlled constants
    (e.g. 0.0 / 0.5 / 1.0) -- this function never fabricates or derives
    Criticality from real evidence; the real Criticality Estimator
    (`ai/criticality/`) remains untouched and unused here.

    Parameters
    ----------
    authority_sequence : Sequence[tuple[int, float]]
        (frame_index, authority) pairs, e.g. from
        `compute_real_authority_sequence`. Each `authority` must already
        be in [0, 1] (guaranteed by `compute_authority_from_evidence`'s
        own contract).
    criticality_values : Sequence[float]
        Controlled experimental constants, each in [0, 1].
    tahs : TAHS | None
        The TAHS instance to use (defaults to `TAHS()`, i.e. the real
        `configs/model.yaml`-driven parameters). Never modified.

    Returns
    -------
    list[SubstitutionExperimentRow]
    """
    engine = tahs if tahs is not None else TAHS()
    rows: list[SubstitutionExperimentRow] = []
    for frame_index, authority in authority_sequence:
        for criticality in criticality_values:
            horizon = engine.select_horizon(trust=authority, criticality=criticality)
            rows.append(
                SubstitutionExperimentRow(
                    frame_index=frame_index,
                    authority_as_trust_substitution=authority,
                    criticality=criticality,
                    horizon=horizon,
                )
            )
    return rows
