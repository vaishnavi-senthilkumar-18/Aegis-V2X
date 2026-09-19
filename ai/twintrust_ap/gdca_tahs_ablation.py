"""
GDCA-only ablation: Authority arm vs. neutral-control arm, Phase 5.

Ablates the already-committed GDCA -> TAHS substitution experiment
(`ai/twintrust_ap/gdca_tahs_experiment.py`, commit `6dad0ee`) by adding
an explicit control arm, per the 2026-09-20 Phase 5 architecture
review's recommended next experiment: for every real `(frame, authority)`
row already produced by that experiment's own pipeline, TAHS is
evaluated BOTH with GDCA Authority substituted into its `trust`
parameter (ARM A, identical in method to the committed experiment) AND
with a fixed NEUTRAL trust value (ARM B), so the two resulting horizons
can be directly compared, row for row.

Goal: determine whether GDCA Authority actually changes TAHS's horizon
decision relative to a neutral/no-signal baseline -- not merely whether
Authority "propagates" through TAHS (already shown by the committed
experiment), but whether it does anything DIFFERENT from a value that
carries no directional information at all.

NEUTRAL CONTROL VALUE = 0.5, chosen from existing repository evidence,
not invented:
------------------------------------------------------------------------
- `ai/twintrust_ap/tahs.py`'s own `trust` parameter is documented as a
  probability in the closed interval [0, 1]; 0.5 is the exact midpoint
  of that domain. `trust=0.0` already carries a strong, specific
  meaning in TAHS's own semantics ("zero trust" -- the value that most
  shortens the horizon), so it cannot serve as a "no information" or
  "neutral" baseline; 0.5 is the only point in [0,1] with no directional
  bias toward either extreme.
- `configs/model.yaml`'s `trust_estimator.calibration_bins` (read only
  for this justification; NOT loaded or modified by this module) labels
  the band `[0.4, 0.6]` "moderate confidence" -- the repository's own
  trust calibration scheme already treats the region around 0.5 as the
  neutral/moderate point on the same [0,1] scale TAHS consumes, distinct
  from "very unreliable" (`[0.0, 0.2]`) or "highly reliable"
  (`[0.8, 1.0]`).
- Mathematically, in TAHS's `H_t = H_min + (H_max-H_min) *
  sigmoid(beta*trust - gamma*criticality)`, `trust=0.5` is the value
  around which the trust-driven contribution to the sigmoid argument is
  symmetric (`beta*0.5` is exactly half of `beta*1.0`), so it neither
  favors the shortest nor the longest achievable horizon from trust's
  side of the formula.

This module reuses `ai.twintrust_ap.gdca` (via
`ai.twintrust_ap.gdca_tahs_experiment.compute_real_authority_sequence`,
itself unmodified) and `ai.twintrust_ap.tahs.TAHS` completely
unmodified. The already-committed
`ai/twintrust_ap/gdca_tahs_experiment.py` and
`tests/unit/test_gdca_tahs_substitution_experiment.py` are untouched by
this module.

Field naming: every row exposes `authority_as_trust_substitution` (never
a bare `trust` field), plus explicit `gdca_arm_label`/`control_arm_label`
strings, so neither arm's value can be mistaken for a real Trust score
downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ai.twintrust_ap.gdca_tahs_experiment import compute_real_authority_sequence
from ai.twintrust_ap.tahs import TAHS

#: See module docstring for the full justification.
NEUTRAL_CONTROL_TRUST_VALUE: float = 0.5

GDCA_ARM_LABEL = "gdca_authority_substitution"
CONTROL_ARM_LABEL = "neutral_control_trust_0.5"


@dataclass(frozen=True)
class AblationRow:
    """One (frame, criticality) result of the GDCA-only ablation.

    `authority_as_trust_substitution` mirrors the committed substitution
    experiment's own field naming exactly, for consistency; it is never
    named `trust`. `gdca_arm_label`/`control_arm_label` make each arm's
    identity explicit even if a row is logged or serialized in
    isolation.
    """

    frame_id: int
    authority: float
    criticality: float
    gdca_horizon: int
    control_horizon: int
    horizon_delta: int
    authority_as_trust_substitution: float
    gdca_arm_label: str = field(default=GDCA_ARM_LABEL)
    control_arm_label: str = field(default=CONTROL_ARM_LABEL)


def run_gdca_ablation(
    authority_sequence: Sequence[tuple[int, float]],
    criticality_values: Sequence[float],
    tahs: TAHS | None = None,
    neutral_control_value: float = NEUTRAL_CONTROL_TRUST_VALUE,
) -> list[AblationRow]:
    """
    Run the GDCA-only ablation: for every `(frame, authority)` pair and
    every controlled criticality value, call the EXISTING, UNMODIFIED
    `TAHS.select_horizon` twice -- once with `trust=authority` (ARM A)
    and once with `trust=neutral_control_value` (ARM B) -- and record
    both horizons plus their difference.

    Parameters
    ----------
    authority_sequence : Sequence[tuple[int, float]]
        (frame_id, authority) pairs, e.g. from
        `ai.twintrust_ap.gdca_tahs_experiment.compute_real_authority_sequence`.
        Each `authority` must already be in [0, 1].
    criticality_values : Sequence[float]
        Controlled experimental constants (e.g. 0.0 / 0.5 / 1.0), each in
        [0, 1] -- never derived from `ai.criticality`.
    tahs : TAHS | None
        The TAHS instance to use (defaults to `TAHS()`). Never modified.
    neutral_control_value : float
        ARM B's fixed trust input. Defaults to
        `NEUTRAL_CONTROL_TRUST_VALUE` (0.5) -- see module docstring.

    Returns
    -------
    list[AblationRow]
    """
    engine = tahs if tahs is not None else TAHS()
    rows: list[AblationRow] = []
    for frame_id, authority in authority_sequence:
        for criticality in criticality_values:
            gdca_horizon = engine.select_horizon(trust=authority, criticality=criticality)
            control_horizon = engine.select_horizon(
                trust=neutral_control_value, criticality=criticality
            )
            rows.append(
                AblationRow(
                    frame_id=frame_id,
                    authority=authority,
                    criticality=criticality,
                    gdca_horizon=gdca_horizon,
                    control_horizon=control_horizon,
                    horizon_delta=gdca_horizon - control_horizon,
                    authority_as_trust_substitution=authority,
                )
            )
    return rows


__all__ = [
    "NEUTRAL_CONTROL_TRUST_VALUE",
    "GDCA_ARM_LABEL",
    "CONTROL_ARM_LABEL",
    "AblationRow",
    "run_gdca_ablation",
    "compute_real_authority_sequence",
]
