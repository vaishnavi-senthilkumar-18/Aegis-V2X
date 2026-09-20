"""
Trust + Criticality + GDCA Authority -> TAHS three-input EXPERIMENT,
Phase 5.

Implements ONLY the approved three-input TAHS integration (2026-09-20
TAHS three-input design audit and implementation approval,
`claude/project_status.md`): exercise the extended
`TAHS.select_horizon(trust, criticality, authority)` using

    Authority  = REAL GDCA output (real 92-frame position sequence,
                 identical mechanism/window to
                 `ai/twintrust_ap/gdca_tahs_experiment.py`, reused via
                 import, unmodified)
    Trust      = CONTROLLED EXPERIMENTAL CONSTANT (never a real Trust
                 estimate -- `ai/trust_estimator/` remains untouched and
                 unused here, exactly as in every prior Phase 5 TAHS
                 experiment)
    Criticality = CONTROLLED EXPERIMENTAL CONSTANT (never a real
                 Criticality estimate -- `ai/criticality/` remains
                 untouched and unused here)

THIS IS NOT:
- A modification of GDCA's mathematics. `ai.twintrust_ap.gdca` functions
  are imported and used completely unmodified (via
  `gdca_tahs_experiment.compute_real_authority_sequence`, itself also
  unmodified).
- A modification of TAHS's mathematics. `ai.twintrust_ap.tahs.TAHS` is
  imported and called exactly as its Phase 5 three-input extension
  already defines it -- no interface or formula change happens in this
  module.
- A replacement for, or modification of, the existing GDCA-only ->
  TAHS `trust`-slot substitution experiment
  (`ai/twintrust_ap/gdca_tahs_experiment.py`, Option E) or its ablation
  (`ai/twintrust_ap/gdca_tahs_ablation.py`). Both remain untouched,
  standalone, and valid as their own explicitly-labeled
  authority-as-trust-substitution / ablation experiments. This module is
  a DIFFERENT experiment: Authority enters TAHS's own `authority`
  parameter, never the `trust` parameter, so Trust and Authority are
  never set to the same GDCA value in the same call.
- A claim that Trust or Criticality are real-data estimates. Every
  result row explicitly records `trust_source="controlled_constant"` and
  `criticality_source="controlled_constant"` alongside
  `authority_source="real_gdca"`, so no downstream consumer can mistake
  the controlled constants for real evidence.
- A claim that GDCA improves prediction, communication, or control
  performance, or that delta=1.0 is calibrated. `delta` is the
  PROVISIONAL, UNCALIBRATED architectural coefficient documented in
  `ai/twintrust_ap/tahs.py` and `configs/model.yaml` -- this experiment
  does not tune it and does not claim its value is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ai.twintrust_ap.gdca_tahs_experiment import compute_real_authority_sequence
from ai.twintrust_ap.tahs import TAHS

TRUST_SOURCE_LABEL = "controlled_constant"
CRITICALITY_SOURCE_LABEL = "controlled_constant"
AUTHORITY_SOURCE_LABEL = "real_gdca"


@dataclass(frozen=True)
class ThreeInputExperimentRow:
    """One (frame, trust, criticality) result of the three-input TAHS
    experiment. `authority` is the real GDCA value for this frame;
    `trust`/`criticality` are controlled experimental constants, never
    real estimates -- their `*_source` fields make this explicit even if
    a row is logged, printed, or serialized in isolation."""

    frame_index: int
    trust: float
    criticality: float
    authority: float
    horizon: int
    trust_source: str = TRUST_SOURCE_LABEL
    criticality_source: str = CRITICALITY_SOURCE_LABEL
    authority_source: str = AUTHORITY_SOURCE_LABEL


def run_gdca_tahs_three_input_experiment(
    authority_sequence: Sequence[tuple[int, float]],
    trust_values: Sequence[float],
    criticality_values: Sequence[float],
    tahs: TAHS | None = None,
) -> list[ThreeInputExperimentRow]:
    """
    Run the three-input experiment: for every (frame, authority) pair
    (real GDCA output) and every (trust, criticality) controlled-constant
    combination, call the EXISTING, UNMODIFIED
    `TAHS.select_horizon(trust=trust, criticality=criticality,
    authority=authority)`.

    Parameters
    ----------
    authority_sequence : Sequence[tuple[int, float]]
        (frame_index, authority) pairs, e.g. from
        `ai.twintrust_ap.gdca_tahs_experiment.compute_real_authority_sequence`.
        Each `authority` must already be in [0, 1] (guaranteed by GDCA's
        own contract).
    trust_values : Sequence[float]
        Controlled experimental constants, each in [0, 1]. Never derived
        from `ai/trust_estimator/`, and never set equal to any value in
        `authority_sequence` by this function's own convention (callers
        are responsible for choosing a trust set that does not attempt
        to re-run the trust-slot substitution experiment through this
        module).
    criticality_values : Sequence[float]
        Controlled experimental constants, each in [0, 1]. Never derived
        from `ai/criticality/`.
    tahs : TAHS | None
        The TAHS instance to use (defaults to `TAHS()`, i.e. the real
        `configs/model.yaml`-driven parameters, including the
        provisional `delta`). Never modified.

    Returns
    -------
    list[ThreeInputExperimentRow]
    """
    engine = tahs if tahs is not None else TAHS()
    rows: list[ThreeInputExperimentRow] = []
    for frame_index, authority in authority_sequence:
        for trust in trust_values:
            for criticality in criticality_values:
                horizon = engine.select_horizon(
                    trust=trust, criticality=criticality, authority=authority
                )
                rows.append(
                    ThreeInputExperimentRow(
                        frame_index=frame_index,
                        trust=trust,
                        criticality=criticality,
                        authority=authority,
                        horizon=horizon,
                    )
                )
    return rows


__all__ = [
    "TRUST_SOURCE_LABEL",
    "CRITICALITY_SOURCE_LABEL",
    "AUTHORITY_SOURCE_LABEL",
    "ThreeInputExperimentRow",
    "run_gdca_tahs_three_input_experiment",
    "compute_real_authority_sequence",
]
