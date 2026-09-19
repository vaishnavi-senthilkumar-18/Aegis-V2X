"""Tests for `ai.twintrust_ap.gdca_tahs_ablation` -- the GDCA-only
ablation (Authority arm vs. neutral-control arm) recommended by the
2026-09-20 Phase 5 architecture review, ablating the already-committed
GDCA -> TAHS substitution experiment (`gdca_tahs_experiment.py`,
commit `6dad0ee`).

SYNTHETIC INPUTS: values here are plain floats/tuples chosen to exercise
the ablation mechanism's correctness, determinism, and mathematically
expected horizon relationships -- NOT measurements from any real CARLA
scene. The real 92-frame `Vehicle147` ablation run is exercised
separately, offline from pytest (see `claude/project_status.md`'s dated
entry for that run's results), for the same reason the original
substitution experiment's real-data run is offline: it requires a live
backend connection, and this suite must remain fully deterministic and
reproducible without one.

Neither `ai.trust_estimator` nor `ai.criticality` is imported anywhere in
this file -- no real Trust or Criticality evidence is fabricated.
"""

from __future__ import annotations

import math

import pytest

from ai.twintrust_ap.gdca_tahs_ablation import (
    CONTROL_ARM_LABEL,
    GDCA_ARM_LABEL,
    NEUTRAL_CONTROL_TRUST_VALUE,
    AblationRow,
    run_gdca_ablation,
)
from ai.twintrust_ap.tahs import TAHS

CONTROLLED_CRITICALITY_VALUES = (0.0, 0.5, 1.0)
ALLOWED_HORIZONS = {1, 2, 3, 5, 8, 10}


# --------------------------------------------------------------------------
# Neutral control value sanity
# --------------------------------------------------------------------------


def test_neutral_control_value_is_the_domain_midpoint():
    assert NEUTRAL_CONTROL_TRUST_VALUE == 0.5


def test_control_horizon_matches_direct_tahs_call_with_neutral_value():
    """The ablation's control arm must produce exactly what calling TAHS
    directly with the neutral value produces -- no separate/duplicated
    TAHS logic."""
    tahs = TAHS()
    rows = run_gdca_ablation([(1, 0.73)], [0.42], tahs=tahs)
    direct_control = tahs.select_horizon(trust=NEUTRAL_CONTROL_TRUST_VALUE, criticality=0.42)
    assert rows[0].control_horizon == direct_control


def test_gdca_horizon_matches_direct_tahs_call_with_authority():
    tahs = TAHS()
    rows = run_gdca_ablation([(1, 0.73)], [0.42], tahs=tahs)
    direct_gdca = tahs.select_horizon(trust=0.73, criticality=0.42)
    assert rows[0].gdca_horizon == direct_gdca


# --------------------------------------------------------------------------
# Row structure / labeling
# --------------------------------------------------------------------------


def test_rows_are_ablation_row_instances_with_required_fields():
    rows = run_gdca_ablation([(1, 0.5)], [0.5])
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, AblationRow)
    assert row.frame_id == 1
    assert row.authority == 0.5
    assert row.criticality == 0.5
    assert hasattr(row, "gdca_horizon")
    assert hasattr(row, "control_horizon")
    assert hasattr(row, "horizon_delta")


def test_rows_never_use_a_bare_trust_field():
    rows = run_gdca_ablation([(1, 0.5)], [0.5])
    assert hasattr(rows[0], "authority_as_trust_substitution")
    assert not hasattr(rows[0], "trust")


def test_rows_carry_explicit_arm_labels():
    rows = run_gdca_ablation([(1, 0.5)], [0.5])
    assert rows[0].gdca_arm_label == GDCA_ARM_LABEL
    assert rows[0].control_arm_label == CONTROL_ARM_LABEL
    assert rows[0].gdca_arm_label != rows[0].control_arm_label


def test_produces_one_row_per_authority_criticality_pair():
    authority_sequence = [(1, 0.1), (2, 0.5), (3, 0.9)]
    rows = run_gdca_ablation(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    assert len(rows) == len(authority_sequence) * len(CONTROLLED_CRITICALITY_VALUES)


# --------------------------------------------------------------------------
# horizon_delta correctness
# --------------------------------------------------------------------------


def test_horizon_delta_equals_gdca_minus_control():
    authority_sequence = [(i, a / 10) for i, a in enumerate(range(11))]
    rows = run_gdca_ablation(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    for row in rows:
        assert row.horizon_delta == row.gdca_horizon - row.control_horizon


def test_authority_equal_to_neutral_value_gives_zero_delta():
    """When authority == the neutral control value exactly, both arms
    call TAHS with the identical trust input, so delta must be exactly 0
    -- a direct sanity check on the ablation's own consistency."""
    rows = run_gdca_ablation([(1, NEUTRAL_CONTROL_TRUST_VALUE)], CONTROLLED_CRITICALITY_VALUES)
    for row in rows:
        assert row.horizon_delta == 0


# --------------------------------------------------------------------------
# Validity / boundedness / determinism
# --------------------------------------------------------------------------


def test_both_arms_produce_only_discretized_horizons():
    authority_sequence = [(i, a / 10) for i, a in enumerate(range(11))]
    rows = run_gdca_ablation(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    for row in rows:
        assert row.gdca_horizon in ALLOWED_HORIZONS
        assert row.control_horizon in ALLOWED_HORIZONS


def test_ablation_is_deterministic():
    authority_sequence = [(1, 0.2), (2, 0.6), (3, 0.95)]
    rows1 = run_gdca_ablation(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    rows2 = run_gdca_ablation(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    assert rows1 == rows2


# --------------------------------------------------------------------------
# Mathematically expected GDCA-vs-control relationship (TAHS monotonicity
# implies these directions; not invented, derived from TAHS's own
# already-tested invariant)
# --------------------------------------------------------------------------


def test_authority_above_neutral_never_yields_a_shorter_horizon_than_control():
    """Since TAHS.select_horizon is monotonically non-decreasing in trust
    (test_tahs_monotonicity.py, unchanged), any authority > 0.5 must
    produce gdca_horizon >= control_horizon (control uses trust=0.5) for
    the SAME criticality."""
    above_neutral = [0.6, 0.7, 0.8, 0.9, 1.0]
    for criticality in CONTROLLED_CRITICALITY_VALUES:
        rows = run_gdca_ablation([(i, a) for i, a in enumerate(above_neutral)], [criticality])
        for row in rows:
            assert row.gdca_horizon >= row.control_horizon


def test_authority_below_neutral_never_yields_a_longer_horizon_than_control():
    below_neutral = [0.0, 0.1, 0.2, 0.3, 0.4]
    for criticality in CONTROLLED_CRITICALITY_VALUES:
        rows = run_gdca_ablation([(i, a) for i, a in enumerate(below_neutral)], [criticality])
        for row in rows:
            assert row.gdca_horizon <= row.control_horizon


def test_high_authority_produces_longer_horizon_than_control_when_it_matters():
    """Concrete, hand-verified case: authority=1.0, criticality=0.5 ->
    gdca_horizon=8, control_horizon=5 (control uses trust=0.5,
    criticality=0.5 -> continuous_h=5.5 -> nearest=5; authority=1.0,
    criticality=0.5 -> continuous_h=7.58 -> nearest=8). A genuine,
    nonzero, positive delta -- not merely a >= 0 inequality."""
    rows = run_gdca_ablation([(1, 1.0)], [0.5])
    assert rows[0].gdca_horizon == 8
    assert rows[0].control_horizon == 5
    assert rows[0].horizon_delta == 3


def test_zero_authority_reduces_horizon_relative_to_control_when_it_matters():
    """Concrete, hand-verified case: authority=0.0, criticality=1.0 ->
    gdca_horizon=3, control_horizon=5 (control: trust=0.5,
    criticality=1.0 -> continuous_h=1+9*sigmoid(-0.5)=1+9*0.3775=4.4 ->
    nearest=5; authority=0.0, criticality=1.0 ->
    continuous_h=1+9*sigmoid(-1.0)=1+9*0.2689=3.42 -> nearest=3). A
    genuine, nonzero, negative delta."""
    rows = run_gdca_ablation([(1, 0.0)], [1.0])
    assert rows[0].gdca_horizon == 3
    assert rows[0].control_horizon == 5
    assert rows[0].horizon_delta == -2


# --------------------------------------------------------------------------
# Real-shaped (synthetic) sequence: effect persists across criticality
# --------------------------------------------------------------------------


def _stationary_then_moving_sequence() -> list[tuple[int, float]]:
    """Directly-specified synthetic authority sequence shaped like the
    real Vehicle147 trajectory's own high-then-zero authority pattern
    (see gdca_tahs_experiment.py's own synthetic test fixture for the
    position-level construction this mirrors) -- avoids re-deriving
    positions here since only the authority values matter for this
    ablation-level test."""
    return [(i, a) for i, a in enumerate([0.8, 0.95, 0.99, 0.72, 0.9, 0.0, 0.0, 0.0])]


def test_effect_persists_across_all_three_controlled_criticality_values():
    """Nonzero horizon_delta rows must exist at EACH of the 3 controlled
    criticality values for a sequence with genuine authority variation --
    confirms the ablation effect is not an artifact of one particular
    criticality choice."""
    authority_sequence = _stationary_then_moving_sequence()
    for criticality in CONTROLLED_CRITICALITY_VALUES:
        rows = run_gdca_ablation(authority_sequence, [criticality])
        nonzero_deltas = [r for r in rows if r.horizon_delta != 0]
        assert len(nonzero_deltas) > 0, f"no nonzero delta at criticality={criticality}"
