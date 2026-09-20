"""Tests for `ai.twintrust_ap.gdca_tahs_trust_criticality_authority_experiment`
-- the three-input Trust + Criticality + GDCA Authority -> TAHS EXPERIMENT
(Phase 5 three-input TAHS integration).

SYNTHETIC INPUTS: the position sequences here are plain, hand-constructed
tuples (same shape as
`tests/unit/test_gdca_tahs_substitution_experiment.py`'s own synthetic
sequence) chosen to exercise the three-input mechanism's monotonicity/
boundedness/determinism -- NOT measurements from any real CARLA scene.

Every assertion in this file treats `ai.trust_estimator` and
`ai.criticality` as OUT OF SCOPE -- neither module is imported anywhere
here. `trust`/`criticality` values used are explicit controlled constants,
never claimed as real evidence -- see the `*_source` field assertions
below.
"""

from __future__ import annotations

import math

import pytest

from ai.twintrust_ap.gdca_tahs_trust_criticality_authority_experiment import (
    AUTHORITY_SOURCE_LABEL,
    CRITICALITY_SOURCE_LABEL,
    TRUST_SOURCE_LABEL,
    ThreeInputExperimentRow,
    compute_real_authority_sequence,
    run_gdca_tahs_three_input_experiment,
)
from ai.twintrust_ap.tahs import TAHS

CONTROLLED_TRUST_VALUES = (0.0, 0.5, 1.0)
CONTROLLED_CRITICALITY_VALUES = (0.0, 0.5, 1.0)
ALLOWED_HORIZONS = {1, 2, 3, 5, 8, 10}


def _stationary_then_moving_sequence() -> list[tuple[int, tuple[float, float, float]]]:
    """Same synthetic shape as
    test_gdca_tahs_substitution_experiment.py's own sequence -- a varied
    small-jitter stationary phase followed by a steady large-displacement
    moving phase -- NOT real data."""
    displacements = [0.15, 0.05, 0.002, 0.007, 0.006, 0.0015] + [0.5] * 8
    positions = [0.0]
    for d in displacements:
        positions.append(positions[-1] + d)
    return [(1000 + 10 * i, (p, 0.0, 0.0)) for i, p in enumerate(positions)]


# --------------------------------------------------------------------------
# Source labeling -- the scientific-integrity requirement
# --------------------------------------------------------------------------


def test_rows_label_trust_and_criticality_as_controlled_constants_and_authority_as_real_gdca():
    rows = run_gdca_tahs_three_input_experiment([(1, 0.5)], [0.3], [0.2])
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, ThreeInputExperimentRow)
    assert row.trust_source == "controlled_constant" == TRUST_SOURCE_LABEL
    assert row.criticality_source == "controlled_constant" == CRITICALITY_SOURCE_LABEL
    assert row.authority_source == "real_gdca" == AUTHORITY_SOURCE_LABEL


def test_row_fields_are_not_confused_with_each_other():
    """trust, criticality, and authority must be stored as distinct
    fields -- none silently aliasing another (the exact failure mode the
    Option E substitution experiment deliberately avoided by never naming
    its field 'trust')."""
    rows = run_gdca_tahs_three_input_experiment([(7, 0.9)], [0.1], [0.2])
    row = rows[0]
    assert row.trust == 0.1
    assert row.criticality == 0.2
    assert row.authority == 0.9
    assert row.frame_index == 7


# --------------------------------------------------------------------------
# Row generation shape
# --------------------------------------------------------------------------


def test_produces_one_row_per_authority_trust_criticality_combination():
    authority_sequence = [(1, 0.1), (2, 0.5), (3, 0.9)]
    rows = run_gdca_tahs_three_input_experiment(
        authority_sequence, CONTROLLED_TRUST_VALUES, CONTROLLED_CRITICALITY_VALUES
    )
    assert len(rows) == len(authority_sequence) * len(CONTROLLED_TRUST_VALUES) * len(CONTROLLED_CRITICALITY_VALUES)


def test_horizon_is_always_from_the_discrete_set():
    authority_sequence = [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)]
    rows = run_gdca_tahs_three_input_experiment(
        authority_sequence, CONTROLLED_TRUST_VALUES, CONTROLLED_CRITICALITY_VALUES
    )
    for row in rows:
        assert row.horizon in ALLOWED_HORIZONS


def test_horizon_is_within_configured_range():
    authority_sequence = [(1, a / 10) for a in range(11)]
    rows = run_gdca_tahs_three_input_experiment(
        authority_sequence, CONTROLLED_TRUST_VALUES, CONTROLLED_CRITICALITY_VALUES
    )
    for row in rows:
        assert 1 <= row.horizon <= 10


def test_uses_the_real_unmodified_tahs_class():
    """Confirms this experiment calls the actual, unmodified three-input
    TAHS.select_horizon (not a reimplementation) -- same output as
    calling TAHS directly with all three arguments."""
    tahs = TAHS()
    authority_sequence = [(1, 0.73)]
    rows = run_gdca_tahs_three_input_experiment(authority_sequence, [0.2], [0.42], tahs=tahs)
    direct = tahs.select_horizon(trust=0.2, criticality=0.42, authority=0.73)
    assert rows[0].horizon == direct


def test_is_deterministic():
    authority_sequence = [(1, 0.3), (2, 0.6), (3, 0.9)]
    rows1 = run_gdca_tahs_three_input_experiment(
        authority_sequence, CONTROLLED_TRUST_VALUES, CONTROLLED_CRITICALITY_VALUES
    )
    rows2 = run_gdca_tahs_three_input_experiment(
        authority_sequence, CONTROLLED_TRUST_VALUES, CONTROLLED_CRITICALITY_VALUES
    )
    assert rows1 == rows2


def test_trust_is_always_the_explicitly_supplied_controlled_constant_never_aliased_from_authority():
    """Sanity/regression guard for the task's explicit constraint: a
    row's `trust` must always be exactly one of the caller-supplied
    `trust_values`, never silently derived from or equal to that same
    row's real GDCA `authority` by construction (the Option E trust-slot
    substitution pattern this module must not reintroduce). Numeric
    coincidence between a controlled constant and a real authority value
    at a shared boundary (e.g. both 0.0) is expected and is NOT itself a
    bug -- see GDCA's own semantics for why 0.0 is a meaningful authority
    value, not a marker of substitution -- so this test checks the
    STRUCTURAL guarantee (trust comes from trust_values, never from
    authority) rather than asserting the two value sets are numerically
    disjoint."""
    authority_sequence = compute_real_authority_sequence(_stationary_then_moving_sequence())
    real_authorities = {a for _, a in authority_sequence}
    trust_values = (0.2, 0.6)  # deliberately NOT present anywhere in this authority sequence
    assert not (set(trust_values) & real_authorities)  # sanity check on the test's own fixture
    rows = run_gdca_tahs_three_input_experiment(authority_sequence, trust_values, [0.5])
    for row in rows:
        assert row.trust in trust_values
        assert row.authority in real_authorities


# --------------------------------------------------------------------------
# Monotonicity requirements, all three axes
# --------------------------------------------------------------------------


def test_higher_authority_never_produces_a_shorter_horizon_for_fixed_trust_and_criticality():
    authorities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for trust in CONTROLLED_TRUST_VALUES:
        for criticality in CONTROLLED_CRITICALITY_VALUES:
            authority_sequence = [(i, a) for i, a in enumerate(authorities)]
            rows = run_gdca_tahs_three_input_experiment(authority_sequence, [trust], [criticality])
            horizons = [row.horizon for row in rows]
            for h_prev, h_next in zip(horizons, horizons[1:]):
                assert h_next >= h_prev, (
                    f"trust={trust}, criticality={criticality}: horizon decreased as "
                    f"authority increased ({horizons})"
                )


def test_higher_trust_never_produces_a_shorter_horizon_for_fixed_criticality_and_authority():
    trusts_ascending = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for criticality in CONTROLLED_CRITICALITY_VALUES:
        for authority in (0.0, 0.5, 1.0):
            rows = run_gdca_tahs_three_input_experiment([(0, authority)], trusts_ascending, [criticality])
            horizons = [row.horizon for row in rows]
            for h_prev, h_next in zip(horizons, horizons[1:]):
                assert h_next >= h_prev, (
                    f"criticality={criticality}, authority={authority}: horizon decreased "
                    f"as trust increased ({horizons})"
                )


def test_higher_criticality_never_produces_a_longer_horizon_for_fixed_trust_and_authority():
    criticalities_ascending = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for trust in CONTROLLED_TRUST_VALUES:
        for authority in (0.0, 0.5, 1.0):
            rows = run_gdca_tahs_three_input_experiment([(0, authority)], [trust], criticalities_ascending)
            horizons = [row.horizon for row in rows]
            for h_prev, h_next in zip(horizons, horizons[1:]):
                assert h_next <= h_prev, (
                    f"trust={trust}, authority={authority}: horizon increased as "
                    f"criticality increased ({horizons})"
                )


def test_real_shaped_sequence_high_authority_period_yields_longer_or_equal_horizons_than_low_authority_period():
    """Same stationary-then-moving synthetic shape as the substitution
    experiment's own equivalent test, run through the three-input path at
    fixed controlled trust/criticality."""
    seq = _stationary_then_moving_sequence()
    authority_sequence = compute_real_authority_sequence(seq)
    rows = run_gdca_tahs_three_input_experiment(authority_sequence, [0.4], [0.3])
    stationary_horizons = [r.horizon for r in rows[:5]]
    moving_horizons = [r.horizon for r in rows[-3:]]
    assert min(stationary_horizons) >= max(moving_horizons)


def test_authority_sequence_still_bounded_and_finite_reused_unmodified():
    """Confirms compute_real_authority_sequence (reused, not reimplemented)
    still behaves per its own existing contract."""
    seq = _stationary_then_moving_sequence()
    authorities = compute_real_authority_sequence(seq)
    assert len(authorities) > 0
    for _, a in authorities:
        assert 0.0 <= a <= 1.0
        assert math.isfinite(a)
