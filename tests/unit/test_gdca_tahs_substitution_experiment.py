"""Tests for `ai.twintrust_ap.gdca_tahs_experiment` -- the Option E
GDCA-Authority -> TAHS substitution EXPERIMENT approved by the
2026-09-19 architecture design review (`claude/project_status.md`).

SYNTHETIC INPUTS: the position sequences here are plain, hand-constructed
tuples chosen to exercise the substitution mechanism's monotonicity/
boundedness/determinism -- NOT measurements from any real CARLA scene.
The real 92-frame `Vehicle147` sequence is exercised separately, offline
from pytest (see `claude/project_status.md`'s dated entry for that run's
results), because it requires a live backend connection and this suite
must remain fully deterministic and reproducible without one.

Every assertion in this file treats `ai.trust_estimator` and
`ai.criticality` as OUT OF SCOPE -- neither module is imported anywhere
here, and no real Trust or Criticality evidence is fabricated.
"""

from __future__ import annotations

import math

import pytest

from ai.twintrust_ap.gdca import compute_authority_from_evidence
from ai.twintrust_ap.gdca_tahs_experiment import (
    SubstitutionExperimentRow,
    compute_real_authority_sequence,
    run_gdca_tahs_substitution_experiment,
)
from ai.twintrust_ap.tahs import TAHS

CONTROLLED_CRITICALITY_VALUES = (0.0, 0.5, 1.0)
ALLOWED_HORIZONS = {1, 2, 3, 5, 8, 10}


def _stationary_then_moving_sequence() -> list[tuple[int, tuple[float, float, float]]]:
    """A synthetic (frame_index, position_xyz) sequence with a clearly
    stationary phase (small, VARIED jitter -- not all-identical, which
    would trivially zero out the rolling spread and mask any contrast)
    followed by a clearly moving phase (steady, repeated large
    displacement) -- deliberately shaped like the real Vehicle147
    trajectory's own stationary-then-accelerating pattern (small varied
    residuals settling low, then a large STEADY residual that collapses
    the rolling spread toward zero and correctly triggers a low-authority
    response), without using any real data. Numerically verified before
    use: stationary-phase authority ranges ~0.72-0.9996, moving-phase
    authority collapses to 0.0 (per this module's own mechanism, applied
    exactly as already validated in claude/project_status.md's real-data
    checkpoint)."""
    displacements = [0.15, 0.05, 0.002, 0.007, 0.006, 0.0015] + [0.5] * 8
    positions = [0.0]
    for d in displacements:
        positions.append(positions[-1] + d)
    return [(1000 + 10 * i, (p, 0.0, 0.0)) for i, p in enumerate(positions)]


# --------------------------------------------------------------------------
# compute_real_authority_sequence
# --------------------------------------------------------------------------


def test_authority_sequence_is_bounded_in_zero_one():
    seq = _stationary_then_moving_sequence()
    authorities = compute_real_authority_sequence(seq)
    assert len(authorities) > 0
    for _, a in authorities:
        assert 0.0 <= a <= 1.0
        assert math.isfinite(a)


def test_authority_sequence_no_nan_or_inf():
    seq = _stationary_then_moving_sequence()
    authorities = compute_real_authority_sequence(seq)
    for _, a in authorities:
        assert not math.isnan(a)
        assert not math.isinf(a)


def test_authority_sequence_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        compute_real_authority_sequence([(0, (0.0, 0.0, 0.0))])


def test_authority_sequence_is_deterministic():
    seq = _stationary_then_moving_sequence()
    a1 = compute_real_authority_sequence(seq)
    a2 = compute_real_authority_sequence(seq)
    assert a1 == a2


def test_authority_high_during_stationary_phase_low_during_moving_phase():
    """The stationary phase's tiny, near-repeated residuals should score
    noticeably higher authority than the steady-large-displacement moving
    phase, mirroring the real Vehicle147 demonstration's own finding."""
    seq = _stationary_then_moving_sequence()
    authorities = compute_real_authority_sequence(seq)
    stationary_scores = [a for _, a in authorities[:5]]
    moving_scores = [a for _, a in authorities[-3:]]
    assert min(stationary_scores) > max(moving_scores)


# --------------------------------------------------------------------------
# run_gdca_tahs_substitution_experiment -- the substitution itself
# --------------------------------------------------------------------------


def test_substitution_rows_use_authority_as_trust_substitution_field_name():
    """Explicit labeling requirement: the field must never be named
    'trust' -- it must be unambiguous that this is a substitution."""
    rows = run_gdca_tahs_substitution_experiment([(1, 0.5)], [0.5])
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, SubstitutionExperimentRow)
    assert hasattr(row, "authority_as_trust_substitution")
    assert not hasattr(row, "trust")
    assert row.authority_as_trust_substitution == 0.5


def test_substitution_produces_one_row_per_authority_criticality_pair():
    authority_sequence = [(1, 0.1), (2, 0.5), (3, 0.9)]
    rows = run_gdca_tahs_substitution_experiment(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    assert len(rows) == len(authority_sequence) * len(CONTROLLED_CRITICALITY_VALUES)


def test_substitution_horizon_is_always_from_the_discrete_set():
    authority_sequence = [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)]
    rows = run_gdca_tahs_substitution_experiment(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    for row in rows:
        assert row.horizon in ALLOWED_HORIZONS


def test_substitution_horizon_is_within_configured_range():
    authority_sequence = [(1, a / 10) for a in range(11)]
    rows = run_gdca_tahs_substitution_experiment(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    for row in rows:
        assert 1 <= row.horizon <= 10


def test_substitution_uses_the_real_unmodified_tahs_class():
    """Confirms this experiment calls the actual TAHS class (not a
    reimplementation) -- same output as calling TAHS directly."""
    tahs = TAHS()
    authority_sequence = [(1, 0.73)]
    rows = run_gdca_tahs_substitution_experiment(authority_sequence, [0.42], tahs=tahs)
    direct = tahs.select_horizon(trust=0.73, criticality=0.42)
    assert rows[0].horizon == direct


def test_substitution_is_deterministic():
    authority_sequence = [(1, 0.3), (2, 0.6), (3, 0.9)]
    rows1 = run_gdca_tahs_substitution_experiment(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    rows2 = run_gdca_tahs_substitution_experiment(authority_sequence, CONTROLLED_CRITICALITY_VALUES)
    assert rows1 == rows2


# --------------------------------------------------------------------------
# Monotonicity requirements (mandated by the task)
# --------------------------------------------------------------------------


def test_higher_authority_never_produces_a_shorter_horizon_for_fixed_criticality():
    authorities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for criticality in CONTROLLED_CRITICALITY_VALUES:
        authority_sequence = [(i, a) for i, a in enumerate(authorities)]
        rows = run_gdca_tahs_substitution_experiment(authority_sequence, [criticality])
        horizons = [row.horizon for row in rows]
        for h_prev, h_next in zip(horizons, horizons[1:]):
            assert h_next >= h_prev, (
                f"criticality={criticality}: horizon decreased as authority increased "
                f"({horizons})"
            )


def test_higher_criticality_never_produces_a_longer_horizon_for_fixed_authority():
    fixed_authorities = [0.0, 0.25, 0.5, 0.75, 1.0]
    criticalities_ascending = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for authority in fixed_authorities:
        rows = run_gdca_tahs_substitution_experiment([(0, authority)], criticalities_ascending)
        horizons = [row.horizon for row in rows]
        for h_prev, h_next in zip(horizons, horizons[1:]):
            assert h_next <= h_prev, (
                f"authority={authority}: horizon increased as criticality increased "
                f"({horizons})"
            )


def test_real_shaped_sequence_high_authority_period_yields_longer_or_equal_horizons_than_low_authority_period():
    """Uses the same stationary-then-moving synthetic shape as the
    authority-sequence tests above, run through TAHS at a fixed
    criticality, to confirm the stationary (high-authority) phase yields
    horizons >= the moving (low-authority) phase's horizons -- the
    concrete TAHS-level version of requirement #8."""
    seq = _stationary_then_moving_sequence()
    authority_sequence = compute_real_authority_sequence(seq)
    rows = run_gdca_tahs_substitution_experiment(authority_sequence, [0.3])
    stationary_horizons = [r.horizon for r in rows[:5]]
    moving_horizons = [r.horizon for r in rows[-3:]]
    assert min(stationary_horizons) >= max(moving_horizons)
