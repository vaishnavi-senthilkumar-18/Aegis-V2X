"""Focused tests for `ai.twintrust_ap.tahs.TAHS`, Phase 5's concrete
Trust-Adaptive Horizon Selection.

SYNTHETIC / CODE-CORRECTNESS VALIDATION ONLY. `trust`/`criticality`/
`authority` values here are plain floats chosen to exercise the formula's
shape, boundary, determinism, and monotonicity behaviour -- not
measurements from any real CARLA scene. TAHS's public contract is
`select_horizon(trust: float, criticality: float, authority: float = 0.0)
-> int`; it never touches `DigitalTwinState`, tensors, V2X-ViT embeddings,
or raw LiDAR (see docs/interfaces.md), so there is no real/synthetic
evidence distinction to make here the way there is for Trust/Criticality
-- these are exactly the inputs TAHS is documented to take.

Covers docs/interfaces.md invariant #2 (monotonicity, all three inputs)
plus the boundedness and determinism properties implied by the module's
own docstring. `authority`/`delta` are the Phase 5 three-input extension
(see tahs.py's "PHASE 5 THREE-INPUT EXTENSION" docstring) -- `delta` is a
PROVISIONAL/UNCALIBRATED architectural coefficient, not an optimized or
validated value.
"""

from __future__ import annotations

import pytest

from ai.twintrust_ap.tahs import TAHS, TAHSParams

#: Mirrors configs/model.yaml's tahs: section exactly (horizon_min=1,
#: horizon_max=10, beta=1.0, gamma=1.0, delta=1.0,
#: horizon_discretization=[1,2,3,5,8,10]). `delta` is the PROVISIONAL,
#: UNCALIBRATED authority-sensitivity coefficient added for the Phase 5
#: three-input extension.
DEFAULT_PARAMS = TAHSParams(
    horizon_min=1,
    horizon_max=10,
    beta=1.0,
    gamma=1.0,
    delta=1.0,
    horizon_discretization=(1, 2, 3, 5, 8, 10),
)

ALLOWED_HORIZONS = {1, 2, 3, 5, 8, 10}


def make_tahs() -> TAHS:
    return TAHS(DEFAULT_PARAMS)


def test_loads_defaults_from_model_config():
    """The zero-arg constructor path must match configs/model.yaml exactly."""
    tahs = TAHS()
    assert tahs._p == DEFAULT_PARAMS


def test_horizon_is_always_from_the_discrete_set():
    tahs = make_tahs()
    for t in (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
        for c in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            assert tahs.select_horizon(t, c) in ALLOWED_HORIZONS


def test_boundary_trust_and_criticality_values():
    tahs = make_tahs()
    for t in (0.0, 1.0):
        for c in (0.0, 1.0):
            assert tahs.select_horizon(t, c) in ALLOWED_HORIZONS


def test_monotonicity_invariant_for_fixed_criticality():
    """docs/interfaces.md invariant #2: for fixed criticality,
    T1 > T2 => H1 >= H2 -- higher trust never reduces the horizon."""
    tahs = make_tahs()
    for criticality in (0.0, 0.3, 0.5, 0.7, 1.0):
        trusts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        horizons = [tahs.select_horizon(t, criticality) for t in trusts]
        for h_prev, h_next in zip(horizons, horizons[1:]):
            assert h_next >= h_prev


def test_low_trust_high_criticality_selects_a_below_midpoint_horizon():
    """trust=0, criticality=1 => sigmoid(-1)=0.2689 => continuous_h=3.42,
    nearest discrete value is 3 (verified by direct computation, not
    assumed to be the extreme 1)."""
    tahs = make_tahs()
    assert tahs.select_horizon(trust=0.0, criticality=1.0) == 3


def test_high_trust_low_criticality_selects_an_above_midpoint_horizon():
    """trust=1, criticality=0 => sigmoid(1)=0.7311 => continuous_h=7.58,
    nearest discrete value is 8 (verified by direct computation, not
    assumed to be the extreme 10)."""
    tahs = make_tahs()
    assert tahs.select_horizon(trust=1.0, criticality=0.0) == 8


def test_balanced_trust_and_criticality_selects_a_mid_range_horizon():
    tahs = make_tahs()
    horizon = tahs.select_horizon(trust=0.5, criticality=0.5)
    assert horizon in ALLOWED_HORIZONS


def test_selection_is_deterministic_for_identical_inputs():
    tahs = make_tahs()
    first = tahs.select_horizon(0.62, 0.41)
    second = tahs.select_horizon(0.62, 0.41)
    assert first == second


def test_trust_below_zero_raises():
    tahs = make_tahs()
    with pytest.raises(ValueError):
        tahs.select_horizon(-0.01, 0.5)


def test_trust_above_one_raises():
    tahs = make_tahs()
    with pytest.raises(ValueError):
        tahs.select_horizon(1.01, 0.5)


def test_criticality_below_zero_raises():
    tahs = make_tahs()
    with pytest.raises(ValueError):
        tahs.select_horizon(0.5, -0.01)


def test_criticality_above_one_raises():
    tahs = make_tahs()
    with pytest.raises(ValueError):
        tahs.select_horizon(0.5, 1.01)


def test_exact_hand_computed_value_at_trust_equals_criticality():
    """At trust == criticality, beta*T - gamma*C == 0 (with beta=gamma=1),
    so sigmoid(0) == 0.5 exactly, and continuous_h == 1 + 9*0.5 == 5.5,
    which snaps to 5 (closer than 8: |5.5-5|=0.5 < |5.5-8|=2.5)."""
    tahs = make_tahs()
    assert tahs.select_horizon(trust=0.5, criticality=0.5) == 5


# --------------------------------------------------------------------------
# Phase 5 three-input extension: authority / delta
# --------------------------------------------------------------------------


def test_omitting_authority_matches_two_input_formula_exactly():
    """select_horizon(T, C) must equal select_horizon(T, C, 0.0) exactly --
    authority=0.0 is the additive identity for delta*authority, so omitting
    it must reproduce the original two-input formula bit-for-bit, for
    every previously-tested (trust, criticality) pair."""
    tahs = make_tahs()
    for t in (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
        for c in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            assert tahs.select_horizon(t, c) == tahs.select_horizon(t, c, 0.0)
            assert tahs.select_horizon(t, c) == tahs.select_horizon(t, c, authority=0.0)


def test_two_input_exact_values_unchanged_by_the_extension():
    """The pre-existing hand-verified exact values must be bit-for-bit
    unchanged now that select_horizon has a third parameter."""
    tahs = make_tahs()
    assert tahs.select_horizon(trust=0.0, criticality=1.0) == 3
    assert tahs.select_horizon(trust=1.0, criticality=0.0) == 8
    assert tahs.select_horizon(trust=0.5, criticality=0.5) == 5


def test_horizon_is_always_from_the_discrete_set_with_authority():
    tahs = make_tahs()
    for t in (0.0, 0.5, 1.0):
        for c in (0.0, 0.5, 1.0):
            for a in (0.0, 0.1, 0.5, 0.9, 1.0):
                assert tahs.select_horizon(t, c, a) in ALLOWED_HORIZONS


def test_monotonicity_invariant_for_fixed_trust_and_criticality():
    """docs/interfaces.md invariant #2 (Phase 5 extension): for fixed
    trust and criticality, A1 > A2 => H1 >= H2 -- higher authority never
    reduces the horizon."""
    tahs = make_tahs()
    for trust in (0.0, 0.3, 0.5, 0.7, 1.0):
        for criticality in (0.0, 0.3, 0.5, 0.7, 1.0):
            authorities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            horizons = [tahs.select_horizon(trust, criticality, a) for a in authorities]
            for h_prev, h_next in zip(horizons, horizons[1:]):
                assert h_next >= h_prev


def test_exact_hand_computed_value_with_authority():
    """trust=0.5, criticality=0.5, authority=1.0, beta=gamma=delta=1.0 =>
    sigmoid_arg = 0.5 - 0.5 + 1.0*1.0 = 1.0 => sigmoid(1.0) = 0.731058...
    => continuous_h = 1 + 9*0.731058 = 7.5795, nearest discrete value is 8
    (|7.5795-8|=0.42 < |7.5795-5|=2.58) -- matches the existing
    trust=1.0/criticality=0.0 case's continuous value exactly, since both
    produce the same sigmoid argument (1.0)."""
    tahs = make_tahs()
    assert tahs.select_horizon(trust=0.5, criticality=0.5, authority=1.0) == 8


def test_authority_below_zero_raises():
    tahs = make_tahs()
    with pytest.raises(ValueError):
        tahs.select_horizon(0.5, 0.5, -0.01)


def test_authority_above_one_raises():
    tahs = make_tahs()
    with pytest.raises(ValueError):
        tahs.select_horizon(0.5, 0.5, 1.01)


def test_authority_boundary_values_are_accepted():
    tahs = make_tahs()
    for a in (0.0, 1.0):
        assert tahs.select_horizon(0.5, 0.5, a) in ALLOWED_HORIZONS


def test_selection_with_authority_is_deterministic():
    tahs = make_tahs()
    first = tahs.select_horizon(0.62, 0.41, 0.33)
    second = tahs.select_horizon(0.62, 0.41, 0.33)
    assert first == second


def test_delta_loads_from_model_config():
    """The zero-arg constructor path must load delta=1.0 from
    configs/model.yaml's tahs: section, matching DEFAULT_PARAMS exactly
    (see the module docstring: delta is PROVISIONAL/UNCALIBRATED, not
    tuned here or anywhere else)."""
    tahs = TAHS()
    assert tahs._p.delta == 1.0
    assert tahs._p == DEFAULT_PARAMS


def test_custom_delta_changes_authority_sensitivity():
    """A non-default delta must actually change the authority term's
    effect (sanity check that delta is wired into the formula, not
    ignored) -- delta=0.0 makes authority contribute nothing, matching
    the plain two-input result regardless of the authority value passed."""
    zero_delta_params = TAHSParams(
        horizon_min=1, horizon_max=10, beta=1.0, gamma=1.0, delta=0.0,
        horizon_discretization=(1, 2, 3, 5, 8, 10),
    )
    tahs = TAHS(zero_delta_params)
    assert tahs.select_horizon(0.5, 0.5, 1.0) == tahs.select_horizon(0.5, 0.5, 0.0)
    assert tahs.select_horizon(0.5, 0.5, 1.0) == 5
