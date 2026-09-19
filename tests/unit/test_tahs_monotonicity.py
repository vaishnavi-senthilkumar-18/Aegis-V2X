"""Focused tests for `ai.twintrust_ap.tahs.TAHS`, Phase 5's concrete
Trust-Adaptive Horizon Selection.

SYNTHETIC / CODE-CORRECTNESS VALIDATION ONLY. `trust`/`criticality` values
here are plain floats chosen to exercise the formula's shape, boundary,
determinism, and monotonicity behaviour -- not measurements from any real
CARLA scene. TAHS's public contract is `select_horizon(trust: float,
criticality: float) -> int`; it never touches `DigitalTwinState`, tensors,
V2X-ViT embeddings, or raw LiDAR (see docs/interfaces.md), so there is no
real/synthetic evidence distinction to make here the way there is for
Trust/Criticality -- these are exactly the inputs TAHS is documented to
take.

Covers docs/interfaces.md invariant #2 (monotonicity) plus the boundedness
and determinism properties implied by the module's own docstring.
"""

from __future__ import annotations

import pytest

from ai.twintrust_ap.tahs import TAHS, TAHSParams

#: Mirrors configs/model.yaml's tahs: section exactly (horizon_min=1,
#: horizon_max=10, beta=1.0, gamma=1.0, horizon_discretization=[1,2,3,5,8,10]).
DEFAULT_PARAMS = TAHSParams(
    horizon_min=1,
    horizon_max=10,
    beta=1.0,
    gamma=1.0,
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
