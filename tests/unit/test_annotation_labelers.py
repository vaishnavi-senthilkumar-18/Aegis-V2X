"""Unit tests for simulation.annotation — Phase 2's ground-truth label
generators. NOTE: these test the Phase 2 *bootstrap* TAHS/FSDP used to
sanity-check dataset labels, not Phase 4/5's trained/implemented estimators
(those get their own tests per docs/interfaces.md: test_trust_estimator.py,
test_tahs_monotonicity.py, test_fsdp_determinism.py, written in Phase 4/5).
"""

import pytest

from ai.twintrust_ap.fsdp import CommunicationAction, CriticalityBin, TrustBin
from simulation.annotation.decision_labeler import BootstrapFSDP, BootstrapTAHS
from simulation.annotation.trust_criticality_labeler import (
    CriticalityWeights,
    TrustCriticalityLabeler,
    TrustWeights,
)


def test_trust_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        TrustWeights(w_error=0.5, w_uncertainty=0.5, w_sync_age=0.5, w_comm_quality=0.5)


def test_criticality_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        CriticalityWeights(
            alpha_relative_speed=0.5,
            alpha_blockage_prob=0.5,
            alpha_sync_age=0.5,
            alpha_channel_degradation=0.5,
            alpha_traffic_density=0.5,
        )


def test_default_weights_match_configs_model_yaml():
    """Defaults must mirror configs/model.yaml exactly (frozen, Phase 1)."""
    tw = TrustWeights()
    assert (tw.w_error, tw.w_uncertainty, tw.w_sync_age, tw.w_comm_quality) == (
        0.35,
        0.25,
        0.20,
        0.20,
    )
    assert tw.temperature == 0.8

    cw = CriticalityWeights()
    assert (
        cw.alpha_relative_speed,
        cw.alpha_blockage_prob,
        cw.alpha_sync_age,
        cw.alpha_channel_degradation,
        cw.alpha_traffic_density,
    ) == (0.25, 0.25, 0.15, 0.20, 0.15)


def test_trust_is_bounded_in_zero_one_across_input_grid():
    labeler = TrustCriticalityLabeler()
    for e in (0.0, 0.5, 1.0):
        for u in (0.0, 0.5, 1.0):
            for a in (0.0, 0.5, 1.0):
                for q in (0.0, 0.5, 1.0):
                    trust, _ = labeler.compute_trust(e, u, a, q)
                    assert 0.0 <= trust <= 1.0


def test_trust_increases_as_error_uncertainty_age_decrease_and_quality_increases():
    labeler = TrustCriticalityLabeler()
    bad, _ = labeler.compute_trust(
        prediction_error=0.9,
        prediction_uncertainty=0.9,
        sync_age_normalized=0.9,
        comm_quality_normalized=0.1,
    )
    good, _ = labeler.compute_trust(
        prediction_error=0.1,
        prediction_uncertainty=0.1,
        sync_age_normalized=0.1,
        comm_quality_normalized=0.9,
    )
    assert good > bad


def test_criticality_is_bounded_and_monotonic_in_each_feature():
    labeler = TrustCriticalityLabeler()
    low = labeler.compute_criticality(0.0, 0.0, 0.0, 0.0, 0.0)
    high = labeler.compute_criticality(1.0, 1.0, 1.0, 1.0, 1.0)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= low <= high <= 1.0


# -- Bootstrap TAHS/FSDP invariants, per docs/interfaces.md ------------------------


def test_bootstrap_tahs_horizon_is_always_from_discrete_set():
    tahs = BootstrapTAHS()
    allowed = {1, 2, 3, 5, 8, 10}
    for t in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        for c in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            assert tahs.select_horizon(t, c) in allowed


def test_bootstrap_tahs_monotonicity_invariant():
    """docs/interfaces.md invariant #2: for fixed criticality, T1 > T2 => H1 >= H2."""
    tahs = BootstrapTAHS()
    criticality = 0.3
    trusts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    horizons = [tahs.select_horizon(t, criticality) for t in trusts]
    for h_prev, h_next in zip(horizons, horizons[1:]):
        assert h_next >= h_prev


def test_bootstrap_fsdp_determinism_invariant():
    """docs/interfaces.md invariant #3: same (trust_bin, criticality_bin) => same action."""
    fsdp = BootstrapFSDP()
    action_1 = fsdp.decide(0.85, 0.9)
    action_2 = fsdp.decide(0.85, 0.9)
    assert action_1 == action_2
    assert isinstance(action_1, CommunicationAction)


def test_bootstrap_fsdp_discretize_uses_frozen_enums():
    fsdp = BootstrapFSDP()
    trust_bin, criticality_bin = fsdp.discretize(0.95, 0.05)
    assert trust_bin == TrustBin.HIGH
    assert criticality_bin == CriticalityBin.LOW


def test_bootstrap_fsdp_worst_case_state_is_synchronize():
    fsdp = BootstrapFSDP()
    assert fsdp.decide(trust=0.05, criticality=0.95) == CommunicationAction.SYNCHRONIZE


def test_bootstrap_fsdp_lookup_action_covers_all_nine_states():
    fsdp = BootstrapFSDP()
    for trust_bin in TrustBin:
        for criticality_bin in CriticalityBin:
            action = fsdp.lookup_action(trust_bin, criticality_bin)
            assert isinstance(action, CommunicationAction)
