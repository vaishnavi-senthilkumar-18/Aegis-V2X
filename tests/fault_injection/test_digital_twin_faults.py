from simulation.fault_injection.fault_types.digital_twin import (
    apply_partial_failure,
    apply_stale_state,
    apply_state_drift,
    apply_sync_delay,
)


def test_stale_state_increases_age_with_severity(synthetic_frame, rng):
    low = apply_stale_state(synthetic_frame, severity=0.2, rng=rng)
    high = apply_stale_state(synthetic_frame, severity=0.8, rng=rng)
    assert high["twin_age_ms"] > low["twin_age_ms"]


def test_stale_state_zero_severity_no_added_age(synthetic_frame, rng):
    result = apply_stale_state(synthetic_frame, severity=0.0, rng=rng)
    assert result["twin_age_ms"] == synthetic_frame["twin_age_ms"]


def test_sync_delay_adds_to_timestamp(synthetic_frame, rng):
    original = synthetic_frame["sync_timestamp"]
    result = apply_sync_delay(synthetic_frame, severity=0.5, rng=rng)
    assert result["sync_timestamp"] > original


def test_state_drift_zero_severity_no_change(synthetic_frame, rng):
    result = apply_state_drift(synthetic_frame, severity=0.0, rng=rng)
    assert result["mobility_state"] == synthetic_frame["mobility_state"]


def test_state_drift_nonzero_severity_changes_values(synthetic_frame, rng):
    result = apply_state_drift(synthetic_frame, severity=0.5, rng=rng)
    assert result["mobility_state"]["speed_mps"] != synthetic_frame["mobility_state"]["speed_mps"]


def test_partial_failure_zero_severity_drops_nothing(synthetic_frame, rng):
    result = apply_partial_failure(synthetic_frame, severity=0.0, rng=rng)
    log = result["_fault_log"][-1]
    assert log["dropped_fields"] == []


def test_partial_failure_full_severity_drops_all_droppable_fields(synthetic_frame, rng):
    result = apply_partial_failure(synthetic_frame, severity=1.0, rng=rng)
    assert result["csi"] is None
    assert result["mobility_state"] is None


def test_partial_failure_is_reproducible_given_seed(synthetic_frame):
    import numpy as np

    rng1 = np.random.default_rng(seed=11)
    rng2 = np.random.default_rng(seed=11)
    r1 = apply_partial_failure(synthetic_frame, severity=0.5, rng=rng1)
    r2 = apply_partial_failure(synthetic_frame, severity=0.5, rng=rng2)
    assert r1["_fault_log"][-1]["dropped_fields"] == r2["_fault_log"][-1]["dropped_fields"]
