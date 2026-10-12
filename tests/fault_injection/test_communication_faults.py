import numpy as np
import pytest

from simulation.fault_injection.fault_types.communication import (
    apply_bandwidth_reduction,
    apply_jitter,
    apply_latency,
    apply_packet_loss,
)


def test_packet_loss_zero_severity_never_drops(synthetic_frame, rng):
    for _ in range(50):
        result = apply_packet_loss(synthetic_frame, severity=0.0, rng=rng)
        assert result is not None


def test_packet_loss_full_severity_always_drops(synthetic_frame, rng):
    for _ in range(50):
        result = apply_packet_loss(synthetic_frame, severity=1.0, rng=rng)
        assert result is None


def test_packet_loss_is_deterministic_given_seed(synthetic_frame):
    rng1 = np.random.default_rng(seed=7)
    rng2 = np.random.default_rng(seed=7)
    results1 = [apply_packet_loss(synthetic_frame, 0.5, rng1) is None for _ in range(30)]
    results2 = [apply_packet_loss(synthetic_frame, 0.5, rng2) is None for _ in range(30)]
    assert results1 == results2


def test_latency_adds_delay_proportional_to_severity(synthetic_frame, rng):
    original_ts = synthetic_frame["wireless_timestamp"]
    low = apply_latency(synthetic_frame, severity=0.2, rng=rng)
    high = apply_latency(synthetic_frame, severity=0.8, rng=rng)
    assert low["wireless_timestamp"] > original_ts
    assert high["wireless_timestamp"] > low["wireless_timestamp"]


def test_latency_zero_severity_is_no_delay(synthetic_frame, rng):
    original_ts = synthetic_frame["wireless_timestamp"]
    result = apply_latency(synthetic_frame, severity=0.0, rng=rng)
    assert result["wireless_timestamp"] == original_ts


def test_latency_does_not_mutate_input(synthetic_frame, rng):
    original_ts = synthetic_frame["wireless_timestamp"]
    apply_latency(synthetic_frame, severity=0.5, rng=rng)
    assert synthetic_frame["wireless_timestamp"] == original_ts


def test_jitter_is_reproducible_given_seed(synthetic_frame):
    rng1 = np.random.default_rng(seed=3)
    rng2 = np.random.default_rng(seed=3)
    r1 = apply_jitter(synthetic_frame, severity=0.5, rng=rng1)
    r2 = apply_jitter(synthetic_frame, severity=0.5, rng=rng2)
    assert r1["wireless_timestamp"] == r2["wireless_timestamp"]


def test_bandwidth_reduction_zero_severity_keeps_all_frames(synthetic_frame, rng):
    for i in range(10):
        result = apply_bandwidth_reduction(synthetic_frame, severity=0.0, rng=rng, frame_index=i)
        assert result is not None


def test_bandwidth_reduction_full_severity_drops_all(synthetic_frame, rng):
    result = apply_bandwidth_reduction(synthetic_frame, severity=1.0, rng=rng, frame_index=5)
    assert result is None


def test_invalid_severity_raises():
    from simulation.fault_injection.severity import validate_severity

    with pytest.raises(ValueError):
        validate_severity(1.5)
    with pytest.raises(ValueError):
        validate_severity(-0.1)
