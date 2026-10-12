import numpy as np

from simulation.fault_injection.fault_types.perception import (
    apply_gps_drift,
    apply_lidar_noise,
    apply_missing_detections,
    apply_occlusion,
)


def test_lidar_noise_zero_severity_leaves_points_unchanged(synthetic_frame, rng):
    result = apply_lidar_noise(synthetic_frame, severity=0.0, rng=rng)
    np.testing.assert_array_almost_equal(result["lidar"], synthetic_frame["lidar"])


def test_lidar_noise_increases_with_severity(synthetic_frame):
    rng1 = np.random.default_rng(seed=1)
    rng2 = np.random.default_rng(seed=1)
    low = apply_lidar_noise(synthetic_frame, severity=0.1, rng=rng1)
    high = apply_lidar_noise(synthetic_frame, severity=0.9, rng=rng2)
    low_deviation = np.abs(low["lidar"] - synthetic_frame["lidar"]).mean()
    high_deviation = np.abs(high["lidar"] - synthetic_frame["lidar"]).mean()
    assert high_deviation > low_deviation


def test_lidar_noise_does_not_mutate_input(synthetic_frame, rng):
    original = synthetic_frame["lidar"].copy()
    apply_lidar_noise(synthetic_frame, severity=0.5, rng=rng)
    np.testing.assert_array_equal(synthetic_frame["lidar"], original)


def test_occlusion_zero_severity_keeps_all_points(synthetic_frame, rng):
    result = apply_occlusion(synthetic_frame, severity=0.0, rng=rng)
    assert result["lidar"].shape[0] == synthetic_frame["lidar"].shape[0]


def test_occlusion_removes_points_proportional_to_severity(synthetic_frame):
    rng1 = np.random.default_rng(seed=5)
    rng2 = np.random.default_rng(seed=5)
    low = apply_occlusion(synthetic_frame, severity=0.1, rng=rng1)
    high = apply_occlusion(synthetic_frame, severity=0.5, rng=rng2)
    assert high["lidar"].shape[0] <= low["lidar"].shape[0]


def test_missing_detections_zero_severity_keeps_all(synthetic_frame, rng):
    result = apply_missing_detections(synthetic_frame, severity=0.0, rng=rng)
    assert len(result["detections"]) == len(synthetic_frame["detections"])


def test_missing_detections_full_severity_drops_all(synthetic_frame, rng):
    result = apply_missing_detections(synthetic_frame, severity=1.0, rng=rng)
    assert len(result["detections"]) == 0


def test_gps_drift_zero_severity_no_change(synthetic_frame, rng):
    result = apply_gps_drift(synthetic_frame, severity=0.0, rng=rng)
    assert result["gps_lat"] == synthetic_frame["gps_lat"]
    assert result["gps_lon"] == synthetic_frame["gps_lon"]


def test_gps_drift_nonzero_severity_changes_position(synthetic_frame, rng):
    result = apply_gps_drift(synthetic_frame, severity=0.8, rng=rng)
    # Extremely unlikely to be bit-identical given nonzero sigma.
    assert (
        result["gps_lat"] != synthetic_frame["gps_lat"]
        or result["gps_lon"] != synthetic_frame["gps_lon"]
    )
