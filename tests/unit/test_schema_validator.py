"""Unit tests for simulation.dataset_pipeline.schema (DatasetSample, matching
configs/dataset.yaml exactly) and .validator (Chapter 15 quality checks)."""

import numpy as np
import pytest
from pydantic import ValidationError

from simulation.dataset_pipeline.schema import (
    DatasetSample,
    TrafficDensity,
    WeatherCondition,
    build_sample_id,
)
from simulation.dataset_pipeline.validator import DatasetValidator


def _make_sample(**overrides) -> DatasetSample:
    defaults = dict(
        sample=build_sample_id("Scene00", 1, 1),
        scene_id="Scene00",
        frame_id=1,
        vehicle_id=1,
        timestamp=0.1,
        lidar="synchronized/Scene00/x.npz",
        gps=(1.0, 2.0, 10.0),
        imu=((0.0, 0.0, 9.8), (0.0, 0.0, 0.0)),
        speed=10.0,
        csi="synchronized/Scene00/x.npz",
        snr=20.0,
        rssi=-60.0,
        path_loss=90.0,
        beam_index=5,
        traffic_density=TrafficDensity.DENSE,
        weather=WeatherCondition.CLEAR_DAY,
        ground_truth_future_csi=0.5,
        ground_truth_future_beam=6,
        ground_truth_trust=0.8,
        ground_truth_criticality=0.2,
        sync_offset_ms=3.0,
    )
    defaults.update(overrides)
    return DatasetSample(**defaults)


def test_valid_sample_constructs():
    assert _make_sample().ground_truth_trust == 0.8


def test_sample_id_matches_naming_convention():
    sample = _make_sample()
    assert sample.sample == "Scene00_Vehicle01_Frame000001"


def test_trust_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        _make_sample(ground_truth_trust=1.5)


def test_negative_path_loss_rejected():
    with pytest.raises(ValidationError):
        _make_sample(path_loss=-1.0)


def test_only_frozen_traffic_density_values_accepted():
    with pytest.raises(ValidationError):
        _make_sample(traffic_density="medium")  # "medium" is not in the frozen enum


def test_only_frozen_weather_values_accepted():
    with pytest.raises(ValidationError):
        _make_sample(weather="fog_dense")  # compound variants are not in the frozen enum


def test_validator_flags_sync_tolerance_violation():
    validator = DatasetValidator(sync_tolerance_ms=10.0)
    report = validator.validate_samples([_make_sample(sync_offset_ms=25.0)])
    assert not report.passed
    assert "sync_tolerance" in report.issues_by_category


def test_validator_flags_duplicate_frames():
    validator = DatasetValidator()
    s1 = _make_sample(sample="A")
    s2 = _make_sample(sample="B")  # same scene/vehicle/frame as s1
    report = validator.validate_samples([s1, s2])
    assert "duplicate_frame" in report.issues_by_category


def test_validator_flags_gps_out_of_range():
    validator = DatasetValidator()
    report = validator.validate_samples([_make_sample(gps=(200.0, 2.0, 10.0))])
    assert "gps_validity" in report.issues_by_category


def test_validator_passes_clean_dataset():
    validator = DatasetValidator(sync_tolerance_ms=10.0)
    s1 = _make_sample(sample="A", frame_id=1)
    s2 = _make_sample(sample="B", frame_id=2, timestamp=0.2)
    report = validator.validate_samples([s1, s2])
    assert report.passed
    assert report.issue_count == 0


def test_validator_array_payload_flags_short_lidar_and_nan_csi():
    validator = DatasetValidator(min_lidar_points=1000)
    issues = validator.validate_array_payload(
        "Scene00_Vehicle01_Frame000001", np.zeros((10, 4)), np.array([np.nan + 0j])
    )
    categories = {i.category for i in issues}
    assert "lidar_completeness" in categories
    assert "csi_integrity" in categories


def test_validator_array_payload_accepts_clean_arrays():
    validator = DatasetValidator(min_lidar_points=10)
    lidar = np.random.rand(1000, 4).astype(np.float32)
    csi = (np.random.rand(128) + 1j * np.random.rand(128)).astype(np.complex64)
    assert validator.validate_array_payload("ok", lidar, csi) == []
