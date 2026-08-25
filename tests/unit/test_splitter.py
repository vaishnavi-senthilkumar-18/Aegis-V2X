"""Unit tests for simulation.dataset_pipeline.splitter — scene-disjoint
70/15/15 split, ratios sourced from configs/dataset.yaml `dataset_targets`."""

import pytest

from simulation.dataset_pipeline.schema import (
    DatasetSample,
    TrafficDensity,
    WeatherCondition,
    build_sample_id,
)
from simulation.dataset_pipeline.splitter import DatasetSplitter


def _make_samples(num_scenes: int, frames_per_scene: int) -> list[DatasetSample]:
    samples = []
    for scene_idx in range(num_scenes):
        scene_id = f"Scene{scene_idx:02d}"
        for frame_idx in range(frames_per_scene):
            samples.append(
                DatasetSample(
                    sample=build_sample_id(scene_id, 1, frame_idx),
                    scene_id=scene_id,
                    frame_id=frame_idx,
                    vehicle_id=1,
                    timestamp=frame_idx * 0.1,
                    lidar="x.npz",
                    gps=(1.0, 2.0, 10.0),
                    imu=((0.0, 0.0, 9.8), (0.0, 0.0, 0.0)),
                    speed=10.0,
                    csi="x.npz",
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
            )
    return samples


def test_split_ratios_default_to_configs_dataset_yaml():
    scene_ids = [f"Scene{i:02d}" for i in range(100)]
    assignment = DatasetSplitter(seed=1).assign(
        scene_ids
    )  # no explicit ratios -> reads configs/dataset.yaml
    assert len(assignment.train_scenes) == 70
    assert len(assignment.validation_scenes) == 15
    assert len(assignment.test_scenes) == 15


def test_all_scenes_assigned_exactly_once():
    scene_ids = [f"Scene{i:02d}" for i in range(37)]
    assignment = DatasetSplitter(seed=7).assign(scene_ids)
    all_assigned = assignment.train_scenes + assignment.validation_scenes + assignment.test_scenes
    assert sorted(all_assigned) == sorted(scene_ids)
    assert len(set(all_assigned)) == len(scene_ids)


def test_split_is_deterministic_for_fixed_seed():
    scene_ids = [f"Scene{i:02d}" for i in range(50)]
    a = DatasetSplitter(seed=42).assign(scene_ids)
    b = DatasetSplitter(seed=42).assign(scene_ids)
    assert (a.train_scenes, a.validation_scenes, a.test_scenes) == (
        b.train_scenes,
        b.validation_scenes,
        b.test_scenes,
    )


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        DatasetSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_split_samples_has_no_scene_leakage():
    samples = _make_samples(num_scenes=20, frames_per_scene=5)
    splits = DatasetSplitter(seed=3).split_samples(samples)
    assert DatasetSplitter.verify_no_leakage(splits)
    assert sum(len(v) for v in splits.values()) == len(samples)


def test_split_of_unknown_scene_raises():
    assignment = DatasetSplitter(seed=1).assign(["Scene00", "Scene01"])
    with pytest.raises(KeyError):
        assignment.split_of("SceneUnknown")
