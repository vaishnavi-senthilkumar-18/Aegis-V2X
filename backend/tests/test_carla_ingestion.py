"""Tests for `app.services.carla_ingestion`.

Uses real sample data taken directly from
`docs/phase2_phase3_reconciliation_2026-09-10.md` (straight_road scene,
vehicle147, frame_122396) rather than invented fixtures, so this actually
exercises the real-data shape the reconciliation found -- including the
GPS/IMU frame-index lag.
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.services.carla_ingestion import (
    SOURCE_CARLA_ONLY,
    assemble_frame_payload,
    assemble_scene_payloads,
    discover_vehicle_ids,
)


@pytest.fixture
def scene_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def vehicle_id() -> uuid.UUID:
    return uuid.uuid4()


def _write_json(path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


@pytest.fixture
def frame_dir_with_full_vehicle(tmp_path):
    """One frame directory with all 3 JSON sensor files for vehicle147.

    Real values copied verbatim from the reconciliation doc -- note GPS's
    internal frame (122387) deliberately does NOT match the directory name
    (frame_122396), reproducing the real ~90ms lag pattern found on both
    completed scenes.
    """
    frame_dir = tmp_path / "frame_122396"
    frame_dir.mkdir()

    _write_json(
        frame_dir / "vehicle147_gps.json",
        {
            "timestamp": 13.3709183963947,
            "frame": 122387,
            "lat": 0.002597277550805188,
            "lon": 0.00018045771035802462,
            "alt": 1.8815484456717968,
        },
    )
    _write_json(
        frame_dir / "vehicle147_imu.json",
        {
            "timestamp": 13.460918394383043,
            "frame": 122396,
            "accel": [0.26350677013397217, 3.9009602069854736, 9.763330459594727],
            "gyro": [0.011471222154796124, 0.013331711292266846, 0.16280075907707214],
            "compass": 0.48884472250938416,
        },
    )
    _write_json(
        frame_dir / "vehicle147_vehicle_speed.json",
        {"timestamp": 13.4609, "frame": 122396, "speed_mps": 12.4},
    )
    (frame_dir / "vehicle147_lidar.npz").write_bytes(b"")
    (frame_dir / "vehicle147_camera.npz").write_bytes(b"")

    return frame_dir


class TestDiscoverVehicleIds:
    def test_finds_single_vehicle(self, frame_dir_with_full_vehicle):
        ids = discover_vehicle_ids(frame_dir_with_full_vehicle)
        assert ids == ["147"]

    def test_finds_multiple_vehicles_sorted_numerically(self, tmp_path):
        frame_dir = tmp_path / "frame_100000"
        frame_dir.mkdir()
        for vid in ["9", "150", "23"]:
            (frame_dir / f"vehicle{vid}_gps.json").write_text("{}")
        assert discover_vehicle_ids(frame_dir) == ["9", "23", "150"]

    def test_empty_directory_returns_no_vehicles(self, tmp_path):
        frame_dir = tmp_path / "frame_000000"
        frame_dir.mkdir()
        assert discover_vehicle_ids(frame_dir) == []


class TestAssembleFramePayload:
    def test_full_vehicle_data_assembles_correctly(
        self, frame_dir_with_full_vehicle, scene_id, vehicle_id
    ):
        payload = assemble_frame_payload(
            frame_dir_with_full_vehicle, "147", scene_id, vehicle_id
        )

        assert payload is not None
        assert payload.scene_id == scene_id
        assert payload.vehicle_id == vehicle_id
        assert payload.source == SOURCE_CARLA_ONLY
        assert payload.frame_index == 122387
        assert payload.simulation_timestamp == pytest.approx(13.460918394383043)
        assert payload.wireless_timestamp == payload.simulation_timestamp
        assert payload.gps_lat == pytest.approx(0.002597277550805188)
        assert payload.gps_lon == pytest.approx(0.00018045771035802462)
        assert payload.gps_alt == pytest.approx(1.8815484456717968)
        assert payload.imu_data["accel"] == [
            0.26350677013397217,
            3.9009602069854736,
            9.763330459594727,
        ]
        assert payload.speed_mps == pytest.approx(12.4)
        assert payload.lidar_path is not None
        assert payload.lidar_path.endswith("vehicle147_lidar.npz")

    def test_missing_gps_falls_back_to_imu_frame_and_timestamp(self, tmp_path, scene_id, vehicle_id):
        frame_dir = tmp_path / "frame_500000"
        frame_dir.mkdir()
        _write_json(
            frame_dir / "vehicle1_imu.json",
            {"timestamp": 9.0, "frame": 500000, "accel": [0, 0, 9.8], "gyro": [0, 0, 0], "compass": 0.0},
        )

        payload = assemble_frame_payload(frame_dir, "1", scene_id, vehicle_id)

        assert payload is not None
        assert payload.frame_index == 500000
        assert payload.simulation_timestamp == pytest.approx(9.0)
        assert payload.gps_lat is None
        assert payload.gps_lon is None
        assert payload.lidar_path is None

    def test_no_sensor_files_at_all_returns_none(self, tmp_path, scene_id, vehicle_id):
        frame_dir = tmp_path / "frame_999999"
        frame_dir.mkdir()
        (frame_dir / "vehicle2_gps.json").write_text("{}")

        payload = assemble_frame_payload(frame_dir, "1", scene_id, vehicle_id)

        assert payload is None

    def test_gps_and_imu_missing_but_speed_present_without_timestamp_returns_none(
        self, tmp_path, scene_id, vehicle_id
    ):
        """speed_mps alone can't supply a timestamp -- assembly must skip, not crash."""
        frame_dir = tmp_path / "frame_1"
        frame_dir.mkdir()
        _write_json(frame_dir / "vehicle1_vehicle_speed.json", {"speed_mps": 5.0})

        payload = assemble_frame_payload(frame_dir, "1", scene_id, vehicle_id)

        assert payload is None


class TestAssembleScenePayloads:
    def test_assembles_across_multiple_frame_directories(self, tmp_path, scene_id):
        scene_dir = tmp_path / "some_scene_Scene00"
        scene_dir.mkdir()

        for frame_idx, ts in [(100, 1.0), (110, 2.0)]:
            frame_dir = scene_dir / f"frame_{frame_idx}"
            frame_dir.mkdir()
            _write_json(
                frame_dir / "vehicle5_imu.json",
                {"timestamp": ts, "frame": frame_idx, "accel": [0, 0, 9.8], "gyro": [0, 0, 0], "compass": 0.0},
            )

        vehicle_id = uuid.uuid4()
        payloads = assemble_scene_payloads(scene_dir, scene_id, {"5": vehicle_id})

        assert len(payloads) == 2
        assert {p.frame_index for p in payloads} == {100, 110}
        assert all(p.vehicle_id == vehicle_id for p in payloads)

    def test_unknown_vehicle_id_is_skipped_not_erroring(self, tmp_path, scene_id):
        scene_dir = tmp_path / "some_scene_Scene00"
        scene_dir.mkdir()
        frame_dir = scene_dir / "frame_1"
        frame_dir.mkdir()
        _write_json(
            frame_dir / "vehicle999_imu.json",
            {"timestamp": 1.0, "frame": 1, "accel": [0, 0, 9.8], "gyro": [0, 0, 0], "compass": 0.0},
        )

        payloads = assemble_scene_payloads(scene_dir, scene_id, {})

        assert payloads == []

    def test_ignores_non_frame_directories(self, tmp_path, scene_id):
        scene_dir = tmp_path / "some_scene_Scene00"
        scene_dir.mkdir()
        (scene_dir / "not_a_frame_dir").mkdir()
        (scene_dir / "geometry_snapshot.json").write_text("{}")

        payloads = assemble_scene_payloads(scene_dir, scene_id, {})

        assert payloads == []
