"""Tests for frame ingestion and the sync-tolerance logic (Ch. 9, <=10ms)."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from tests.conftest import make_scene_and_vehicle as _make_scene_and_vehicle


def test_frame_within_sync_tolerance_is_valid(client: TestClient) -> None:
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    response = client.post(
        "/api/v1/frames",
        json={
            "scene_id": scene_id,
            "vehicle_id": vehicle_id,
            "frame_index": 0,
            "simulation_timestamp": 1.000,
            "wireless_timestamp": 1.005,  # 5ms offset, within 10ms tolerance
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["is_sync_valid"] is True
    assert abs(body["sync_offset_ms"] - 5.0) < 1e-6


def test_frame_outside_sync_tolerance_is_flagged_not_rejected(client: TestClient) -> None:
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    response = client.post(
        "/api/v1/frames",
        json={
            "scene_id": scene_id,
            "vehicle_id": vehicle_id,
            "frame_index": 0,
            "simulation_timestamp": 1.000,
            "wireless_timestamp": 1.050,  # 50ms offset, outside 10ms tolerance
        },
    )
    # Out-of-tolerance frames are stored (not rejected), flagged for the
    # dashboard's sync-health panel.
    assert response.status_code == 201
    body = response.json()
    assert body["is_sync_valid"] is False
    assert abs(body["sync_offset_ms"] - 50.0) < 1e-6


def test_unsynchronized_count_stats(client: TestClient) -> None:
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    offsets = [0.001, 0.001, 0.050, 0.001]  # 3 in-tolerance, 1 out
    for i, wireless_offset in enumerate(offsets):
        client.post(
            "/api/v1/frames",
            json={
                "scene_id": scene_id,
                "vehicle_id": vehicle_id,
                "frame_index": i,
                "simulation_timestamp": float(i),
                "wireless_timestamp": float(i) + wireless_offset,
            },
        )
    response = client.get(f"/api/v1/frames/stats/unsynchronized-count?scene_id={scene_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["total_frames"] == 4
    assert body["unsynchronized_frames"] == 1
    assert abs(body["unsynchronized_ratio"] - 0.25) < 1e-6


def test_stats_route_not_shadowed_by_frame_id_route(client: TestClient) -> None:
    """Regression test for the route-ordering bug found during Phase 3 verification:
    `/frames/stats/unsynchronized-count` must resolve to the stats handler, not be
    matched as an invalid UUID by `/frames/{frame_id}`.
    """
    response = client.get("/api/v1/frames/stats/unsynchronized-count")
    assert response.status_code == 200
    assert "total_frames" in response.json()


def test_get_frame_not_found_returns_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/frames/{uuid.uuid4()}")
    assert response.status_code == 404


def test_latest_per_vehicle(client: TestClient) -> None:
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    for i in range(5):
        client.post(
            "/api/v1/frames",
            json={
                "scene_id": scene_id,
                "vehicle_id": vehicle_id,
                "frame_index": i,
                "simulation_timestamp": float(i),
                "wireless_timestamp": float(i),
            },
        )
    response = client.get(f"/api/v1/frames/scene/{scene_id}/latest-per-vehicle")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["frame_index"] == 4  # latest frame for the single vehicle
    assert body[0]["vehicle_code"] == "Vehicle00"
