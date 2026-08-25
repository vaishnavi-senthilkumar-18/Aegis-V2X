"""Tests for POST /frames/bulk.

Added to close the ingestion-scale gap flagged in
`claude/project_status.md`'s "Open architecture questions" item 2: the
original one-row-per-call `POST /frames` was never load-tested at Phase 2's
target dataset scale (10,000-20,000 frames across 100-150 scenes). See
`app/crud/frame.py::create_frames_bulk` and `app/schemas/frame.py`'s
`FrameBulkCreate` for the implementation this exercises.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import make_scene_and_vehicle as _make_scene_and_vehicle


def _frame_payload(
    scene_id: str, vehicle_id: str, frame_index: int, wireless_offset: float = 0.0
) -> dict:
    return {
        "scene_id": scene_id,
        "vehicle_id": vehicle_id,
        "frame_index": frame_index,
        "simulation_timestamp": float(frame_index),
        "wireless_timestamp": float(frame_index) + wireless_offset,
    }


def test_bulk_insert_creates_all_frames(client: TestClient) -> None:
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    frames = [_frame_payload(scene_id, vehicle_id, i) for i in range(50)]

    response = client.post("/api/v1/frames/bulk", json={"frames": frames})

    assert response.status_code == 201
    body = response.json()
    assert body["created"] == 50
    assert body["unsynchronized"] == 0
    assert len(body["frame_ids"]) == 50
    assert len(set(body["frame_ids"])) == 50  # every id is real and distinct

    listed = client.get(f"/api/v1/frames?scene_id={scene_id}&limit=100")
    assert listed.status_code == 200
    assert len(listed.json()) == 50


def test_bulk_insert_flags_out_of_sync_frames_instead_of_rejecting(client: TestClient) -> None:
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    frames = [
        _frame_payload(scene_id, vehicle_id, 0, wireless_offset=0.001),  # 1ms: in tolerance
        _frame_payload(scene_id, vehicle_id, 1, wireless_offset=0.050),  # 50ms: out of tolerance
        _frame_payload(scene_id, vehicle_id, 2, wireless_offset=0.001),
    ]

    response = client.post("/api/v1/frames/bulk", json={"frames": frames})

    # All three are stored -- out-of-sync frames are flagged, not rejected,
    # matching the single-frame POST /frames contract (see
    # app/crud/frame.py::_build_frame, shared by both code paths).
    assert response.status_code == 201
    body = response.json()
    assert body["created"] == 3
    assert body["unsynchronized"] == 1
    assert len(body["frame_ids"]) == 3


def test_bulk_insert_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/api/v1/frames/bulk", json={"frames": []})
    assert response.status_code == 422


def test_bulk_insert_rejects_over_2000_items(client: TestClient) -> None:
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    frames = [_frame_payload(scene_id, vehicle_id, i) for i in range(2001)]

    response = client.post("/api/v1/frames/bulk", json={"frames": frames})

    assert response.status_code == 422


def test_bulk_route_not_shadowed_by_frame_id_route(client: TestClient) -> None:
    """Regression test for the route-ordering rule in `app/api/v1/frames.py`'s
    module docstring: `/frames/bulk` must resolve to the bulk-create handler.

    `/frames/bulk` is a POST-only route and `/frames/{frame_id}` is GET-only,
    so they can never collide on method+path the way the GET-only literal
    routes (`/frames/stats/...`, `/frames/scene/...`) could have -- this test
    instead confirms the response actually has the bulk handler's shape
    (`created`/`unsynchronized`/`frame_ids`), not `FrameRead`'s shape, which
    is what a misrouted request would return instead.
    """
    scene_id, vehicle_id = _make_scene_and_vehicle(client)
    frames = [_frame_payload(scene_id, vehicle_id, 0)]

    response = client.post("/api/v1/frames/bulk", json={"frames": frames})

    assert response.status_code == 201
    body = response.json()
    assert set(body.keys()) == {"created", "unsynchronized", "frame_ids"}
