"""Tests for the synthetic data generator: schema conformance and record counts."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_generate_synthetic_scene_creates_full_chain(client: TestClient) -> None:
    response = client.post(
        "/api/v1/synthetic/scenes",
        json={"scene_code": "PytestScene", "num_vehicles": 3, "num_frames": 10},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["vehicles_created"] == 3
    assert body["frames_created"] == 30
    assert body["trust_records_created"] == 30
    assert body["criticality_records_created"] == 30
    assert body["decisions_created"] == 30


def test_synthetic_frames_are_tagged_and_schema_conformant(client: TestClient) -> None:
    gen_resp = client.post(
        "/api/v1/synthetic/scenes",
        json={"scene_code": "PytestScene2", "num_vehicles": 1, "num_frames": 5},
    ).json()

    frames_resp = client.get(f"/api/v1/frames?scene_id={gen_resp['scene_id']}")
    assert frames_resp.status_code == 200
    frames = frames_resp.json()
    assert len(frames) == 5
    for frame in frames:
        assert frame["source"] == "synthetic"
        # Every frame must have the full multimodal field set populated
        # (schema conformance with the real ingestion contract).
        assert frame["csi"] is not None
        assert frame["snr_db"] is not None
        assert frame["position_x"] is not None
        assert frame["lane_id"] is not None

        trust_resp = client.get(f"/api/v1/trust/frame/{frame['id']}")
        assert trust_resp.status_code == 200

        criticality_resp = client.get(f"/api/v1/criticality/frame/{frame['id']}")
        assert criticality_resp.status_code == 200

        decision_resp = client.get(f"/api/v1/decisions/frame/{frame['id']}")
        assert decision_resp.status_code == 200
        assert decision_resp.json()["policy_source"] == "synthetic"


def test_synthetic_scene_populates_latest_per_vehicle(client: TestClient) -> None:
    gen_resp = client.post(
        "/api/v1/synthetic/scenes",
        json={"scene_code": "PytestScene3", "num_vehicles": 4, "num_frames": 20},
    ).json()
    response = client.get(f"/api/v1/frames/scene/{gen_resp['scene_id']}/latest-per-vehicle")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 4
    for row in body:
        assert row["frame_index"] == 19  # last frame index for every vehicle
