"""Tests for scene and vehicle CRUD endpoints."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_create_and_get_scene(client: TestClient) -> None:
    payload = {"scene_code": "TestScene01", "map_name": "Town05", "num_vehicles_target": 3}
    create_resp = client.post("/api/v1/scenes", json=payload)
    assert create_resp.status_code == 201
    scene = create_resp.json()
    assert scene["scene_code"] == "TestScene01"

    get_resp = client.get(f"/api/v1/scenes/{scene['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == scene["id"]


def test_get_scene_not_found(client: TestClient) -> None:
    response = client.get(f"/api/v1/scenes/{uuid.uuid4()}")
    assert response.status_code == 404


def test_list_scenes(client: TestClient) -> None:
    client.post("/api/v1/scenes", json={"scene_code": "A", "num_vehicles_target": 1})
    client.post("/api/v1/scenes", json={"scene_code": "B", "num_vehicles_target": 1})
    response = client.get("/api/v1/scenes")
    assert response.status_code == 200
    codes = {s["scene_code"] for s in response.json()}
    assert {"A", "B"}.issubset(codes)


def test_add_and_list_vehicles(client: TestClient) -> None:
    scene_payload = {"scene_code": "VehScene", "num_vehicles_target": 2}
    scene = client.post("/api/v1/scenes", json=scene_payload).json()

    v1 = client.post(
        f"/api/v1/scenes/{scene['id']}/vehicles",
        json={"vehicle_code": "Vehicle00", "vehicle_type": "car", "is_ego": True},
    )
    assert v1.status_code == 201
    assert v1.json()["vehicle_code"] == "Vehicle00"
    assert v1.json()["is_ego"] is True

    list_resp = client.get(f"/api/v1/scenes/{scene['id']}/vehicles")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


def test_add_vehicle_to_missing_scene(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/scenes/{uuid.uuid4()}/vehicles",
        json={"vehicle_code": "Vehicle00"},
    )
    assert response.status_code == 404
