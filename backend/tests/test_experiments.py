"""Tests for the experiment lifecycle (create -> running -> completed)."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_experiment_lifecycle(client: TestClient) -> None:
    create_payload = {
        "name": f"baseline-{uuid.uuid4()}",
        "description": "Baseline run",
        "status": "planned",
    }
    create_resp = client.post("/api/v1/experiments", json=create_payload)
    assert create_resp.status_code == 201
    experiment = create_resp.json()
    assert experiment["status"] == "planned"

    running_url = f"/api/v1/experiments/{experiment['id']}"
    patch_running = client.patch(running_url, json={"status": "running"})
    assert patch_running.status_code == 200
    assert patch_running.json()["status"] == "running"

    patch_completed = client.patch(
        f"/api/v1/experiments/{experiment['id']}",
        json={"status": "completed", "latency_ms": 12.5, "reliability_score": 0.98},
    )
    assert patch_completed.status_code == 200
    body = patch_completed.json()
    assert body["status"] == "completed"
    assert body["latency_ms"] == 12.5
    assert body["reliability_score"] == 0.98


def test_duplicate_experiment_name_rejected(client: TestClient) -> None:
    name = f"dup-{uuid.uuid4()}"
    assert client.post("/api/v1/experiments", json={"name": name}).status_code == 201
    assert client.post("/api/v1/experiments", json={"name": name}).status_code == 409


def test_update_missing_experiment_returns_404(client: TestClient) -> None:
    response = client.patch(f"/api/v1/experiments/{uuid.uuid4()}", json={"status": "running"})
    assert response.status_code == 404


def test_list_experiments(client: TestClient) -> None:
    client.post("/api/v1/experiments", json={"name": f"exp-a-{uuid.uuid4()}"})
    client.post("/api/v1/experiments", json={"name": f"exp-b-{uuid.uuid4()}"})
    response = client.get("/api/v1/experiments")
    assert response.status_code == 200
    assert len(response.json()) >= 2
