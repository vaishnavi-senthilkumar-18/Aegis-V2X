"""Tests for decision creation, validation, and the action-distribution stat."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import make_frame as _make_frame


def test_create_valid_decision(client: TestClient) -> None:
    frame_id = _make_frame(client)
    response = client.post(
        "/api/v1/decisions",
        json={
            "frame_id": frame_id,
            "prediction_horizon": 5,
            "fsdp_action": "maintain_beam",
            "trust_probability_used": 0.7,
            "criticality_score_used": 0.2,
            "policy_source": "synthetic",
        },
    )
    assert response.status_code == 201
    assert response.json()["prediction_horizon"] == 5


def test_invalid_prediction_horizon_rejected(client: TestClient) -> None:
    frame_id = _make_frame(client)
    response = client.post(
        "/api/v1/decisions",
        json={
            "frame_id": frame_id,
            "prediction_horizon": 4,  # not in {1,2,3,5,8,10}
            "fsdp_action": "maintain_beam",
            "trust_probability_used": 0.7,
            "criticality_score_used": 0.2,
        },
    )
    assert response.status_code == 422


def test_invalid_fsdp_action_rejected(client: TestClient) -> None:
    frame_id = _make_frame(client)
    response = client.post(
        "/api/v1/decisions",
        json={
            "frame_id": frame_id,
            "prediction_horizon": 5,
            "fsdp_action": "not_a_real_action",
            "trust_probability_used": 0.7,
            "criticality_score_used": 0.2,
        },
    )
    assert response.status_code == 422


def test_duplicate_decision_rejected(client: TestClient) -> None:
    frame_id = _make_frame(client)
    payload = {
        "frame_id": frame_id,
        "prediction_horizon": 5,
        "fsdp_action": "maintain_beam",
        "trust_probability_used": 0.7,
        "criticality_score_used": 0.2,
    }
    assert client.post("/api/v1/decisions", json=payload).status_code == 201
    assert client.post("/api/v1/decisions", json=payload).status_code == 409


def test_action_distribution(client: TestClient) -> None:
    actions = ["maintain_beam", "maintain_beam", "reselect_beam"]
    for action in actions:
        frame_id = _make_frame(client)
        client.post(
            "/api/v1/decisions",
            json={
                "frame_id": frame_id,
                "prediction_horizon": 5,
                "fsdp_action": action,
                "trust_probability_used": 0.7,
                "criticality_score_used": 0.2,
            },
        )
    response = client.get("/api/v1/decisions/stats/action-distribution")
    assert response.status_code == 200
    body = response.json()
    assert body["total_decisions"] == 3
    counts = {entry["fsdp_action"]: entry["count"] for entry in body["distribution"]}
    assert counts["maintain_beam"] == 2
    assert counts["reselect_beam"] == 1
