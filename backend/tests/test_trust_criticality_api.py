"""API-level tests for trust and criticality record ingestion."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from tests.conftest import make_frame as _make_frame


def test_create_trust_record(client: TestClient) -> None:
    frame_id = _make_frame(client)
    response = client.post(
        "/api/v1/trust",
        json={
            "frame_id": frame_id,
            "prediction_error": 0.1,
            "prediction_uncertainty": 0.1,
            "sync_age_penalty": 0.1,
            "comm_quality": 0.9,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert 0.0 <= body["trust_probability"] <= 1.0
    assert body["interpretation"] in {
        "very_unreliable", "unreliable", "moderate", "reliable", "highly_reliable"
    }

    fetch = client.get(f"/api/v1/trust/frame/{frame_id}")
    assert fetch.status_code == 200
    assert fetch.json()["id"] == body["id"]


def test_duplicate_trust_record_rejected(client: TestClient) -> None:
    frame_id = _make_frame(client)
    payload = {
        "frame_id": frame_id,
        "prediction_error": 0.1,
        "prediction_uncertainty": 0.1,
        "sync_age_penalty": 0.1,
        "comm_quality": 0.9,
    }
    first = client.post("/api/v1/trust", json=payload)
    assert first.status_code == 201
    second = client.post("/api/v1/trust", json=payload)
    assert second.status_code == 409


def test_trust_record_for_missing_frame_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/v1/trust",
        json={
            "frame_id": str(uuid.uuid4()),
            "prediction_error": 0.1,
            "prediction_uncertainty": 0.1,
            "sync_age_penalty": 0.1,
            "comm_quality": 0.9,
        },
    )
    assert response.status_code == 404


def test_create_criticality_record(client: TestClient) -> None:
    frame_id = _make_frame(client)
    response = client.post(
        "/api/v1/criticality",
        json={
            "frame_id": frame_id,
            "relative_speed_score": 0.4,
            "blockage_probability_score": 0.2,
            "sync_age_score": 0.1,
            "channel_degradation_score": 0.3,
            "traffic_density_score": 0.5,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert 0.0 <= body["criticality_score"] <= 1.0

    fetch = client.get(f"/api/v1/criticality/frame/{frame_id}")
    assert fetch.status_code == 200
