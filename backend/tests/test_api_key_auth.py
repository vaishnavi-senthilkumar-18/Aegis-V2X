"""Tests for the optional `X-API-Key` write-endpoint gate.

Covers `app.core.security.require_api_key`: unset by default (unauthenticated
writes, matching the backend's original local-dev-only behavior), and when
`settings.api_key` is set, write endpoints require a matching `X-API-Key`
header while GET endpoints stay open either way.

`require_api_key` does `from app.core.config import settings` and reads
`settings.api_key` directly -- it does not call `get_settings()` fresh on
every request. That means toggling the key via `os.environ` + clearing
`get_settings`'s `lru_cache` (the pattern `tests/conftest.py` uses once, at
import time, for `DATABASE_URL`) would produce a *new* `Settings` instance
that `app.core.security`'s already-bound `settings` name would never see.
Instead, these tests `monkeypatch.setattr` the `api_key` attribute directly
on that shared singleton -- `app.core.security.settings` and
`app.core.config.settings` are the same object -- and `monkeypatch` reverts
the attribute automatically after each test, so no test leaks an API-key
requirement into a later one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.security import settings as security_settings
from tests.conftest import make_scene_and_vehicle as _make_scene_and_vehicle

TEST_API_KEY = "test-secret-key-123"


def test_writes_unauthenticated_by_default(client: TestClient) -> None:
    """No API_KEY configured (the default) -- writes must keep working
    unauthenticated. This must not regress existing behavior."""
    assert security_settings.api_key is None

    response = client.post(
        "/api/v1/scenes", json={"scene_code": "S-noauth", "num_vehicles_target": 1}
    )

    assert response.status_code == 201


def test_write_without_header_rejected_when_api_key_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(security_settings, "api_key", TEST_API_KEY)

    response = client.post(
        "/api/v1/scenes", json={"scene_code": "S-noheader", "num_vehicles_target": 1}
    )

    assert response.status_code == 401


def test_write_with_wrong_header_rejected_when_api_key_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(security_settings, "api_key", TEST_API_KEY)

    response = client.post(
        "/api/v1/scenes",
        json={"scene_code": "S-wrongkey", "num_vehicles_target": 1},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_write_with_correct_header_succeeds_when_api_key_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(security_settings, "api_key", TEST_API_KEY)

    response = client.post(
        "/api/v1/scenes",
        json={"scene_code": "S-correctkey", "num_vehicles_target": 1},
        headers={"X-API-Key": TEST_API_KEY},
    )

    assert response.status_code == 201


def test_get_endpoints_unaffected_when_api_key_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET endpoints must stay open regardless of whether API_KEY is set."""
    scene_id, _vehicle_id = _make_scene_and_vehicle(client)  # write, before the gate is armed
    monkeypatch.setattr(security_settings, "api_key", TEST_API_KEY)

    response = client.get(f"/api/v1/scenes/{scene_id}")

    assert response.status_code == 200


def test_get_endpoints_unaffected_when_api_key_unset(client: TestClient) -> None:
    scene_id, _vehicle_id = _make_scene_and_vehicle(client)

    response = client.get(f"/api/v1/scenes/{scene_id}")

    assert response.status_code == 200
