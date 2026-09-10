"""Shared pytest fixtures.

Tests run against a real local PostgreSQL 16 database
(`aegis_v2x_test`, separate from the development `aegis_v2x` database)
rather than SQLite, so JSON columns, UUID types, and `DISTINCT ON` queries
behave exactly as they do in production. The schema is created fresh via
`Base.metadata.create_all()` at the start of the test session and dropped
at the end; each test function gets a clean set of tables via truncation
in `db_session`.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg2://aegis:aegis@localhost:5432/aegis_v2x_test"
)
os.environ["AEGIS_SKIP_DASHBOARD_MOUNT"] = "1"

# These imports must come after the os.environ lines above -- app.main (and
# anything importing app.core.config) reads DATABASE_URL/settings at import
# time, so importing it first would bake in the wrong database. flake8's
# E402 ("module level import not at top of file") doesn't know that, hence
# the noqa on each: reordering to satisfy it would break the test database
# isolation this file exists to provide.
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402

get_settings.cache_clear()
settings = get_settings()

engine = create_engine(settings.database_url, future=True)
TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@pytest.fixture(scope="session", autouse=True)
def _create_schema() -> Generator[None, None, None]:
    """Create all tables once per test session, drop them at the end."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _truncate_tables() -> Generator[None, None, None]:
    """Truncate every table before each test for isolation."""
    yield
    with engine.begin() as conn:
        table_names = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
        conn.execute(text(f"TRUNCATE {table_names} RESTART IDENTITY CASCADE"))


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """A raw SQLAlchemy session for tests that want to set up data directly."""
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """A FastAPI TestClient wired to the test database via dependency override."""

    def _override_get_db() -> Generator[Session, None, None]:
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_scene_and_vehicle(client: TestClient) -> tuple[str, str]:
    """Test helper: create a scene + single vehicle, return their ids as a tuple."""
    scene_payload = {"scene_code": f"S-{uuid.uuid4()}", "num_vehicles_target": 1}
    scene = client.post("/api/v1/scenes", json=scene_payload).json()
    vehicle_payload = {"vehicle_code": "Vehicle00"}
    vehicle = client.post(f"/api/v1/scenes/{scene['id']}/vehicles", json=vehicle_payload).json()
    return scene["id"], vehicle["id"]


def make_frame(client: TestClient) -> str:
    """Test helper: create a scene, vehicle, and single frame; return the frame id."""
    scene_id, vehicle_id = make_scene_and_vehicle(client)
    frame_payload = {
        "scene_id": scene_id,
        "vehicle_id": vehicle_id,
        "frame_index": 0,
        "simulation_timestamp": 1.0,
        "wireless_timestamp": 1.0,
    }
    frame = client.post("/api/v1/frames", json=frame_payload).json()
    return frame["id"]
