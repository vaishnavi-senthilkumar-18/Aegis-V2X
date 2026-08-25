# Aegis-V2X Backend

Phase 3 deliverable (Backend & Dashboard, Logapriya). FastAPI + PostgreSQL
service that owns the Digital Twin data model — scenes, vehicles, frames,
calibrated trust/criticality records, TwinTrust-AP decisions, and research
experiments — and hosts the React dashboard SPA (see `../dashboard/`).

Full schema/API documentation with cross-references to the specification
docs: `../docs/backend_api_documentation.md`.

## Requirements

- Python 3.11+
- PostgreSQL 16 (a local instance is fine for development)

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env if your PostgreSQL credentials differ from the defaults

# create the database/role once (adjust to your local Postgres admin flow):
#   createuser aegis --pwprompt
#   createdb aegis_v2x -O aegis

alembic upgrade head
```

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

- API docs (Swagger): http://localhost:8000/api/docs
- API docs (ReDoc): http://localhost:8000/api/redoc
- Prometheus metrics: http://localhost:8000/metrics
- Dashboard (after `cd ../dashboard && npm install && npm run build`):
  http://localhost:8000/dashboard/

Set `AEGIS_SKIP_DASHBOARD_MOUNT=1` to boot the API without attempting to
mount `dashboard/dist` (useful before the frontend has been built, or when
running the backend standalone for API development).

## Generating demo data (no Phase 2 dependency)

Phase 2 (Simulation & Dataset) has not been built yet. Use the synthetic
data endpoint to populate a realistic scene for local development and
dashboard demos — every row is tagged `source="synthetic"` and uses the
exact same schema real CARLA/Sionna RT ingestion will use:

```bash
curl -X POST http://localhost:8000/api/v1/synthetic/scenes \
  -H "Content-Type: application/json" \
  -d '{"scene_code": "IntersectionDemo01", "num_vehicles": 5, "num_frames": 200}'
```

## Authentication (optional)

Write endpoints (`POST`/`PATCH`) are gated behind an optional shared
`X-API-Key` header, implemented in `app/core/security.py`. By default
`API_KEY` is unset in `.env`/the environment, which makes
`require_api_key` a no-op — writes stay unauthenticated, matching the
original local-dev-only scope. `GET` endpoints are never gated.

To turn the gate on (e.g. before exposing the backend beyond localhost,
such as through a dev tunnel):

```bash
# in backend/.env
API_KEY=some-shared-secret
```

Then every write request must include the header:

```bash
curl -X POST http://localhost:8000/api/v1/scenes \
  -H "Content-Type: application/json" \
  -H "X-API-Key: some-shared-secret" \
  -d '{"scene_code": "Demo01"}'
```

A request without the header, or with the wrong value, gets `401`.

## Bulk frame ingestion

`POST /frames/bulk` accepts `{"frames": [...]}` (1-2000 `FrameCreate`
objects) and inserts them all in a single database transaction, instead of
one HTTP round trip per frame — built for Phase 2's target dataset scale
(10,000-20,000 frames across 100-150 scenes). Out-of-sync frames within
the batch are stored and flagged (`is_sync_valid=False`), never rejected,
same as the single-frame `POST /frames`. See
`app/crud/frame.py::create_frames_bulk` and
`../docs/backend_api_documentation.md` §4 for the response shape.

## Testing

```bash
# create a separate test database once:
#   createdb aegis_v2x_test -O aegis
pytest
ruff check .
```

## Monitoring (optional)

```bash
cd ../configs/monitoring
docker compose -f docker-compose.monitoring.yml up
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / aegis) — "Aegis-V2X Backend
  Overview" dashboard is pre-provisioned.

## Project layout

```
backend/
├── app/
│   ├── core/        # config, database session, logging, Prometheus metrics
│   ├── models/       # SQLAlchemy ORM models (one file per table/domain)
│   ├── schemas/       # Pydantic request/response schemas
│   ├── crud/           # data-access + domain math (trust/criticality calibration)
│   ├── api/v1/           # FastAPI routers, one per resource
│   ├── services/           # cross-cutting business logic (synthetic data generator)
│   └── main.py               # app assembly: CORS, routers, metrics, dashboard mount
├── alembic/                    # migrations
└── tests/                        # pytest suite (real PostgreSQL, not SQLite)
```

## Design decisions worth knowing

- **Route ordering matters.** Literal-path routes (e.g.
  `/frames/stats/unsynchronized-count`) are registered before parameterized
  routes (`/frames/{frame_id}`) in every router — reversing this shadows
  the literal route (see `app/api/v1/frames.py` module docstring).
- **Out-of-tolerance frames are stored, not rejected.** `is_sync_valid` and
  `sync_offset_ms` flag frames outside the 10ms sync tolerance so the
  dashboard's sync-health panel can surface them, rather than silently
  dropping data (`app/crud/frame.py`).
- **The synthetic generator uses the real CRUD layer**, not a shortcut
  path, so metrics, sync validation, and calibration math run identically
  regardless of data source (`app/services/synthetic_data_service.py`).
