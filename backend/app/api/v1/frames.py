"""Frame endpoints.

IMPORTANT — route ordering: literal-path routes (`/frames/stats/...`,
`/frames/scene/...`) MUST be declared before the parameterized
`/frames/{frame_id}` route. FastAPI/Starlette match routes in declaration
order, so registering `{frame_id}` first would shadow every literal path
under it (a real bug caught and fixed during Phase 3 verification — see
`docs/backend_api_documentation.md` §6).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_api_key
from app.crud import frame as frame_crud
from app.schemas.frame import (
    FrameBulkCreate,
    FrameBulkCreateResponse,
    FrameCreate,
    FrameRead,
    LatestVehicleFrame,
    UnsyncedCountResponse,
)

router = APIRouter(prefix="/frames", tags=["frames"])


@router.post(
    "", response_model=FrameRead, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_frame(payload: FrameCreate, db: Session = Depends(get_db)) -> FrameRead:
    """Ingest one synchronized multimodal observation."""
    frame = frame_crud.create_frame(db, payload)
    return FrameRead.model_validate(frame)


@router.post(
    "/bulk", response_model=FrameBulkCreateResponse, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_frames_bulk(
    payload: FrameBulkCreate, db: Session = Depends(get_db)
) -> FrameBulkCreateResponse:
    """Ingest many synchronized multimodal observations in one request.

    A literal-path route (`/frames/bulk`) registered before the
    parameterized `/frames/{frame_id}` route, same ordering rule as the
    other literal routes below (see module docstring) -- otherwise
    `/frames/bulk` would be shadowed and matched as an invalid frame-id
    UUID instead.

    Built for Phase 2's target ingestion volume; see
    `app.crud.frame.create_frames_bulk` for why this is a single
    transaction rather than N calls to `create_frame`.
    """
    frames = frame_crud.create_frames_bulk(db, payload.frames)
    unsynchronized = sum(1 for f in frames if not f.is_sync_valid)
    return FrameBulkCreateResponse(
        created=len(frames),
        unsynchronized=unsynchronized,
        frame_ids=[f.id for f in frames],
    )


@router.get("", response_model=list[FrameRead])
def list_frames(
    scene_id: uuid.UUID | None = None,
    vehicle_id: uuid.UUID | None = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[FrameRead]:
    """List frames, optionally filtered by scene and/or vehicle."""
    frames = frame_crud.list_frames(
        db, scene_id=scene_id, vehicle_id=vehicle_id, skip=skip, limit=limit
    )
    return [FrameRead.model_validate(f) for f in frames]


# --- Literal-path routes registered BEFORE `/{frame_id}` (see module docstring) ---


@router.get("/stats/unsynchronized-count", response_model=UnsyncedCountResponse)
def get_unsynchronized_count(
    scene_id: uuid.UUID | None = None, db: Session = Depends(get_db)
) -> UnsyncedCountResponse:
    """Return sync-health stats: total frames vs. frames outside the 10ms tolerance."""
    total, unsynced = frame_crud.get_unsynchronized_count(db, scene_id=scene_id)
    ratio = (unsynced / total) if total > 0 else 0.0
    return UnsyncedCountResponse(
        total_frames=total, unsynchronized_frames=unsynced, unsynchronized_ratio=ratio
    )


@router.get("/scene/{scene_id}/latest-per-vehicle", response_model=list[LatestVehicleFrame])
def get_latest_per_vehicle(
    scene_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[LatestVehicleFrame]:
    """Return the most recent frame for every vehicle in a scene.

    Backs the Digital Twin dashboard page's radar map: one Postgres
    `DISTINCT ON` query instead of N per-vehicle round trips (added during
    the Phase 3 dashboard rebuild).
    """
    rows = frame_crud.get_latest_frame_per_vehicle_with_codes(db, scene_id)
    return [
        LatestVehicleFrame.model_validate(
            {**FrameRead.model_validate(f).model_dump(), "vehicle_code": code}
        )
        for f, code in rows
    ]


# --- Parameterized route registered LAST ---


@router.get("/{frame_id}", response_model=FrameRead)
def get_frame(frame_id: uuid.UUID, db: Session = Depends(get_db)) -> FrameRead:
    """Fetch a single frame by id."""
    frame = frame_crud.get_frame(db, frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    return FrameRead.model_validate(frame)
