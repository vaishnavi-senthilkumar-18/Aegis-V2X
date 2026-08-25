"""Trust record endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_api_key
from app.crud import frame as frame_crud
from app.crud import trust as trust_crud
from app.schemas.trust import TrustRecordCreate, TrustRecordRead

router = APIRouter(prefix="/trust", tags=["trust"])


@router.post(
    "", response_model=TrustRecordRead, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_trust_record(
    payload: TrustRecordCreate, db: Session = Depends(get_db)
) -> TrustRecordRead:
    """Compute and store a calibrated trust record for a frame.

    A `Frame` must already exist (409 if the frame doesn't exist would be
    ambiguous with a duplicate-record 409, so a missing frame is reported
    as 404 and a duplicate trust record for an already-scored frame as
    409, matching the ingestion-order contract documented in
    `docs/backend_api_documentation.md` §4).
    """
    if frame_crud.get_frame(db, payload.frame_id) is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    if trust_crud.get_trust_record_by_frame(db, payload.frame_id) is not None:
        raise HTTPException(status_code=409, detail="Trust record already exists for this frame")
    try:
        record = trust_crud.create_trust_record(db, payload)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="Trust record already exists for this frame"
        ) from exc
    return TrustRecordRead.model_validate(record)


@router.get("", response_model=list[TrustRecordRead])
def list_trust_records(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
) -> list[TrustRecordRead]:
    """List trust records, most recent first."""
    records = trust_crud.list_trust_records(db, skip=skip, limit=limit)
    return [TrustRecordRead.model_validate(r) for r in records]


@router.get("/frame/{frame_id}", response_model=TrustRecordRead)
def get_trust_record_by_frame(
    frame_id: uuid.UUID, db: Session = Depends(get_db)
) -> TrustRecordRead:
    """Fetch the trust record for a given frame."""
    record = trust_crud.get_trust_record_by_frame(db, frame_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Trust record not found for this frame")
    return TrustRecordRead.model_validate(record)
