"""Criticality record endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_api_key
from app.crud import criticality as criticality_crud
from app.crud import frame as frame_crud
from app.schemas.criticality import CriticalityRecordCreate, CriticalityRecordRead

router = APIRouter(prefix="/criticality", tags=["criticality"])


@router.post(
    "", response_model=CriticalityRecordRead, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_criticality_record(
    payload: CriticalityRecordCreate, db: Session = Depends(get_db)
) -> CriticalityRecordRead:
    """Compute and store a criticality record for a frame."""
    if frame_crud.get_frame(db, payload.frame_id) is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    if criticality_crud.get_criticality_record_by_frame(db, payload.frame_id) is not None:
        raise HTTPException(
            status_code=409, detail="Criticality record already exists for this frame"
        )
    try:
        record = criticality_crud.create_criticality_record(db, payload)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="Criticality record already exists for this frame"
        ) from exc
    return CriticalityRecordRead.model_validate(record)


@router.get("", response_model=list[CriticalityRecordRead])
def list_criticality_records(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
) -> list[CriticalityRecordRead]:
    """List criticality records, most recent first."""
    records = criticality_crud.list_criticality_records(db, skip=skip, limit=limit)
    return [CriticalityRecordRead.model_validate(r) for r in records]


@router.get("/frame/{frame_id}", response_model=CriticalityRecordRead)
def get_criticality_record_by_frame(
    frame_id: uuid.UUID, db: Session = Depends(get_db)
) -> CriticalityRecordRead:
    """Fetch the criticality record for a given frame."""
    record = criticality_crud.get_criticality_record_by_frame(db, frame_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Criticality record not found for this frame")
    return CriticalityRecordRead.model_validate(record)
