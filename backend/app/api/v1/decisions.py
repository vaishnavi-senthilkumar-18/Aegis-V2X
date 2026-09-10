"""Decision endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_api_key
from app.crud import decision as decision_crud
from app.crud import frame as frame_crud
from app.schemas.decision import (
    ActionDistributionEntry,
    ActionDistributionResponse,
    DecisionCreate,
    DecisionRead,
)

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.post(
    "", response_model=DecisionRead, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_decision(payload: DecisionCreate, db: Session = Depends(get_db)) -> DecisionRead:
    """Record the joint TAHS + FSDP decision for a frame."""
    if frame_crud.get_frame(db, payload.frame_id) is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    if decision_crud.get_decision_by_frame(db, payload.frame_id) is not None:
        raise HTTPException(status_code=409, detail="Decision already exists for this frame")
    try:
        decision = decision_crud.create_decision(db, payload)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="Decision already exists for this frame"
        ) from exc
    return DecisionRead.model_validate(decision)


@router.get("", response_model=list[DecisionRead])
def list_decisions(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
) -> list[DecisionRead]:
    """List decisions, most recent first."""
    decisions = decision_crud.list_decisions(db, skip=skip, limit=limit)
    return [DecisionRead.model_validate(d) for d in decisions]


@router.get("/stats/action-distribution", response_model=ActionDistributionResponse)
def get_action_distribution(db: Session = Depends(get_db)) -> ActionDistributionResponse:
    """Return the count of decisions per FSDP action, descending by count.

    Registered before `/{decision handled by frame}` style dynamic routes
    are not applicable here since decisions only expose `/frame/{frame_id}`
    (string-literal collision-free), but this stays above it for
    readability/consistency with the `frames` router's ordering rule.
    """
    distribution = decision_crud.get_action_distribution(db)
    total = sum(count for _, count in distribution)
    return ActionDistributionResponse(
        total_decisions=total,
        distribution=[ActionDistributionEntry(fsdp_action=a, count=c) for a, c in distribution],
    )


@router.get("/frame/{frame_id}", response_model=DecisionRead)
def get_decision_by_frame(frame_id: uuid.UUID, db: Session = Depends(get_db)) -> DecisionRead:
    """Fetch the decision for a given frame."""
    decision = decision_crud.get_decision_by_frame(db, frame_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found for this frame")
    return DecisionRead.model_validate(decision)
