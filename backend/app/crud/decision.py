"""CRUD operations for Decision, including the action-distribution stat."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.metrics import fsdp_decisions_counter, prediction_horizon_gauge
from app.models.decision import Decision
from app.schemas.decision import DecisionCreate


def create_decision(db: Session, payload: DecisionCreate) -> Decision:
    """Persist a joint TAHS + FSDP decision for a frame."""
    decision = Decision(**payload.model_dump())
    db.add(decision)
    db.commit()
    db.refresh(decision)

    fsdp_decisions_counter.labels(action=decision.fsdp_action).inc()
    prediction_horizon_gauge.set(decision.prediction_horizon)
    return decision


def get_decision(db: Session, decision_id: uuid.UUID) -> Decision | None:
    """Fetch a single decision by id."""
    return db.get(Decision, decision_id)


def get_decision_by_frame(db: Session, frame_id: uuid.UUID) -> Decision | None:
    """Fetch the decision belonging to a given frame (1:1)."""
    stmt = select(Decision).where(Decision.frame_id == frame_id)
    return db.execute(stmt).scalar_one_or_none()


def list_decisions(db: Session, skip: int = 0, limit: int = 100) -> list[Decision]:
    """List decisions, most recent first."""
    stmt = select(Decision).order_by(Decision.created_at.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def get_action_distribution(db: Session) -> list[tuple[str, int]]:
    """Return (fsdp_action, count) pairs across all decisions, descending by count."""
    stmt = (
        select(Decision.fsdp_action, func.count().label("count"))
        .group_by(Decision.fsdp_action)
        .order_by(func.count().desc())
    )
    return [(row[0], row[1]) for row in db.execute(stmt).all()]
