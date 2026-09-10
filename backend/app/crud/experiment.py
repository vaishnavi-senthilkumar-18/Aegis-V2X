"""CRUD operations for Experiment."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate


def create_experiment(db: Session, payload: ExperimentCreate) -> Experiment:
    """Register a new experiment."""
    experiment = Experiment(**payload.model_dump())
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def get_experiment(db: Session, experiment_id: uuid.UUID) -> Experiment | None:
    """Fetch a single experiment by id."""
    return db.get(Experiment, experiment_id)


def list_experiments(db: Session, skip: int = 0, limit: int = 100) -> list[Experiment]:
    """List experiments, most recently updated first."""
    stmt = select(Experiment).order_by(Experiment.updated_at.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def update_experiment(
    db: Session, experiment_id: uuid.UUID, payload: ExperimentUpdate
) -> Experiment | None:
    """Apply a partial update to an experiment. Returns None if not found."""
    experiment = get_experiment(db, experiment_id)
    if experiment is None:
        return None
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(experiment, field, value)
    db.commit()
    db.refresh(experiment)
    return experiment
