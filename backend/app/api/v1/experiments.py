"""Experiment endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_api_key
from app.crud import experiment as experiment_crud
from app.schemas.experiment import ExperimentCreate, ExperimentRead, ExperimentUpdate

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post(
    "", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_experiment(payload: ExperimentCreate, db: Session = Depends(get_db)) -> ExperimentRead:
    """Register a new named, config-versioned research run."""
    try:
        experiment = experiment_crud.create_experiment(db, payload)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="An experiment with this name already exists"
        ) from exc
    return ExperimentRead.model_validate(experiment)


@router.get("", response_model=list[ExperimentRead])
def list_experiments(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
) -> list[ExperimentRead]:
    """List experiments, most recently updated first."""
    experiments = experiment_crud.list_experiments(db, skip=skip, limit=limit)
    return [ExperimentRead.model_validate(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentRead)
def get_experiment(experiment_id: uuid.UUID, db: Session = Depends(get_db)) -> ExperimentRead:
    """Fetch a single experiment by id."""
    experiment = experiment_crud.get_experiment(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentRead.model_validate(experiment)


@router.patch(
    "/{experiment_id}", response_model=ExperimentRead,
    dependencies=[Depends(require_api_key)],
)
def update_experiment(
    experiment_id: uuid.UUID, payload: ExperimentUpdate, db: Session = Depends(get_db)
) -> ExperimentRead:
    """Apply a partial update to an experiment (status transitions, metrics, etc.)."""
    experiment = experiment_crud.update_experiment(db, experiment_id, payload)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentRead.model_validate(experiment)
