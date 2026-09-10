"""Pydantic schemas for Experiment."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.experiment import VALID_EXPERIMENT_STATUSES


class ExperimentCreate(BaseModel):
    """Payload to register a new experiment."""

    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=1024)
    config: dict | None = None
    status: str = Field(default="planned", max_length=32)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in VALID_EXPERIMENT_STATUSES:
            raise ValueError(f"status must be one of {VALID_EXPERIMENT_STATUSES}, got {value!r}")
        return value


class ExperimentUpdate(BaseModel):
    """Partial-update payload for an experiment (PATCH)."""

    description: str | None = None
    config: dict | None = None
    status: str | None = None
    latency_ms: float | None = None
    sync_overhead_ms: float | None = None
    energy_j: float | None = None
    reliability_score: float | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is not None and value not in VALID_EXPERIMENT_STATUSES:
            raise ValueError(f"status must be one of {VALID_EXPERIMENT_STATUSES}, got {value!r}")
        return value


class ExperimentRead(BaseModel):
    """Experiment as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    config: dict | None
    status: str
    latency_ms: float | None
    sync_overhead_ms: float | None
    energy_j: float | None
    reliability_score: float | None
    created_at: datetime
    updated_at: datetime
