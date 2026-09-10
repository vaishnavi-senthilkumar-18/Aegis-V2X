"""Pydantic schemas for Decision."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.decision import VALID_FSDP_ACTIONS, VALID_PREDICTION_HORIZONS


class DecisionCreate(BaseModel):
    """Payload to record a joint TAHS + FSDP decision for a frame."""

    frame_id: uuid.UUID
    prediction_horizon: int
    fsdp_action: str
    trust_probability_used: float = Field(..., ge=0.0, le=1.0)
    criticality_score_used: float = Field(..., ge=0.0, le=1.0)
    policy_source: str = Field(default="synthetic", max_length=32)
    rationale: str | None = Field(default=None, max_length=512)

    @field_validator("prediction_horizon")
    @classmethod
    def validate_horizon(cls, value: int) -> int:
        """Reject any horizon not in the discretized set from Sec. 7."""
        if value not in VALID_PREDICTION_HORIZONS:
            raise ValueError(
                f"prediction_horizon must be one of {VALID_PREDICTION_HORIZONS}, got {value}"
            )
        return value

    @field_validator("fsdp_action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        """Reject any FSDP action not in the six defined by Sec. 8."""
        if value not in VALID_FSDP_ACTIONS:
            raise ValueError(f"fsdp_action must be one of {VALID_FSDP_ACTIONS}, got {value!r}")
        return value


class DecisionRead(BaseModel):
    """Decision as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    frame_id: uuid.UUID
    prediction_horizon: int
    fsdp_action: str
    trust_probability_used: float
    criticality_score_used: float
    policy_source: str
    rationale: str | None
    created_at: datetime


class ActionDistributionEntry(BaseModel):
    """One bucket of the FSDP action-distribution stats endpoint."""

    fsdp_action: str
    count: int


class ActionDistributionResponse(BaseModel):
    """Response for `GET /decisions/stats/action-distribution`."""

    total_decisions: int
    distribution: list[ActionDistributionEntry]
