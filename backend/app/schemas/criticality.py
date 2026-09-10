"""Pydantic schemas for CriticalityRecord."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CriticalityRecordCreate(BaseModel):
    """Payload for computing and storing a criticality record for a frame."""

    frame_id: uuid.UUID
    relative_speed_score: float = Field(..., ge=0.0, le=1.0)
    blockage_probability_score: float = Field(..., ge=0.0, le=1.0)
    sync_age_score: float = Field(..., ge=0.0, le=1.0)
    channel_degradation_score: float = Field(..., ge=0.0, le=1.0)
    traffic_density_score: float = Field(..., ge=0.0, le=1.0)


class CriticalityRecordRead(BaseModel):
    """Criticality record as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    frame_id: uuid.UUID
    relative_speed_score: float
    blockage_probability_score: float
    sync_age_score: float
    channel_degradation_score: float
    traffic_density_score: float
    criticality_score: float
    created_at: datetime
