"""Pydantic schemas for TrustRecord."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TrustRecordCreate(BaseModel):
    """Payload for computing and storing a trust record for a frame.

    Only the four raw inputs (e_t, u_t, a_t, q_t) are supplied by the
    caller; `raw_score`, `trust_probability`, and `interpretation` are
    computed server-side by `app.crud.trust.compute_trust_probability`.
    """

    frame_id: uuid.UUID
    prediction_error: float = Field(..., ge=0.0)
    prediction_uncertainty: float = Field(..., ge=0.0)
    sync_age_penalty: float = Field(..., ge=0.0)
    comm_quality: float = Field(..., ge=0.0, le=1.0)


class TrustRecordRead(BaseModel):
    """Trust record as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    frame_id: uuid.UUID
    prediction_error: float
    prediction_uncertainty: float
    sync_age_penalty: float
    comm_quality: float
    raw_score: float
    trust_probability: float
    interpretation: str
    created_at: datetime
