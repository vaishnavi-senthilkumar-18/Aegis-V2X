"""Pydantic schemas for the synthetic-data generation endpoint (dev/demo only)."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class SyntheticSceneRequest(BaseModel):
    """Payload requesting a synthetic scene be generated end-to-end.

    Generates a `Scene`, `num_vehicles` `Vehicle`s converging on a 4-way
    intersection, `num_frames` per vehicle with physically plausible
    motion, and a full `TrustRecord` / `CriticalityRecord` / `Decision`
    chain per frame — a stand-in for the real CARLA + Sionna RT pipeline
    delivered in Phase 2, using identical schema and API contracts so the
    dashboard and downstream phases don't care which source produced the
    data (every synthetic row is tagged `source="synthetic"`).
    """

    scene_code: str = Field(..., min_length=1, max_length=128)
    num_vehicles: int = Field(default=4, ge=1, le=16)
    num_frames: int = Field(default=200, ge=1, le=5000)
    map_name: str = Field(default="Town05", max_length=64)
    weather_preset: str = Field(default="ClearNoon", max_length=64)


class SyntheticSceneResponse(BaseModel):
    """Summary of what the synthetic generator produced."""

    scene_id: uuid.UUID
    scene_code: str
    vehicles_created: int
    frames_created: int
    trust_records_created: int
    criticality_records_created: int
    decisions_created: int
