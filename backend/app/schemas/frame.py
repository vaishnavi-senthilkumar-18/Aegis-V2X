"""Pydantic schemas for Frame."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FrameCreate(BaseModel):
    """Payload to ingest one synchronized multimodal observation."""

    scene_id: uuid.UUID
    vehicle_id: uuid.UUID
    frame_index: int = Field(..., ge=0)

    simulation_timestamp: float
    wireless_timestamp: float

    lidar_path: str | None = None

    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_alt: float | None = None

    position_x: float | None = None
    position_y: float | None = None
    position_z: float | None = None
    lane_id: int | None = None

    imu_data: dict | None = None
    speed_mps: float | None = None

    csi: dict | None = None
    snr_db: float | None = None
    rssi_dbm: float | None = None
    path_loss_db: float | None = None
    beam_index: int | None = None

    traffic_density: float | None = None
    weather: str | None = None

    mobility_state: dict | None = None
    environmental_context: dict | None = None
    prediction_uncertainty: float | None = None
    twin_age_ms: float | None = None

    gt_future_csi: dict | None = None
    gt_future_beam: int | None = None

    source: str = Field(default="synthetic", max_length=32)


class FrameRead(BaseModel):
    """Frame as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    scene_id: uuid.UUID
    vehicle_id: uuid.UUID
    frame_index: int

    simulation_timestamp: float
    wireless_timestamp: float
    sync_timestamp: float
    sync_offset_ms: float
    is_sync_valid: bool

    lidar_path: str | None

    gps_lat: float | None
    gps_lon: float | None
    gps_alt: float | None

    position_x: float | None
    position_y: float | None
    position_z: float | None
    lane_id: int | None

    imu_data: dict | None
    speed_mps: float | None

    csi: dict | None
    snr_db: float | None
    rssi_dbm: float | None
    path_loss_db: float | None
    beam_index: int | None

    traffic_density: float | None
    weather: str | None

    mobility_state: dict | None
    environmental_context: dict | None
    prediction_uncertainty: float | None
    twin_age_ms: float | None

    gt_future_csi: dict | None
    gt_future_beam: int | None

    source: str
    created_at: datetime


class LatestVehicleFrame(FrameRead):
    """A frame joined with its owning vehicle's code, for the radar map."""

    vehicle_code: str


class UnsyncedCountResponse(BaseModel):
    """Response for the sync-health stats endpoint."""

    total_frames: int
    unsynchronized_frames: int
    unsynchronized_ratio: float


class FrameBulkCreate(BaseModel):
    """Payload to ingest many frames in a single request.

    Added to close the gap flagged in `claude/project_status.md`'s "Open
    architecture questions" item 2: the original one-row-per-call
    `POST /frames` was never load-tested at Phase 2's target dataset scale
    (10,000-20,000 frames across 100-150 scenes). Capped at 2000 frames per
    request -- large enough to meaningfully batch, small enough to keep a
    single request's payload and transaction size reasonable; a full
    100-150 scene dataset export should be chunked into multiple bulk
    calls rather than sent as one giant request.
    """

    frames: list[FrameCreate] = Field(..., min_length=1, max_length=2000)


class FrameBulkCreateResponse(BaseModel):
    """Response summarizing a bulk ingestion call."""

    created: int
    unsynchronized: int
    frame_ids: list[uuid.UUID]
