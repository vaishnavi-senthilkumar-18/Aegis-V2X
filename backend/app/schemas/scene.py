"""Pydantic schemas for Scene and Vehicle."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VehicleCreate(BaseModel):
    """Payload to register a vehicle within a scene."""

    vehicle_code: str = Field(..., min_length=1, max_length=64)
    vehicle_type: str = Field(default="car", max_length=32)
    is_ego: bool = False


class VehicleRead(BaseModel):
    """Vehicle as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    scene_id: uuid.UUID
    vehicle_code: str
    vehicle_type: str
    is_ego: bool
    created_at: datetime


class SceneCreate(BaseModel):
    """Payload to create a new scene."""

    scene_code: str = Field(..., min_length=1, max_length=128)
    map_name: str = Field(default="Town05", max_length=64)
    weather_preset: str = Field(default="ClearNoon", max_length=64)
    num_vehicles_target: int = Field(default=0, ge=0)
    description: str | None = Field(default=None, max_length=512)


class SceneRead(BaseModel):
    """Scene as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    scene_code: str
    map_name: str
    weather_preset: str
    num_vehicles_target: int
    description: str | None
    created_at: datetime


class SceneWithVehicles(SceneRead):
    """Scene with its nested vehicle roster."""

    vehicles: list[VehicleRead] = []
