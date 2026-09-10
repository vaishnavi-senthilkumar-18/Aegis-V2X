"""Scene and Vehicle ORM models.

Maps to the Dataset Design & Annotation Guide, Ch. 19 (Metadata
Specification): one `Scene` row per simulation run (a CARLA episode /
Sionna RT ray-tracing run in Phase 2), with `Vehicle` rows for every
traffic participant observed within it.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Scene(Base):
    """A single simulated intersection scenario / episode.

    Attributes:
        id: Primary key (UUID).
        scene_code: Human-readable scenario identifier, e.g. "IntersectionDemo01".
        map_name: CARLA map/town used for this scene.
        weather_preset: CARLA weather preset applied during the run.
        num_vehicles_target: Planned vehicle count for the scenario (actual
            count is the length of `vehicles`).
        description: Free-text scenario description.
        created_at: Row creation timestamp (server-generated).
    """

    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scene_code: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    map_name: Mapped[str] = mapped_column(String(64), nullable=False, default="Town05")
    weather_preset: Mapped[str] = mapped_column(String(64), nullable=False, default="ClearNoon")
    num_vehicles_target: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    vehicles: Mapped[list[Vehicle]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )
    frames: Mapped[list[Frame]] = relationship(  # noqa: F821 - forward ref, see frame.py
        back_populates="scene", cascade="all, delete-orphan"
    )


class Vehicle(Base):
    """A traffic participant (ego or remote) observed within a `Scene`.

    Attributes:
        id: Primary key (UUID).
        scene_id: Owning scene.
        vehicle_code: Scene-scoped identifier, e.g. "Vehicle00". The dashboard
            strips the redundant "Vehicle " literal prefix when rendering
            this since the code already contains it (fixed during the
            Phase 3 dashboard rebuild).
        vehicle_type: e.g. "car", "truck", "motorcycle".
        is_ego: Whether this vehicle carries the Digital Twin / OBU under test.
        created_at: Row creation timestamp.
    """

    __tablename__ = "vehicles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vehicle_code: Mapped[str] = mapped_column(String(64), nullable=False)
    vehicle_type: Mapped[str] = mapped_column(String(32), nullable=False, default="car")
    is_ego: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scene: Mapped[Scene] = relationship(back_populates="vehicles")
    frames: Mapped[list[Frame]] = relationship(  # noqa: F821
        back_populates="vehicle", cascade="all, delete-orphan"
    )
