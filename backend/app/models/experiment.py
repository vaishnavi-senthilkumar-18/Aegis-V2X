"""Experiment ORM model.

Tracks named, config-versioned research runs for Phase 6/7 evaluation and
reproducibility. Objective-function terms (latency, sync overhead, energy,
reliability) correspond to Sec. 9 of the Mathematical Formulation.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

#: Experiment lifecycle states.
VALID_EXPERIMENT_STATUSES: tuple[str, ...] = (
    "planned",
    "running",
    "completed",
    "failed",
    "archived",
)


class Experiment(Base):
    """A named, reproducible research run (baseline or ablation).

    Attributes:
        id: Primary key (UUID).
        name: Human-readable, unique experiment name.
        description: Free-text description of what this experiment tests.
        config: Full experiment configuration (JSON) — scenes referenced,
            hyperparameters, baseline/ablation flags, etc.
        status: One of `VALID_EXPERIMENT_STATUSES`.
        latency_ms: Mean end-to-end latency objective-function term.
        sync_overhead_ms: Mean Digital Twin sync overhead objective-function term.
        energy_j: Mean energy consumption objective-function term (joules).
        reliability_score: Mean reliability objective-function term, in [0, 1].
        created_at: Row creation timestamp.
        updated_at: Row last-update timestamp.
    """

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="planned")

    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    sync_overhead_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_j: Mapped[float | None] = mapped_column(Float, nullable=True)
    reliability_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
