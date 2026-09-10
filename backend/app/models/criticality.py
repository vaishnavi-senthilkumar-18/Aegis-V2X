"""CriticalityRecord ORM model.

Implements `C_t = sum_i alpha_i * f_i` (Mathematical Formulation Sec. 6)
over five features: relative speed, blockage probability, sync age,
channel degradation, and traffic density. Weights default to a uniform
prior (0.2 each, sum = 1); Phases 4/5 may learn non-uniform weights from
data.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

#: Uniform-prior feature weights (alpha_i), sum(alpha_i) == 1.
DEFAULT_CRITICALITY_WEIGHTS: dict[str, float] = {
    "relative_speed": 0.2,
    "blockage_probability": 0.2,
    "sync_age": 0.2,
    "channel_degradation": 0.2,
    "traffic_density": 0.2,
}


class CriticalityRecord(Base):
    """Scene/frame criticality score used to gate resource-intensive actions.

    Attributes:
        id: Primary key (UUID).
        frame_id: The frame this criticality computation was derived from (1:1).
        relative_speed_score: f_1 — normalized relative speed feature.
        blockage_probability_score: f_2 — LoS blockage probability feature.
        sync_age_score: f_3 — normalized Digital Twin sync age feature.
        channel_degradation_score: f_4 — normalized channel degradation feature.
        traffic_density_score: f_5 — normalized local traffic density feature.
        criticality_score: C_t, weighted sum in [0, 1].
        created_at: Row creation timestamp.
    """

    __tablename__ = "criticality_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    frame_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("frames.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    relative_speed_score: Mapped[float] = mapped_column(Float, nullable=False)
    blockage_probability_score: Mapped[float] = mapped_column(Float, nullable=False)
    sync_age_score: Mapped[float] = mapped_column(Float, nullable=False)
    channel_degradation_score: Mapped[float] = mapped_column(Float, nullable=False)
    traffic_density_score: Mapped[float] = mapped_column(Float, nullable=False)

    criticality_score: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    frame: Mapped[Frame] = relationship(back_populates="criticality_record")  # noqa: F821
