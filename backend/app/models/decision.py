"""Decision ORM model.

Stores the joint TAHS (Trust-Adaptive Horizon Selection) + FSDP output per
frame: `prediction_horizon` (discretized per Sec. 7) and `fsdp_action`
(one of the six actions in Sec. 8). `policy_source` distinguishes Phase
3's placeholder heuristic policy from Phase 5's offline-optimized policy
table — the schema does not change between phases, only who writes to it.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

#: Discretized prediction horizons allowed by Sec. 7 of the Mathematical Formulation.
VALID_PREDICTION_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 8, 10)

#: The six FSDP actions defined in Sec. 8 of the Mathematical Formulation.
VALID_FSDP_ACTIONS: tuple[str, ...] = (
    "maintain_beam",
    "reselect_beam",
    "trigger_resync",
    "downgrade_mode",
    "upgrade_mode",
    "handover",
)


class Decision(Base):
    """The joint prediction-horizon + FSDP-action decision for one `Frame`.

    Attributes:
        id: Primary key (UUID).
        frame_id: The frame this decision was derived from (1:1).
        prediction_horizon: H_t, one of `VALID_PREDICTION_HORIZONS`.
        fsdp_action: One of `VALID_FSDP_ACTIONS`.
        trust_probability_used: T_t snapshot at decision time (denormalized
            for fast dashboard queries without a join).
        criticality_score_used: C_t snapshot at decision time.
        policy_source: "synthetic" (Phase 3 heuristic) or "fsdp_table" (Phase 5).
        rationale: Free-text explanation of why this action was chosen (audit trail).
        created_at: Row creation timestamp.
    """

    __tablename__ = "decisions"

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

    prediction_horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    fsdp_action: Mapped[str] = mapped_column(String(32), nullable=False)

    trust_probability_used: Mapped[float] = mapped_column(Float, nullable=False)
    criticality_score_used: Mapped[float] = mapped_column(Float, nullable=False)

    policy_source: Mapped[str] = mapped_column(String(32), nullable=False, default="synthetic")
    rationale: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    frame: Mapped[Frame] = relationship(back_populates="decision")  # noqa: F821
