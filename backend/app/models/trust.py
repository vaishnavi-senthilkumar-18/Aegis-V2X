"""TrustRecord ORM model.

Implements the Calibrated Digital Twin Trust Estimator,
`03_Mathematical_Formulation.docx` Sec. 5:

    z_t = [w1*e_t, w2*u_t, w3*a_t, w4*q_t]
    S_t = sum_i w_i * z_i
    T_t = sigmoid(S_t / tau)

See `app/crud/trust.py::compute_trust_probability` for the implementation,
including the reward/penalty sign convention documented there.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

#: Qualitative interpretation bands reproduced from Sec. 5 of the
#: Mathematical Formulation, in ascending order of trust probability.
TRUST_BANDS: list[tuple[float, str]] = [
    (0.2, "very_unreliable"),
    (0.4, "unreliable"),
    (0.6, "moderate"),
    (0.8, "reliable"),
    (1.01, "highly_reliable"),
]


class TrustRecord(Base):
    """Calibrated trust probability computed for one `Frame`.

    Attributes:
        id: Primary key (UUID).
        frame_id: The frame this trust computation was derived from (1:1).
        prediction_error: e_t — channel/state prediction error (penalty).
        prediction_uncertainty: u_t — model uncertainty (penalty).
        sync_age_penalty: a_t — Digital Twin synchronization age penalty.
        comm_quality: q_t — communication quality signal (reward).
        raw_score: S_t before calibration.
        trust_probability: T_t = sigmoid(S_t / tau), in [0, 1].
        interpretation: Qualitative band from `TRUST_BANDS`.
        created_at: Row creation timestamp.
    """

    __tablename__ = "trust_records"

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

    prediction_error: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_uncertainty: Mapped[float] = mapped_column(Float, nullable=False)
    sync_age_penalty: Mapped[float] = mapped_column(Float, nullable=False)
    comm_quality: Mapped[float] = mapped_column(Float, nullable=False)

    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    trust_probability: Mapped[float] = mapped_column(Float, nullable=False)
    interpretation: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    frame: Mapped[Frame] = relationship(back_populates="trust_record")  # noqa: F821
