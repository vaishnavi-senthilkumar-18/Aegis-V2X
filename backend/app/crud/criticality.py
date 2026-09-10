"""CRUD + weighted-sum math for CriticalityRecord.

Implements `C_t = sum_i alpha_i * f_i` (Mathematical Formulation Sec. 6)
over five normalized features. See `app.models.criticality` for the
default uniform-prior weights.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.metrics import criticality_score_histogram
from app.models.criticality import DEFAULT_CRITICALITY_WEIGHTS, CriticalityRecord
from app.schemas.criticality import CriticalityRecordCreate


def compute_criticality_score(
    relative_speed_score: float,
    blockage_probability_score: float,
    sync_age_score: float,
    channel_degradation_score: float,
    traffic_density_score: float,
    weights: dict[str, float] = DEFAULT_CRITICALITY_WEIGHTS,
) -> float:
    """Compute C_t as the weighted sum of five normalized [0,1] features."""
    return (
        weights["relative_speed"] * relative_speed_score
        + weights["blockage_probability"] * blockage_probability_score
        + weights["sync_age"] * sync_age_score
        + weights["channel_degradation"] * channel_degradation_score
        + weights["traffic_density"] * traffic_density_score
    )


def create_criticality_record(
    db: Session, payload: CriticalityRecordCreate
) -> CriticalityRecord:
    """Compute and persist a criticality record for a frame."""
    criticality_score = compute_criticality_score(
        relative_speed_score=payload.relative_speed_score,
        blockage_probability_score=payload.blockage_probability_score,
        sync_age_score=payload.sync_age_score,
        channel_degradation_score=payload.channel_degradation_score,
        traffic_density_score=payload.traffic_density_score,
    )
    record = CriticalityRecord(
        frame_id=payload.frame_id,
        relative_speed_score=payload.relative_speed_score,
        blockage_probability_score=payload.blockage_probability_score,
        sync_age_score=payload.sync_age_score,
        channel_degradation_score=payload.channel_degradation_score,
        traffic_density_score=payload.traffic_density_score,
        criticality_score=criticality_score,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    criticality_score_histogram.observe(criticality_score)
    return record


def get_criticality_record(db: Session, record_id: uuid.UUID) -> CriticalityRecord | None:
    """Fetch a single criticality record by id."""
    return db.get(CriticalityRecord, record_id)


def get_criticality_record_by_frame(
    db: Session, frame_id: uuid.UUID
) -> CriticalityRecord | None:
    """Fetch the criticality record belonging to a given frame (1:1)."""
    stmt = select(CriticalityRecord).where(CriticalityRecord.frame_id == frame_id)
    return db.execute(stmt).scalar_one_or_none()


def list_criticality_records(
    db: Session, skip: int = 0, limit: int = 100
) -> list[CriticalityRecord]:
    """List criticality records, most recent first."""
    stmt = (
        select(CriticalityRecord)
        .order_by(CriticalityRecord.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(db.execute(stmt).scalars().all())
