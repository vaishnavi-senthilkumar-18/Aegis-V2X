"""CRUD + calibration math for TrustRecord.

Implements the Calibrated Digital Twin Trust Estimator,
`03_Mathematical_Formulation.docx` Sec. 5:

    z_t = [w1*e_t, w2*u_t, w3*a_t, w4*q_t]
    S_t = sum_i w_i * z_i
    T_t = sigmoid(S_t / tau)

`prediction_error` (e_t), `prediction_uncertainty` (u_t), and
`sync_age_penalty` (a_t) are *penalties*, while `comm_quality` (q_t) is a
*reward*. The spec's summation notation doesn't make this reward/penalty
distinction explicit, so the weighted score here is formed as
`w4*q_t - w1*e_t - w2*u_t - w3*a_t` before calibration — documented inline
at the call site so a reader of the code (not just the paper) understands
the sign convention. The qualitative interpretation bands
(`TRUST_BANDS` in `app.models.trust`) reproduce Sec. 5's table exactly.
"""

from __future__ import annotations

import math
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.metrics import trust_score_histogram
from app.models.trust import TRUST_BANDS, TrustRecord
from app.schemas.trust import TrustRecordCreate

#: Default weights (w1..w4) applied to (e_t, u_t, a_t, q_t) respectively,
#: matching the uniform-prior convention used elsewhere in the schema
#: (criticality weights) pending a learned weighting in Phase 4/5.
DEFAULT_TRUST_WEIGHTS: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

#: Temperature parameter tau controlling calibration sharpness.
DEFAULT_TAU: float = 1.0


def _interpret(trust_probability: float) -> str:
    """Map a trust probability to its qualitative band from Sec. 5."""
    for upper_bound, label in TRUST_BANDS:
        if trust_probability < upper_bound:
            return label
    return TRUST_BANDS[-1][1]


def compute_trust_probability(
    prediction_error: float,
    prediction_uncertainty: float,
    sync_age_penalty: float,
    comm_quality: float,
    weights: tuple[float, float, float, float] = DEFAULT_TRUST_WEIGHTS,
    tau: float = DEFAULT_TAU,
) -> tuple[float, float, str]:
    """Compute (raw_score, trust_probability, interpretation) for one frame.

    Args:
        prediction_error: e_t, channel/state prediction error (penalty).
        prediction_uncertainty: u_t, model uncertainty (penalty).
        sync_age_penalty: a_t, Digital Twin synchronization age penalty.
        comm_quality: q_t, communication quality signal in [0, 1] (reward).
        weights: (w1, w2, w3, w4) applied to (e_t, u_t, a_t, q_t).
        tau: Calibration temperature; higher tau -> softer (less confident) T_t.

    Returns:
        A tuple of (S_t, T_t, interpretation_label).
    """
    w1, w2, w3, w4 = weights
    raw_score = (w4 * comm_quality) - (w1 * prediction_error) - (w2 * prediction_uncertainty) - (
        w3 * sync_age_penalty
    )
    trust_probability = 1.0 / (1.0 + math.exp(-raw_score / tau))
    interpretation = _interpret(trust_probability)
    return raw_score, trust_probability, interpretation


def create_trust_record(db: Session, payload: TrustRecordCreate) -> TrustRecord:
    """Compute and persist a trust record for a frame."""
    raw_score, trust_probability, interpretation = compute_trust_probability(
        prediction_error=payload.prediction_error,
        prediction_uncertainty=payload.prediction_uncertainty,
        sync_age_penalty=payload.sync_age_penalty,
        comm_quality=payload.comm_quality,
    )
    record = TrustRecord(
        frame_id=payload.frame_id,
        prediction_error=payload.prediction_error,
        prediction_uncertainty=payload.prediction_uncertainty,
        sync_age_penalty=payload.sync_age_penalty,
        comm_quality=payload.comm_quality,
        raw_score=raw_score,
        trust_probability=trust_probability,
        interpretation=interpretation,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    trust_score_histogram.observe(trust_probability)
    return record


def get_trust_record(db: Session, record_id: uuid.UUID) -> TrustRecord | None:
    """Fetch a single trust record by id."""
    return db.get(TrustRecord, record_id)


def get_trust_record_by_frame(db: Session, frame_id: uuid.UUID) -> TrustRecord | None:
    """Fetch the trust record belonging to a given frame (1:1)."""
    stmt = select(TrustRecord).where(TrustRecord.frame_id == frame_id)
    return db.execute(stmt).scalar_one_or_none()


def list_trust_records(db: Session, skip: int = 0, limit: int = 100) -> list[TrustRecord]:
    """List trust records, most recent first."""
    stmt = select(TrustRecord).order_by(TrustRecord.created_at.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())
