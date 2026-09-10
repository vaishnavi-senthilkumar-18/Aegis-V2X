"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict[str, str]:
    """Report service and database liveness.

    Returns `status: "ok"` only if a trivial query against the configured
    PostgreSQL database succeeds, so this doubles as a DB connectivity
    check for orchestration/liveness probes.
    """
    db.execute(text("SELECT 1"))
    return {"status": "ok", "service": "aegis-v2x-backend"}
