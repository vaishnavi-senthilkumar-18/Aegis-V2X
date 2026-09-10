"""SQLAlchemy engine, session factory, and declarative base.

Single source of truth for database connectivity. `get_db` is the FastAPI
dependency every router uses to obtain a request-scoped session.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Declarative base class shared by every ORM model."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped database session.

    Ensures the session is always closed, even if the request handler
    raises, preventing connection-pool exhaustion under load.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
