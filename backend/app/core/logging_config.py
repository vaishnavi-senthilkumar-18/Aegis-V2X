"""Structured logging setup, applied once at application startup."""

from __future__ import annotations

import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    """Configure root logging with a consistent, timestamped format.

    Idempotent: safe to call multiple times (e.g. once from `app.main` and
    once from test fixtures) without duplicating handlers.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())
