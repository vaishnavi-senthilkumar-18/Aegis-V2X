"""
Communication fault injectors: packet_loss, latency, jitter, bandwidth_reduction.

Each function takes a frame (dict representing one synchronized multimodal sample, per the
Dataset Design Guide's Chapter 10 schema), a severity in [0, 1], and a seeded numpy Generator,
and returns a *new* frame dict with the fault applied. Frames are never mutated in place, so
the same base frame can be reused across multiple fault configs without cross-contamination.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

# Default max-value constants (see docs/fault_taxonomy.md Sec. 3.A for rationale).
# These are intentionally centralized and named, not hardcoded inline, so Vaishnavi can
# tune them without touching injection logic.
MAX_LATENCY_MS = 500.0
MAX_JITTER_MS = 150.0


def apply_packet_loss(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any] | None:
    """Drop the frame entirely with probability equal to severity.

    Returns
    -------
    dict | None
        The original frame if not dropped, or None if dropped (simulating a lost V2X packet).
    """
    drop_probability = severity
    if rng.random() < drop_probability:
        return None
    return copy.deepcopy(frame)


def apply_latency(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Add a fixed delay to the frame's wireless_timestamp, proportional to severity."""
    delay_ms = severity * MAX_LATENCY_MS
    out = copy.deepcopy(frame)
    out["wireless_timestamp"] = out.get("wireless_timestamp", 0.0) + delay_ms
    out.setdefault("_fault_log", []).append(
        {"fault": "latency", "severity": severity, "delay_ms": delay_ms}
    )
    return out


def apply_jitter(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Add zero-mean Gaussian timing noise to the frame's wireless_timestamp."""
    sigma_ms = severity * MAX_JITTER_MS
    jitter_ms = float(rng.normal(loc=0.0, scale=sigma_ms)) if sigma_ms > 0 else 0.0
    out = copy.deepcopy(frame)
    out["wireless_timestamp"] = out.get("wireless_timestamp", 0.0) + jitter_ms
    out.setdefault("_fault_log", []).append(
        {"fault": "jitter", "severity": severity, "jitter_ms": jitter_ms}
    )
    return out


def apply_bandwidth_reduction(
    frame: dict[str, Any],
    severity: float,
    rng: np.random.Generator,
    frame_index: int,
    max_rate_hz: float = 10.0,
) -> dict[str, Any] | None:
    """Simulate a reduced message rate by deterministically dropping frames that exceed
    the reduced capacity, based on frame_index modulo the effective rate ratio.

    Parameters
    ----------
    frame_index : int
        The index of this frame within the scenario's frame sequence (0-based).
    max_rate_hz : float
        The nominal, unthrottled message rate (matches sensor frequency in the Dataset Guide,
        Chapter 8 — CSI at 10 Hz by default).
    """
    effective_rate = max_rate_hz * (1.0 - severity)
    if effective_rate <= 0:
        return None
    keep_every_n = max(1, round(max_rate_hz / effective_rate))
    if frame_index % keep_every_n != 0:
        return None
    return copy.deepcopy(frame)
