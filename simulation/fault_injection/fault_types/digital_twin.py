"""
Digital Twin degradation fault injectors: stale_state, sync_delay, state_drift, partial_failure.

These directly target the inputs to the Calibrated Twin Trust Estimator defined in
03_Mathematical_Formulation.docx Sec. 5 (e_t, u_t, a_t, q_t) — the point of this fault
category is to test whether the Trust Estimator actually detects and responds to a twin
that has become unreliable. See docs/fault_taxonomy.md Sec. 3.C.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

MAX_STALE_MS = 2000.0
MAX_SYNC_DELAY_MS = 500.0
MAX_DRIFT_UNITS = 1.0  # normalized drift magnitude; see docs Sec. 6 open question re: units


def apply_stale_state(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Inject artificial age into the Digital Twin state (Age_t in the math formulation)."""
    out = copy.deepcopy(frame)
    injected_age_ms = severity * MAX_STALE_MS
    out["twin_age_ms"] = out.get("twin_age_ms", 0.0) + injected_age_ms
    out.setdefault("_fault_log", []).append(
        {"fault": "stale_state", "severity": severity, "injected_age_ms": injected_age_ms}
    )
    return out


def apply_sync_delay(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Delay the twin's sync_timestamp update, simulating a slow synchronization cycle."""
    out = copy.deepcopy(frame)
    delay_ms = severity * MAX_SYNC_DELAY_MS
    out["sync_timestamp"] = out.get("sync_timestamp", 0.0) + delay_ms
    out.setdefault("_fault_log", []).append(
        {"fault": "sync_delay", "severity": severity, "delay_ms": delay_ms}
    )
    return out


def apply_state_drift(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Inject a divergence between the twin's reported state and ground truth.

    Applies to frame["mobility_state"] if present (expected to be a dict of numeric fields,
    e.g. {"speed_mps": ..., "heading_deg": ...}). Each numeric field is perturbed by
    severity-scaled Gaussian noise, representing the twin's internal model diverging from
    physical reality.
    """
    out = copy.deepcopy(frame)
    drift_scale = severity * MAX_DRIFT_UNITS
    mobility = out.get("mobility_state")
    if isinstance(mobility, dict) and drift_scale > 0:
        drifted = {}
        for key, value in mobility.items():
            if isinstance(value, (int, float)):
                drifted[key] = value + rng.normal(0, drift_scale * abs(value) + 1e-6)
            else:
                drifted[key] = value
        out["mobility_state"] = drifted
    out.setdefault("_fault_log", []).append(
        {"fault": "state_drift", "severity": severity, "drift_scale": drift_scale}
    )
    return out


def apply_partial_failure(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Randomly null out individual Digital Twin state fields, simulating partial data loss.

    Only operates on a predefined set of twin-state fields so required scaffolding keys
    (like _fault_log, scene_id, frame_index) are never touched.
    """
    out = copy.deepcopy(frame)
    droppable_fields = [
        "csi", "snr_db", "beam_index", "mobility_state",
        "environmental_context", "prediction_uncertainty",
    ]
    drop_probability = severity
    dropped = []
    for field in droppable_fields:
        if field in out and rng.random() < drop_probability:
            out[field] = None
            dropped.append(field)
    out.setdefault("_fault_log", []).append(
        {"fault": "partial_failure", "severity": severity, "dropped_fields": dropped}
    )
    return out
