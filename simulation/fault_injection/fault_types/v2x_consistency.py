"""
V2X inconsistency fault injectors: conflicting_info, stale_cooperative_info, misreported_position.

These model the "external trust" side of the project (Section 12 of the Master Prompt) —
information received from other vehicles that may be contradictory, stale, or wrong. See
docs/fault_taxonomy.md Sec. 3.D.

All functions expect frame["cooperative_info"] to be a list of per-vehicle report dicts, e.g.:
    [{"vehicle_id": "V02", "position": [x, y, z], "timestamp": t, "reported_objects": [...]}, ...]
If absent, each function is a safe no-op aside from fault logging.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

MAX_STALE_MS = 2000.0
MAX_POSITION_ERROR_M = 15.0


def apply_conflicting_info(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """With probability == severity, duplicate a cooperative report with contradictory
    object data, simulating two vehicles disagreeing about the same object's state.
    """
    out = copy.deepcopy(frame)
    coop = out.get("cooperative_info")
    conflict_probability = severity
    conflicts_injected = 0
    if isinstance(coop, list) and coop:
        new_reports = []
        for report in coop:
            new_reports.append(report)
            if rng.random() < conflict_probability:
                conflicting = copy.deepcopy(report)
                conflicting["vehicle_id"] = f"{report.get('vehicle_id', 'unknown')}_conflict"
                # Flip reported objects' positions to simulate contradictory state.
                for obj in conflicting.get("reported_objects", []):
                    if "position" in obj:
                        obj["position"] = [-c for c in obj["position"]]
                new_reports.append(conflicting)
                conflicts_injected += 1
        out["cooperative_info"] = new_reports
    out.setdefault("_fault_log", []).append(
        {"fault": "conflicting_info", "severity": severity, "conflicts_injected": conflicts_injected}
    )
    return out


def apply_stale_cooperative_info(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Age every received cooperative report's timestamp, simulating delayed V2X relay."""
    out = copy.deepcopy(frame)
    coop = out.get("cooperative_info")
    age_ms = severity * MAX_STALE_MS
    if isinstance(coop, list):
        aged_reports = []
        for report in coop:
            aged = copy.deepcopy(report)
            if "timestamp" in aged:
                aged["timestamp"] = aged["timestamp"] - age_ms
            aged_reports.append(aged)
        out["cooperative_info"] = aged_reports
    out.setdefault("_fault_log", []).append(
        {"fault": "stale_cooperative_info", "severity": severity, "age_ms": age_ms}
    )
    return out


def apply_misreported_position(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Inject positional error into each reporting vehicle's self-reported position."""
    out = copy.deepcopy(frame)
    coop = out.get("cooperative_info")
    error_m = severity * MAX_POSITION_ERROR_M
    if isinstance(coop, list):
        corrupted_reports = []
        for report in coop:
            corrupted = copy.deepcopy(report)
            if "position" in corrupted and error_m > 0:
                position = np.array(corrupted["position"], dtype=float)
                noise = rng.normal(0, error_m, size=position.shape)
                corrupted["position"] = (position + noise).tolist()
            corrupted_reports.append(corrupted)
        out["cooperative_info"] = corrupted_reports
    out.setdefault("_fault_log", []).append(
        {"fault": "misreported_position", "severity": severity, "error_m": error_m}
    )
    return out
