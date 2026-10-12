"""
Perception fault injectors: lidar_noise, occlusion, missing_detections, gps_drift.

See docs/fault_taxonomy.md Sec. 3.B for the severity-to-parameter mapping rationale.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

MAX_LIDAR_NOISE_M = 0.5
MAX_GPS_DRIFT_M = 10.0


def apply_lidar_noise(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Add zero-mean Gaussian noise to each point in the frame's LiDAR point cloud.

    Expects frame["lidar"] to be an (N, 3) or (N, 4) numpy array (x, y, z[, intensity]).
    If lidar data is absent (e.g. stored by file reference only), this is a no-op aside
    from logging, since the actual perturbation would need to happen at load time.
    """
    out = copy.deepcopy(frame)
    sigma = severity * MAX_LIDAR_NOISE_M
    lidar = out.get("lidar")
    if isinstance(lidar, np.ndarray) and sigma > 0:
        noise = rng.normal(loc=0.0, scale=sigma, size=lidar[:, :3].shape)
        lidar = lidar.copy()
        lidar[:, :3] = lidar[:, :3] + noise
        out["lidar"] = lidar
    out.setdefault("_fault_log", []).append(
        {"fault": "lidar_noise", "severity": severity, "sigma_m": sigma}
    )
    return out


def apply_occlusion(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Mask out a contiguous angular fraction of the LiDAR field of view.

    The occluded fraction of points (by azimuth angle) is removed. severity directly
    controls the fraction of the 360-degree field of view blocked.
    """
    out = copy.deepcopy(frame)
    fraction = severity
    lidar = out.get("lidar")
    if isinstance(lidar, np.ndarray) and fraction > 0 and lidar.shape[0] > 0:
        azimuth = np.arctan2(lidar[:, 1], lidar[:, 0])  # radians, range (-pi, pi]
        occlusion_start = rng.uniform(-np.pi, np.pi)
        occlusion_width = fraction * 2 * np.pi
        occlusion_end = occlusion_start + occlusion_width
        # Normalize angles for wraparound comparison
        relative = (azimuth - occlusion_start) % (2 * np.pi)
        mask_occluded = relative <= occlusion_width
        out["lidar"] = lidar[~mask_occluded]
    out.setdefault("_fault_log", []).append(
        {"fault": "occlusion", "severity": severity, "fraction": fraction}
    )
    return out


def apply_missing_detections(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Drop each detected object independently with probability == severity.

    Expects frame["detections"] to be a list of per-object dicts/records. If absent,
    this is a no-op aside from logging.
    """
    out = copy.deepcopy(frame)
    drop_probability = severity
    detections = out.get("detections")
    if isinstance(detections, list) and detections:
        keep_mask = rng.random(len(detections)) >= drop_probability
        out["detections"] = [d for d, keep in zip(detections, keep_mask) if keep]
    out.setdefault("_fault_log", []).append(
        {"fault": "missing_detections", "severity": severity, "drop_probability": drop_probability}
    )
    return out


def apply_gps_drift(
    frame: dict[str, Any], severity: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Add a persistent random-walk-style positional bias to GPS lat/lon.

    Note: bias is drawn once per call (per-frame), not accumulated across frames. A
    scenario-level accumulated drift model is a v2.0 candidate once Phase 2 defines exact
    GPS coordinate conventions.
    """
    out = copy.deepcopy(frame)
    bias_m = severity * MAX_GPS_DRIFT_M
    if "gps_lat" in out and "gps_lon" in out and bias_m > 0:
        # Approximate meters-to-degrees conversion (good enough for synthetic fault injection;
        # not geodetically precise).
        meters_per_degree_lat = 111_320.0
        dlat = rng.normal(0, bias_m) / meters_per_degree_lat
        dlon = rng.normal(0, bias_m) / meters_per_degree_lat
        out["gps_lat"] = out["gps_lat"] + dlat
        out["gps_lon"] = out["gps_lon"] + dlon
    out.setdefault("_fault_log", []).append(
        {"fault": "gps_drift", "severity": severity, "bias_m": bias_m}
    )
    return out
