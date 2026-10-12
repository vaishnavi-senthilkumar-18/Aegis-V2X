"""
Shared pytest fixtures: synthetic frame generators so fault types can be unit-tested
without needing real CARLA/Sionna RT data from Phase 2.
"""

from __future__ import annotations

import numpy as np
import pytest


def make_synthetic_frame(frame_index: int = 0, num_lidar_points: int = 100) -> dict:
    """Build one synthetic frame matching the Dataset Design Guide's Chapter 10 schema
    closely enough to exercise every fault type.
    """
    rng = np.random.default_rng(seed=1000 + frame_index)
    return {
        "scene_id": "SyntheticScene01",
        "frame_index": frame_index,
        "simulation_timestamp": 1000.0 + frame_index * 100.0,
        "wireless_timestamp": 1000.0 + frame_index * 100.0,
        "sync_timestamp": 1000.0 + frame_index * 100.0,
        "lidar": rng.uniform(-50, 50, size=(num_lidar_points, 3)),
        "gps_lat": 12.9716,
        "gps_lon": 77.5946,
        "csi": {"real": 0.5, "imag": 0.1},
        "snr_db": 25.0,
        "beam_index": 3,
        "twin_age_ms": 0.0,
        "prediction_uncertainty": 0.1,
        "mobility_state": {"speed_mps": 15.0, "heading_deg": 90.0},
        "environmental_context": {"weather": "rain", "traffic_density": "dense"},
        "detections": [
            {"object_id": f"obj_{i}", "class": "vehicle", "position": [i * 2.0, 0.0, 0.0]}
            for i in range(5)
        ],
        "cooperative_info": [
            {
                "vehicle_id": "V02",
                "position": [10.0, 5.0, 0.0],
                "timestamp": 1000.0 + frame_index * 100.0,
                "reported_objects": [{"object_id": "obj_0", "position": [0.0, 0.0, 0.0]}],
            },
            {
                "vehicle_id": "V03",
                "position": [-10.0, 5.0, 0.0],
                "timestamp": 1000.0 + frame_index * 100.0,
                "reported_objects": [{"object_id": "obj_1", "position": [2.0, 0.0, 0.0]}],
            },
        ],
    }


@pytest.fixture
def synthetic_frame() -> dict:
    return make_synthetic_frame(frame_index=0)


@pytest.fixture
def synthetic_scenario() -> list[dict]:
    """A 20-frame synthetic scenario."""
    return [make_synthetic_frame(frame_index=i) for i in range(20)]


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)
