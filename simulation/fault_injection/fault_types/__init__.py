"""
Dispatch registry mapping FaultType -> implementation function.

This is the single place that needs updating when a new fault type is added — see
docs on "how to add a new fault type" in the module README.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..schemas import FaultType
from . import communication, digital_twin, perception, v2x_consistency

# Every entry has the signature: (frame: dict, severity: float, rng: Generator) -> dict | None
# bandwidth_reduction is handled specially in injector.py since it additionally needs frame_index.
FAULT_DISPATCH: dict[FaultType, Callable[..., dict[str, Any] | None]] = {
    FaultType.PACKET_LOSS: communication.apply_packet_loss,
    FaultType.LATENCY: communication.apply_latency,
    FaultType.JITTER: communication.apply_jitter,
    FaultType.BANDWIDTH_REDUCTION: communication.apply_bandwidth_reduction,
    FaultType.LIDAR_NOISE: perception.apply_lidar_noise,
    FaultType.OCCLUSION: perception.apply_occlusion,
    FaultType.MISSING_DETECTIONS: perception.apply_missing_detections,
    FaultType.GPS_DRIFT: perception.apply_gps_drift,
    FaultType.STALE_STATE: digital_twin.apply_stale_state,
    FaultType.SYNC_DELAY: digital_twin.apply_sync_delay,
    FaultType.STATE_DRIFT: digital_twin.apply_state_drift,
    FaultType.PARTIAL_FAILURE: digital_twin.apply_partial_failure,
    FaultType.CONFLICTING_INFO: v2x_consistency.apply_conflicting_info,
    FaultType.STALE_COOPERATIVE_INFO: v2x_consistency.apply_stale_cooperative_info,
    FaultType.MISREPORTED_POSITION: v2x_consistency.apply_misreported_position,
}

# Fault types whose implementation needs the frame_index as an extra argument.
FAULTS_REQUIRING_FRAME_INDEX = {FaultType.BANDWIDTH_REDUCTION}
