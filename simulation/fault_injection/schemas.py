"""
Schema definitions for fault injection configs.

Every fault applied to a scenario is described by a FaultConfig. A full injection run is
described by a ScenarioFaultPlan, which bundles a scenario_id, a reproducibility seed, and
one or more FaultConfig entries. This is the YAML-serializable contract referenced in
docs/fault_taxonomy.md Sec. 4.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FaultCategory(str, Enum):
    COMMUNICATION = "communication"
    PERCEPTION = "perception"
    DIGITAL_TWIN = "digital_twin"
    V2X_CONSISTENCY = "v2x_consistency"


class FaultType(str, Enum):
    # Communication
    PACKET_LOSS = "packet_loss"
    LATENCY = "latency"
    JITTER = "jitter"
    BANDWIDTH_REDUCTION = "bandwidth_reduction"
    # Perception
    LIDAR_NOISE = "lidar_noise"
    OCCLUSION = "occlusion"
    MISSING_DETECTIONS = "missing_detections"
    GPS_DRIFT = "gps_drift"
    # Digital Twin
    STALE_STATE = "stale_state"
    SYNC_DELAY = "sync_delay"
    STATE_DRIFT = "state_drift"
    PARTIAL_FAILURE = "partial_failure"
    # V2X consistency
    CONFLICTING_INFO = "conflicting_info"
    STALE_COOPERATIVE_INFO = "stale_cooperative_info"
    MISREPORTED_POSITION = "misreported_position"


# Maps each FaultType to the category it belongs to. Used to validate config consistency.
FAULT_TYPE_CATEGORY: dict[FaultType, FaultCategory] = {
    FaultType.PACKET_LOSS: FaultCategory.COMMUNICATION,
    FaultType.LATENCY: FaultCategory.COMMUNICATION,
    FaultType.JITTER: FaultCategory.COMMUNICATION,
    FaultType.BANDWIDTH_REDUCTION: FaultCategory.COMMUNICATION,
    FaultType.LIDAR_NOISE: FaultCategory.PERCEPTION,
    FaultType.OCCLUSION: FaultCategory.PERCEPTION,
    FaultType.MISSING_DETECTIONS: FaultCategory.PERCEPTION,
    FaultType.GPS_DRIFT: FaultCategory.PERCEPTION,
    FaultType.STALE_STATE: FaultCategory.DIGITAL_TWIN,
    FaultType.SYNC_DELAY: FaultCategory.DIGITAL_TWIN,
    FaultType.STATE_DRIFT: FaultCategory.DIGITAL_TWIN,
    FaultType.PARTIAL_FAILURE: FaultCategory.DIGITAL_TWIN,
    FaultType.CONFLICTING_INFO: FaultCategory.V2X_CONSISTENCY,
    FaultType.STALE_COOPERATIVE_INFO: FaultCategory.V2X_CONSISTENCY,
    FaultType.MISREPORTED_POSITION: FaultCategory.V2X_CONSISTENCY,
}


class FaultConfig(BaseModel):
    """A single fault to apply within a scenario."""

    type: FaultType
    category: FaultCategory
    severity: float = Field(..., ge=0.0, le=1.0)
    start_frame: int = Field(default=0, ge=0)
    end_frame: Optional[int] = Field(
        default=None, description="Inclusive end frame; null means 'rest of scenario'."
    )

    @field_validator("category")
    @classmethod
    def category_must_match_type(cls, v: FaultCategory, info):
        fault_type = info.data.get("type")
        if fault_type is not None:
            expected = FAULT_TYPE_CATEGORY[fault_type]
            if v != expected:
                raise ValueError(
                    f"Fault type '{fault_type}' belongs to category '{expected}', "
                    f"not '{v}'."
                )
        return v

    @field_validator("end_frame")
    @classmethod
    def end_must_be_after_start(cls, v: Optional[int], info):
        start = info.data.get("start_frame", 0)
        if v is not None and v < start:
            raise ValueError(f"end_frame ({v}) must be >= start_frame ({start}).")
        return v

    def is_active_at(self, frame_index: int) -> bool:
        """Whether this fault is active at the given frame index."""
        if frame_index < self.start_frame:
            return False
        if self.end_frame is not None and frame_index > self.end_frame:
            return False
        return True


class ScenarioFaultPlan(BaseModel):
    """A full, reproducible fault-injection plan for one scenario."""

    scenario_id: str
    seed: int
    faults: list[FaultConfig] = Field(default_factory=list)

    def faults_active_at(self, frame_index: int) -> list[FaultConfig]:
        """Return every fault active at a given frame index (faults are composable)."""
        return [f for f in self.faults if f.is_active_at(frame_index)]
