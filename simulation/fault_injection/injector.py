"""
Core fault injection engine.

FaultInjector takes a ScenarioFaultPlan and applies it, frame by frame, to a sequence of
scenario frames. It is the single entry point the rest of the project (Phase 2 scenario
pipeline, Phase 7 evaluation harness) should import.

Reproducibility contract: given the same ScenarioFaultPlan.seed and the same input frames,
calling inject() twice must produce identical output, every time. This is enforced by
deriving all randomness from a single numpy.random.Generator seeded once per run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fault_types import FAULT_DISPATCH, FAULTS_REQUIRING_FRAME_INDEX
from .schemas import FaultConfig, ScenarioFaultPlan


@dataclass
class InjectionResult:
    """Result of running a ScenarioFaultPlan over a sequence of frames."""

    scenario_id: str
    seed: int
    frames: list[dict[str, Any]]
    # Per-frame record of which faults were active and what they did, for the scenario
    # catalog / experiment tracking (see docs/fault_taxonomy.md Sec. 4).
    frame_fault_log: list[list[dict[str, Any]]] = field(default_factory=list)
    dropped_frame_indices: list[int] = field(default_factory=list)


class FaultInjector:
    """Applies a ScenarioFaultPlan to a scenario's frame sequence, deterministically."""

    def __init__(self, plan: ScenarioFaultPlan):
        self.plan = plan
        # One Generator per injector instance, seeded once. Re-running inject() on a fresh
        # FaultInjector built from the same plan reproduces identical results; reusing the
        # same injector instance across two inject() calls does NOT (the generator's state
        # advances), which is intentional — construct a new FaultInjector per run.
        self._rng = np.random.default_rng(plan.seed)

    def inject(self, frames: list[dict[str, Any]]) -> InjectionResult:
        """Apply this injector's fault plan to a sequence of frames.

        Parameters
        ----------
        frames : list[dict]
            Scenario frames in frame-index order (frame 0 first), matching the Dataset
            Design Guide's Chapter 10 schema.

        Returns
        -------
        InjectionResult
            The fault-injected frames, plus per-frame fault logs and any frame indices
            that were dropped entirely (e.g. by packet_loss or bandwidth_reduction).
        """
        result_frames: list[dict[str, Any]] = []
        frame_fault_log: list[list[dict[str, Any]]] = []
        dropped_indices: list[int] = []

        for frame_index, frame in enumerate(frames):
            active_faults = self.plan.faults_active_at(frame_index)
            current_frame: dict[str, Any] | None = frame

            for fault_config in active_faults:
                if current_frame is None:
                    # Already dropped by an earlier fault this frame; nothing left to degrade.
                    break
                current_frame = self._apply_single_fault(
                    current_frame, fault_config, frame_index
                )

            if current_frame is None:
                dropped_indices.append(frame_index)
                frame_fault_log.append(
                    [f.model_dump() for f in active_faults]
                )
                continue

            result_frames.append(current_frame)
            frame_fault_log.append(current_frame.get("_fault_log", []))

        return InjectionResult(
            scenario_id=self.plan.scenario_id,
            seed=self.plan.seed,
            frames=result_frames,
            frame_fault_log=frame_fault_log,
            dropped_frame_indices=dropped_indices,
        )

    def _apply_single_fault(
        self,
        frame: dict[str, Any],
        fault_config: FaultConfig,
        frame_index: int,
    ) -> dict[str, Any] | None:
        fn = FAULT_DISPATCH.get(fault_config.type)
        if fn is None:
            raise ValueError(
                f"No implementation registered for fault type: {fault_config.type}"
            )
        if fault_config.type in FAULTS_REQUIRING_FRAME_INDEX:
            return fn(frame, fault_config.severity, self._rng, frame_index=frame_index)
        return fn(frame, fault_config.severity, self._rng)
