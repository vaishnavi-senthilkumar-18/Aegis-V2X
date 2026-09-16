"""Resumable, checkpointed wireless dataset generation pipeline (Stage B Step 6).

Wraps `ChannelSimulator` (Sionna RT 2.1.0, validated Stage B Step 5/5.1) to
generate per-link CSI + metadata output, reusing the existing
`run_channel_sim.py` per-link output convention (one `link_<id>.npz` holding
`csi`, one `link_<id>.json` holding the scalar metadata) so this does not
introduce a second, incompatible on-disk schema.

What `run_channel_sim.py` does not provide, and this module adds:
  - a JSON checkpoint manifest so an interrupted/resumed run skips samples
    already completed, instead of silently redoing (and re-billing storage
    for) them
  - per-sample retry with bounded attempts on failure, with failures
    recorded explicitly (status="failed", with the error), never silently
    dropped
  - per-sample timing/RSS-memory instrumentation, for cost/runtime
    projection before committing to a full production run

This module is scene-agnostic: callers supply an already-built Sionna
`Scene` and a list of `SampleSpec` (per-sample TX/RX positions). It does not
build scenes or pick TX/RX geometry itself -- that stays the caller's
responsibility, since the correct scene/geometry source is itself still an
open question (see Stage B Step 5 finding: the validated Stage A/B 52-mesh
scene uses a synthetic, non-overlapping layout, not real per-actor Town04
world placement; `run_channel_sim.py`'s `GeometryAdapter` path uses
fabricated placeholder static geometry, which this project has explicitly
ruled out as a real-CARLA-geometry source).
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from .channel_simulator import ChannelSimulationResult, ChannelSimulator

logger = logging.getLogger("aegis_v2x.simulation.sionna_configs.wireless_dataset_generator")


@dataclass
class SampleSpec:
    """One generation unit: a (frame, rsu_positions, vehicle_positions) tuple.

    `condition_label` is free-form (e.g. "los_near", "nlos_blocked",
    "multipath_wall") -- not consumed by the generator itself, only carried
    through to the run summary so a pilot run's condition coverage can be
    reported.
    """
    sample_id: str
    frame: int
    wireless_timestamp: float
    rsu_positions: Dict[int, Tuple[float, float, float]]
    vehicle_positions: Dict[int, Tuple[float, float, float]]
    condition_label: str = ""


@dataclass
class SampleOutcome:
    sample_id: str
    status: str  # "completed" | "failed" | "skipped_no_links"
    condition_label: str = ""
    num_links: int = 0
    runtime_s: float = 0.0
    peak_rss_mb: float = 0.0
    bytes_written: int = 0
    error: Optional[str] = None
    attempts: int = 1


class CheckpointManifest:
    """Durable, resumable record of which samples have already been
    generated. Written atomically (write-to-temp + rename) so a crash
    mid-write can't corrupt the manifest itself.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._completed: Dict[str, dict] = {}
        if self.path.exists():
            try:
                with open(self.path) as fh:
                    data = json.load(fh)
                self._completed = data.get("completed", {})
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read existing checkpoint %s (%s); starting fresh.", self.path, exc)

    def is_done(self, sample_id: str) -> bool:
        return sample_id in self._completed

    def mark_done(self, outcome: SampleOutcome) -> None:
        self._completed[outcome.sample_id] = {
            "status": outcome.status, "condition_label": outcome.condition_label,
            "num_links": outcome.num_links, "runtime_s": outcome.runtime_s,
            "peak_rss_mb": outcome.peak_rss_mb, "bytes_written": outcome.bytes_written,
            "error": outcome.error, "attempts": outcome.attempts,
        }
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump({"completed": self._completed}, fh, indent=2)
        os.replace(tmp, self.path)  # atomic on both POSIX and Windows (same volume)

    def summary(self) -> dict:
        by_status: Dict[str, int] = {}
        for v in self._completed.values():
            by_status[v["status"]] = by_status.get(v["status"], 0) + 1
        return {"total_recorded": len(self._completed), "by_status": by_status}


class WirelessDatasetGenerator:
    """Resumable, checkpointed, retrying driver around
    `ChannelSimulator.simulate_frame()`.
    """

    def __init__(self, scene, simulator: ChannelSimulator, output_dir: Path,
                 checkpoint_path: Optional[Path] = None, max_retries: int = 2,
                 retry_backoff_s: float = 1.0):
        self.scene = scene
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = CheckpointManifest(checkpoint_path or (self.output_dir / "_checkpoint.json"))
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._proc = psutil.Process()

    def generate(self, samples: List[SampleSpec], resume: bool = True) -> List[SampleOutcome]:
        outcomes = []
        for spec in samples:
            if resume and self.checkpoint.is_done(spec.sample_id):
                logger.info("Skipping already-completed sample %s (resume)", spec.sample_id)
                continue
            outcome = self._generate_one_with_retry(spec)
            self.checkpoint.mark_done(outcome)
            outcomes.append(outcome)
        return outcomes

    def _generate_one_with_retry(self, spec: SampleSpec) -> SampleOutcome:
        last_error = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return self._generate_one(spec, attempt)
            except Exception as exc:  # noqa: BLE001 -- one bad sample must not abort the run
                last_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
                logger.error("Sample %s attempt %d/%d failed: %s", spec.sample_id, attempt,
                             self.max_retries + 1, exc)
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff_s)
        return SampleOutcome(sample_id=spec.sample_id, status="failed", condition_label=spec.condition_label,
                              error=last_error, attempts=self.max_retries + 1)

    def _generate_one(self, spec: SampleSpec, attempt: int) -> SampleOutcome:
        t0 = time.time()
        rss_before = self._proc.memory_info().rss
        results = self.simulator.simulate_frame(self.scene, spec.rsu_positions, spec.vehicle_positions,
                                                  spec.frame, spec.wireless_timestamp)
        rss_after = self._proc.memory_info().rss
        runtime = time.time() - t0

        if not results:
            return SampleOutcome(sample_id=spec.sample_id, status="skipped_no_links",
                                  condition_label=spec.condition_label, runtime_s=runtime,
                                  peak_rss_mb=rss_after / 1e6, attempts=attempt)

        sample_dir = self.output_dir / spec.sample_id
        sample_dir.mkdir(exist_ok=True)
        bytes_written = sum(self._write_result(sample_dir, r) for r in results)

        return SampleOutcome(
            sample_id=spec.sample_id, status="completed", condition_label=spec.condition_label,
            num_links=len(results), runtime_s=runtime, peak_rss_mb=max(rss_before, rss_after) / 1e6,
            bytes_written=bytes_written, attempts=attempt,
        )

    @staticmethod
    def _write_result(sample_dir: Path, result: ChannelSimulationResult) -> int:
        """Same on-disk shape as run_channel_sim.py's _write_result -- one
        npz (csi) + one json (scalar metadata) per link."""
        prefix = sample_dir / f"link_{result.link_id}"
        csi_path = f"{prefix}.npz"
        np.savez_compressed(csi_path, csi=result.csi)
        metadata = {
            "link_id": result.link_id, "vehicle_id": result.vehicle_id, "frame": result.frame,
            "wireless_timestamp": result.wireless_timestamp, "snr_db": result.snr_db,
            "rssi_dbm": result.rssi_dbm, "path_loss_db": result.path_loss_db,
            "delay_spread_s": result.delay_spread_s, "propagation_delay_s": result.propagation_delay_s,
            "beam_index": result.beam_index, "los": result.los,
            "num_multipath_components": result.num_multipath_components,
        }
        json_path = f"{prefix}.json"
        with open(json_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        return os.path.getsize(csi_path) + os.path.getsize(json_path)
