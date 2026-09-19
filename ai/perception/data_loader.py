"""Phase 4 data loader — Perception, Trust & Intelligence.

Combines two REAL data sources for every vehicle/frame:
  1. Structured telemetry, fetched live from the Phase 3 backend API
     (position_x/y/z, lane_id, speed_mps, environmental_context, etc.)
  2. Real LiDAR point clouds, loaded directly from disk via each frame's
     `lidar_path` field — confirmed on 2026-09-15 to be real (N, 4)
     float32 x/y/z/intensity arrays, e.g. 5,170 points for one vehicle at
     one frame in the straight-road scene. NOT synthetic/placeholder data.

Designed to run LOCALLY on the machine that has both:
  - Network access to the backend (default http://127.0.0.1:8000)
  - Filesystem access to the paths referenced by `lidar_path`
    (currently C:\\AegisTemp\\... on Vaishnavi's machine — this is a real
    constraint, not an oversight; see the 2026-09-15 handoff discussion).

NOT YET RUN END-TO-END against the live backend as of this file's creation.
Treat every code path here as "should work based on the confirmed API
response shape and confirmed .npz format" rather than "verified" until
someone actually runs it and reports real output. Do not report results
from this module as verified without that step.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import requests

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


@dataclasses.dataclass
class SceneRef:
    """A real scene, as returned by GET /api/v1/scenes. Both fields are
    taken directly from the live backend response — not hardcoded."""

    id: str
    scene_code: str
    num_vehicles_target: int


@dataclasses.dataclass
class PerceptionSample:
    """One real vehicle, at one real frame: telemetry + LiDAR, aligned.

    `lidar_points` is None (not a fabricated empty array) if the .npz
    file referenced by `lidar_path` could not be loaded — callers must
    handle this explicitly rather than assuming every sample has points.
    """

    scene_id: str
    vehicle_code: str
    frame_index: int
    position_xyz: tuple[float, float, float] | None
    lane_id: int | None
    speed_mps: float | None
    lidar_path: str | None
    lidar_points: np.ndarray | None  # (N, 4) float32: x, y, z, intensity
    environmental_context: dict[str, Any] | None
    raw_frame: dict[str, Any]  # full original API response, for anything not modeled above


class Phase4DataLoader:
    """Fetches real telemetry from the backend and real LiDAR from disk,
    for a given scene. Every method either returns real data or raises/
    logs a clear error — nothing here silently substitutes fabricated
    values for missing data.
    """

    def __init__(self, backend_url: str = DEFAULT_BACKEND_URL, timeout_s: float = 30.0):
        self.backend_url = backend_url.rstrip("/")
        self.timeout_s = timeout_s

    def list_scenes(self) -> list[SceneRef]:
        """GET /api/v1/scenes — real scene list, both straight-road and
        roundabout should appear here once the backend is running."""
        resp = requests.get(f"{self.backend_url}/api/v1/scenes", timeout=self.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        # NOTE: earlier manual testing via PowerShell's Invoke-RestMethod
        # showed a {"value": [...], "Count": N} shape -- that turned out to
        # be PowerShell's own display formatting for JSON arrays, not the
        # real API response. The actual endpoint returns a plain JSON list.
        # Handle both shapes defensively rather than assuming either.
        if isinstance(payload, list):
            scenes_raw = payload
        elif isinstance(payload, dict):
            scenes_raw = payload.get("value", [])
        else:
            scenes_raw = []
        return [
            SceneRef(
                id=s["id"],
                scene_code=s["scene_code"],
                num_vehicles_target=s.get("num_vehicles_target", 0),
            )
            for s in scenes_raw
        ]

    def fetch_latest_per_vehicle(self, scene_id: str) -> list[dict[str, Any]]:
        """GET /api/v1/frames/scene/{scene_id}/latest-per-vehicle — real
        per-vehicle telemetry, confirmed working against the live backend
        on 2026-09-15 for the straight-road scene (54 vehicles returned).
        """
        url = f"{self.backend_url}/api/v1/frames/scene/{scene_id}/latest-per-vehicle"
        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("value", [])
        return []

    @staticmethod
    def load_lidar_points(lidar_path: str | None) -> np.ndarray | None:
        """Loads the real point cloud referenced by a frame's lidar_path.

        Confirmed format (2026-09-15, straight-road scene, vehicle156,
        frame_122416): keys ['data', 'timestamp', 'frame'], data shape
        (5170, 4) float32 -- x, y, z, intensity.

        Returns None (not a fabricated empty array) if the path is missing,
        the file doesn't exist, or it can't be parsed as expected -- the
        caller decides how to handle a genuinely missing point cloud rather
        than this function silently inventing one.
        """
        if not lidar_path:
            return None
        path = Path(lidar_path)
        if not path.exists():
            return None
        try:
            with np.load(path) as npz:
                if "data" not in npz:
                    return None
                points = npz["data"]
            if points.ndim != 2 or points.shape[1] not in (3, 4):
                # Real data was confirmed as (N, 4); guard against an
                # unexpected shape rather than silently accepting garbage.
                return None
            return points
        except Exception:
            # Deliberately broad: a corrupted/malformed .npz (as already
            # seen once in the roundabout dataset -- vehicle6724,
            # frame_129013, a genuine BadZipFile) must not crash the whole
            # loader. The missing point cloud is reported as None, not
            # fabricated or silently skipped without trace.
            return None

    def load_scene_samples(self, scene_id: str) -> list[PerceptionSample]:
        """Builds one PerceptionSample per vehicle for the CURRENT latest
        frame of a scene (telemetry + LiDAR, aligned by the same frame
        each vehicle's record already carries).

        NOTE: this loads one (latest) frame per vehicle, not the full
        frame history. A full-sequence loader (for GRU trajectory
        training) is a separate, not-yet-built method -- see TODO below.
        """
        frames = self.fetch_latest_per_vehicle(scene_id)
        samples: list[PerceptionSample] = []
        for frame in frames:
            position = None
            if frame.get("position_x") is not None and frame.get("position_y") is not None:
                position = (
                    frame["position_x"],
                    frame["position_y"],
                    frame.get("position_z", 0.0),
                )
            lidar_path = frame.get("lidar_path")
            samples.append(
                PerceptionSample(
                    scene_id=scene_id,
                    vehicle_code=frame.get("vehicle_code", "unknown"),
                    frame_index=frame.get("frame_index", -1),
                    position_xyz=position,
                    lane_id=frame.get("lane_id"),
                    speed_mps=frame.get("speed_mps"),
                    lidar_path=lidar_path,
                    lidar_points=self.load_lidar_points(lidar_path),
                    environmental_context=frame.get("environmental_context"),
                    raw_frame=frame,
                )
            )
        return samples

    def _resolve_vehicle_id(self, scene_id: str, vehicle_code: str) -> str:
        """GET /api/v1/scenes/{scene_id}/vehicles -- resolves a real vehicle's
        code (e.g. "Vehicle179") to its database id, needed because `GET
        /frames` filters by `vehicle_id` (a UUID), not by code.
        """
        url = f"{self.backend_url}/api/v1/scenes/{scene_id}/vehicles"
        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        vehicles = payload if isinstance(payload, list) else payload.get("value", [])
        for v in vehicles:
            if v.get("vehicle_code") == vehicle_code:
                return v["id"]
        raise ValueError(f"No vehicle with code {vehicle_code!r} found in scene {scene_id}")

    def load_full_sequence(
        self, scene_id: str, vehicle_code: str, page_size: int = 500
    ) -> list[PerceptionSample]:
        """Builds one PerceptionSample per real frame for a single vehicle,
        across its ENTIRE recorded history in a scene -- for GRU trajectory
        training, which needs the full sequence, not just the latest frame.

        Uses the existing `GET /frames?scene_id=&vehicle_id=` endpoint
        (`app.api.v1.frames.list_frames`), which already orders results by
        `Frame.frame_index` ascending (see `app/crud/frame.py::list_frames`)
        -- no new backend endpoint was needed. Paginates via `skip`/`limit`
        since that endpoint defaults to `limit=100` and this scene's real
        vehicles can have far more frames than that in their history.
        """
        vehicle_id = self._resolve_vehicle_id(scene_id, vehicle_code)
        url = f"{self.backend_url}/api/v1/frames"
        samples: list[PerceptionSample] = []
        skip = 0
        while True:
            resp = requests.get(
                url,
                params={"scene_id": scene_id, "vehicle_id": vehicle_id, "skip": skip, "limit": page_size},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            for frame in page:
                position = None
                if frame.get("position_x") is not None and frame.get("position_y") is not None:
                    position = (
                        frame["position_x"],
                        frame["position_y"],
                        frame.get("position_z", 0.0),
                    )
                lidar_path = frame.get("lidar_path")
                samples.append(
                    PerceptionSample(
                        scene_id=scene_id,
                        vehicle_code=vehicle_code,
                        frame_index=frame.get("frame_index", -1),
                        position_xyz=position,
                        lane_id=frame.get("lane_id"),
                        speed_mps=frame.get("speed_mps"),
                        lidar_path=lidar_path,
                        lidar_points=self.load_lidar_points(lidar_path),
                        environmental_context=frame.get("environmental_context"),
                        raw_frame=frame,
                    )
                )
            if len(page) < page_size:
                break
            skip += page_size
        return samples


if __name__ == "__main__":
    # Manual smoke-test entry point. NOT a substitute for real pytest
    # coverage (see Phase 4 acceptance criteria) -- run this first just to
    # sanity-check the loader against your live backend before writing
    # PointPillars/V2X-ViT/GRU code on top of it.
    loader = Phase4DataLoader()
    scenes = loader.list_scenes()
    print(f"Found {len(scenes)} real scene(s):")
    for scene in scenes:
        print(f"  - {scene.scene_code} ({scene.id}), target vehicles: {scene.num_vehicles_target}")

    for scene in scenes:
        print(f"\nLoading samples for {scene.scene_code}...")
        samples = loader.load_scene_samples(scene.id)
        with_lidar = sum(1 for s in samples if s.lidar_points is not None)
        print(f"  {len(samples)} vehicle samples, {with_lidar} with real loaded LiDAR points")
        if samples and samples[0].lidar_points is not None:
            print(f"  Example: {samples[0].vehicle_code} has {samples[0].lidar_points.shape[0]} real points")
