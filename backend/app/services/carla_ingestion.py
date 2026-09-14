"""Assembles real Phase 2 CARLA scene output into `FrameCreate` payloads.

Real scene exports (see `docs/phase2_phase3_reconciliation_2026-09-10.md`)
do not match `FrameCreate`'s one-combined-record shape. Each
`frame_<index>/` directory instead contains, per vehicle:

    vehicle{ID}_gps.json
    vehicle{ID}_imu.json
    vehicle{ID}_vehicle_speed.json
    vehicle{ID}_lidar.npz
    vehicle{ID}_camera.npz

plus one shared, non-per-vehicle file per frame directory:

    vehicle_states.json   -- flat JSON array, one object per vehicle,
                              written by scene_exporter.py from
                              vehicle.get_transform(). Contains
                              position_xyz, velocity_xyz, heading_deg,
                              speed_mps, road_id, lane_id, traffic light
                              state.

This module bridges the gap between that real layout and `FrameCreate`,
without touching `_build_frame`/`create_frame` in `app.crud.frame` at all.

Known open issue, carried over verbatim from the reconciliation doc --
do not silently "fix" this without a team decision:

* No wireless-side timestamp exists in real data yet (Sionna RT has not
  been run for any completed scene). Frames assembled by this module are
  tagged `source="carla_only"` and `wireless_timestamp` is set equal to
  `simulation_timestamp`, which makes `sync_offset_ms == 0` and
  `is_sync_valid == True` by construction. This is a placeholder, not a
  real synchronization measurement -- do not present `is_sync_valid` for
  `carla_only` frames as meaning anything about real sync quality until
  Sionna integration lands.
* `vehicle{ID}_camera.npz` has no home in `FrameCreate` yet. It is
  deliberately not referenced here; add a field/decision before wiring it
  in, rather than silently dropping or silently inventing a mapping.

RESOLVED (2026-09-14, Vaishnavi -- confirmed directly from
`simulation/carla_scenarios/sensors.py` and `scene_exporter.py`, not
guessed):

* `gps_lat`/`gps_lon` ARE genuine GPS-degree values, not a units bug.
  `sensors.py`'s `_attach_gnss` reads CARLA's built-in `sensor.other.gnss`,
  which converts local position to lat/lon via the map's geo-reference
  transform. The tiny values (~0.0026, ~0.00018) are real degrees --
  CARLA's default Town05 map's geo-reference origin sits near (0,0)
  (equator/prime meridian), not any real-world city. No conversion
  needed.
* The Digital Twin radar map plots `position_x`/`position_y`, NOT
  `gps_lat`/`gps_lon` (per `phase3_backend_api_documentation.md`: "GPS is
  too coarse for an intersection-scale radar visualization"). This
  module previously left `position_x/y/z` and `lane_id` as None for
  every frame, which is why the radar map showed no vehicles despite
  scene/frame data being present and correctly ingested. Fixed below:
  `position_xyz` and `lane_id` are now read from each frame's
  `vehicle_states.json` (written by `scene_exporter.py` from
  `vehicle.get_transform().location`) and mapped into
  `position_x`/`position_y`/`position_z`/`lane_id`.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

from app.schemas.frame import FrameCreate

logger = logging.getLogger(__name__)

#: Marks frames assembled from real CARLA-only export data (no Sionna RT
#: wireless data yet), distinct from "synthetic" and the eventual
#: "carla_sionna" value once wireless integration lands.
SOURCE_CARLA_ONLY = "carla_only"

_VEHICLE_ID_RE = re.compile(r"^vehicle(\d+)_")


def discover_vehicle_ids(frame_dir: Path) -> list[str]:
    """Return the distinct vehicle IDs present in one frame directory.

    Derived from filenames (e.g. `vehicle6718_gps.json` -> `"6718"`)
    rather than assumed, since vehicle IDs vary per scene/frame and are
    not zero-padded or globally fixed.
    """
    ids: set[str] = set()
    for f in frame_dir.iterdir():
        m = _VEHICLE_ID_RE.match(f.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids, key=int)


def _read_json(path: Path) -> dict | None:
    """Read one sensor JSON file, or None if it doesn't exist for this vehicle/frame.

    Real exports are not guaranteed to have every sensor file for every
    vehicle in every frame (observed directly during reconciliation, e.g.
    a missing `_gps.json` in some frames) -- a missing file is treated as
    "no reading this frame", not an error.
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_vehicle_states(frame_dir: Path) -> dict[str, dict]:
    """Read `vehicle_states.json` and index it by vehicle_id (as a string).

    `vehicle_states.json` is a flat JSON array of per-vehicle state
    objects (position_xyz, velocity_xyz, heading_deg, speed_mps, road_id,
    lane_id, traffic light state) -- one shared file per frame directory,
    not per-vehicle like the other sensor files. Returns an empty dict if
    the file doesn't exist (older exports, or a frame with no state
    captured).
    """
    path = frame_dir / "vehicle_states.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        states = json.load(fh)
    return {str(s["vehicle_id"]): s for s in states}


def assemble_frame_payload(
    frame_dir: Path,
    vehicle_id_str: str,
    scene_id: uuid.UUID,
    vehicle_id: uuid.UUID,
    vehicle_states: dict[str, dict] | None = None,
) -> FrameCreate | None:
    """Assemble one `FrameCreate` from one vehicle's files in one frame directory.

    Args:
        frame_dir: Path to a `frame_<index>/` directory, e.g.
            `.../straight_road_dense_clear_day_Scene00/frame_122396`.
        vehicle_id_str: The vehicle ID as it appears in filenames (e.g.
            `"147"`), NOT the database `vehicle_id` UUID.
        scene_id: The already-created `Scene` row's UUID this frame belongs to.
        vehicle_id: The already-created `Vehicle` row's UUID this frame belongs to.
        vehicle_states: Pre-loaded, pre-indexed result of
            `_load_vehicle_states(frame_dir)`, keyed by vehicle_id string.
            Pass this in (rather than re-reading `vehicle_states.json`
            once per vehicle) when assembling many vehicles for the same
            frame -- see `assemble_scene_payloads`. If None, this function
            loads it itself (convenient for calling this function alone,
            e.g. in tests).

    Returns:
        A `FrameCreate` ready to pass to `create_frame`/`create_frames_bulk`,
        or None if no sensor files at all were found for this vehicle in
        this frame directory (nothing to ingest).

    `frame_index` is taken from the GPS or IMU file's own internal
    `"frame"` field when available (matching the real, sometimes-lagged
    frame numbering found during reconciliation), falling back to the
    frame directory's own name only if neither file is present.
    """
    prefix = f"vehicle{vehicle_id_str}_"

    gps = _read_json(frame_dir / f"{prefix}gps.json")
    imu = _read_json(frame_dir / f"{prefix}imu.json")
    speed = _read_json(frame_dir / f"{prefix}vehicle_speed.json")

    lidar_file = frame_dir / f"{prefix}lidar.npz"
    lidar_path = str(lidar_file) if lidar_file.exists() else None

    if vehicle_states is None:
        vehicle_states = _load_vehicle_states(frame_dir)
    state = vehicle_states.get(vehicle_id_str)

    if gps is None and imu is None and speed is None and lidar_path is None and state is None:
        logger.warning(
            "No sensor files found for vehicle %s in %s -- skipping",
            vehicle_id_str,
            frame_dir,
        )
        return None

    # Prefer the frame index recorded inside the sensor data itself over
    # the directory name, since reconciliation found these can legitimately
    # differ (the ~90ms GPS lag pattern) -- GPS first, then IMU, then
    # vehicle_states, then the directory name as a last resort.
    frame_index = None
    if gps is not None:
        frame_index = gps.get("frame")
    if frame_index is None and imu is not None:
        frame_index = imu.get("frame")
    if frame_index is None and state is not None:
        frame_index = state.get("frame")
    if frame_index is None:
        dir_match = re.search(r"frame_(\d+)", frame_dir.name)
        frame_index = int(dir_match.group(1)) if dir_match else 0

    # simulation_timestamp: prefer IMU's timestamp (matches its own frame_index
    # exactly in every sample seen so far), fall back to GPS's.
    simulation_timestamp = None
    if imu is not None:
        simulation_timestamp = imu.get("timestamp")
    if simulation_timestamp is None and gps is not None:
        simulation_timestamp = gps.get("timestamp")
    if simulation_timestamp is None:
        logger.warning(
            "No timestamp available for vehicle %s in %s -- skipping",
            vehicle_id_str,
            frame_dir,
        )
        return None

    # No wireless-side timestamp exists in real data yet -- see module
    # docstring. This is a placeholder equal to simulation_timestamp, not
    # a real measurement.
    wireless_timestamp = simulation_timestamp

    position_x = position_y = position_z = None
    lane_id = None
    environmental_context = None
    if state is not None:
        pos = state.get("position_xyz")
        if pos and len(pos) == 3:
            position_x, position_y, position_z = pos
        lane_id = state.get("lane_id")
        # Real traffic-light interaction data, confirmed present in
        # vehicle_states.json (171 real (vehicle, frame) instances found
        # across this scene during manual inspection, 2026-09-14).
        # Stored in environmental_context (the existing generic "E_t"
        # JSON column) rather than adding new dedicated columns, since
        # this is exploratory/first use of the field for real data.
        environmental_context = {
            "is_at_traffic_light": state.get("is_at_traffic_light"),
            "traffic_light_state": state.get("traffic_light_state"),
            "road_id": state.get("road_id"),
        }

    return FrameCreate(
        scene_id=scene_id,
        vehicle_id=vehicle_id,
        frame_index=int(frame_index),
        simulation_timestamp=float(simulation_timestamp),
        wireless_timestamp=float(wireless_timestamp),
        lidar_path=lidar_path,
        gps_lat=gps.get("lat") if gps else None,
        gps_lon=gps.get("lon") if gps else None,
        gps_alt=gps.get("alt") if gps else None,
        position_x=position_x,
        position_y=position_y,
        position_z=position_z,
        lane_id=lane_id,
        environmental_context=environmental_context,
        imu_data=imu,
        speed_mps=speed.get("speed_mps") if speed else None,
        source=SOURCE_CARLA_ONLY,
    )


def assemble_scene_payloads(
    scene_dir: Path,
    scene_id: uuid.UUID,
    vehicle_id_map: dict[str, uuid.UUID],
) -> list[FrameCreate]:
    """Assemble `FrameCreate` payloads for every vehicle in every frame of a scene.

    Args:
        scene_dir: Path to e.g.
            `datasets/raw/carla/.../straight_road_dense_clear_day_Scene00`.
        scene_id: The already-created `Scene` row's UUID.
        vehicle_id_map: Maps filename vehicle-ID strings (e.g. `"147"`) to
            already-created `Vehicle` row UUIDs. Vehicles must be created
            in the database before calling this -- this module does not
            create `Vehicle`/`Scene` rows itself.

    Returns:
        A flat list of `FrameCreate`, ready to pass to `create_frames_bulk`
        in batches of up to 2000 (see `FrameBulkCreate`'s cap).
    """
    payloads: list[FrameCreate] = []
    frame_dirs = sorted(
        (d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")),
        key=lambda d: d.name,
    )

    for frame_dir in frame_dirs:
        vehicle_states = _load_vehicle_states(frame_dir)
        vehicle_id_strs = discover_vehicle_ids(frame_dir)
        for vid_str in vehicle_id_strs:
            db_vehicle_id = vehicle_id_map.get(vid_str)
            if db_vehicle_id is None:
                logger.warning(
                    "Vehicle %s in %s has no corresponding DB vehicle_id -- skipping. "
                    "Create the Vehicle row first.",
                    vid_str,
                    frame_dir,
                )
                continue
            payload = assemble_frame_payload(
                frame_dir, vid_str, scene_id, db_vehicle_id, vehicle_states
            )
            if payload is not None:
                payloads.append(payload)

    logger.info(
        "Assembled %d frame payloads from %d frame directories in %s",
        len(payloads),
        len(frame_dirs),
        scene_dir,
    )
    return payloads
