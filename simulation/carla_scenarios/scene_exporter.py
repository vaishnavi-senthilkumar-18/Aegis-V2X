"""Runs the CARLA tick loop for one scene and exports per-frame physical
state + scene geometry to datasets/raw/carla/<scene_id>/.

Outputs match 09_Dataset_Design_and_Annotation_Guide Chapter 6: vehicle
position/velocity/acceleration/heading, traffic light state, road/lane ID,
weather, obstacle/pedestrian positions, LiDAR, GPS, IMU, vehicle speed,
simulation timestamp.

Also writes a per-frame scene geometry snapshot (dynamic actor bounding
boxes + semantic material tags) consumed by sionna_configs.geometry_adapter
to build the ray-tracing scene for the same frame.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise ImportError("carla_scenarios.scene_exporter requires the 'carla' package.") from exc

from simulation.traffic_generation.traffic_generator import SpawnedActors
from .sensors import SensorReading, SensorRig

logger = logging.getLogger("aegis_v2x.simulation.carla_scenarios.scene_exporter")


@dataclass
class VehicleFrameState:
    vehicle_id: int
    frame: int
    timestamp: float
    position_xyz: List[float]
    velocity_xyz: List[float]
    acceleration_xyz: List[float]
    heading_deg: float
    speed_mps: float
    road_id: int
    lane_id: int
    is_at_traffic_light: bool
    traffic_light_state: str


class SceneExporter:
    """Drains sensor rigs and world state once per tick and writes frame files."""

    def __init__(self, world: "carla.World", actors: SpawnedActors,
                 sensor_rigs: Dict[int, SensorRig], scene_id: str, output_dir: Path):
        self._world = world
        self._actors = actors
        self._sensor_rigs = sensor_rigs
        self._scene_id = scene_id
        self._output_dir = Path(output_dir) / scene_id
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._map = world.get_map()

    def export_frame(self, frame: int, timestamp: float) -> None:
        vehicle_states = self._export_vehicle_states(frame, timestamp)
        readings_by_vehicle = self._drain_sensors(frame, timestamp)
        geometry = self._export_geometry_snapshot()

        self._write_physical_frame(frame, vehicle_states, readings_by_vehicle)
        self._write_geometry_frame(frame, geometry)

    def _export_vehicle_states(self, frame: int, timestamp: float) -> List[VehicleFrameState]:
        states: List[VehicleFrameState] = []
        for vehicle in self._actors.vehicles:
            if not vehicle.is_alive:
                continue
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            accel = vehicle.get_acceleration()
            waypoint = self._map.get_waypoint(transform.location, project_to_road=True)
            speed = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))

            states.append(VehicleFrameState(
                vehicle_id=vehicle.id, frame=frame, timestamp=timestamp,
                position_xyz=[transform.location.x, transform.location.y, transform.location.z],
                velocity_xyz=[velocity.x, velocity.y, velocity.z],
                acceleration_xyz=[accel.x, accel.y, accel.z],
                heading_deg=transform.rotation.yaw, speed_mps=speed,
                road_id=waypoint.road_id if waypoint else -1,
                lane_id=waypoint.lane_id if waypoint else 0,
                is_at_traffic_light=bool(vehicle.is_at_traffic_light()) if hasattr(vehicle, "is_at_traffic_light") else False,
                traffic_light_state=str(vehicle.get_traffic_light_state()) if vehicle.get_traffic_light() else "None",
            ))
        return states

    def _drain_sensors(self, frame: int, timestamp: float) -> Dict[int, List[SensorReading]]:
        readings: Dict[int, List[SensorReading]] = {}
        for vehicle_id, rig in self._sensor_rigs.items():
            rig.read_speed(frame, timestamp)
            readings[vehicle_id] = rig.drain()
        return readings

    def _export_geometry_snapshot(self) -> List[dict]:
        snapshot: List[dict] = []
        for vehicle in self._actors.vehicles:
            if not vehicle.is_alive:
                continue
            bbox = vehicle.bounding_box
            transform = vehicle.get_transform()
            snapshot.append({
                "actor_id": vehicle.id, "material": "vehicle",
                "center_xyz": [transform.location.x, transform.location.y, transform.location.z],
                "extent_xyz": [bbox.extent.x, bbox.extent.y, bbox.extent.z],
                "yaw_deg": transform.rotation.yaw,
            })
        for rsu in self._actors.roadside_units:
            snapshot.append({
                "actor_id": rsu.actor_id, "material": "default", "role": "rsu",
                "center_xyz": [rsu.location.x, rsu.location.y, rsu.location.z],
                "extent_xyz": [0.1, 0.1, 0.1], "yaw_deg": 0.0,
            })
        return snapshot

    def _write_physical_frame(self, frame: int, vehicle_states: List[VehicleFrameState],
                               readings_by_vehicle: Dict[int, List[SensorReading]]) -> None:
        frame_dir = self._output_dir / f"frame_{frame:06d}"
        frame_dir.mkdir(exist_ok=True)

        with open(frame_dir / "vehicle_states.json", "w") as fh:
            json.dump([asdict(s) for s in vehicle_states], fh, indent=2)

        for vehicle_id, readings in readings_by_vehicle.items():
            for reading in readings:
                self._write_sensor_reading(frame_dir, vehicle_id, reading)

    def _write_sensor_reading(self, frame_dir: Path, vehicle_id: int, reading: SensorReading) -> None:
        prefix = frame_dir / f"vehicle{vehicle_id}_{reading.sensor_type}"
        if reading.sensor_type in ("lidar", "camera"):
            np.savez(f"{prefix}.npz", data=reading.data, timestamp=reading.timestamp, frame=reading.frame)
        else:
            payload = {"timestamp": reading.timestamp, "frame": reading.frame, **reading.data}
            with open(f"{prefix}.json", "w") as fh:
                json.dump(payload, fh, indent=2)

    def _write_geometry_frame(self, frame: int, geometry: List[dict]) -> None:
        frame_dir = self._output_dir / f"frame_{frame:06d}"
        frame_dir.mkdir(exist_ok=True)
        with open(frame_dir / "geometry_snapshot.json", "w") as fh:
            json.dump(geometry, fh, indent=2)
