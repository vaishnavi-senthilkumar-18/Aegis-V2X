"""Populates a CARLA world with vehicles, pedestrians, and roadside units (RSUs).

Density and vehicle-count range come from `configs/simulation.yaml`
(`carla.traffic_density: [dense, sparse]`, `carla.vehicles_per_scene: [30, 60]`)
— Phase 1's frozen parameter contract, owned by Vaishnavi. This module does
not invent its own density levels or ranges; "medium" density does not
exist in the frozen config and must not be reintroduced without updating
`configs/simulation.yaml` and `docs/module_decomposition.md` first.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise ImportError("traffic_generation.traffic_generator requires the 'carla' package.") from exc

# Logger name rooted under "aegis_v2x" so configs/logging.yaml's handlers
# (console + rotating file) apply via propagation, per the frozen logging config.
logger = logging.getLogger("aegis_v2x.simulation.traffic_generation")


@dataclass
class RoadsideUnit:
    """A static CARLA prop actor that also acts as a Sionna RT transmitter/receiver point."""

    actor_id: int
    location: "carla.Location"
    height_m: float


@dataclass
class SpawnedActors:
    vehicles: List["carla.Actor"] = field(default_factory=list)
    pedestrians: List["carla.Actor"] = field(default_factory=list)
    pedestrian_controllers: List["carla.Actor"] = field(default_factory=list)
    roadside_units: List[RoadsideUnit] = field(default_factory=list)

    def all_actor_ids(self) -> List[int]:
        ids = [a.id for a in self.vehicles + self.pedestrians + self.pedestrian_controllers]
        ids += [rsu.actor_id for rsu in self.roadside_units]
        return ids


class TrafficGenerator:
    """Spawns vehicles/pedestrians/RSUs for one scene, per the frozen density model."""

    def __init__(self, world: "carla.World", traffic_manager: "carla.TrafficManager",
                 vehicles_per_scene_range: Tuple[int, int] = (30, 60), seed: int = 42,
                 center_location: "carla.Location" = None, max_radius_m: float = None):
        self._world = world
        self._tm = traffic_manager
        self._vehicles_per_scene_range = vehicles_per_scene_range
        self._rng = random.Random(seed)
        self._center_location = center_location
        self._max_radius_m = max_radius_m

    def _vehicle_count_for_density(self, density: str) -> int:
        lo, hi = self._vehicles_per_scene_range
        midpoint = (lo + hi) / 2.0
        if density == "dense":
            return self._rng.randint(int(round(midpoint)), hi)
        if density == "sparse":
            return self._rng.randint(lo, int(round(midpoint)))
        raise ValueError(f"Unknown traffic_density '{density}'; frozen config only defines dense/sparse.")

    def spawn_vehicles(self, density: str, autopilot: bool = True) -> List["carla.Actor"]:
        blueprint_library = self._world.get_blueprint_library()
        vehicle_blueprints = [bp for bp in blueprint_library.filter("vehicle.*")
                               if int(bp.get_attribute("number_of_wheels")) == 4]

        spawn_points = self._world.get_map().get_spawn_points()
        if self._center_location is not None and self._max_radius_m is not None:
            spawn_points = [p for p in spawn_points if
                            ((p.location.x - self._center_location.x) ** 2 +
                             (p.location.y - self._center_location.y) ** 2) ** 0.5 <= self._max_radius_m]
        self._rng.shuffle(spawn_points)

        target_count = min(self._vehicle_count_for_density(density), len(spawn_points))
        spawned: List["carla.Actor"] = []

        for spawn_point in spawn_points[:target_count]:
            blueprint = self._rng.choice(vehicle_blueprints)
            if blueprint.has_attribute("color"):
                color = self._rng.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)

            actor = self._world.try_spawn_actor(blueprint, spawn_point)
            if actor is None:
                continue
            actor.set_autopilot(autopilot, self._tm.get_port())
            spawned.append(actor)

        if len(spawned) < target_count:
            logger.warning("Requested %d vehicles (%s density), only spawned %d (insufficient spawn points).",
                            target_count, density, len(spawned))
        return spawned

    def spawn_pedestrians(self, density: str, pedestrian_ratio: float = 0.5):
        """Spawns pedestrians at `pedestrian_ratio` x the vehicle count for this density.

        KNOWN ISSUE (flagged 2026-09-05): get_random_location_from_navigation()
        segfaults the whole client process on this CARLA install/map combo,
        very likely a missing/unbuilt pedestrian nav-mesh for this package.
        Pedestrian spawning is disabled until this is root-caused. Vehicles
        and RSUs are unaffected and spawn normally.
        """
        return [], []

    def _spawn_pedestrians_DISABLED(self, density: str, pedestrian_ratio: float = 0.5):
        blueprint_library = self._world.get_blueprint_library()
        walker_blueprints = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        target_count = int(round(self._vehicle_count_for_density(density) * pedestrian_ratio))
        pedestrians: List["carla.Actor"] = []
        controllers: List["carla.Actor"] = []

        for _ in range(target_count):
            spawn_location = self._world.get_random_location_from_navigation()
            if spawn_location is None:
                continue
            blueprint = self._rng.choice(walker_blueprints)
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")

            walker = self._world.try_spawn_actor(blueprint, carla.Transform(spawn_location))
            if walker is None:
                continue

            controller = self._world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller is not None:
                controller.start()
                controller.go_to_location(self._world.get_random_location_from_navigation())
                controller.set_max_speed(1.0 + self._rng.random())
                controllers.append(controller)
            pedestrians.append(walker)

        return pedestrians, controllers

    def spawn_roadside_units(self, count: int, height_m: float = 6.0,
                              placement: str = "intersection_corners") -> List[RoadsideUnit]:
        rsu_locations = self._select_rsu_locations(count, placement)
        rsus: List[RoadsideUnit] = []
        blueprint = self._world.get_blueprint_library().find("static.prop.streetsign")

        for loc in rsu_locations:
            elevated = carla.Location(loc.x, loc.y, loc.z + height_m)
            actor = self._world.try_spawn_actor(blueprint, carla.Transform(elevated))
            if actor is None:
                continue
            rsus.append(RoadsideUnit(actor_id=actor.id, location=elevated, height_m=height_m))
        return rsus

    def _select_rsu_locations(self, count: int, placement: str) -> List["carla.Location"]:
        topology = self._world.get_map().get_topology()
        candidate_points = [seg[0].transform.location for seg in topology]
        if placement == "intersection_corners":
            junctions = {wp.get_junction().id: wp for wp, _ in topology if wp.is_junction and wp.get_junction()}
            candidate_points = [wp.transform.location for wp in junctions.values()] or candidate_points

        if self._center_location is not None and self._max_radius_m is not None:
            filtered = [p for p in candidate_points if
                       ((p.x - self._center_location.x) ** 2 +
                        (p.y - self._center_location.y) ** 2) ** 0.5 <= self._max_radius_m]
            if filtered:
                candidate_points = filtered

        self._rng.shuffle(candidate_points)
        if not candidate_points:
            return []
        return [candidate_points[i % len(candidate_points)] for i in range(count)]

    def build_scene(self, density: str, num_roadside_units: int, autopilot: bool = True) -> SpawnedActors:
        actors = SpawnedActors()
        actors.vehicles = self.spawn_vehicles(density, autopilot=autopilot)
        actors.pedestrians, actors.pedestrian_controllers = self.spawn_pedestrians(density)
        actors.roadside_units = self.spawn_roadside_units(num_roadside_units)
        logger.info("Scene populated: %d vehicles, %d pedestrians, %d RSUs (density=%s)",
                    len(actors.vehicles), len(actors.pedestrians), len(actors.roadside_units), density)
        return actors
