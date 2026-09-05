"""Converts CARLA scene geometry (static map mesh + per-frame dynamic actors)
into a Sionna RT `Scene` ready for ray tracing.

Design note on static geometry: Sionna RT needs a Mitsuba-format static mesh
(buildings, roads, ground) for each CARLA map. CARLA's Python API does not
expose town meshes directly, so the static mesh for each map used in
simulation/carla_scenarios/town_maps.py is exported **once, offline** (Unreal
asset export or the community `carla-scene-export` Blender pipeline) into
`simulation/sionna_configs/static_scenes/<map_name>.xml` (Mitsuba XML). This
module loads that static scene once per map and inserts per-frame dynamic
objects (vehicles, RSUs) from carla_scenarios.scene_exporter's geometry
snapshot on top of it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

try:
    from sionna.rt import Scene, SceneObject, load_scene
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sionna_configs.geometry_adapter requires the 'sionna' package (Sionna RT). "
        "Install it per requirements/simulation.txt on a GPU workstation."
    ) from exc

logger = logging.getLogger("aegis_v2x.simulation.sionna_configs.geometry_adapter")

# Material mapping is a documented internal default: configs/simulation.yaml
# does not specify per-surface radio materials, only that ray tracing is on
# and which outputs to compute.
MATERIAL_MAP = {
    "road": "itu_concrete", "building": "itu_brick", "vehicle": "itu_metal",
    "vegetation": "itu_wood", "ground": "itu_wet_ground", "default": "itu_concrete",
}

_TEMPLATE_MESH = {
    "vehicle": "simulation/sionna_configs/static_scenes/templates/vehicle_box.ply",
    "default": "simulation/sionna_configs/static_scenes/templates/rsu_pole.ply",
}

# --- Real per-map coordinate offsets (queried from live CARLA, 2026-09-05) ---
MAP_OFFSETS = {
    "Town04":   (-24.7, 80.4, 0.0),
    "Town10HD": (4.0, -28.1, 0.0),
}
TOWN03_JUNCTION_ANCHOR = (94.1, -73.4)
TOWN03_ROUNDABOUT_ANCHOR = (-0.4, 23.8)
TOWN03_JUNCTION_LOCAL = (-60.0, 0.0)
TOWN03_ROUNDABOUT_LOCAL = (60.0, 0.0)


def _apply_map_offset(map_name, center_xyz):
    import math
    x, y, z = center_xyz
    if map_name == "Town03":
        d_j = math.hypot(x - TOWN03_JUNCTION_ANCHOR[0], y - TOWN03_JUNCTION_ANCHOR[1])
        d_r = math.hypot(x - TOWN03_ROUNDABOUT_ANCHOR[0], y - TOWN03_ROUNDABOUT_ANCHOR[1])
        if d_j <= d_r:
            ax, ay = TOWN03_JUNCTION_ANCHOR
            lx, ly = TOWN03_JUNCTION_LOCAL
        else:
            ax, ay = TOWN03_ROUNDABOUT_ANCHOR
            lx, ly = TOWN03_ROUNDABOUT_LOCAL
        return (x - ax + lx, y - ay + ly, z)
    ox, oy, oz = MAP_OFFSETS.get(map_name, (0.0, 0.0, 0.0))
    return (x + ox, y + oy, z + oz)



class GeometryAdapterError(RuntimeError):
    """Raised when a required static scene or template asset is missing."""


class GeometryAdapter:
    """Loads a map's static scene once and rebuilds the dynamic-object layer per frame."""

    def __init__(self, static_scene_dir: Path = Path("simulation/sionna_configs/static_scenes"),
                 material_map: Dict[str, str] = None):
        self._static_scene_dir = Path(static_scene_dir)
        self._material_map = material_map or MATERIAL_MAP
        self._scene_cache: Dict[str, Scene] = {}

    def load_static_scene(self, map_name: str) -> Scene:
        if map_name in self._scene_cache:
            return self._scene_cache[map_name]

        scene_path = self._static_scene_dir / f"{map_name}.xml"
        if not scene_path.exists():
            raise GeometryAdapterError(
                f"Static Mitsuba scene for map '{map_name}' not found at {scene_path}. "
                "Export it once offline (see module docstring) before running the channel simulator."
            )
        scene = load_scene(str(scene_path))
        self._scene_cache[map_name] = scene
        logger.info("Loaded static scene for map '%s' from %s", map_name, scene_path)
        return scene

    def build_frame_scene(self, map_name: str, geometry_snapshot: List[dict]) -> Scene:
        scene = self.load_static_scene(map_name)
        scene.edit(remove=[name for name in scene.objects.keys() if name.startswith("dyn_")])
        offset_snapshot = []
        for entry in geometry_snapshot:
            entry = dict(entry)
            entry["center_xyz"] = _apply_map_offset(map_name, entry["center_xyz"])
            offset_snapshot.append(entry)
        scene.edit(add=[self._to_scene_object(entry) for entry in offset_snapshot])
        return scene

    def _to_scene_object(self, entry: dict) -> SceneObject:
        material_class = self._material_map.get(entry.get("material", "default"), self._material_map["default"])
        template_key = "vehicle" if entry.get("material") == "vehicle" else "default"

        obj = SceneObject(fname=_TEMPLATE_MESH[template_key], name=f"dyn_{entry['actor_id']}",
                           radio_material=material_class)
        obj.position = entry["center_xyz"]
        obj.orientation = [entry.get("yaw_deg", 0.0), 0.0, 0.0]
        obj.scaling = [max(entry["extent_xyz"][0] * 2.0, 0.1), max(entry["extent_xyz"][1] * 2.0, 0.1),
                       max(entry["extent_xyz"][2] * 2.0, 0.1)]
        return obj
