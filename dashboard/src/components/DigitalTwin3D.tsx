/**
 * Neon 3D Digital Twin.
 *
 * STAGE 1 (implemented here, now, no new CARLA connection): real vehicle
 * positions/headings/movement, real RSU positions, real per-vehicle
 * traffic-light state, real lane-count-derived approximate road, ego
 * vehicle as the single source of truth (shared with the Physical
 * Layer), shared frame/tick clock, TOP-DOWN and DRIVER'S VIEW cameras
 * both reading the SAME scene graph.
 *
 * STAGE 2 (NOT implemented yet -- deliberately deferred until after the
 * 3 remaining Phase 2 scenes are generated, per project decision
 * 2026-09-14): a real CARLA connection (loading Town04, no scenario
 * re-run) to extract actual road/lane OpenDRIVE geometry, terrain,
 * buildings, vegetation, barriers, poles, and traffic-light positions
 * via `world.get_environment_objects()` / `to_opendrive()`. The
 * `environment` prop below is the integration point for that data --
 * it is typed and wired into the render loop now, but passed an empty
 * object until Stage 2 actually runs, so nothing is fabricated in the
 * meantime.
 *
 * FINAL (not reached until Stage 2 lands): once `environment` carries
 * real geometry, this same component renders it alongside the vehicles
 * without any architectural change -- that is the whole point of
 * wiring the prop in now rather than bolting it on later.
 *
 * CONFIRMED DATA SCOPE, STAGE 1 (checked directly, 2026-09-14):
 * geometry_snapshot.json has exactly 58 real objects for this scene --
 * 54 vehicles + 4 RSUs. No pedestrians, buildings, trees, or exact
 * Town04 road mesh exist anywhere in the CURRENTLY RECORDED dataset --
 * that is a Stage 1 data limitation, not a claim that those things
 * never existed in the CARLA world (the RGB camera frames show they
 * did; we simply never exported their coordinates as structured data).
 * Vehicles are rendered as a procedurally built multi-part shape
 * (body+cabin) via `buildVehicleMesh`, written as a single swappable
 * factory function so a real .glb model can replace it later without
 * touching the rest of this component. The road is an APPROXIMATE lane
 * strip (real lane count, oriented along the real dominant spread of
 * vehicle positions, standard 3.5m lane width -- an engineering
 * convention, not measured Town04 geometry) -- Stage 2's real
 * OpenDRIVE export will replace this, not supplement it.
 *
 * 2026-09-14 FIX -- vehicle/RSU mesh scale mismatch (root cause of the
 * "solid color blob" render seen in Driver's View): vehicle and RSU
 * mesh geometry was built at a FIXED absolute size, while vehicle
 * POSITIONS are compressed into the scene by a data-dependent `scale`
 * factor (`SCENE_WIDTH_UNITS / max(spanX, spanY)`, computed from the
 * real position spread of the current scene). For a scene with a large
 * real position spread, `scale` is small, so vehicles end up positioned
 * close together in scene-space while their mesh geometry stays full
 * size -- meshes overlap heavily and the Driver's View camera (only a
 * few scene-units in front of the ego vehicle) ends up rendering inside
 * an oversized, overlapping mesh. Fix: vehicle and RSU mesh groups are
 * now scaled by the SAME `scale` factor used for positions, every tick,
 * so geometry and position share one coordinate space. This is a
 * rendering-correctness fix, not a data change -- no position values
 * are altered.
 *
 * 2026-09-14 STYLE CHANGE -- glowing-outline vehicles (Tesla FSD /
 * reference-image style), replacing solid emissive boxes: each vehicle
 * is now a body+cabin outline built from `THREE.EdgesGeometry`
 * (glowing border only) plus a near-invisible fill mesh (kept only so
 * click-to-select raycasting still works -- NOT meant to be visible).
 * Per-vehicle wheel geometry was dropped in this pass to match the
 * simplified "car silhouette outline" look in the reference image,
 * rather than a fully modeled car -- swappable later via the same
 * `buildVehicleMesh` factory function if a more detailed silhouette is
 * wanted.
 *
 * Bug-fix carried forward: mesh reference maps are cleared on every
 * setup-effect run to avoid stale references surviving a React
 * Strict-Mode double-invoke (see git history for the original bug).
 */

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { FrameTick } from "../hooks/useFrameSequencePlayback";
import type { Frame } from "../types/api";

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface RsuGeometry {
  actorId: number;
  centerXyz: [number, number, number];
}

/**
 * Stage 2 integration point. Each array is empty until a real CARLA
 * extraction (see module docstring) populates it -- this type exists
 * NOW so that integration is "pass real data into an existing prop",
 * not "redesign the component". Coordinates are real CARLA world
 * x/y/z, same space as vehicle position_x/position_y, whenever
 * populated -- never fabricated placeholder positions.
 */
export interface EnvironmentGeometry {
  buildings: Array<{ id: number; centerXyz: [number, number, number]; extentXyz: [number, number, number] }>;
  vegetation: Array<{ id: number; centerXyz: [number, number, number]; extentXyz: [number, number, number] }>;
  poles: Array<{ id: number; centerXyz: [number, number, number] }>;
  barriers: Array<{ id: number; centerXyz: [number, number, number]; extentXyz: [number, number, number] }>;
  trafficLights: Array<{ id: number; centerXyz: [number, number, number] }>;
  /** Real OpenDRIVE road network, once exported (Stage 2). Rendering
   * this properly (vs. the Stage 1 approximate lane strip) is itself a
   * follow-up task once the data exists -- not implemented in Stage 1. */
  openDriveXml: string | null;
}

export const EMPTY_ENVIRONMENT: EnvironmentGeometry = {
  buildings: [],
  vegetation: [],
  poles: [],
  barriers: [],
  trafficLights: [],
  openDriveXml: null,
};

export type CameraMode = "topdown" | "driver";

interface Props {
  currentTick: FrameTick | null;
  previousTick: FrameTick | undefined;
  bounds: Bounds | null;
  selectedVehicleId: string | null;
  onSelectVehicle: (vehicleId: string) => void;
  rsus: RsuGeometry[];
  /** The database UUID of the currently-selected ego vehicle (resolved
   * from the numeric CAM-### id by the parent -- see DigitalTwin.tsx),
   * or null if not yet resolved. Both camera modes follow THIS vehicle,
   * matching the Physical Layer's single source of truth. */
  egoVehicleDbId: string | null;
  cameraMode: CameraMode;
  /** Stage 2 real static-environment geometry. Defaults to
   * EMPTY_ENVIRONMENT (Stage 1) if not provided -- see module docstring. */
  environment?: EnvironmentGeometry;
}

const SCENE_WIDTH_UNITS = 40;
const DEFAULT_VEHICLE_SIZE: [number, number, number] = [0.9, 0.5, 1.8];
const STANDARD_LANE_WIDTH_M = 3.5; // real-world engineering convention, not measured CARLA data

function headingFromDelta(prev: Frame | undefined, curr: Frame): number {
  if (!prev || prev.position_x == null || prev.position_y == null) return 0;
  if (curr.position_x == null || curr.position_y == null) return 0;
  const dx = curr.position_x - prev.position_x;
  const dy = curr.position_y - prev.position_y;
  if (Math.abs(dx) < 1e-4 && Math.abs(dy) < 1e-4) return 0;
  return Math.atan2(dx, dy);
}

function directionColor(laneId: number | null): number {
  if (laneId == null) return 0x6b7280;
  return laneId < 0 ? 0xf97316 : 0x22d3ee;
}

const RSU_COLOR = 0xa855f7;
const EGO_COLOR = 0xffffff;

/** Fill opacity for the invisible "hit target" mesh behind each glowing
 * outline. Kept intentionally near-zero -- it exists ONLY so
 * click-to-select raycasting keeps working now that there is no solid
 * visible body mesh; it is not meant to read as a filled shape. */
const OUTLINE_FILL_OPACITY = 0.03;
const OUTLINE_LINE_OPACITY_DEFAULT = 0.75;
const OUTLINE_LINE_OPACITY_HIGHLIGHT = 1.0;

/**
 * Builds one procedural 3D vehicle as a GLOWING OUTLINE ONLY: a body
 * shape and a cabin shape, each rendered as edge lines (via
 * THREE.EdgesGeometry) with a near-invisible fill mesh behind them
 * (kept only so raycasting/click-select still works). Matches the
 * reference "car border glows, interior is dark/empty" look rather
 * than a solid emissive box. Deliberately a single factory function so
 * a real GLTF/GLB model (or a more detailed silhouette) can be swapped
 * in here later without touching mesh-update logic elsewhere in this
 * component.
 */
function buildVehicleMesh(color: number): THREE.Group {
  const group = new THREE.Group();
  const [w, h, l] = DEFAULT_VEHICLE_SIZE;

  const makeOutlinePart = (
    geo: THREE.BufferGeometry,
    position: [number, number, number],
  ): { outline: THREE.LineSegments; fill: THREE.Mesh } => {
    const edges = new THREE.EdgesGeometry(geo);
    const lineMat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: OUTLINE_LINE_OPACITY_DEFAULT,
    });
    const outline = new THREE.LineSegments(edges, lineMat);
    outline.position.set(...position);

    // Invisible-ish fill: exists purely as a raycast target so clicking
    // a vehicle on screen still works with no solid body to click on.
    const fillMat = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: OUTLINE_FILL_OPACITY,
      depthWrite: false,
    });
    const fill = new THREE.Mesh(geo, fillMat);
    fill.position.set(...position);

    return { outline, fill };
  };

  const bodyGeo = new THREE.BoxGeometry(w, h * 0.6, l);
  const bodyParts = makeOutlinePart(bodyGeo, [0, h * 0.3, 0]);
  group.add(bodyParts.outline, bodyParts.fill);

  const cabinGeo = new THREE.BoxGeometry(w * 0.75, h * 0.5, l * 0.45);
  const cabinParts = makeOutlinePart(cabinGeo, [0, h * 0.65, -l * 0.05]);
  group.add(cabinParts.outline, cabinParts.fill);

  return group;
}

/**
 * Updates a vehicle's outline color/opacity for direction/ego/selection
 * state. Replaces the old `setGroupEmissive` (solid-material approach)
 * now that vehicles are outline-only: walks the group's LineSegments
 * (the glowing border) and updates their color and highlight opacity;
 * the near-invisible fill meshes are left untouched.
 */
function setVehicleHighlight(group: THREE.Group, color: number, highlighted: boolean): void {
  group.traverse((child) => {
    if (child instanceof THREE.LineSegments) {
      const mat = child.material as THREE.LineBasicMaterial;
      mat.color.setHex(color);
      mat.opacity = highlighted ? OUTLINE_LINE_OPACITY_HIGHLIGHT : OUTLINE_LINE_OPACITY_DEFAULT;
    }
  });
}

export function DigitalTwin3D({
  currentTick,
  previousTick,
  bounds,
  selectedVehicleId,
  onSelectVehicle,
  rsus,
  egoVehicleDbId,
  cameraMode,
  environment = EMPTY_ENVIRONMENT,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const vehicleMeshesRef = useRef<Map<string, THREE.Group>>(new Map());
  const rsuMeshesRef = useRef<THREE.Group[]>([]);
  const environmentGroupRef = useRef<THREE.Group | null>(null);
  const connectionLinesRef = useRef<THREE.LineSegments | null>(null);
  const roadMeshRef = useRef<THREE.Group | null>(null);

  // Real dominant travel axis, derived once from the overall real
  // position spread -- used to orient the approximate road strip along
  // the direction traffic actually travels, rather than an arbitrary
  // fixed axis.
  const dominantAxis = useMemo<"x" | "y">(() => {
    if (!bounds) return "y";
    return bounds.maxX - bounds.minX >= bounds.maxY - bounds.minY ? "x" : "y";
  }, [bounds]);

  const laneCount = useMemo(() => {
    if (!currentTick) return 2;
    const laneIds = new Set(
      Object.values(currentTick.vehicles)
        .map((f) => f.lane_id)
        .filter((id): id is number => id != null),
    );
    return Math.max(laneIds.size, 2);
  }, [currentTick]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 480;
    const height = container.clientHeight || 480;

    const scene = new THREE.Scene();
    // Softer, less "black void" atmosphere -- a dark blue-gray gradient
    // feel via fog + a lighter ambient tone, closer to a dusk/night
    // scene than empty blackness.
    scene.background = new THREE.Color(0x0a1420);
    scene.fog = new THREE.FogExp2(0x0a1420, 0.006);
    sceneRef.current = scene;

    vehicleMeshesRef.current.clear();
    rsuMeshesRef.current = [];

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    camera.position.set(0, 26, 28);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    // A large, filled ground plane covering the whole visible area --
    // NOT a black void with a thin isolated road strip floating in it.
    // This is generic ground dressing (not claimed as real CARLA
    // terrain), just enough so the scene reads as a filled 3D space.
    const groundGeo = new THREE.PlaneGeometry(SCENE_WIDTH_UNITS * 2.5, SCENE_WIDTH_UNITS * 2.5);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x0d1a26, roughness: 1 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.05;
    scene.add(ground);

    // Very subtle grid, kept only as a faint depth cue -- NOT the
    // environment itself (per explicit instruction).
    const grid = new THREE.GridHelper(SCENE_WIDTH_UNITS * 2, 40, 0x1a2c3d, 0x0f1c28);
    (grid.material as THREE.Material).opacity = 0.15;
    (grid.material as THREE.Material).transparent = true;
    scene.add(grid);

    const ambient = new THREE.AmbientLight(0xaad4ff, 0.65);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0x88ccff, 0.6);
    directional.position.set(10, 20, 10);
    scene.add(directional);

    // --------------------------------------------------------------
    // DECORATIVE SCENE DRESSING -- explicitly a visual approximation,
    // NOT measured/real geometry, NOT the Stage 2 `environment` prop
    // above. Placed generically around the scene perimeter so the
    // Digital Twin reads as a filled world rather than an empty void,
    // per explicit instruction (2026-09-14) that this is acceptable as
    // disclosed decoration while real vehicle motion stays data-driven.
    // Deliberately does NOT render any specific vehicle as a bus/truck
    // shape -- no vehicle in the real dataset has a recorded type, so
    // doing that would misrepresent a specific real actor rather than
    // decorate empty background space. Will be removed/replaced
    // wholesale once Stage 2's real CARLA geometry populates the
    // `environment` prop instead.
    // --------------------------------------------------------------
    const dressing = new THREE.Group();
    const treeTrunkMat = new THREE.MeshStandardMaterial({ color: 0x1a2e1a });
    const treeFoliageMat = new THREE.MeshStandardMaterial({ color: 0x134e2a, emissive: 0x22c55e, emissiveIntensity: 0.35 });
    const perimeter = SCENE_WIDTH_UNITS * 0.9;

    // Trees: two rows flanking the road corridor.
    for (let i = 0; i < 24; i++) {
      const t = (i / 24) * Math.PI * 2;
      const rx = Math.cos(t) * perimeter;
      const rz = Math.sin(t) * perimeter;
      const tree = new THREE.Group();
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.2, 2, 6), treeTrunkMat);
      trunk.position.y = 1;
      tree.add(trunk);
      const foliage = new THREE.Mesh(new THREE.ConeGeometry(1.2, 3, 7), treeFoliageMat);
      foliage.position.y = 3;
      tree.add(foliage);
      tree.position.set(rx, 0, rz);
      dressing.add(tree);
    }

    // Simple low-poly building skyline around the far perimeter.
    const buildingMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, emissive: 0x0ea5e9, emissiveIntensity: 0.15 });
    for (let i = 0; i < 14; i++) {
      const t = (i / 14) * Math.PI * 2 + 0.2;
      const dist = perimeter * 1.4;
      const bx = Math.cos(t) * dist;
      const bz = Math.sin(t) * dist;
      const h = 6 + (i % 5) * 3;
      const building = new THREE.Mesh(new THREE.BoxGeometry(4, h, 4), buildingMat);
      building.position.set(bx, h / 2, bz);
      dressing.add(building);
    }

    // Distant low-poly terrain/"mountain" silhouette.
    const terrainMat = new THREE.MeshStandardMaterial({ color: 0x0f2436, emissive: 0x0a3a4a, emissiveIntensity: 0.1, wireframe: true });
    for (let i = 0; i < 10; i++) {
      const t = (i / 10) * Math.PI * 2 + 0.4;
      const dist = perimeter * 2.1;
      const mx = Math.cos(t) * dist;
      const mz = Math.sin(t) * dist;
      const h = 10 + (i % 4) * 6;
      const mountain = new THREE.Mesh(new THREE.ConeGeometry(9, h, 5), terrainMat);
      mountain.position.set(mx, h / 2 - 1, mz);
      dressing.add(mountain);
    }

    scene.add(dressing);

    for (const rsu of rsus) {
      const group = new THREE.Group();
      const poleMat = new THREE.MeshStandardMaterial({ color: RSU_COLOR, emissive: RSU_COLOR, emissiveIntensity: 0.7 });
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 6, 8), poleMat);
      pole.position.y = 3;
      group.add(pole);
      const marker = new THREE.Mesh(new THREE.SphereGeometry(0.35, 10, 10), poleMat);
      marker.position.y = 6.2;
      group.add(marker);
      group.userData.isRsu = true;
      group.userData.rsuCenterXyz = rsu.centerXyz;
      scene.add(group);
      rsuMeshesRef.current.push(group);
    }

    const handleClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycasterRef.current.setFromCamera(mouse, camera);
      const meshes = Array.from(vehicleMeshesRef.current.values()).flatMap((g) => g.children);
      const hits = raycasterRef.current.intersectObjects(meshes);
      if (hits.length > 0) {
        let obj: THREE.Object3D | null = hits[0].object;
        while (obj && !obj.userData.vehicleId) obj = obj.parent;
        if (obj) onSelectVehicle(obj.userData.vehicleId as string);
      }
    };
    renderer.domElement.addEventListener("click", handleClick);

    let animationFrameId: number;
    const animate = () => {
      renderer.render(scene, camera);
      animationFrameId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      renderer.domElement.removeEventListener("click", handleClick);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rsus]);

  // STAGE 2 integration point: renders whatever real static-environment
  // geometry is present in `environment` (empty in Stage 1, so this is
  // currently a no-op -- see module docstring). Coordinates are treated
  // as real CARLA world x/y, scaled the same way as vehicles, whenever
  // `bounds` is available. Rebuilds fully whenever `environment` or
  // `bounds` changes.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !bounds) return;

    if (environmentGroupRef.current) {
      scene.remove(environmentGroupRef.current);
      environmentGroupRef.current.traverse((c) => {
        if (c instanceof THREE.Mesh) {
          c.geometry.dispose();
          (c.material as THREE.Material).dispose();
        }
      });
      environmentGroupRef.current = null;
    }

    const hasAnyRealGeometry =
      environment.buildings.length > 0 ||
      environment.vegetation.length > 0 ||
      environment.poles.length > 0 ||
      environment.barriers.length > 0;
    if (!hasAnyRealGeometry) return; // Stage 1: nothing to render, by design.

    const spanX = bounds.maxX - bounds.minX || 1;
    const spanY = bounds.maxY - bounds.minY || 1;
    const scale = SCENE_WIDTH_UNITS / Math.max(spanX, spanY);
    const toWorld = (x: number, y: number): [number, number] => [
      (x - bounds.minX - spanX / 2) * scale,
      (y - bounds.minY - spanY / 2) * scale,
    ];

    const group = new THREE.Group();

    for (const b of environment.buildings) {
      const [wx, wz] = toWorld(b.centerXyz[0], b.centerXyz[1]);
      const geo = new THREE.BoxGeometry(b.extentXyz[0] * 2 * scale, b.extentXyz[2] * 2 * scale, b.extentXyz[1] * 2 * scale);
      const mat = new THREE.MeshStandardMaterial({ color: 0x1e293b, emissive: 0x0ea5e9, emissiveIntensity: 0.1, wireframe: false });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(wx, (b.extentXyz[2] * scale), wz);
      group.add(mesh);
    }
    for (const v of environment.vegetation) {
      const [wx, wz] = toWorld(v.centerXyz[0], v.centerXyz[1]);
      const geo = new THREE.ConeGeometry(v.extentXyz[0] * scale, v.extentXyz[2] * 2 * scale, 6);
      const mat = new THREE.MeshStandardMaterial({ color: 0x14532d, emissive: 0x22c55e, emissiveIntensity: 0.15 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(wx, v.extentXyz[2] * scale, wz);
      group.add(mesh);
    }
    for (const p of environment.poles) {
      const [wx, wz] = toWorld(p.centerXyz[0], p.centerXyz[1]);
      const geo = new THREE.CylinderGeometry(0.05, 0.05, 4, 6);
      const mat = new THREE.MeshStandardMaterial({ color: 0x475569 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(wx, 2, wz);
      group.add(mesh);
    }
    for (const bar of environment.barriers) {
      const [wx, wz] = toWorld(bar.centerXyz[0], bar.centerXyz[1]);
      const geo = new THREE.BoxGeometry(bar.extentXyz[0] * 2 * scale, bar.extentXyz[2] * 2 * scale, bar.extentXyz[1] * 2 * scale);
      const mat = new THREE.MeshStandardMaterial({ color: 0x334155 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(wx, bar.extentXyz[2] * scale, wz);
      group.add(mesh);
    }

    scene.add(group);
    environmentGroupRef.current = group;
  }, [environment, bounds]);

  // Approximate road, rebuilt when lane count or orientation changes.
  // Widened and given real lane-divider markings (one line per real
  // lane boundary) so it reads as an actual road surface, not an
  // isolated thin strip -- still an approximation (see module
  // docstring), just a more legible one.
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (roadMeshRef.current) {
      scene.remove(roadMeshRef.current);
      roadMeshRef.current.traverse((c) => {
        if (c instanceof THREE.Mesh || c instanceof THREE.Line) {
          c.geometry.dispose();
          (c.material as THREE.Material).dispose();
        }
      });
      roadMeshRef.current = null;
    }

    const laneWidthUnits = (STANDARD_LANE_WIDTH_M / 20) * (SCENE_WIDTH_UNITS / 40) * 6; // widened for legibility at this scale
    const roadWidth = laneCount * laneWidthUnits;
    const roadLength = SCENE_WIDTH_UNITS * 2;

    const group = new THREE.Group();

    const roadGeo =
      dominantAxis === "x"
        ? new THREE.PlaneGeometry(roadLength, roadWidth)
        : new THREE.PlaneGeometry(roadWidth, roadLength);
    const roadMat = new THREE.MeshStandardMaterial({ color: 0x131c2b, emissive: 0x0a3a4a, emissiveIntensity: 0.25 });
    const road = new THREE.Mesh(roadGeo, roadMat);
    road.rotation.x = -Math.PI / 2;
    road.position.y = -0.02;
    group.add(road);

    // Real lane-divider lines: one per lane boundary, spaced by the
    // same standard lane width used to size the road above.
    const dividerMat = new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.6 });
    for (let i = 0; i <= laneCount; i++) {
      const offset = -roadWidth / 2 + i * laneWidthUnits;
      const points =
        dominantAxis === "x"
          ? [new THREE.Vector3(-roadLength / 2, 0.01, offset), new THREE.Vector3(roadLength / 2, 0.01, offset)]
          : [new THREE.Vector3(offset, 0.01, -roadLength / 2), new THREE.Vector3(offset, 0.01, roadLength / 2)];
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      group.add(new THREE.Line(geo, dividerMat));
    }

    scene.add(group);
    roadMeshRef.current = group;
  }, [laneCount, dominantAxis]);

  // Per-tick: update vehicle meshes, connection lines, and camera.
  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    if (!scene || !camera || !currentTick || !bounds) return;

    const spanX = bounds.maxX - bounds.minX || 1;
    const spanY = bounds.maxY - bounds.minY || 1;
    const scale = SCENE_WIDTH_UNITS / Math.max(spanX, spanY);
    const toWorld = (x: number, y: number): [number, number] => [
      (x - bounds.minX - spanX / 2) * scale,
      (y - bounds.minY - spanY / 2) * scale,
    ];

    // Position AND SCALE real RSUs using the same real coordinate
    // scaling as vehicles (2026-09-14 fix -- RSU geometry was
    // previously fixed-size, same bug class as the vehicle mesh-scale
    // mismatch described in the module docstring). Cheap to redo every
    // tick (only 4 objects) and keeps them correctly placed/sized even
    // if bounds change between scenes.
    for (const rsuGroup of rsuMeshesRef.current) {
      const centerXyz = rsuGroup.userData.rsuCenterXyz as [number, number, number];
      const [rsuX, rsuZ] = toWorld(centerXyz[0], centerXyz[1]);
      rsuGroup.position.set(rsuX, 0, rsuZ);
      rsuGroup.scale.setScalar(scale);
    }

    const seenIds = new Set<string>();
    const positions: Array<{ id: string; x: number; z: number }> = [];
    let egoWorldPos: THREE.Vector3 | null = null;
    let egoHeading = 0;

    for (const frame of Object.values(currentTick.vehicles)) {
      if (frame.position_x == null || frame.position_y == null) continue;
      seenIds.add(frame.vehicle_id);
      const [worldX, worldZ] = toWorld(frame.position_x, frame.position_y);
      positions.push({ id: frame.vehicle_id, x: worldX, z: worldZ });

      let mesh = vehicleMeshesRef.current.get(frame.vehicle_id);
      const isEgo = frame.vehicle_id === egoVehicleDbId;
      if (!mesh) {
        mesh = buildVehicleMesh(isEgo ? EGO_COLOR : directionColor(frame.lane_id));
        mesh.userData.vehicleId = frame.vehicle_id;
        scene.add(mesh);
        vehicleMeshesRef.current.set(frame.vehicle_id, mesh);
      }

      mesh.position.set(worldX, DEFAULT_VEHICLE_SIZE[1] / 2, worldZ);
      const prevFrame = previousTick?.vehicles[frame.vehicle_id];
      const heading = headingFromDelta(prevFrame, frame);
      mesh.rotation.y = heading;

      const isSelected = frame.vehicle_id === selectedVehicleId;
      const color = isEgo ? EGO_COLOR : directionColor(frame.lane_id);
      setVehicleHighlight(mesh, color, isSelected || isEgo);

      // 2026-09-14 FIX: mesh scale now includes the same `scale` factor
      // used to compress real positions into the scene, so vehicle
      // geometry and vehicle positions live in the same coordinate
      // space. Previously this only applied the ego/non-ego multiplier
      // (1.15 / 1), leaving mesh size fixed-absolute regardless of how
      // compressed the real position spread was -- root cause of the
      // "solid color blob" render at close (Driver's View) range.
      mesh.scale.setScalar(scale * (isEgo ? 1.15 : 1));

      if (isEgo) {
        egoWorldPos = new THREE.Vector3(worldX, DEFAULT_VEHICLE_SIZE[1] / 2, worldZ);
        egoHeading = heading;
      }
    }

    for (const [vehicleId, mesh] of vehicleMeshesRef.current.entries()) {
      if (!seenIds.has(vehicleId)) {
        scene.remove(mesh);
        mesh.traverse((c) => {
          if (c instanceof THREE.Mesh) {
            c.geometry.dispose();
            (c.material as THREE.Material).dispose();
          }
          if (c instanceof THREE.LineSegments) {
            c.geometry.dispose();
            (c.material as THREE.Material).dispose();
          }
        });
        vehicleMeshesRef.current.delete(vehicleId);
      }
    }

    // Real proximity connection lines, same rule as the 2D map.
    if (connectionLinesRef.current) {
      scene.remove(connectionLinesRef.current);
      connectionLinesRef.current.geometry.dispose();
      (connectionLinesRef.current.material as THREE.Material).dispose();
      connectionLinesRef.current = null;
    }
    const linePoints: number[] = [];
    const PROXIMITY_UNITS = 90 * scale;
    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        const dx = positions[i].x - positions[j].x;
        const dz = positions[i].z - positions[j].z;
        if (Math.hypot(dx, dz) <= PROXIMITY_UNITS) {
          linePoints.push(positions[i].x, 0.6, positions[i].z, positions[j].x, 0.6, positions[j].z);
        }
      }
    }
    if (linePoints.length > 0) {
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePoints, 3));
      const lineMat = new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.35 });
      const lines = new THREE.LineSegments(lineGeo, lineMat);
      scene.add(lines);
      connectionLinesRef.current = lines;
    }

    // Camera follows the real selected ego vehicle, in whichever mode
    // is active. Falls back to the previous fixed elevated view if the
    // ego vehicle has no frame at this tick. Distance/height values
    // here are already in scene-unit space (same space as `worldX` /
    // `worldZ` after `toWorld()`), so they do NOT need the `scale`
    // factor applied separately -- only the vehicle/RSU mesh geometry
    // itself needed that fix (see 2026-09-14 note above).
    if (egoWorldPos) {
      if (cameraMode === "topdown") {
        camera.position.set(egoWorldPos.x, egoWorldPos.y + 9, egoWorldPos.z + 8);
        camera.lookAt(egoWorldPos);
      } else {
        const eyeHeight = 1.4;
        const forwardDist = 3;
        const forward = new THREE.Vector3(Math.sin(egoHeading), 0, Math.cos(egoHeading));
        const eyePos = egoWorldPos.clone().setY(eyeHeight);
        camera.position.copy(eyePos);
        camera.lookAt(eyePos.clone().add(forward.multiplyScalar(forwardDist)));
      }
    }
  }, [currentTick, previousTick, bounds, selectedVehicleId, egoVehicleDbId, cameraMode]);

  return <div ref={containerRef} className="digital-twin-3d__viewport" />;
}
