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
 * 2026-09-15 FIX -- per-frame road rebuild causing visible "blinking"
 * on every tick during playback. Two independent causes, both fixed
 * here, neither touching the vehicle/RSU/environment rendering:
 *
 *   1. `dominantAxis` (which axis the approximate road strip is
 *      oriented along) had been changed to recompute from the tiny
 *      position DELTA between consecutive ticks, instead of the
 *      overall real position spread across the whole scene. With ~54
 *      vehicles moving in mixed directions, that per-tick comparison
 *      can flip between "x" and "y" from one frame to the next, and
 *      every flip destroys and rebuilds the entire road mesh (see the
 *      road-building effect below, which depends on `dominantAxis`).
 *      Reverted to the original stable form: computed once from
 *      `bounds` (the whole scene's real position spread), which only
 *      changes when the scene itself changes -- not every tick.
 *
 *   2. `laneCount` recomputed from ONLY the current tick's visible
 *      lane IDs, every tick. Since which lanes are occupied
 *      legitimately changes frame to frame as vehicles move, enter,
 *      and leave view, this value could change on nearly every tick --
 *      and the same road-building effect also depends on `laneCount`,
 *      so it would rebuild just as often. Fixed by tracking the
 *      MAXIMUM distinct lane count seen so far during this scene's
 *      playback (a monotonically non-decreasing value, via
 *      `maxLaneCountSeenRef`), so the road's lane count stabilizes
 *      once the true lane count has been observed, instead of
 *      fluctuating with momentary occupancy. Reset to the scene's
 *      default whenever `bounds` changes (i.e. a new scene loads).
 *
 * 2026-09-15 FIX -- Top-Down camera was effectively a drone/satellite
 * view, not the "few floors up" perspective intended. The previous
 * offset (`egoWorldPos.y + 12`, `egoWorldPos.z + 9`) was a RAW
 * scene-unit value, not real-world meters -- for a scene with a large
 * real position spread (this one is ~928m), the `scale` factor
 * compressing real meters into scene units is small, so "12 scene
 * units" corresponds to roughly 60+ real meters of altitude. Driver's
 * View already handled this correctly (`1.4 * scale`, i.e. a real
 * 1.4m eye height converted into scene units); Top-Down did not.
 * Fixed by expressing the Top-Down offset in real meters
 * (`TOPDOWN_HEIGHT_M`, `TOPDOWN_BACK_M` below) and multiplying by
 * `scale`, the same pattern Driver's View already used -- no change
 * to Driver's View's own camera math, which was already correct.
 *
 * All other behavior (vehicle rendering, RSU rendering, procedural
 * environment dressing, resize handling, click-to-select) is
 * UNCHANGED from the previous version of this file.
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

const SCENE_WIDTH_UNITS = 180;
const DEFAULT_VEHICLE_SIZE: [number, number, number] = [0.9, 0.5, 1.8];
const STANDARD_LANE_WIDTH_M = 3.5; // real-world engineering convention, not measured CARLA data

/** Top-Down camera offset, expressed in REAL METERS (see 2026-09-15 fix
 * note above), then converted to scene units via `* scale` at the
 * point of use -- same pattern Driver's View already used for its eye
 * height. ~14m is roughly a 4th-5th floor vantage point above the ego,
 * not a drone/satellite altitude. */
const TOPDOWN_HEIGHT_M = 14;
const TOPDOWN_BACK_M = 10;

/** 2026-09-15 FIX -- Driver's View felt "distant / not first-person"
 * even though its height math was already correct. Two causes:
 *
 *   1. The eye position was placed at the vehicle's CENTER, not toward
 *      the windshield/front -- like floating in the middle of the car
 *      looking out, rather than sitting up front like a driver.
 *      `DRIVER_FORWARD_OFFSET_M` shifts the eye forward along the
 *      vehicle's own heading before setting its height.
 *
 *   2. Driver's View shared the same 55-degree FOV as Top-Down. A wide
 *      FOV makes everything read as smaller/farther away -- this alone
 *      can make a technically-close camera feel distant. Driver's View
 *      now uses its own narrower FOV (`DRIVER_VIEW_FOV`), updated only
 *      when in that mode; Top-Down's FOV is untouched.
 */
const DRIVER_FORWARD_OFFSET_M = 0.9; // roughly half a vehicle length forward, toward the windshield
const DRIVER_VIEW_FOV = 42;
const TOPDOWN_VIEW_FOV = 55;

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

  // 2026-09-15 FIX: real dominant travel axis, derived from the OVERALL
  // real position spread across the whole scene (`bounds`), not from a
  // per-tick velocity comparison. `bounds` only changes when the scene
  // itself changes (see DigitalTwin.tsx's computeBounds), so this value
  // is stable across playback -- it will not flip mid-scene and trigger
  // a road rebuild on every frame the way the previous per-tick version
  // could. See module docstring's 2026-09-15 fix note for the full
  // explanation of the blinking this caused.
  const dominantAxis = useMemo<"x" | "y">(() => {
    if (!bounds) return "y";
    return bounds.maxX - bounds.minX >= bounds.maxY - bounds.minY ? "x" : "y";
  }, [bounds]);

  // 2026-09-15 FIX: tracks the MAXIMUM distinct lane count seen so far
  // during this scene's playback. `laneCount` below is monotonically
  // non-decreasing as a result, instead of fluctuating every tick with
  // momentary lane occupancy (which vehicle happens to be visible on
  // which lane at this exact instant) -- that fluctuation was the
  // second, independent cause of the per-frame road rebuild/blink (see
  // module docstring). Reset whenever `bounds` changes, i.e. a new
  // scene has loaded and the previous scene's lane count no longer
  // applies.
  const maxLaneCountSeenRef = useRef<number>(2);
  useEffect(() => {
    maxLaneCountSeenRef.current = 2;
  }, [bounds]);

  const laneCount = useMemo(() => {
    if (!currentTick) return maxLaneCountSeenRef.current;
    const laneIds = new Set(
      Object.values(currentTick.vehicles)
        .map((f) => f.lane_id)
        .filter((id): id is number => id != null),
    );
    const seenThisTick = Math.max(laneIds.size, 2);
    if (seenThisTick > maxLaneCountSeenRef.current) {
      maxLaneCountSeenRef.current = seenThisTick;
    }
    return maxLaneCountSeenRef.current;
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
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  renderer.domElement.style.display = "block";

  const resizeObserver = new ResizeObserver(() => {
    const nextWidth = container.clientWidth;
    const nextHeight = container.clientHeight;
    if (nextWidth <= 0 || nextHeight <= 0) return;
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(nextWidth, nextHeight, false);
  });
  resizeObserver.observe(container);
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
    // above.
    // --------------------------------------------------------------
    const dressing = new THREE.Group();

    const treeTrunkMat = new THREE.MeshStandardMaterial({
      color: 0x26351f,
      roughness: 0.95,
    });

    const treeFoliageMat = new THREE.MeshStandardMaterial({
      color: 0x14532d,
      emissive: 0x0b3d24,
      emissiveIntensity: 0.3,
      roughness: 0.9,
    });

    const buildingMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      emissive: 0x075985,
      emissiveIntensity: 0.22,
      roughness: 0.75,
    });

    const windowMat = new THREE.MeshStandardMaterial({
      color: 0x67e8f9,
      emissive: 0x22d3ee,
      emissiveIntensity: 1.0,
    });

    const barrierMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      emissive: 0x0e7490,
      emissiveIntensity: 0.18,
      roughness: 0.8,
    });

    const poleMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      emissive: 0x0e7490,
      emissiveIntensity: 0.2,
    });

    const mountainMat = new THREE.MeshStandardMaterial({
      color: 0x163047,
      emissive: 0x082f49,
      emissiveIntensity: 0.22,
      roughness: 1,
      flatShading: true,
    });

    const roadHalfWidth = Math.max(
      6,
      Math.min(22, laneCount * 2.2)
    );

    const roadHalfLength = SCENE_WIDTH_UNITS * 0.47;

    const addTree = (
      x: number,
      z: number,
      scaleValue: number
    ) => {
      const tree = new THREE.Group();

      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.22, 0.3, 2.4, 7),
        treeTrunkMat
      );
      trunk.position.y = 1.2;
      tree.add(trunk);

      const lowerFoliage = new THREE.Mesh(
        new THREE.ConeGeometry(1.45, 2.7, 8),
        treeFoliageMat
      );
      lowerFoliage.position.y = 3.0;
      tree.add(lowerFoliage);

      const upperFoliage = new THREE.Mesh(
        new THREE.SphereGeometry(1.15, 8, 6),
        treeFoliageMat
      );
      upperFoliage.position.y = 4.15;
      tree.add(upperFoliage);

      tree.position.set(x, 0, z);
      tree.scale.setScalar(scaleValue);
      dressing.add(tree);
    };

    const addBuilding = (
      x: number,
      z: number,
      width: number,
      depth: number,
      height: number
    ) => {
      const building = new THREE.Group();

      const body = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, depth),
        buildingMat
      );
      body.position.y = height / 2;
      building.add(body);

      for (
        let level = 1;
        level < Math.min(6, Math.floor(height / 3));
        level++
      ) {
        const window = new THREE.Mesh(
          new THREE.BoxGeometry(width * 0.7, 0.08, 0.035),
          windowMat
        );
        window.position.set(
          0,
          level * 3 - 1,
          depth / 2 + 0.03
        );
        building.add(window);
      }

      building.position.set(x, 0, z);
      dressing.add(building);
    };

    const addBarrier = (
      x: number,
      z: number,
      length: number,
      rotation: number
    ) => {
      const barrier = new THREE.Mesh(
        new THREE.BoxGeometry(length, 0.65, 0.22),
        barrierMat
      );
      barrier.position.set(x, 0.33, z);
      barrier.rotation.y = rotation;
      dressing.add(barrier);
    };

    const addPole = (x: number, z: number) => {
      const pole = new THREE.Group();

      const shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(0.09, 0.12, 5.5, 8),
        poleMat
      );
      shaft.position.y = 2.75;
      pole.add(shaft);

      const arm = new THREE.Mesh(
        new THREE.BoxGeometry(1.2, 0.08, 0.08),
        poleMat
      );
      arm.position.set(0.45, 5.2, 0);
      pole.add(arm);

      const lamp = new THREE.Mesh(
        new THREE.SphereGeometry(0.13, 8, 8),
        windowMat
      );
      lamp.position.set(1.0, 5.15, 0);
      pole.add(lamp);

      pole.position.set(x, 0, z);
      dressing.add(pole);
    };

    // Roadside trees.
    for (let i = 0; i < 16; i++) {
      const longitudinal =
        -roadHalfLength + (i / 15) * roadHalfLength * 2;

      const sideOffset =
        roadHalfWidth + 7 + (i % 3) * 2;

      const treeScale =
        0.8 + (i % 4) * 0.12;

      if (dominantAxis === "x") {
        addTree(longitudinal, sideOffset, treeScale);
        addTree(
          longitudinal,
          -sideOffset,
          treeScale * 0.95
        );
      } else {
        addTree(sideOffset, longitudinal, treeScale);
        addTree(
          -sideOffset,
          longitudinal,
          treeScale * 0.95
        );
      }
    }

    // Second vegetation band for scene depth.
    for (let i = 0; i < 10; i++) {
      const longitudinal =
        -roadHalfLength * 0.9 +
        (i / 9) * roadHalfLength * 1.8;

      const sideOffset =
        roadHalfWidth + 23 + (i % 2) * 4;

      if (dominantAxis === "x") {
        addTree(longitudinal, sideOffset, 1.15);
        addTree(longitudinal, -sideOffset, 1.1);
      } else {
        addTree(sideOffset, longitudinal, 1.15);
        addTree(-sideOffset, longitudinal, 1.1);
      }
    }

    // Low-rise city blocks outside the immediate road corridor.
    const buildingPositions = [
      [-52, 32, 9, 10, 9],
      [-28, 40, 12, 11, 13],
      [8, 38, 10, 12, 10],
      [42, 31, 13, 11, 16],
      [-48, -34, 11, 10, 12],
      [-14, -41, 13, 10, 9],
      [22, -38, 10, 12, 14],
      [52, -30, 12, 11, 11],
    ] as const;

    for (const [x, z, width, depth, height] of buildingPositions) {
      addBuilding(x, z, width, depth, height);
    }

    // Roadside barriers.
    if (dominantAxis === "x") {
      addBarrier(
        0,
        roadHalfWidth + 3,
        roadHalfLength * 0.65,
        0
      );
      addBarrier(
        0,
        -roadHalfWidth - 3,
        roadHalfLength * 0.65,
        0
      );
    } else {
      addBarrier(
        roadHalfWidth + 3,
        0,
        roadHalfLength * 0.65,
        Math.PI / 2
      );
      addBarrier(
        -roadHalfWidth - 3,
        0,
        roadHalfLength * 0.65,
        Math.PI / 2
      );
    }

    // Street-light/pole rhythm.
    for (let i = 0; i < 10; i++) {
      const longitudinal =
        -roadHalfLength * 0.85 +
        (i / 9) * roadHalfLength * 1.7;

      const sideOffset = roadHalfWidth + 1.8;

      if (dominantAxis === "x") {
        addPole(longitudinal, sideOffset);
        if (i % 2 === 0) {
          addPole(longitudinal, -sideOffset);
        }
      } else {
        addPole(sideOffset, longitudinal);
        if (i % 2 === 0) {
          addPole(-sideOffset, longitudinal);
        }
      }
    }

    // Distant terrain for depth.
    const terrainDistance = SCENE_WIDTH_UNITS * 0.58;

    for (let i = 0; i < 9; i++) {
      const angle =
        (i / 9) * Math.PI * 2 + 0.25;

      const distance =
        terrainDistance + (i % 3) * 9;

      const mountainHeight =
        12 + (i % 4) * 5;

      const mountain = new THREE.Mesh(
        new THREE.ConeGeometry(
          13 + (i % 3) * 4,
          mountainHeight,
          7
        ),
        mountainMat
      );

      mountain.position.set(
        Math.cos(angle) * distance,
        mountainHeight / 2 - 1,
        Math.sin(angle) * distance
      );

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
    resizeObserver.disconnect();
      cancelAnimationFrame(animationFrameId);
      renderer.domElement.removeEventListener("click", handleClick);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rsus]);

  // STAGE 2 integration point: renders whatever real static-environment
  // geometry is present in `environment` (empty in Stage 1, so this is
  // currently a no-op). Rebuilds fully whenever `environment` or
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
  //
  // 2026-09-15: this effect's dependency array is exactly why the two
  // fixes above (stable dominantAxis, stabilized laneCount) matter --
  // this effect fully rebuilds the road every time either value
  // changes. Before the fix, both could change on nearly every tick;
  // now both are stable across a scene's playback.
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

  if (!bounds) return;
  const sceneBounds = bounds;
  const sceneSpan = Math.max(sceneBounds.maxX - sceneBounds.minX, sceneBounds.maxY - sceneBounds.minY) || 1; const scale = SCENE_WIDTH_UNITS / sceneSpan; const laneWidthUnits = STANDARD_LANE_WIDTH_M * scale;
  const roadLength = (dominantAxis === "x" ? sceneBounds.maxX - sceneBounds.minX : sceneBounds.maxY - sceneBounds.minY) * scale;
  const roadWidth = laneCount * laneWidthUnits;

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
  }, [laneCount, dominantAxis, bounds]);

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

    // 2026-09-15 FIX: Top-Down offset is now expressed in real meters
    // (TOPDOWN_HEIGHT_M / TOPDOWN_BACK_M, declared near the top of this
    // file) and multiplied by `scale`, matching the pattern Driver's
    // View already used for its eye height -- see module docstring's
    // 2026-09-15 fix note. Driver's View itself is UNCHANGED here.
    if (egoWorldPos) {
      if (cameraMode === "topdown") {
        camera.position.set(
          egoWorldPos.x,
          egoWorldPos.y + TOPDOWN_HEIGHT_M * scale,
          egoWorldPos.z + TOPDOWN_BACK_M * scale,
        );
        camera.lookAt(egoWorldPos);
      } else {
        const eyeHeight = 1.4 * scale;
        const forwardDist = 18;
        const forward = new THREE.Vector3(Math.sin(egoHeading), 0, Math.cos(egoHeading));
        // 2026-09-15 FIX: eye position shifted forward (toward the
        // windshield) before setting height, instead of sitting at the
        // vehicle's exact center -- see DRIVER_FORWARD_OFFSET_M note
        // near the top of this file.
        const eyePos = egoWorldPos
          .clone()
          .add(forward.clone().multiplyScalar(DRIVER_FORWARD_OFFSET_M * scale))
          .setY(eyeHeight);
        camera.position.copy(eyePos);
        camera.lookAt(eyePos.clone().add(forward.multiplyScalar(forwardDist)));
      }

      // 2026-09-15 FIX: Driver's View uses its own narrower FOV instead
      // of sharing Top-Down's wide 55-degree FOV -- a wide FOV was
      // making an otherwise-correct close camera read as distant.
      const targetFov = cameraMode === "driver" ? DRIVER_VIEW_FOV : TOPDOWN_VIEW_FOV;
      if (camera.fov !== targetFov) {
        camera.fov = targetFov;
        camera.updateProjectionMatrix();
      }
    }
  }, [currentTick, previousTick, bounds, selectedVehicleId, egoVehicleDbId, cameraMode]);

  return <div ref={containerRef} className="digital-twin-3d__viewport" />;
}
