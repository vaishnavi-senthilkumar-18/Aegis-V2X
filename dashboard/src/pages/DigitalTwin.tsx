import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "../components/PageHeader";
import { EmptyState } from "../components/EmptyState";
import { Badge } from "../components/Badge";
import { PhysicalLayer } from "../components/PhysicalLayer";
import { DigitalTwin3D } from "../components/DigitalTwin3D";
import type { RsuGeometry, CameraMode } from "../components/DigitalTwin3D";
import { listScenes, listVehicles } from "../lib/api";
import {
  useFrameSequencePlayback,
  type FrameTick,
  type PlaybackSpeed,
} from "../hooks/useFrameSequencePlayback";
import type { Frame } from "../types/api";

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Real scenes span a much larger area than the synthetic generator's
 * fixed 60m half-extent (this scene alone is ~928m x~776m -- see
 * `docs/phase2_phase3_reconciliation_2026-09-10.md`'s ingestion notes).
 * Bounds are therefore computed from the actual loaded sequence, not a
 * hardcoded constant, so real scenes of any size render correctly in
 * the 3D view without clipping or a mis-scaled camera.
 */
function computeBounds(ticks: FrameTick[]): Bounds | null {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  let found = false;
  for (const tick of ticks) {
    for (const frame of Object.values(tick.vehicles)){
      if (frame.position_x == null || frame.position_y == null) continue;
      found = true;
      minX = Math.min(minX, frame.position_x);
      maxX = Math.max(maxX, frame.position_x);
      minY = Math.min(minY, frame.position_y);
      maxY = Math.max(maxY, frame.position_y);
    }
  }
  return found ? { minX, maxX, minY, maxY } : null;
}

/**
 * Real traffic-light interaction, read from `environmental_context`
 * (populated by carla_ingestion.py from vehicle_states.json's real
 * is_at_traffic_light/traffic_light_state fields -- see that module's
 * comment for how this was confirmed present, not assumed).
 */
function trafficLightState(frame: Frame): { atLight: boolean; state: string | null } {
  const ctx = frame.environmental_context as
    | { is_at_traffic_light?: boolean; traffic_light_state?: string }
    | null;
  if (!ctx) return { atLight: false, state: null };
  return {
    atLight: Boolean(ctx.is_at_traffic_light),
    state: ctx.traffic_light_state ?? null,
  };
}

const SPEED_OPTIONS: PlaybackSpeed[] = [0.5, 1, 2];

/**
 * Extracts the numeric vehicle ID from a vehicle_code like "Vehicle181"
 * or "Vehicle6724" -- works for any scene's numbering, not just the
 * straight-road scene's 147-200 range.
 */
function numericIdFromVehicleCode(code: string | null | undefined): string | null {
  if (!code) return null;
  const match = code.match(/(\d+)$/);
  return match ? match[1] : null;
}

export function DigitalTwin() {
  const scenesQuery = useQuery({ queryKey: ["scenes"], queryFn: listScenes });
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | null>(null);
  const [egoVehicleId, setEgoVehicleId] = useState<string>("181");
  const [cameraMode, setCameraMode] = useState<CameraMode>("topdown");
  const [rsus, setRsus] = useState<RsuGeometry[]>([]);

  const scenes = scenesQuery.data ?? [];
  const activeSceneId = selectedSceneId ?? scenes[0]?.id ?? undefined;

  // Real vehicle roster for this scene, used to resolve the ego
  // dropdown's numeric CAM-### id (e.g. "181") into the actual database
  // UUID the per-tick Frame data is keyed by -- vehicle_code was set as
  // "Vehicle{numeric_id}" during ingestion (see ingest_real_scene.py),
  // so this is a real, confirmed mapping, not a guess.
  const vehiclesQuery = useQuery({
    queryKey: ["vehicles", activeSceneId],
    queryFn: () => listVehicles(activeSceneId as string),
    enabled: !!activeSceneId,
  });
  const egoVehicleDbId = useMemo(() => {
    const vehicles = vehiclesQuery.data ?? [];
    const match = vehicles.find((v) => numericIdFromVehicleCode(v.vehicle_code) === egoVehicleId);
    return match?.id ?? null;
  }, [vehiclesQuery.data, egoVehicleId]);

  // Real per-scene vehicle ID list, derived from the actual roster
  // returned for the currently selected scene -- NOT a hardcoded range.
  // The straight-road scene has vehicles 147-200; the roundabout scene
  // has 6718-6756; a hardcoded list broke ego vehicle resolution (and
  // therefore Physical Layer's camera lookup) on any scene other than
  // the one the constant happened to match.
  const egoVehicleIdOptions = useMemo(() => {
    const vehicles = vehiclesQuery.data ?? [];
    const ids = vehicles
      .map((v) => numericIdFromVehicleCode(v.vehicle_code))
      .filter((id): id is string => id != null);
    return Array.from(new Set(ids)).sort((a, b) => Number(a) - Number(b));
  }, [vehiclesQuery.data]);

  // If the current ego vehicle isn't valid for the newly selected scene
  // (e.g. it was "181" and the scene just changed to the roundabout,
  // which has no vehicle 181), snap to the first real vehicle in the
  // new scene's roster instead of silently keeping a stale, nonexistent
  // ego vehicle id.
  useEffect(() => {
    if (egoVehicleIdOptions.length === 0) return;
    if (!egoVehicleIdOptions.includes(egoVehicleId)) {
      setEgoVehicleId(egoVehicleIdOptions[0]);
    }
  }, [egoVehicleIdOptions, egoVehicleId]);

  // Real RSU positions (from geometry_snapshot.json,extracted once via
  // scripts/extract_rsus.py -- see that script's docstring). Fetched
  // once; RSUs are static for the whole scene, they don't move per tick.
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}rsus.json`)
      .then((res) => (res.ok ? res.json() : []))
      .then((data: RsuGeometry[]) => setRsus(data))
      .catch(() => setRsus([]));
  }, []);

  const {
    isLoading,
    error,
    sequence,
    currentTickIndex,
    currentTick,
    isPlaying,
    speed,
    play,
    pause,
    setSpeed,
    seekToTickIndex,
    reset,
  } = useFrameSequencePlayback(activeSceneId);

  const bounds = useMemo(() => computeBounds(sequence.ticks), [sequence.ticks]);
  const previousTick = useMemo(
    () => (currentTickIndex > 0 ? sequence.ticks[currentTickIndex - 1] : undefined),
    [sequence.ticks, currentTickIndex],
  );

  const selectedFrame = useMemo(
    () => (currentTick && selectedVehicleId ? currentTick.vehicles[selectedVehicleId] ?? null : null),
    [currentTick, selectedVehicleId],
  );


  return (
    <div>
      <PageHeader
        title="Digital Twin"
        subtitle="Real recorded CARLA scene ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬ play, pause, and scrub through actual simulation frames"
        actions={
          scenes.length > 0 ? (
            <select
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-primary"
              value={activeSceneId ?? ""}
              onChange={(e) => {
                setSelectedSceneId(e.target.value);
                setSelectedVehicleId(null);
                reset();
              }}
            >
              {scenes.map((scene) => (
                <option key={scene.id} value={scene.id}>
                  {scene.scene_code}
                </option>
              ))}
            </select>
          ) : undefined
        }
      />

      {scenes.length === 0 ? (
        <EmptyState
          title="No scenes to visualize"
          detail="Generate a synthetic scene on the Simulation page, or wait for Phase 2 to deliver real CARLA scenes."
        />
      ) : isLoading ? (
        <div className="mt-4 text-sm text-text-secondary">Loading real scene frames...</div>
      ) : error ? (
        <div className="mt-4 text-sm text-danger">
          Failed to load scene frames: {error.message}
        </div>
      ) : !bounds || !currentTick ? (
        <EmptyState
          title="No positioned frames"
          detail="This scene has no frames with position_x/position_y set."
        />
      ) : (
        <>
        {/*
          Two-column 50/50 layout (Physical Layer | Digital Twin) is now
          isolated in its own grid, separate from Vehicle Inspector below.
          Previously Vehicle Inspector was a third child of this same
          grid-cols-2 container, which caused it to auto-wrap into column 1
          of a second row, leaving an empty gap in column 2 next to it.
        */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 w-full">
          <div className="min-w-0 w-full h-full overflow-hidden flex flex-col">
            <PhysicalLayer
              frameIndex={currentTick.frameIndex}
              egoVehicleId={egoVehicleId}
              availableVehicleIds={egoVehicleIdOptions}
              onChangeEgoVehicle={setEgoVehicleId}
            />
          </div>
          <div className="min-w-0 w-full h-full rounded-lg border border-border bg-surface p-4 flex flex-col">
            <div className="mb-3 flex items-center justify-between gap-2">
              <span className="text-sm font-semibold tracking-wide text-text-primary">
                DIGITAL TWIN LAYER
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setCameraMode("topdown")}
                  aria-pressed={cameraMode === "topdown"}
                  className={`rounded-md border px-3 py-1 text-xs ${
                    cameraMode === "topdown"
                      ? "border-accent bg-accent text-white"
                      : "border-border bg-surface-raised text-text-primary hover:bg-surface"
                  }`}
                >
                  TOP-DOWN
                </button>
                <button
                  type="button"
                  onClick={() => setCameraMode("driver")}
                  aria-pressed={cameraMode === "driver"}
                  className={`rounded-md border px-3 py-1 text-xs ${
                    cameraMode === "driver"
                      ? "border-accent bg-accent text-white"
                      : "border-border bg-surface-raised text-text-primary hover:bg-surface"
                  }`}
                >
                  DRIVER'S VIEW
                </button>
              </div>
            </div>
            <div className="w-full" style={{ aspectRatio: "16 / 9" }}>
            <DigitalTwin3D
              currentTick={currentTick}
              previousTick={previousTick}
              bounds={bounds}
              selectedVehicleId={selectedVehicleId}
              onSelectVehicle={setSelectedVehicleId}
              rsus={rsus}
              egoVehicleDbId={egoVehicleDbId}
              cameraMode={cameraMode}
            />
            </div>
            {/* Playback controls */}
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={isPlaying ? pause : play}
                className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-text-primary hover:bg-surface"
              >
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                onClick={reset}
                className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-text-primary hover:bg-surface"
              >
                Reset
              </button>
              {SPEED_OPTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSpeed(s)}
                  aria-pressed={speed === s}
                  className={`rounded-md border px-3 py-1.5 text-sm ${
                    speed === s
                      ? "border-accent bg-accent text-white"
                      : "border-border bg-surface-raised text-text-primary hover:bg-surface"
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(sequence.ticks.length - 1, 0)}
              value={currentTickIndex}
              onChange={(e) => seekToTickIndex(Number(e.target.value))}
              className="mt-3 w-full"
            />
            <div className="mt-1 text-xs text-text-secondary mono">
              Frame {currentTick.frameIndex} Ãƒâ€šÃ‚Â· tick {currentTickIndex + 1}/{sequence.ticks.length}
            </div>
          </div>
        </div>

        {/* Vehicle Inspector: full-width row below the 50/50 pair, not part of that grid */}
        <div className="mt-4 w-full rounded-lg border border-border bg-surface p-5">
            <h2 className="text-sm font-medium text-text-primary">Vehicle Inspector</h2>
            {!selectedFrame ? (
              <div className="mt-4">
                <EmptyState
                  title="No vehicle selected"
                  detail="Click a vehicle on the map to inspect its state at the current frame."
                />
              </div>
            ) : (
              <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                <dt className="text-text-secondary">Vehicle</dt>
                <dd className="mono">{selectedFrame.vehicle_id}</dd>
                <dt className="text-text-secondary">Frame Index</dt>
                <dd className="mono">{selectedFrame.frame_index}</dd>
                <dt className="text-text-secondary">Speed</dt>
                <dd className="mono">
                  {selectedFrame.speed_mps != null
                    ? `${(selectedFrame.speed_mps * 3.6).toFixed(1)} km/h`
                    : "Not reported"}
                </dd>
                <dt className="text-text-secondary">Position (x, y, z)</dt>
                <dd className="mono">
                  {selectedFrame.position_x?.toFixed(1)}, {selectedFrame.position_y?.toFixed(1)},{" "}
                  {selectedFrame.position_z?.toFixed(1)}
                </dd>
                <dt className="text-text-secondary">Lane</dt>
                <dd className="mono">
                  {selectedFrame.lane_id ?? "Not reported"}
                  {selectedFrame.lane_id != null && (
                    <span className="ml-2 text-xs text-text-secondary">
                      ({selectedFrame.lane_id < 0 ? "opposite direction" : "primary direction"})
                    </span>
                  )}
                </dd>
                <dt className="text-text-secondary">Traffic Light</dt>
                <dd className="mono">
                  {(() => {
                    const { atLight, state } = trafficLightState(selectedFrame);
                    return atLight ? state ?? "Unknown" : "Not at a light";
                  })()}
                </dd>
                <dt className="text-text-secondary">SNR</dt>
                <dd className="mono">{selectedFrame.snr_db?.toFixed(1) ?? "Not reported"} dB</dd>
                <dt className="text-text-secondary">Sync Status</dt>
                <dd>
                  {selectedFrame.source === "carla_only" ? (
                    <Badge color="var(--color-text-secondary)">
                      carla_only Ãƒâ€šÃ‚Â· no wireless data yet
                    </Badge>
                  ) : (
                    <Badge
                      color={
                        selectedFrame.is_sync_valid
                          ? "var(--color-success)"
                          : "var(--color-danger)"
                      }
                    >
                      {selectedFrame.is_sync_valid ? "synced" : "out of tolerance"} (
                      {selectedFrame.sync_offset_ms.toFixed(1)}ms)
                    </Badge>
                  )}
                </dd>
                <dt className="text-text-secondary">Source</dt>
                <dd>
                  <Badge>{selectedFrame.source}</Badge>
                </dd>
              </dl>
            )}
        </div>
        </>
      )}
    </div>
  );
}
