import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "../components/PageHeader";
import { EmptyState } from "../components/EmptyState";
import { Badge } from "../components/Badge";
import { PhysicalLayer } from "../components/PhysicalLayer";
import { listScenes } from "../lib/api";
import {
  useFrameSequencePlayback,
  type FrameTick,
  type PlaybackSpeed,
} from "../hooks/useFrameSequencePlayback";
import type { Frame } from "../types/api";

/**
 * Real scenes span a much larger area than the synthetic generator's
 * fixed 60m half-extent (this scene alone is ~928m x ~776m -- see
 * `docs/phase2_phase3_reconciliation_2026-09-10.md`'s ingestion notes).
 * Bounds are therefore computed from the actual loaded sequence, not a
 * hardcoded constant, so real scenes of any size render correctly
 * without clipping or a mis-scaled view.
 */
const SVG_SIZE = 480;
const SVG_PADDING = 24;

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function computeBounds(ticks: FrameTick[]): Bounds | null {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  let found = false;
  for (const tick of ticks) {
    for (const frame of Object.values(tick.vehicles)) {
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

function worldToSvg(x: number, y: number, bounds: Bounds): { cx: number; cy: number } {
  const usable = SVG_SIZE - SVG_PADDING * 2;
  const spanX = bounds.maxX - bounds.minX || 1;
  const spanY = bounds.maxY - bounds.minY || 1;
  const cx = SVG_PADDING + ((x - bounds.minX) / spanX) * usable;
  // Flip Y so it reads naturally on screen (CARLA's Y grows opposite to screen-down SVG Y).
  const cy = SVG_PADDING + (1 - (y - bounds.minY) / spanY) * usable;
  return { cx, cy };
}

/** Real heading derived from the actual position delta between two
 * consecutive real frames -- there is no `heading` field on `Frame`
 * (vehicle_states.json's heading_deg was never mapped into the schema),
 * so this is computed, not fabricated. Returns null (no rotation) when
 * there's no prior frame or the vehicle hasn't moved. */
function headingFromDelta(prev: Frame | undefined, curr: Frame): number | null {
  if (!prev || prev.position_x == null || prev.position_y == null) return null;
  if (curr.position_x == null || curr.position_y == null) return null;
  const dx = curr.position_x - prev.position_x;
  const dy = curr.position_y - prev.position_y;
  if (Math.abs(dx) < 1e-4 && Math.abs(dy) < 1e-4) return null;
  return (Math.atan2(-dy, dx) * 180) / Math.PI;
}

/**
 * Real direction-of-travel color, derived from `lane_id`'s sign.
 * CARLA/OpenDRIVE convention: negative and positive lane_id values on
 * the same road represent opposite driving directions (confirmed by
 * inspecting this scene's real data directly -- distinct negative
 * values -4..-1 and positive values 1..6 both appear across the
 * scene). This is a genuine structural fact from the data, not a
 * decorative color choice.
 */
function directionColor(laneId: number | null): string {
  if (laneId == null) return "var(--color-text-muted)";
  return laneId < 0 ? "#f97316" /* orange: negative-lane direction */ : "var(--color-accent)" /* blue: positive-lane direction */;
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

function trafficLightColor(state: string | null): string {
  if (state === "Red") return "#ef4444";
  if (state === "Green") return "#22c55e";
  if (state === "Yellow") return "#eab308";
  return "var(--color-text-muted)";
}

const SPEED_OPTIONS: PlaybackSpeed[] = [0.5, 1, 2];

/**
 * Real vehicle ID range for this scene, confirmed directly from
 * convert_all_ego_candidates.py's output (54 distinct vehicles,
 * contiguous 147-200) -- not assumed. If a different scene has a
 * different range, this constant needs updating to match; there is no
 * live vehicle-list endpoint wired in here yet.
 */
const EGO_VEHICLE_ID_OPTIONS = Array.from({ length: 54 }, (_, i) => String(147 + i));

/**
 * V2V-style connection lines: drawn between any two vehicles within a
 * real proximity threshold at the CURRENT tick only, using their actual
 * position_x/position_y. This is a visual proxy for "vehicles close
 * enough to plausibly be in V2V range" -- it is derived from genuine
 * distance, not a decorative/random effect, but it is NOT the same as
 * a real V2V link (that would need actual CSI/SNR data, which does not
 * exist yet for source="carla_only" frames -- see carla_ingestion.py).
 * Do not present these lines as confirmed communication links in any
 * screenshot/report without that caveat.
 */
const V2V_PROXIMITY_THRESHOLD_M = 90;

function computeProximityPairs(
  vehicles: Frame[],
): Array<[Frame, Frame]> {
  const pairs: Array<[Frame, Frame]> = [];
  for (let i = 0; i < vehicles.length; i++) {
    const a = vehicles[i];
    if (a.position_x == null || a.position_y == null) continue;
    for (let j = i + 1; j < vehicles.length; j++) {
      const b = vehicles[j];
      if (b.position_x == null || b.position_y == null) continue;
      const dx = a.position_x - b.position_x;
      const dy = a.position_y - b.position_y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist <= V2V_PROXIMITY_THRESHOLD_M) {
        pairs.push([a, b]);
      }
    }
  }
  return pairs;
}

export function DigitalTwin() {
  const scenesQuery = useQuery({ queryKey: ["scenes"], queryFn: listScenes });
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | null>(null);
  const [egoVehicleId, setEgoVehicleId] = useState<string>("181");

  const scenes = scenesQuery.data ?? [];
  const activeSceneId = selectedSceneId ?? scenes[0]?.id ?? undefined;

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

  const allFrameIndices = useMemo(
    () => sequence.ticks.map((t) => t.frameIndex),
    [sequence.ticks],
  );

  return (
    <div>
      <PageHeader
        title="Digital Twin"
        subtitle="Real recorded CARLA scene â€” play, pause, and scrub through actual simulation frames"
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
          <div className="mb-6">
            <PhysicalLayer
              frameIndex={currentTick.frameIndex}
              egoVehicleId={egoVehicleId}
              availableVehicleIds={EGO_VEHICLE_ID_OPTIONS}
              allFrameIndices={allFrameIndices}
              onChangeEgoVehicle={setEgoVehicleId}
            />
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[480px_1fr]">
          <div className="rounded-lg border border-border bg-surface p-4">
            <svg
              width={SVG_SIZE}
              height={SVG_SIZE}
              viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
              className="w-full"
              role="img"
              aria-label="Digital Twin playback of real recorded scene"
            >
              <rect
                x={0}
                y={0}
                width={SVG_SIZE}
                height={SVG_SIZE}
                fill="var(--color-surface-raised)"
                opacity={0.3}
              />

              {/* Real proximity-based V2V-style connection lines -- see
                  computeProximityPairs's own comment on what these
                  genuinely represent vs. a real communication link. */}
              {computeProximityPairs(Object.values(currentTick.vehicles)).map(
                ([a, b], idx) => {
                  const pa = worldToSvg(a.position_x as number, a.position_y as number, bounds);
                  const pb = worldToSvg(b.position_x as number, b.position_y as number, bounds);
                  return (
                    <line
                      key={`${a.vehicle_id}-${b.vehicle_id}-${idx}`}
                      x1={pa.cx}
                      y1={pa.cy}
                      x2={pb.cx}
                      y2={pb.cy}
                      className="twin-connection-line"
                      strokeWidth={1}
                    />
                  );
                },
              )}

              {Object.values(currentTick.vehicles).map((frame) => {
                if (frame.position_x == null || frame.position_y == null) return null;
                const { cx, cy } = worldToSvg(frame.position_x, frame.position_y, bounds);
                const prevFrame = previousTick?.vehicles[frame.vehicle_id];
                const heading = headingFromDelta(prevFrame, frame);
                const isSelected = frame.vehicle_id === selectedVehicleId;
                const color = directionColor(frame.lane_id);
                const { atLight, state } = trafficLightState(frame);

                return (
                  <g
                    key={frame.vehicle_id}
                    transform={`translate(${cx}, ${cy})${heading != null ? ` rotate(${heading})` : ""}`}
                    onClick={() => setSelectedVehicleId(frame.vehicle_id)}
                    className={`cursor-pointer twin-glow-vehicle${isSelected ? " twin-glow-vehicle--selected" : ""}`}
                  >
                    <circle
                      r={isSelected ? 9 : 6}
                      fill={color}
                      stroke={isSelected ? "var(--color-text-primary)" : "none"}
                      strokeWidth={2}
                    />
                    {heading != null && (
                      <path
                        d="M 6 0 L 12 -3 L 12 3 Z"
                        fill={color}
                      />
                    )}
                    {/* Real traffic-light marker: a small dot above the
                        vehicle, colored by that vehicle's actual real
                        traffic_light_state -- only rendered when the
                        vehicle is genuinely at a light (is_at_traffic_light). */}
                    {atLight && (
                      <circle cx={0} cy={-14} r={3} fill={trafficLightColor(state)} />
                    )}
                  </g>
                );
              })}
            </svg>

            <div className="mt-2 flex items-center gap-4 text-xs text-text-secondary">
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: "var(--color-accent)" }} />
                Lane +
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#f97316" }} />
                Lane − (opposite direction)
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#ef4444" }} />
                At red light
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#22c55e" }} />
                At green light
              </span>
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
              Frame {currentTick.frameIndex} Â· tick {currentTickIndex + 1}/{sequence.ticks.length}
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-5">
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
                      carla_only Â· no wireless data yet
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
        </div>
        </>
      )}
    </div>
  );
}
