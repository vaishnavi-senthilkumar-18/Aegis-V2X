import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { EmptyState } from "../components/EmptyState";
import { Badge } from "../components/Badge";
import { getLatestFramePerVehicle, listScenes } from "../lib/api";
import { formatVehicleLabel } from "../lib/format";
import type { LatestVehicleFrame } from "../types/api";

const VIEW_EXTENT_M = 75; // slightly wider than the synthetic generator's 60m half-extent
const SVG_SIZE = 480;

function worldToSvg(x: number, y: number): { cx: number; cy: number } {
  const scale = (SVG_SIZE / 2 - 20) / VIEW_EXTENT_M;
  return { cx: SVG_SIZE / 2 + x * scale, cy: SVG_SIZE / 2 + y * scale };
}

export function DigitalTwin() {
  const scenesQuery = useQuery({ queryKey: ["scenes"], queryFn: listScenes });
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const [selectedVehicleId, setSelectedVehicleId] = useState<string | null>(null);

  const scenes = scenesQuery.data ?? [];
  const activeSceneId = selectedSceneId ?? scenes[0]?.id ?? null;

  const framesQuery = useQuery({
    queryKey: ["latest-per-vehicle", activeSceneId],
    queryFn: () => getLatestFramePerVehicle(activeSceneId as string),
    enabled: !!activeSceneId,
    refetchInterval: 3000,
  });

  const frames = framesQuery.data ?? [];
  const selectedFrame = useMemo(
    () => frames.find((f) => f.vehicle_id === selectedVehicleId) ?? null,
    [frames, selectedVehicleId],
  );

  return (
    <div>
      <PageHeader
        title="Digital Twin"
        subtitle="Live intersection radar map — position data from the most recent frame per vehicle"
        actions={
          scenes.length > 0 ? (
            <select
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-primary"
              value={activeSceneId ?? ""}
              onChange={(e) => {
                setSelectedSceneId(e.target.value);
                setSelectedVehicleId(null);
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
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[480px_1fr]">
          <div className="rounded-lg border border-border bg-surface p-4">
            <svg
              width={SVG_SIZE}
              height={SVG_SIZE}
              viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
              className="w-full"
              role="img"
              aria-label="Digital Twin intersection radar map"
            >
              {/* Intersection roads */}
              <rect
                x={0}
                y={SVG_SIZE / 2 - 40}
                width={SVG_SIZE}
                height={80}
                fill="var(--color-surface-raised)"
              />
              <rect
                x={SVG_SIZE / 2 - 40}
                y={0}
                width={80}
                height={SVG_SIZE}
                fill="var(--color-surface-raised)"
              />
              {/* Range rings */}
              {[20, 40, 60].map((r) => {
                const { cx, cy } = worldToSvg(0, 0);
                const scale = (SVG_SIZE / 2 - 20) / VIEW_EXTENT_M;
                return (
                  <circle
                    key={r}
                    cx={cx}
                    cy={cy}
                    r={r * scale}
                    fill="none"
                    stroke="var(--color-border)"
                    strokeDasharray="4 4"
                  />
                );
              })}
              {frames.map((frame: LatestVehicleFrame) => {
                if (frame.position_x === null || frame.position_y === null) return null;
                const { cx, cy } = worldToSvg(frame.position_x, frame.position_y);
                const isSelected = frame.vehicle_id === selectedVehicleId;
                return (
                  <g
                    key={frame.vehicle_id}
                    onClick={() => setSelectedVehicleId(frame.vehicle_id)}
                    className="cursor-pointer"
                  >
                    <circle
                      cx={cx}
                      cy={cy}
                      r={isSelected ? 9 : 7}
                      fill="var(--color-accent)"
                      stroke={isSelected ? "var(--color-text-primary)" : "none"}
                      strokeWidth={2}
                    />
                    <text
                      x={cx}
                      y={cy - 14}
                      textAnchor="middle"
                      fontSize={10}
                      fill="var(--color-text-secondary)"
                      className="mono"
                    >
                      {formatVehicleLabel(frame.vehicle_code)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="rounded-lg border border-border bg-surface p-5">
            <h2 className="text-sm font-medium text-text-primary">Vehicle Inspector</h2>
            {!selectedFrame ? (
              <div className="mt-4">
                <EmptyState
                  title="No vehicle selected"
                  detail="Click a vehicle on the radar map to inspect its latest frame."
                />
              </div>
            ) : (
              <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                <dt className="text-text-secondary">Vehicle</dt>
                <dd className="mono">{formatVehicleLabel(selectedFrame.vehicle_code)}</dd>
                <dt className="text-text-secondary">Frame Index</dt>
                <dd className="mono">{selectedFrame.frame_index}</dd>
                <dt className="text-text-secondary">Speed</dt>
                <dd className="mono">{selectedFrame.speed_mps?.toFixed(1) ?? "—"} m/s</dd>
                <dt className="text-text-secondary">Position (x, y, z)</dt>
                <dd className="mono">
                  {selectedFrame.position_x?.toFixed(1)}, {selectedFrame.position_y?.toFixed(1)},{" "}
                  {selectedFrame.position_z?.toFixed(1)}
                </dd>
                <dt className="text-text-secondary">Lane</dt>
                <dd className="mono">{selectedFrame.lane_id ?? "—"}</dd>
                <dt className="text-text-secondary">SNR</dt>
                <dd className="mono">{selectedFrame.snr_db?.toFixed(1) ?? "—"} dB</dd>
                <dt className="text-text-secondary">Sync Status</dt>
                <dd>
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
                </dd>
                <dt className="text-text-secondary">Source</dt>
                <dd>
                  <Badge>{selectedFrame.source}</Badge>
                </dd>
              </dl>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
