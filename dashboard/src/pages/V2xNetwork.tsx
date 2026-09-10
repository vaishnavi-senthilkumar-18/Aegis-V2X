import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { StatTile } from "../components/StatTile";
import { EmptyState } from "../components/EmptyState";
import { Sparkline } from "../components/Sparkline";
import { Badge } from "../components/Badge";
import { listFrames, listScenes } from "../lib/api";

export function V2xNetwork() {
  const scenesQuery = useQuery({ queryKey: ["scenes"], queryFn: listScenes });
  const [selectedSceneId, setSelectedSceneId] = useState<string | null>(null);
  const scenes = scenesQuery.data ?? [];
  const activeSceneId = selectedSceneId ?? scenes[0]?.id ?? null;

  const framesQuery = useQuery({
    queryKey: ["frames", "v2x", activeSceneId],
    queryFn: () => listFrames({ scene_id: activeSceneId ?? undefined, limit: 200 }),
    enabled: !!activeSceneId,
    refetchInterval: 4000,
  });

  const frames = framesQuery.data ?? [];
  const framesWithSnr = frames.filter((f) => f.snr_db !== null);
  const avgSnr =
    framesWithSnr.length > 0
      ? framesWithSnr.reduce((sum, f) => sum + (f.snr_db ?? 0), 0) / framesWithSnr.length
      : null;
  const avgRssi =
    framesWithSnr.length > 0
      ? frames.reduce((sum, f) => sum + (f.rssi_dbm ?? 0), 0) / framesWithSnr.length
      : null;
  const avgPathLoss =
    framesWithSnr.length > 0
      ? frames.reduce((sum, f) => sum + (f.path_loss_db ?? 0), 0) / framesWithSnr.length
      : null;

  return (
    <div>
      <PageHeader
        title="V2X Network"
        subtitle="Channel state, SNR/RSSI history, and beam allocation"
        actions={
          scenes.length > 0 ? (
            <select
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-primary"
              value={activeSceneId ?? ""}
              onChange={(e) => setSelectedSceneId(e.target.value)}
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

      {frames.length === 0 ? (
        <EmptyState
          title="No channel data yet"
          detail="Frames carry CSI/SNR/RSSI/beam data once ingested. Generate a synthetic scene or wait for Phase 2's Sionna RT pipeline."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
            <StatTile label="Avg SNR" value={avgSnr !== null ? `${avgSnr.toFixed(1)} dB` : "—"} />
            <StatTile
              label="Avg RSSI"
              value={avgRssi !== null ? `${avgRssi.toFixed(1)} dBm` : "—"}
            />
            <StatTile
              label="Avg Path Loss"
              value={avgPathLoss !== null ? `${avgPathLoss.toFixed(1)} dB` : "—"}
            />
          </div>

          <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
            <div className="rounded-lg border border-border bg-surface p-5">
              <h2 className="text-sm font-medium text-text-primary">SNR History (dB)</h2>
              <div className="mt-3">
                <Sparkline
                  values={framesWithSnr.map((f) => f.snr_db as number)}
                  width={400}
                  height={80}
                  color="var(--color-accent)"
                />
              </div>
            </div>
            <div className="rounded-lg border border-border bg-surface p-5">
              <h2 className="text-sm font-medium text-text-primary">RSSI History (dBm)</h2>
              <div className="mt-3">
                <Sparkline
                  values={framesWithSnr.map((f) => f.rssi_dbm as number)}
                  width={400}
                  height={80}
                  color="var(--color-warning)"
                />
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-lg border border-border bg-surface p-5">
            <h2 className="text-sm font-medium text-text-primary">Recent Frames</h2>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-text-secondary">
                    <th className="pb-2 pr-4">Vehicle</th>
                    <th className="pb-2 pr-4">Frame</th>
                    <th className="pb-2 pr-4">SNR</th>
                    <th className="pb-2 pr-4">RSSI</th>
                    <th className="pb-2 pr-4">Beam</th>
                    <th className="pb-2 pr-4">Sync</th>
                  </tr>
                </thead>
                <tbody className="mono">
                  {frames.slice(0, 15).map((frame) => (
                    <tr key={frame.id} className="border-t border-border">
                      <td className="py-1.5 pr-4">{frame.vehicle_id.slice(0, 8)}</td>
                      <td className="py-1.5 pr-4">{frame.frame_index}</td>
                      <td className="py-1.5 pr-4">{frame.snr_db?.toFixed(1) ?? "—"}</td>
                      <td className="py-1.5 pr-4">{frame.rssi_dbm?.toFixed(1) ?? "—"}</td>
                      <td className="py-1.5 pr-4">{frame.beam_index ?? "—"}</td>
                      <td className="py-1.5 pr-4">
                        <Badge
                          color={
                            frame.is_sync_valid ? "var(--color-success)" : "var(--color-danger)"
                          }
                        >
                          {frame.is_sync_valid ? "ok" : "drift"}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
