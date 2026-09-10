import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { StatTile } from "../components/StatTile";
import { EmptyState } from "../components/EmptyState";
import {
  getActionDistribution,
  getUnsyncedCount,
  listDecisions,
  listScenes,
} from "../lib/api";
import { formatPercent, formatTimestamp } from "../lib/format";
import { Link } from "react-router-dom";

export function Overview() {
  const scenesQuery = useQuery({ queryKey: ["scenes"], queryFn: listScenes, refetchInterval: 5000 });
  const unsyncedQuery = useQuery({
    queryKey: ["unsynced-count"],
    queryFn: () => getUnsyncedCount(),
    refetchInterval: 5000,
  });
  const distributionQuery = useQuery({
    queryKey: ["action-distribution"],
    queryFn: getActionDistribution,
    refetchInterval: 5000,
  });
  const decisionsQuery = useQuery({
    queryKey: ["decisions", "recent"],
    queryFn: () => listDecisions(10),
    refetchInterval: 5000,
  });

  // Trend deltas require >=2 real historical points; we only ever have the
  // current snapshot from polling, so trends are honestly "—" rather than
  // a fabricated delta (design-integrity rule from the Phase 3 rebuild).
  const [firstLoadTimestamp] = useState(() => new Date().toISOString());

  const scenes = scenesQuery.data ?? [];
  const totalDecisions = distributionQuery.data?.total_decisions ?? 0;

  return (
    <div>
      <PageHeader
        title="Overview"
        subtitle="Live snapshot of the Aegis-V2X Digital Twin backend"
      />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Scenes" value={String(scenes.length)} trend="—" />
        <StatTile
          label="Sync Health"
          value={
            unsyncedQuery.data
              ? formatPercent(1 - unsyncedQuery.data.unsynchronized_ratio)
              : "—"
          }
          trend="—"
        />
        <StatTile label="Decisions Logged" value={String(totalDecisions)} trend="—" />
        <StatTile
          label="Session Started"
          value={formatTimestamp(firstLoadTimestamp)}
        />
      </div>

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-surface p-5">
          <h2 className="text-sm font-medium text-text-primary">Scenes</h2>
          {scenes.length === 0 ? (
            <div className="mt-4">
              <EmptyState
                title="No scenes yet"
                detail="Generate a synthetic scene from the Simulation page to seed the backend with demo data, or wait for Phase 2 (CARLA + Sionna RT) to deliver real scenes."
              />
            </div>
          ) : (
            <ul className="mt-4 divide-y divide-border">
              {scenes.slice(0, 6).map((scene) => (
                <li key={scene.id} className="flex items-center justify-between py-2 text-sm">
                  <span>{scene.scene_code}</span>
                  <span className="mono text-xs text-text-secondary">
                    {scene.num_vehicles_target} vehicles · {scene.map_name}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <Link
            to="/simulation"
            className="mt-4 inline-block text-xs text-accent hover:underline"
          >
            Manage scenes →
          </Link>
        </div>

        <div className="rounded-lg border border-border bg-surface p-5">
          <h2 className="text-sm font-medium text-text-primary">Recent Decisions</h2>
          {(decisionsQuery.data?.length ?? 0) === 0 ? (
            <div className="mt-4">
              <EmptyState
                title="No decisions yet"
                detail="Decisions appear once frames have trust and criticality records and TwinTrust-AP (currently a Phase 3 placeholder heuristic) has scored them."
              />
            </div>
          ) : (
            <ul className="mt-4 divide-y divide-border">
              {decisionsQuery.data?.slice(0, 6).map((decision) => (
                <li key={decision.id} className="flex items-center justify-between py-2 text-sm">
                  <span className="mono text-xs">{decision.fsdp_action}</span>
                  <span className="text-xs text-text-secondary">
                    H={decision.prediction_horizon} · T=
                    {decision.trust_probability_used.toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <Link
            to="/twintrust-ap"
            className="mt-4 inline-block text-xs text-accent hover:underline"
          >
            View TwinTrust-AP pipeline →
          </Link>
        </div>
      </div>
    </div>
  );
}
