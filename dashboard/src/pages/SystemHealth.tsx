import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "../components/PageHeader";
import { StatTile } from "../components/StatTile";
import { StatusDot } from "../components/StatusDot";
import { EmptyState } from "../components/EmptyState";
import { useSystemStatus } from "../hooks/useSystemStatus";
import { useApiTelemetry } from "../hooks/useApiTelemetry";
import { getUnsyncedCount, listDecisions } from "../lib/api";
import { formatMs, formatPercent, formatTimestamp } from "../lib/format";

const STATUS_LABEL: Record<string, string> = {
  online: "Online",
  offline: "Offline",
  not_built: "Not built yet",
};

export function SystemHealth() {
  const { subsystems } = useSystemStatus();
  const telemetry = useApiTelemetry();
  const unsyncedQuery = useQuery({
    queryKey: ["unsynced-count"],
    queryFn: () => getUnsyncedCount(),
    refetchInterval: 5000,
  });
  const recentDecisionsQuery = useQuery({
    queryKey: ["decisions", "activity-feed"],
    queryFn: () => listDecisions(15),
    refetchInterval: 4000,
  });

  return (
    <div>
      <PageHeader
        title="System Health"
        subtitle="Live subsystem status, API latency, database sync health, and recent activity"
      />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Latest API Latency" value={formatMs(telemetry.latestLatencyMs)} />
        <StatTile label="Avg API Latency" value={formatMs(telemetry.avgLatencyMs)} />
        <StatTile
          label="Sync Health"
          value={
            unsyncedQuery.data
              ? formatPercent(1 - unsyncedQuery.data.unsynchronized_ratio)
              : "—"
          }
        />
        <StatTile
          label="Out-of-Tolerance Frames"
          value={unsyncedQuery.data ? String(unsyncedQuery.data.unsynchronized_frames) : "—"}
        />
      </div>

      <div className="mt-6 rounded-lg border border-border bg-surface p-5">
        <h2 className="text-sm font-medium text-text-primary">Subsystems</h2>
        <p className="mt-1 text-xs text-text-secondary">
          Reported honestly — subsystems from phases not yet built are always shown as such, never
          faked as online.
        </p>
        <ul className="mt-4 divide-y divide-border">
          {subsystems.map((subsystem) => (
            <li key={subsystem.label} className="flex items-center justify-between py-3 text-sm">
              <div className="flex items-center gap-2">
                <StatusDot status={subsystem.status} />
                <span>{subsystem.label}</span>
              </div>
              <div className="text-right">
                <p className="text-xs">{STATUS_LABEL[subsystem.status]}</p>
                <p className="text-[11px] text-text-muted">{subsystem.detail}</p>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-6 rounded-lg border border-border bg-surface p-5">
        <h2 className="text-sm font-medium text-text-primary">Live Activity Feed</h2>
        {(recentDecisionsQuery.data?.length ?? 0) === 0 ? (
          <div className="mt-4">
            <EmptyState
              title="No recent activity"
              detail="This feed shows real decisions as they're logged, not simulated activity."
            />
          </div>
        ) : (
          <ul className="mt-4 divide-y divide-border">
            {recentDecisionsQuery.data?.map((decision) => (
              <li key={decision.id} className="flex items-center justify-between py-2 text-xs">
                <span className="mono">
                  {decision.fsdp_action} · H={decision.prediction_horizon}
                </span>
                <span className="text-text-muted">{formatTimestamp(decision.created_at)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
