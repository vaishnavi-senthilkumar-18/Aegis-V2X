import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "../components/PageHeader";
import { EmptyState } from "../components/EmptyState";
import { Badge } from "../components/Badge";
import { listExperiments } from "../lib/api";
import type { ExperimentStatus } from "../types/api";

const STATUS_COLOR: Record<ExperimentStatus, string> = {
  planned: "var(--color-text-secondary)",
  running: "var(--color-accent)",
  completed: "var(--color-success)",
  failed: "var(--color-danger)",
  archived: "var(--color-text-muted)",
};

export function Experiments() {
  const experimentsQuery = useQuery({
    queryKey: ["experiments"],
    queryFn: listExperiments,
    refetchInterval: 8000,
  });
  const experiments = experimentsQuery.data ?? [];

  return (
    <div>
      <PageHeader
        title="Experiments"
        subtitle="Named, config-versioned research runs — populated in Phase 6/7 (baselines, ablations, statistical evaluation)"
      />

      {experiments.length === 0 ? (
        <EmptyState
          title="No experiments registered yet"
          detail="This is an honest empty state, not a loading error: Phase 3 built the experiment registry, but Phase 6/7 (Edge Deployment, Integration & Evaluation) is what actually runs baseline/ablation experiments against it."
        />
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-border text-xs text-text-secondary">
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Latency</th>
                <th className="px-4 py-3">Sync Overhead</th>
                <th className="px-4 py-3">Energy</th>
                <th className="px-4 py-3">Reliability</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((experiment) => (
                <tr key={experiment.id} className="border-b border-border last:border-0">
                  <td className="px-4 py-3">{experiment.name}</td>
                  <td className="px-4 py-3">
                    <Badge color={STATUS_COLOR[experiment.status]}>{experiment.status}</Badge>
                  </td>
                  <td className="mono px-4 py-3 text-xs">
                    {experiment.latency_ms !== null ? `${experiment.latency_ms}ms` : "—"}
                  </td>
                  <td className="mono px-4 py-3 text-xs">
                    {experiment.sync_overhead_ms !== null
                      ? `${experiment.sync_overhead_ms}ms`
                      : "—"}
                  </td>
                  <td className="mono px-4 py-3 text-xs">
                    {experiment.energy_j !== null ? `${experiment.energy_j}J` : "—"}
                  </td>
                  <td className="mono px-4 py-3 text-xs">
                    {experiment.reliability_score !== null
                      ? experiment.reliability_score.toFixed(3)
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
