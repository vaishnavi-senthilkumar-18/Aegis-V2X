import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "../components/PageHeader";
import { StatTile } from "../components/StatTile";
import { EmptyState } from "../components/EmptyState";
import { Badge } from "../components/Badge";
import { getActionDistribution, listDecisions, listTrustRecords } from "../lib/api";
import { trustColor, trustLabel } from "../lib/format";

export function TwinTrustAp() {
  const trustQuery = useQuery({
    queryKey: ["trust-records"],
    queryFn: () => listTrustRecords(100),
    refetchInterval: 4000,
  });
  const decisionsQuery = useQuery({
    queryKey: ["decisions", "twintrust"],
    queryFn: () => listDecisions(20),
    refetchInterval: 4000,
  });
  const distributionQuery = useQuery({
    queryKey: ["action-distribution"],
    queryFn: getActionDistribution,
    refetchInterval: 4000,
  });

  const trustRecords = trustQuery.data ?? [];
  const avgTrust =
    trustRecords.length > 0
      ? trustRecords.reduce((sum, t) => sum + t.trust_probability, 0) / trustRecords.length
      : null;

  const decisions = decisionsQuery.data ?? [];
  const distribution = distributionQuery.data;
  const maxCount = distribution ? Math.max(1, ...distribution.distribution.map((d) => d.count)) : 1;

  return (
    <div>
      <PageHeader
        title="TwinTrust-AP"
        subtitle="Calibrated trust estimation → criticality scoring → joint decision pipeline (Phase 3 placeholder policy; Phase 5 delivers TAHS + FSDP optimization)"
      />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile
          label="Avg Trust Probability (T_t)"
          value={avgTrust !== null ? avgTrust.toFixed(3) : "—"}
        />
        <StatTile label="Trust Records Scored" value={String(trustRecords.length)} />
        <StatTile label="Decisions Logged" value={String(distribution?.total_decisions ?? 0)} />
        <StatTile label="Policy Source" value="synthetic (Phase 3)" />
      </div>

      <div className="mt-6 rounded-lg border border-warning/30 bg-surface p-4 text-xs text-text-secondary">
        The decisions below use a placeholder heuristic policy (
        <code className="mono">policy_source = "synthetic"</code>), not the real optimized
        TAHS/FSDP policy — that arrives in Phase 5 and will write through the same{" "}
        <code className="mono">decisions</code> table with{" "}
        <code className="mono">policy_source = "fsdp_table"</code>.
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-border bg-surface p-5">
          <h2 className="text-sm font-medium text-text-primary">FSDP Action Distribution</h2>
          {!distribution || distribution.distribution.length === 0 ? (
            <div className="mt-4">
              <EmptyState title="No decisions yet" detail="Distribution populates once decisions are logged." />
            </div>
          ) : (
            <div className="mt-4 space-y-2">
              {distribution.distribution.map((entry) => (
                <div key={entry.fsdp_action} className="flex items-center gap-3">
                  <span className="mono w-32 shrink-0 text-xs text-text-secondary">
                    {entry.fsdp_action}
                  </span>
                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-surface-raised">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{ width: `${(entry.count / maxCount) * 100}%` }}
                    />
                  </div>
                  <span className="mono w-10 text-right text-xs text-text-secondary">
                    {entry.count}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-lg border border-border bg-surface p-5">
          <h2 className="text-sm font-medium text-text-primary">Trust Interpretation Bands</h2>
          <p className="mt-1 text-xs text-text-secondary">
            Reproduced from Mathematical Formulation Sec. 5
          </p>
          <ul className="mt-4 space-y-2 text-xs">
            {["very_unreliable", "unreliable", "moderate", "reliable", "highly_reliable"].map(
              (band) => {
                const count = trustRecords.filter((t) => t.interpretation === band).length;
                return (
                  <li key={band} className="flex items-center justify-between">
                    <Badge color={trustColor(band)}>{trustLabel(band)}</Badge>
                    <span className="mono text-text-secondary">{count} frames</span>
                  </li>
                );
              },
            )}
          </ul>
        </div>
      </div>

      <div className="mt-6 rounded-lg border border-border bg-surface p-5">
        <h2 className="text-sm font-medium text-text-primary">Recent Decisions</h2>
        {decisions.length === 0 ? (
          <div className="mt-4">
            <EmptyState title="No decisions logged" detail="Decisions appear once frames are fully scored." />
          </div>
        ) : (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-text-secondary">
                  <th className="pb-2 pr-4">Action</th>
                  <th className="pb-2 pr-4">Horizon</th>
                  <th className="pb-2 pr-4">T_t</th>
                  <th className="pb-2 pr-4">C_t</th>
                  <th className="pb-2 pr-4">Policy</th>
                  <th className="pb-2 pr-4">Rationale</th>
                </tr>
              </thead>
              <tbody className="mono">
                {decisions.map((decision) => (
                  <tr key={decision.id} className="border-t border-border">
                    <td className="py-1.5 pr-4">{decision.fsdp_action}</td>
                    <td className="py-1.5 pr-4">{decision.prediction_horizon}</td>
                    <td className="py-1.5 pr-4">{decision.trust_probability_used.toFixed(3)}</td>
                    <td className="py-1.5 pr-4">{decision.criticality_score_used.toFixed(3)}</td>
                    <td className="py-1.5 pr-4">
                      <Badge>{decision.policy_source}</Badge>
                    </td>
                    <td className="max-w-xs truncate py-1.5 pr-4 text-text-secondary">
                      {decision.rationale ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
