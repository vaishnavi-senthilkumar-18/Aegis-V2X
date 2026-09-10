import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "../components/PageHeader";
import { StatTile } from "../components/StatTile";
import { EmptyState } from "../components/EmptyState";
import { listCriticalityRecords, listDecisions, listTrustRecords } from "../lib/api";

function histogram(values: number[], buckets = 10): number[] {
  const counts = new Array(buckets).fill(0);
  for (const v of values) {
    const idx = Math.min(buckets - 1, Math.max(0, Math.floor(v * buckets)));
    counts[idx] += 1;
  }
  return counts;
}

function Histogram({ values, color }: { values: number[]; color: string }) {
  const counts = histogram(values);
  const max = Math.max(1, ...counts);
  return (
    <div className="flex h-24 items-end gap-1">
      {counts.map((count, i) => (
        <div
          key={i}
          className="flex-1 rounded-t"
          style={{ height: `${(count / max) * 100}%`, backgroundColor: color, minHeight: count > 0 ? 2 : 0 }}
          title={`${(i / counts.length).toFixed(1)}–${((i + 1) / counts.length).toFixed(1)}: ${count}`}
        />
      ))}
    </div>
  );
}

export function Analytics() {
  const trustQuery = useQuery({ queryKey: ["trust-records", "analytics"], queryFn: () => listTrustRecords(500) });
  const criticalityQuery = useQuery({
    queryKey: ["criticality-records", "analytics"],
    queryFn: () => listCriticalityRecords(500),
  });
  const decisionsQuery = useQuery({
    queryKey: ["decisions", "analytics"],
    queryFn: () => listDecisions(500),
  });

  const trustValues = (trustQuery.data ?? []).map((t) => t.trust_probability);
  const criticalityValues = (criticalityQuery.data ?? []).map((c) => c.criticality_score);
  const horizons = (decisionsQuery.data ?? []).map((d) => d.prediction_horizon);
  const horizonCounts = horizons.reduce<Record<number, number>>((acc, h) => {
    acc[h] = (acc[h] ?? 0) + 1;
    return acc;
  }, {});
  const maxHorizonCount = Math.max(1, ...Object.values(horizonCounts));

  const hasData = trustValues.length > 0 || criticalityValues.length > 0;

  return (
    <div>
      <PageHeader
        title="Analytics"
        subtitle="Aggregate distributions across trust, criticality, and prediction horizons"
      />

      {!hasData ? (
        <EmptyState
          title="No records to analyze yet"
          detail="Generate a synthetic scene (Simulation page) or wait for real ingestion to populate trust/criticality/decision records."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatTile label="Trust Records" value={String(trustValues.length)} />
            <StatTile label="Criticality Records" value={String(criticalityValues.length)} />
            <StatTile label="Decisions" value={String(horizons.length)} />
            <StatTile
              label="Mean Criticality (C_t)"
              value={
                criticalityValues.length > 0
                  ? (criticalityValues.reduce((a, b) => a + b, 0) / criticalityValues.length).toFixed(3)
                  : "—"
              }
            />
          </div>

          <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
            <div className="rounded-lg border border-border bg-surface p-5">
              <h2 className="text-sm font-medium text-text-primary">Trust Probability Distribution</h2>
              <div className="mt-4">
                <Histogram values={trustValues} color="var(--color-accent)" />
              </div>
            </div>
            <div className="rounded-lg border border-border bg-surface p-5">
              <h2 className="text-sm font-medium text-text-primary">Criticality Score Distribution</h2>
              <div className="mt-4">
                <Histogram values={criticalityValues} color="var(--color-warning)" />
              </div>
            </div>
          </div>

          <div className="mt-6 rounded-lg border border-border bg-surface p-5">
            <h2 className="text-sm font-medium text-text-primary">Prediction Horizon Distribution</h2>
            <div className="mt-4 space-y-2">
              {[1, 2, 3, 5, 8, 10].map((h) => (
                <div key={h} className="flex items-center gap-3">
                  <span className="mono w-10 shrink-0 text-xs text-text-secondary">H={h}</span>
                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-surface-raised">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{ width: `${((horizonCounts[h] ?? 0) / maxHorizonCount) * 100}%` }}
                    />
                  </div>
                  <span className="mono w-10 text-right text-xs text-text-secondary">
                    {horizonCounts[h] ?? 0}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
