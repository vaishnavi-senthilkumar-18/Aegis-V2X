interface StatTileProps {
  label: string;
  value: string;
  /** "—" when there isn't yet enough real history to compute a delta —
   * never fabricate a trend from a single data point. */
  trend?: string;
  trendPositive?: boolean;
  accent?: string;
}

export function StatTile({ label, value, trend, trendPositive, accent }: StatTileProps) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <p className="text-xs text-text-secondary">{label}</p>
      <p className="mono mt-2 text-2xl font-semibold" style={accent ? { color: accent } : undefined}>
        {value}
      </p>
      {trend !== undefined && (
        <p
          className={`mt-1 text-xs ${
            trend === "—"
              ? "text-text-muted"
              : trendPositive
                ? "text-success"
                : "text-danger"
          }`}
        >
          {trend}
        </p>
      )}
    </div>
  );
}
