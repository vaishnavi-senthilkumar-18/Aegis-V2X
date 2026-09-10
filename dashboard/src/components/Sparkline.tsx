interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}

/** Minimal dependency-free sparkline — one polyline, no axes/labels. */
export function Sparkline({ values, width = 200, height = 48, color = "var(--color-accent)" }: SparklineProps) {
  if (values.length < 2) {
    return (
      <div
        style={{ width, height }}
        className="flex items-center justify-center text-[11px] text-text-muted"
      >
        not enough data
      </div>
    );
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}
