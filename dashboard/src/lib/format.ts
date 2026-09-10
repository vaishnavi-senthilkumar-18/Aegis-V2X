/** Small formatting helpers shared across pages. */

export function formatPercent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(1)}ms`;
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/**
 * Vehicle codes already contain the "Vehicle" literal (e.g. "Vehicle00"),
 * so labeling them as "Vehicle Vehicle00" is redundant. This strips a
 * leading "Vehicle " prefix if present — fixed during the Phase 3
 * dashboard rebuild after the redundant label was noticed in review.
 */
export function formatVehicleLabel(vehicleCode: string): string {
  return vehicleCode.startsWith("Vehicle") ? vehicleCode : `Vehicle ${vehicleCode}`;
}

export function trustColor(interpretation: string): string {
  switch (interpretation) {
    case "highly_reliable":
    case "reliable":
      return "var(--color-trust-high)";
    case "moderate":
      return "var(--color-trust-mid)";
    default:
      return "var(--color-trust-low)";
  }
}

export function trustLabel(interpretation: string): string {
  return interpretation.replace(/_/g, " ");
}
