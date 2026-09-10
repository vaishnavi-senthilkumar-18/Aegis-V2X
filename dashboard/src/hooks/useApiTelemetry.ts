import { useSyncExternalStore } from "react";
import { getLatencySamplesSnapshot, subscribeToLatencySamples } from "../lib/api";

export interface ApiTelemetry {
  latestLatencyMs: number | null;
  avgLatencyMs: number | null;
  sampleCount: number;
}

/**
 * Exposes rolling API latency stats via `useSyncExternalStore`.
 *
 * IMPORTANT: `getSnapshot()` must return a referentially stable object when
 * nothing has changed — `useSyncExternalStore` compares snapshots with
 * `Object.is`. During the Phase 3 dashboard rebuild, an earlier version of
 * this hook computed a fresh object literal on every call, which broke
 * that contract and caused an infinite re-render loop in the bottom status
 * bar (React error #185). The fix: cache the computed snapshot and only
 * replace it when the underlying latency actually changes.
 */
let cachedSnapshot: ApiTelemetry = { latestLatencyMs: null, avgLatencyMs: null, sampleCount: 0 };
let cachedSampleCount = -1;

function computeSnapshot(): ApiTelemetry {
  const samples = getLatencySamplesSnapshot();
  if (samples.length === cachedSampleCount) {
    return cachedSnapshot;
  }
  cachedSampleCount = samples.length;
  const latest = samples.length > 0 ? samples[samples.length - 1] : null;
  const avg =
    samples.length > 0 ? samples.reduce((sum, value) => sum + value, 0) / samples.length : null;
  cachedSnapshot = { latestLatencyMs: latest, avgLatencyMs: avg, sampleCount: samples.length };
  return cachedSnapshot;
}

export function useApiTelemetry(): ApiTelemetry {
  return useSyncExternalStore(subscribeToLatencySamples, computeSnapshot);
}
