/**
 * Thin REST client for the Aegis-V2X backend. Every function maps 1:1 to a
 * single endpoint documented in `docs/backend_api_documentation.md` §4.
 * No mocking, no fallback data — if a request fails, callers see the real
 * error via react-query's error state (design-integrity rule: never
 * fabricate a data point that doesn't exist).
 */

import type {
  ActionDistributionResponse,
  CriticalityRecord,
  Decision,
  Experiment,
  Frame,
  HealthResponse,
  LatestVehicleFrame,
  Scene,
  TrustRecord,
  UnsyncedCountResponse,
  Vehicle,
} from "../types/api";

const API_BASE = "/api/v1";

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const start = performance.now();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const latencyMs = performance.now() - start;
  recordLatencySample(latencyMs);

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // response had no JSON body; fall back to statusText
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

// --- Lightweight latency sample tracking for the System Health page ---
// Kept here (not in a separate store) since it is a direct side effect of
// every request this client makes.
type LatencyListener = (samples: number[]) => void;
const latencyListeners = new Set<LatencyListener>();
const latencySamples: number[] = [];
const MAX_LATENCY_SAMPLES = 50;

function recordLatencySample(ms: number): void {
  latencySamples.push(ms);
  if (latencySamples.length > MAX_LATENCY_SAMPLES) latencySamples.shift();
  latencyListeners.forEach((listener) => listener([...latencySamples]));
}

export function subscribeToLatencySamples(listener: LatencyListener): () => void {
  latencyListeners.add(listener);
  return () => latencyListeners.delete(listener);
}

export function getLatencySamplesSnapshot(): number[] {
  return latencySamples;
}

// --- Health ---
export const getHealth = () => request<HealthResponse>("/health");

// --- Scenes / Vehicles ---
export const listScenes = () => request<Scene[]>("/scenes");
export const getScene = (sceneId: string) => request<Scene>(`/scenes/${sceneId}`);
export const listVehicles = (sceneId: string) =>
  request<Vehicle[]>(`/scenes/${sceneId}/vehicles`);
export const createSyntheticScene = (payload: {
  scene_code: string;
  num_vehicles: number;
  num_frames: number;
  map_name?: string;
  weather_preset?: string;
}) =>
  request<{
    scene_id: string;
    scene_code: string;
    vehicles_created: number;
    frames_created: number;
    trust_records_created: number;
    criticality_records_created: number;
    decisions_created: number;
  }>("/synthetic/scenes", { method: "POST", body: JSON.stringify(payload) });

// --- Frames ---
export const listFrames = (params: { scene_id?: string; vehicle_id?: string; limit?: number }) => {
  const search = new URLSearchParams();
  if (params.scene_id) search.set("scene_id", params.scene_id);
  if (params.vehicle_id) search.set("vehicle_id", params.vehicle_id);
  if (params.limit) search.set("limit", String(params.limit));
  return request<Frame[]>(`/frames?${search.toString()}`);
};
export const getLatestFramePerVehicle = (sceneId: string) =>
  request<LatestVehicleFrame[]>(`/frames/scene/${sceneId}/latest-per-vehicle`);
export const getUnsyncedCount = (sceneId?: string) =>
  request<UnsyncedCountResponse>(
    `/frames/stats/unsynchronized-count${sceneId ? `?scene_id=${sceneId}` : ""}`,
  );

// --- Trust / Criticality ---
export const getTrustByFrame = (frameId: string) =>
  request<TrustRecord>(`/trust/frame/${frameId}`);
export const listTrustRecords = (limit = 100) =>
  request<TrustRecord[]>(`/trust?limit=${limit}`);
export const getCriticalityByFrame = (frameId: string) =>
  request<CriticalityRecord>(`/criticality/frame/${frameId}`);
export const listCriticalityRecords = (limit = 100) =>
  request<CriticalityRecord[]>(`/criticality?limit=${limit}`);

// --- Decisions ---
export const listDecisions = (limit = 100) => request<Decision[]>(`/decisions?limit=${limit}`);
export const getDecisionByFrame = (frameId: string) =>
  request<Decision>(`/decisions/frame/${frameId}`);
export const getActionDistribution = () =>
  request<ActionDistributionResponse>("/decisions/stats/action-distribution");

// --- Experiments ---
export const listExperiments = () => request<Experiment[]>("/experiments");

export { ApiError };
