import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../lib/api";

export type SubsystemStatus = "online" | "offline" | "not_built";

export interface SubsystemState {
  label: string;
  status: SubsystemStatus;
  detail: string;
}

/**
 * Reports live status for every subsystem the dashboard could plausibly
 * show. Design-integrity rule established during the Phase 3 dashboard
 * rebuild and preserved here: never show a subsystem as "online" or
 * fabricate a data point that doesn't exist. CARLA, Sionna RT, and the
 * AI models (PointPillars / V2X-ViT / GRU / Trust Estimator) are Phase
 * 2/4/5 deliverables that do not exist yet in this codebase, so they are
 * always reported `not_built` regardless of what the backend says —
 * there is nothing for the backend to even check.
 */
export function useSystemStatus() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 5000,
    retry: false,
  });

  const subsystems: SubsystemState[] = [
    {
      label: "Backend API",
      status: health.isSuccess ? "online" : health.isError ? "offline" : "offline",
      detail: health.isSuccess
        ? "FastAPI + PostgreSQL reachable"
        : "Unreachable — is uvicorn running?",
    },
    {
      label: "PostgreSQL",
      status: health.isSuccess ? "online" : "offline",
      detail: health.isSuccess ? "Reachable via backend health check" : "Unknown",
    },
    {
      label: "CARLA Simulator",
      status: "not_built",
      detail: "Phase 2 (Simulation & Dataset) — not started",
    },
    {
      label: "NVIDIA Sionna RT",
      status: "not_built",
      detail: "Phase 2 (Simulation & Dataset) — not started",
    },
    {
      label: "AI Models (PointPillars / V2X-ViT / GRU)",
      status: "not_built",
      detail: "Phase 4 (Perception & Trust Estimation) — not started",
    },
    {
      label: "TwinTrust-AP Policy",
      status: "not_built",
      detail: "Phase 5 — not started; decisions currently use a placeholder heuristic",
    },
  ];

  return { subsystems, isLoading: health.isLoading };
}
