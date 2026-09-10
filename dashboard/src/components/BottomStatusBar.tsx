import { useApiTelemetry } from "../hooks/useApiTelemetry";
import { useSystemStatus } from "../hooks/useSystemStatus";
import { formatMs } from "../lib/format";
import { StatusDot } from "./StatusDot";

export function BottomStatusBar() {
  const { latestLatencyMs, avgLatencyMs } = useApiTelemetry();
  const { subsystems } = useSystemStatus();
  const backend = subsystems.find((s) => s.label === "Backend API");

  return (
    <footer className="flex h-8 shrink-0 items-center justify-between border-t border-border bg-surface px-4 text-xs text-text-muted">
      <div className="flex items-center gap-4">
        <span className="flex items-center gap-1.5">
          <StatusDot status={backend?.status ?? "offline"} />
          {backend?.status === "online" ? "Backend connected" : "Backend unreachable"}
        </span>
        <span className="mono">
          latest {formatMs(latestLatencyMs)} · avg {formatMs(avgLatencyMs)}
        </span>
      </div>
      <span>Aegis-V2X Research Console · Phase 3</span>
    </footer>
  );
}
