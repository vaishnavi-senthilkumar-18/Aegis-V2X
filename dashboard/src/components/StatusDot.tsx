import type { SubsystemStatus } from "../hooks/useSystemStatus";

const COLOR: Record<SubsystemStatus, string> = {
  online: "bg-success",
  offline: "bg-danger",
  not_built: "bg-text-muted",
};

export function StatusDot({ status }: { status: SubsystemStatus }) {
  return <span className={`inline-block h-2 w-2 rounded-full ${COLOR[status]}`} aria-hidden />;
}
