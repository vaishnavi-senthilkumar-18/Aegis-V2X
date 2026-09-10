import { NavLink, Outlet } from "react-router-dom";
import { BottomStatusBar } from "./BottomStatusBar";

// Paths are relative to the router's basename ("/dashboard", set in App.tsx
// to match where the backend mounts the built SPA) — react-router prepends
// it automatically, so these must NOT repeat the "/dashboard" prefix.
const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/digital-twin", label: "Digital Twin" },
  { to: "/v2x-network", label: "V2X Network" },
  { to: "/twintrust-ap", label: "TwinTrust-AP" },
  { to: "/simulation", label: "Simulation" },
  { to: "/experiments", label: "Experiments" },
  { to: "/analytics", label: "Analytics" },
  { to: "/system-health", label: "System Health" },
];

export function Layout() {
  return (
    <div className="flex h-screen flex-col bg-bg text-text-primary">
      <div className="flex flex-1 overflow-hidden">
        <aside className="flex w-60 flex-col border-r border-border bg-surface">
          <div className="border-b border-border px-5 py-5">
            <p className="text-sm font-semibold tracking-wide text-text-primary">Aegis-V2X</p>
            <p className="mt-0.5 text-xs text-text-muted">Research Console — Phase 3</p>
          </div>
          <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-4">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `block rounded-md px-3 py-2 text-sm transition-colors ${
                    isActive
                      ? "bg-accent-soft text-text-primary"
                      : "text-text-secondary hover:bg-surface-raised hover:text-text-primary"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="border-t border-border px-5 py-4 text-xs text-text-muted">
            <p>Backend & Dashboard</p>
            <p className="mt-0.5">Lead: Logapriya</p>
          </div>
        </aside>
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-8 py-8">
            <Outlet />
          </div>
        </main>
      </div>
      <BottomStatusBar />
    </div>
  );
}
