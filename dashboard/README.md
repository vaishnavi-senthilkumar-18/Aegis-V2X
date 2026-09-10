# Aegis-V2X Research Console (Dashboard)

Phase 3 deliverable (Backend & Dashboard, Logapriya). A Vite + React 19 +
TypeScript + Tailwind v4 single-page app, served by the FastAPI backend at
`/dashboard/` in production, or standalone via the Vite dev server during
development.

No CDN dependencies — everything is npm-bundled. (The sandbox this project
was built in blocks `cdnjs`/`jsdelivr`; the same constraint likely applies
to any deployment context with restricted egress, e.g. near the Jetson
Orin, so this was kept as a hard rule rather than a one-off workaround.)

## Pages

| Page | Route | What it shows |
|---|---|---|
| Overview | `/` | Scene count, sync health, recent decisions |
| Digital Twin | `/digital-twin` | SVG radar map of the intersection (from `position_x/y`, `lane_id`) + vehicle inspector |
| V2X Network | `/v2x-network` | Channel state, SNR/RSSI history, beam allocation, recent frames |
| TwinTrust-AP | `/twintrust-ap` | Trust probability distribution, criticality, FSDP action distribution, decision log |
| Simulation | `/simulation` | Scene config, vehicle roster, synthetic-scene generator |
| Experiments | `/experiments` | Named research run registry (honest empty state pre-Phase 6/7) |
| Analytics | `/analytics` | Aggregate trust/criticality/prediction-horizon distributions |
| System Health | `/system-health` | API latency, live subsystem status, sync health, activity feed |

## Design-integrity rule

**Never show a subsystem as "online" or fabricate a data point that
doesn't exist.** `src/hooks/useSystemStatus.ts` always reports CARLA,
Sionna RT, and the AI models as `not_built` — those are Phase 2/4/5
deliverables that don't exist in this codebase yet, so there is nothing
for the frontend to even check. Trend indicators show `—` rather than a
fabricated delta until real historical data exists (see
`src/components/StatTile.tsx`). Empty states (`src/components/EmptyState.tsx`)
say plainly what's missing and why, rather than looking like a loading
error. Keep this rule when extending the dashboard in later phases.

## Setup

```bash
cd dashboard
npm install
```

## Development

```bash
npm run dev
```

Runs on `http://localhost:5173` with API requests proxied to
`http://localhost:8000` (see `vite.config.ts`). Run the backend
separately (`cd ../backend && uvicorn app.main:app --reload`).

## Production build

```bash
npm run build
```

Outputs to `dashboard/dist`, which the backend automatically mounts at
`/dashboard/` if present (see `backend/app/main.py`). The build's asset
base path is `/dashboard/` (`vite.config.ts` → `base`) to match.

## Architecture notes

- **State/data:** `@tanstack/react-query` for all server state (polling
  every 3-8s depending on the page); no client-side caching layer beyond
  that. No Redux/Zustand — page-local `useState` is enough for the UI
  state that exists (selected scene, form inputs).
- **Routing:** `react-router-dom`, `BrowserRouter` with `basename="/dashboard"`
  so the backend's static mount and the client router agree. The backend's
  `SPAStaticFiles` class falls back to `index.html` for any extension-less
  path so direct navigation/hard-refresh to a client-side route (e.g.
  `/dashboard/analytics`) doesn't 404 — see the note in
  `backend/app/main.py` about the two ways that fallback was broken before
  it worked (checking a status code that's never returned; then catching
  the wrong `HTTPException` class).
- **Styling:** Tailwind v4 with a small custom theme (`src/index.css`) —
  dark, research-console aesthetic. No component library; hand-built
  primitives in `src/components/` (`StatTile`, `Badge`, `EmptyState`,
  `Sparkline`, `PageHeader`, `StatusDot`).
- **API client:** `src/lib/api.ts` is a thin `fetch` wrapper, one function
  per backend endpoint, typed against `src/types/api.ts` (hand-written
  mirrors of the backend's Pydantic schemas — kept close to the Python
  field names on purpose, so cross-referencing doesn't require a mental
  mapping step). It also tracks a rolling window of request latencies for
  the System Health page.
- **`useApiTelemetry`** (`src/hooks/useApiTelemetry.ts`) uses
  `useSyncExternalStore`. The snapshot function is memoized/cached and
  only produces a new object reference when the underlying data actually
  changes — `useSyncExternalStore` compares snapshots with `Object.is`,
  and an earlier version of this hook that returned a fresh object every
  call caused an infinite re-render loop (React error #185) in
  `BottomStatusBar`. Keep that memoization if you touch this hook.
