# Aegis-V2X Backend — API & Schema Documentation

**Phase:** 3 — Backend & Dashboard
**Lead:** Logapriya
**Status:** Implemented, tested (49/49 pytest — the original 38 plus bulk ingestion and API-key auth added 2026-08-25), verified against live PostgreSQL 16
**Note:** This document reflects the **2026-08-15 full rebuild** of Phase 3 after the
original `Aegis-V2X/` working folder was lost. Schema, API contracts, and design
decisions are unchanged from the original delivery — see §8 for what happened and
what, if anything, differs from the original.

## 1. Purpose

This document specifies the database schema and REST API delivered in
Phase 3, and how each maps back to the authoritative specification docs:

* `02_Aegis-V2X_System_Architecture.pdf.docx` (Sec. 6 — Backend Architecture)
* `03_Mathematical_Formulation.docx` (Sec. 4–8 — Digital Twin state, Trust,
  Criticality, TAHS, FSDP)
* `09_Aegis-V2X_Dataset_Design_and_Annotation_Guide.pdf.docx` (Ch. 10 —
  Dataset Schema)

Auto-generated interactive docs (OpenAPI/Swagger) are always available at
`/api/docs` when the service is running, and are the source of truth for
exact request/response shapes. This document explains the *why* behind the
schema and cross-references the math.

## 2. Entity-relationship summary

```
Scene 1───N Vehicle
Scene 1───N Frame ───N:1─── Vehicle
Frame 1───1 TrustRecord
Frame 1───1 CriticalityRecord
Frame 1───1 Decision
Experiment  (standalone; references scenes via config JSON)
```

## 3. Table-by-table mapping to the specification

### `scenes` / `vehicles`

Maps to the Dataset Design Guide's Chapter 19 (Metadata Specification).
One `Scene` row per simulation run; `Vehicle` rows are the traffic
participants observed within it.

### `frames`

The core table — one row per synchronized multimodal observation. Maps to
**two** spec sections simultaneously:

| Dataset Schema field (Ch. 10) | `frames` column | Math notation (Sec. 4) |
|---|---|---|
| Sample / Frame_ID | `frame_index` | — |
| Timestamp | `simulation_timestamp`, `wireless_timestamp`, `sync_timestamp` | `t` |
| LiDAR | `lidar_path` (file reference, not inline) | — |
| GPS | `gps_lat`, `gps_lon`, `gps_alt` | part of `M_t` |
| IMU | `imu_data` (JSON) | part of `M_t` |
| Speed | `speed_mps` | part of `M_t` |
| CSI | `csi` (JSON) | `CSI_t` |
| SNR | `snr_db` | `SNR_t` |
| RSSI | `rssi_dbm` | — |
| Path Loss | `path_loss_db` | — |
| Beam Index | `beam_index` | `B_t` |
| Traffic Density / Weather | `traffic_density`, `weather` | part of `E_t` |
| Ground Truth Future CSI/Beam | `gt_future_csi`, `gt_future_beam` | — |
| — | `mobility_state` (JSON) | `M_t` |
| — | `environmental_context` (JSON) | `E_t` |
| — | `prediction_uncertainty` | `U_t` |
| — | `twin_age_ms` | `Age_t` |
| — | `position_x`, `position_y`, `position_z`, `lane_id` | CARLA-native local intersection coords, distinct from GPS (needed for the Digital Twin radar map — too coarse otherwise) |

Synchronization tolerance (Ch. 9: ≤10ms) is enforced in
`app/crud/frame.py::create_frame`, which computes `sync_offset_ms =
|simulation_timestamp - wireless_timestamp| * 1000` and sets
`is_sync_valid` accordingly. Out-of-tolerance frames are logged as
warnings and surfaced on the dashboard's sync-health panel — they are
stored, not rejected.

### `trust_records`

Implements the Calibrated Twin Trust Estimator, `03_Mathematical_Formulation.docx`
Sec. 5:

```
z_t = [w1*e_t, w2*u_t, w3*a_t, w4*q_t]
S_t = sum_i w_i * z_i
T_t = sigmoid(S_t / tau)
```

Implemented in `app/crud/trust.py::compute_trust_probability`. Because
`e_t` (prediction error), `u_t` (uncertainty), and `a_t` (sync age) are
*penalties* while `q_t` (comm quality) is a *reward*, the weighted score is
formed as `w4*q_t - w1*e_t - w2*u_t - w3*a_t` before calibration — this
sign convention is documented inline since the spec's summation notation
doesn't make the reward/penalty distinction explicit. The qualitative
interpretation bands (`very_unreliable` … `highly_reliable`) reproduce the
table in Sec. 5 exactly.

### `criticality_records`

Implements `C_t = sum_i alpha_i * f_i` (Sec. 6) over five features:
relative speed, blockage probability, sync age, channel degradation,
traffic density. Weights default to `0.2` each (uniform prior,
`sum(alpha_i) = 1`); Phase 4/5 may learn non-uniform weights.

### `decisions`

Stores the joint TAHS + FSDP output per frame: `prediction_horizon`
(discretized to `{1,2,3,5,8,10}` per Sec. 7) and `fsdp_action` (one of the
six actions in Sec. 8). `policy_source` distinguishes Phase 3's
placeholder policy (`"synthetic"`) from Phase 5's offline-optimized policy
table (`"fsdp_table"`) — **the schema does not change between phases**,
only who writes to it.

### `experiments`

Tracks named, config-versioned research runs for Phase 6/7 evaluation and
reproducibility (latency, sync overhead, energy, reliability — the
objective function terms from Sec. 9).

## 4. REST API summary

All endpoints are versioned under `/api/v1`. Full interactive
documentation: `/api/docs` (Swagger) or `/api/redoc`.

| Resource | Endpoints |
|---|---|
| Health | `GET /health` |
| Scenes | `POST /scenes`, `GET /scenes`, `GET /scenes/{id}`, `POST /scenes/{id}/vehicles`, `GET /scenes/{id}/vehicles` |
| Frames | `POST /frames`, `POST /frames/bulk`, `GET /frames`, `GET /frames/{id}`, `GET /frames/stats/unsynchronized-count`, `GET /frames/scene/{scene_id}/latest-per-vehicle` |
| Trust | `POST /trust`, `GET /trust`, `GET /trust/frame/{frame_id}` |
| Criticality | `POST /criticality`, `GET /criticality`, `GET /criticality/frame/{frame_id}` |
| Decisions | `POST /decisions`, `GET /decisions`, `GET /decisions/frame/{frame_id}`, `GET /decisions/stats/action-distribution` |
| Experiments | `POST /experiments`, `GET /experiments`, `GET /experiments/{id}`, `PATCH /experiments/{id}` |
| Synthetic data (dev/demo only) | `POST /synthetic/scenes` |

**Ingestion order:** a `Frame` must exist before its `TrustRecord`,
`CriticalityRecord`, or `Decision` can be created (each is validated
against the parent frame; duplicate records per frame return `409`).

**Route ordering:** literal-path routes (`/frames/stats/...`,
`/frames/scene/.../latest-per-vehicle`) are registered before the
parameterized `/frames/{frame_id}` route — reversing this order would
shadow the literal routes, since Starlette matches in declaration order.

**Bulk ingestion:** `POST /frames/bulk` accepts `{"frames": [...]}`
(1-2000 `FrameCreate` items) and inserts the whole batch in a single
transaction (`app/crud/frame.py::create_frames_bulk`) instead of one round
trip per frame -- added 2026-08-25 to close the gap flagged in
`claude/project_status.md`'s "Open architecture questions" item 2, since
Phase 2's target scale (10,000-20,000 frames across 100-150 scenes) made
one-row-per-call ingestion impractical. It shares the exact same
sync-tolerance logic as `POST /frames` (both call `_build_frame`), so
out-of-tolerance frames within a batch are stored and flagged, never
rejected. Registered as a literal route (`/frames/bulk`) before the
parameterized `/frames/{frame_id}` route, per the same route-ordering rule.

**Authentication:** write endpoints (`POST`/`PATCH` -- every one listed in
the table above except the `GET`s) are gated behind an optional
`X-API-Key` header, added 2026-08-25 (`app/core/security.py::require_api_key`).
`settings.api_key` is unset by default, in which case the gate is a no-op
and writes stay unauthenticated, matching the backend's original
local-dev-only scope. Setting the `API_KEY` environment variable (see
`backend/.env.example`) turns the gate on for every write endpoint at
once; requests missing the header or sending the wrong value get `401`.
`GET` endpoints are never gated. This was added because the backend, once
exposed beyond localhost via the VS Code Dev Tunnel used for the Lovable
dashboard preview, had no way to stop an arbitrary caller with the tunnel
URL from writing data -- see `claude/project_status.md`'s "Open
architecture questions" item 3.

## 5. Monitoring

Prometheus scrapes `/metrics` (via `prometheus-fastapi-instrumentator` for
HTTP metrics, plus five custom domain metrics in `app/core/metrics.py`):

* `aegis_trust_score` (histogram) — `T_t` distribution
* `aegis_criticality_score` (histogram) — `C_t` distribution
* `aegis_prediction_horizon_latest` (gauge) — most recent `H_t`
* `aegis_fsdp_decisions_total{action}` (counter) — FSDP action counts
* `aegis_ingested_frames_total{source}` (counter) — ingestion throughput

All five are recorded from inside the CRUD layer (`app/crud/*.py`), not
the API handlers, so any code path that writes data — including the
synthetic data generator — contributes to the metrics.

Grafana is pre-provisioned (`configs/monitoring/grafana/`) with a starter
"Aegis-V2X Backend Overview" dashboard covering all five plus HTTP
request rate/latency.

## 6. Dashboard — React SPA

The backend hosts a Vite + React 19 + TypeScript + Tailwind v4 single-page
app at `/dashboard/`, built from `dashboard/` and served as static files
(`dashboard/dist`) through a custom `SPAStaticFiles` class
(`app/main.py`) that falls back to `index.html` for client-side routes.

Eight pages, all wired to live backend data via `@tanstack/react-query`
(polling every 3-8s, no mocking):

1. **Overview** — scene/decision snapshot
2. **Digital Twin** — SVG radar map (from `position_x/y/lane_id`) + vehicle inspector
3. **V2X Network** — channel state, SNR/RSSI history, beam allocation
4. **TwinTrust-AP** — trust/criticality/FSDP decision pipeline
5. **Simulation** — scene config, vehicle roster, synthetic-scene generator UI
6. **Experiments** — registry (honest empty state until Phase 6/7)
7. **Analytics** — aggregate trust/criticality/horizon distributions
8. **System Health** — API latency, subsystem status, sync health, live activity feed

**Design-integrity rule** (established during the original Phase 3
dashboard rebuild, preserved through this full rebuild): never show a
subsystem as "online" or fabricate a data point that doesn't exist.
`useSystemStatus.ts` explicitly reports CARLA, Sionna RT, and the AI
models as `not_built` — those are Phase 2/4/5 deliverables, not
implemented in this codebase. Trend indicators show "—" rather than a
fabricated delta until real historical data exists. Empty states say so
plainly (e.g. the Experiments page before Phase 6/7 populates it).

## 7. Verification performed (2026-08-15 rebuild)

* Migration (`alembic upgrade head`) applied cleanly against a real local
  PostgreSQL 16 instance — all 8 tables created with correct FKs/indexes,
  autogenerated and reviewed (not hand-written).
* `pytest`: 38/38 passing (health; scenes/vehicles CRUD; frame ingestion
  and sync-tolerance logic incl. a regression test for the route-ordering
  bug; trust/criticality calibration math, both as pure unit tests and via
  the API; decision validation incl. rejecting invalid horizons/actions
  and duplicate decisions; experiment lifecycle; synthetic data generation
  and full schema conformance).
* `ruff check .`: clean.
* `tsc -b && vite build`: clean, production bundle builds successfully.
* Playwright sweep across all 8 dashboard routes against the live backend
  (seeded scene with 5 vehicles × 200 frames): zero console errors, zero
  page errors, screenshots reviewed for visual correctness.
* Manual exercise of every endpoint via `curl` (synthetic scene
  generation, sync-health stats, latest-per-vehicle, action distribution).

### Verification performed (2026-08-25 addendum: bulk ingestion + API-key auth)

* `backend/tests/test_frames_bulk.py` (5 tests) and
  `backend/tests/test_api_key_auth.py` (6 tests) added, covering: bulk
  insert correctness, out-of-sync frames flagged not rejected within a
  batch, empty-list and >2000-item rejection (422), the `/frames/bulk`
  route-ordering regression, unauthenticated writes staying unauthenticated
  by default, missing/wrong/correct `X-API-Key` handling once `API_KEY` is
  set, and `GET` endpoints staying open either way.
* `pytest`: **49/49 passing** (the original 38 plus these 11) -- run for
  real, not just syntax-checked, against a genuine local PostgreSQL 16.15
  instance (a portable build stood up for this verification pass, since
  the previous session's device-bridge sandbox had no way to run Postgres
  at all). `ruff check .`: clean.
* Anyone continuing this work on Logapriya's own machine should still run
  `pytest` once against her regular local Postgres/`.venv` before treating
  this as fully closed on that environment specifically -- this pass
  confirms the code is correct, not that her particular local setup is
  configured to match.

### Bugs found and fixed during this rebuild

* **SPA fallback routing didn't work at all.** The first version of
  `SPAStaticFiles.get_response` checked `response.status_code == 404` on
  the return value of `super().get_response()` — but Starlette's
  `StaticFiles.get_response` doesn't return a 404 response for a missing
  path, it *raises* `HTTPException(404)`. The second attempt caught
  `fastapi.HTTPException`, which still didn't work because
  `fastapi.HTTPException` is a *subclass* of
  `starlette.exceptions.HTTPException`, and Starlette raises the base
  class directly — catching the subclass doesn't catch instances of its
  parent. Fixed by catching `starlette.exceptions.HTTPException`
  specifically. Verified via `curl` against all 8 dashboard routes before
  and after the fix.
* **Synthetic vehicle motion made the Digital Twin radar map render
  empty for realistic demo parameters.** The motion model let vehicles
  travel unbounded distance (`speed * frame_index * dt`), so with the
  default demo parameters (200 frames at 8-14 m/s) every vehicle's
  *latest* frame — which is what the radar map and vehicle inspector
  show — sat hundreds of meters past the 75m visible range, regardless of
  how much data existed. Fixed by clamping travel distance to
  `_INTERSECTION_HALF_EXTENT_M` so vehicles converge on and stop at the
  intersection center, matching the "4-way intersection convergence"
  framing the motion model was already documented as producing. Verified
  visually via Playwright screenshot before/after.

## 8. What happened to the original Phase 3 delivery

The original Phase 3 (`Aegis-V2X/` working folder, including the backend,
the React dashboard, and the interim single-file dashboard that preceded
it) was built entirely in a previous session's ephemeral cloud sandbox and
was never pushed to a durable git remote — a risk explicitly flagged in
`claude/project_status.md` at the time ("The backend repo is not yet in
version control / pushed anywhere durable — flag to the team before the
session workspace is reclaimed"). That flag went unaddressed and the
folder was subsequently lost.

This rebuild reconstructs Phase 3 **from the detailed documentation that
survived in the Claude Project** (`claude/project_status.md`,
`claude/phase3_backend_api_documentation.md`,
`claude/phase3_dashboard_rebuild_status.md`) — schema, API contracts,
calibration formulas, dashboard page list, and known bugs were all
specified precisely enough to rebuild faithfully rather than
reinterpreting from scratch. No architectural decisions changed; this is
a like-for-like rebuild, not a redesign.

**This must not happen a third time.** The rebuilt project is delivered as
a zip via this conversation; the team should `git init` and push to a
real remote (GitHub/GitLab) immediately, independent of any Claude
session's storage.
