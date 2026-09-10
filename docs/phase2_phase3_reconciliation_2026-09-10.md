# Phase 2 → Phase 3 Reconciliation: Real Data Structure Summary

**Scene inspected:** `straight_road_dense_clear_day_Scene00`
**Inspected by:** Vaishnavi
**Date:** 2026-09-10

## Context

This reconciles Phase 2's real CARLA scene output against Phase 3's backend
schema — a standing open risk flagged since 2026-08-22 in
`claude/project_status.md`: Logapriya's backend was built and tested only
against synthetic, self-generated data, never a real CARLA/Sionna scene.

## 1. Folder/File Layout (real, not spec-assumed)

```
straight_road_scene00_FULL/
└── straight_road_dense_clear_day_Scene00/
    └── frame_XXXXXX/                          (one folder per exported frame)
        ├── geometry_snapshot.json
        └── vehicle{ID}_camera.npz
        └── vehicle{ID}_gps.json
        └── vehicle{ID}_imu.json
        └── vehicle{ID}_lidar.npz
        └── vehicle{ID}_vehicle_speed.json
```

Per-vehicle, per-sensor files — **not** one combined record per frame as the
Dataset Design Guide's schema (Ch. 10) implies.

## 2. GPS field structure (real)

```json
{
  "timestamp": 13.3709183963947,
  "frame": 122387,
  "lat": 0.002597277550805188,
  "lon": 0.00018045771035802462,
  "alt": 1.8815484456717968
}
```

**Issue:** `lat`/`lon` values are far too small to be real GPS degrees
(±90/±180 range expected). These look like raw CARLA local coordinates, not
converted GPS. **Needs confirmation from Vaishnavi/Haridharani** on whether
this is intentional (placeholder) or a units bug before Logapriya builds
ingestion logic around it.

## 3. IMU field structure (real)

```json
{
  "timestamp": 13.460918394383043,
  "frame": 122396,
  "accel": [0.2635, 3.9010, 9.7633],
  "gyro": [0.0115, 0.0133, 0.1628],
  "compass": 0.48884472250938416
}
```

Matches expectations reasonably well — 3-axis accel/gyro, compass heading,
own timestamp/frame.

## 4. Sync offset finding — real risk, confirmed with data

Sampled GPS files across 4 frames in the same scene, comparing folder name
vs. internal `frame` field:

| Folder | Internal frame | Lag (frames) | Lag (ms, @10Hz tick) |
|---|---|---|---|
| frame_121506 | 121506 | 0 | 0 |
| frame_121526 | 121517 | 9 | 90ms |
| frame_121536 | 121527 | 9 | 90ms |
| frame_121546 | 121537 | 9 | 90ms |

**Finding:** GPS is systematically lagging the frame folder it's stored in
by ~90ms in most samples — **9x over the ≤10ms sync tolerance** assumed in
the Dataset Design Guide (Ch. 9) and implemented in
`backend/app/crud/frame.py`'s `is_sync_valid` check.

**Confirmed on a second, independent scene** (`roundabout_sparse_night_Scene00`,
different map/traffic density/weather): the exact same pattern reproduces —
lag is 0 in the first sampled frame, then settles into a consistent 9-frame
(90ms) lag from then on:

| Folder | Internal frame | Lag (frames) | Lag (ms) |
|---|---|---|---|
| frame_127713 | 127713 | 0 | 0 |
| frame_127733 | 127724 | 9 | 90ms |
| frame_127743 | 127734 | 9 | 90ms |
| frame_127753 | 127744 | 9 | 90ms |
| frame_127763 | 127754 | 9 | 90ms |

Two independent scenes showing the identical 0-then-9-frame pattern is
strong evidence this is a **systemic behavior in the CARLA export pipeline
itself** (likely `scene_exporter.py`'s per-sensor write ordering/buffering),
not a scene-specific fluke or random jitter. Worth a root-cause pass on the
exporter code directly, rather than just working around it downstream.

## 5. Missing entirely

**No CSI/SNR/wireless/beam data anywhere in this scene.** This is CARLA-only
output — Sionna RT has not yet been run/integrated into this scene's export.
The `frames` table schema (`csi`, `snr_db`, `path_loss_db`, `beam_index`
columns) has nothing to ingest from this data yet.

## Code-level comparison (added after reviewing Logapriya's branch,
`feature/logapriya-software-engineering`, commit `b86057e`)

Reviewed `backend/app/schemas/frame.py` (`FrameCreate`/`FrameRead`) and
`backend/app/models/frame.py` (`Frame` ORM model) directly against the real
data above. Three concrete mismatches found:

### Mismatch 1 — record shape
`FrameCreate` expects **one combined record** per `(scene_id, vehicle_id,
frame_index)`, with `gps_lat`/`gps_lon`/`gps_alt`, `imu_data` (dict),
`lidar_path` (string reference) all as fields on a single object.

Real output is **five separate per-vehicle files** per frame
(`_gps.json`, `_imu.json`, `_lidar.npz`, `_camera.npz`,
`_vehicle_speed.json`) with no combined record anywhere. Ingestion will
need a merge step (read all 5 files per vehicle per frame, assemble one
`FrameCreate` payload) before this schema can accept real data at all.

### Mismatch 2 — no wireless timestamp exists yet
The `Frame` model's own docstring defines `wireless_timestamp` as "Sionna
RT wireless-side clock time (seconds)," and `sync_offset_ms`/
`is_sync_valid` are both computed from
`|simulation_timestamp - wireless_timestamp|`.

Real GPS/IMU files each have exactly **one** `timestamp` field — there is
no second, wireless-side timestamp anywhere in this scene, because Sionna
RT has not been run for it. **The sync-validation logic as written cannot
be exercised at all with current real data** — it needs a second timestamp
that doesn't exist yet. This is a bigger gap than "the tolerance is too
strict" — right now there's nothing to check tolerance against.

### Mismatch 3 — GPS units unconfirmed
Model docstring says `gps_lat`/`gps_lon` are "part of `M_t`" (mobility
state) with no unit specified. Real values (~0.0026, ~0.00018) are far too
small for GPS degrees. Still needs confirmation from whoever owns
`scene_exporter.py` (Vaishnavi/Haridharani) on intended units before
Logapriya writes any range validation.

## Recommendation for Logapriya

1. Backend schema/ingestion should assume **per-vehicle, per-sensor files**
   as the real input shape, not a single combined frame record — needs a
   merge/assembly step upstream of `FrameCreate`, not just field renaming.
2. `wireless_timestamp`, `sync_offset_ms`, and `is_sync_valid` cannot be
   populated from real Phase 2 data until Sionna RT integration lands.
   Suggest: either make `wireless_timestamp` nullable and skip sync
   validation when absent, or explicitly gate ingestion of CARLA-only
   frames behind a `source="carla_only"` flag distinct from the existing
   `source="carla_sionna"` value the model docstring already anticipates.
3. Once Sionna does land, the ≤10ms tolerance itself will still need
   revisiting — real CARLA-only GPS/IMU pairs already show a ~90ms lag
   pattern (see Sync offset finding above) unrelated to wireless timing,
   worth understanding before layering wireless sync on top.
4. CSI/wireless fields will stay empty until Sionna RT integration lands —
   consistent with point 2, don't validate against them yet.

## Open follow-ups

- Confirm GPS lat/lon units (real degrees vs. raw local coords) with
  Vaishnavi/Haridharani. Both scenes show the same suspiciously small
  values (`straight_road`: ~0.0026/0.00018; `roundabout`: ~-0.0014/0.00005),
  ruling out a one-off data error — this looks like a real units question,
  not noise.
- Root-cause the 0-then-9-frame GPS lag pattern in `scene_exporter.py` —
  confirmed identical across both completed scenes, so this is now a
  pipeline-level finding, not scene-specific.
- ~~Repeat this same inspection against `roundabout_sparse_night_Scene00`~~
  — done; findings above confirmed on both of Phase 2's completed scenes.
