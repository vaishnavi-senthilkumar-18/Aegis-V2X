"""One-off script: ingest a real Phase 2 CARLA scene into the database.

Usage (from the `backend/` directory, with the venv activated):

    python -m scripts.ingest_real_scene

Edit SCENE_DIR, SCENE_CODE, MAP_NAME below to match the scene you're
ingesting. Run once per scene (straight_road, then separately for
roundabout, etc.) -- it always creates a NEW Scene row, so re-running with
the same SCENE_DIR will duplicate data unless you change SCENE_CODE or
clean up first.

What this does, step by step:
    1. Scans the scene directory to discover every distinct vehicle ID
       that appears anywhere across all frame directories (real scenes
       have been observed to add/lose vehicles between frames -- e.g. new
       vehicles entering later, some finishing their route early -- so
       this scans everything up front rather than assuming frame 0's
       roster is the full roster).
    2. Creates one `Scene` row, now with num_vehicles_target set to the
       REAL discovered count (fixed -- previously this was hardcoded to
       0 because Scene creation ran before vehicle discovery).
    3. Creates one `Vehicle` row per discovered vehicle ID.
    4. Runs `assemble_scene_payloads()` (from
       `app.services.carla_ingestion`) to build `FrameCreate` payloads
       from the real GPS/IMU/speed/lidar files.
    5. Inserts them via `create_frames_bulk()` in batches of 2000 (the
       cap enforced by `FrameBulkCreate`), reporting progress as it goes.
"""

from __future__ import annotations

from pathlib import Path

from app.core.database import SessionLocal
from app.crud.frame import create_frames_bulk
from app.crud.scene import add_vehicle, create_scene
from app.schemas.scene import SceneCreate, VehicleCreate
from app.services.carla_ingestion import assemble_scene_payloads, discover_vehicle_ids

# ---------------------------------------------------------------------------
# Edit these three lines per scene you're ingesting.
# NOTE: pointed at a LOCAL copy (C:\AegisTemp\...), not the Google-Drive
# path -- reading ~25,000 small files directly off Drive was far too slow
# (10+ minutes with barely any CPU progress). Copy the scene locally first
# with robocopy, then point this at the local copy.
# ---------------------------------------------------------------------------
SCENE_DIR = Path(
    r"C:\AegisTemp\roundabout_scene00_FULL\roundabout_sparse_night_Scene00"
)
SCENE_CODE = "roundabout_sparse_night_Scene00"
MAP_NAME = "Town04"  # adjust if you know the actual CARLA map used
# ---------------------------------------------------------------------------

BULK_BATCH_SIZE = 2000


def discover_all_vehicle_ids(scene_dir: Path) -> list[str]:
    """Scan every frame directory and return the union of all vehicle IDs seen.

    Deliberately scans the whole scene rather than just the first frame,
    since real scenes can have vehicles entering/leaving over time.
    """
    all_ids: set[str] = set()
    frame_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")]
    print(f"Scanning {len(frame_dirs)} frame directories for vehicle IDs...")
    for i, frame_dir in enumerate(frame_dirs, 1):
        all_ids.update(discover_vehicle_ids(frame_dir))
        if i % 500 == 0:
            print(f"  ...scanned {i}/{len(frame_dirs)} frame directories")
    return sorted(all_ids, key=int)


def main() -> None:
    if not SCENE_DIR.exists():
        raise SystemExit(f"Scene directory not found: {SCENE_DIR}")

    db = SessionLocal()
    try:
        # Step 1: discover vehicle IDs FIRST, so the real count is known
        # before the Scene row is created. (Fixed: previously this ran
        # AFTER create_scene(), leaving num_vehicles_target hardcoded at
        # 0 regardless of how many vehicles the scene actually had.)
        vehicle_id_strs = discover_all_vehicle_ids(SCENE_DIR)
        print(f"Discovered {len(vehicle_id_strs)} distinct vehicles: {vehicle_id_strs[:10]}"
              f"{'...' if len(vehicle_id_strs) > 10 else ''}")

        # Step 2: create the Scene row, now with the real vehicle count.
        print(f"Creating scene '{SCENE_CODE}'...")
        scene = create_scene(
            db,
            SceneCreate(
                scene_code=SCENE_CODE,
                map_name=MAP_NAME,
                weather_preset="ClearNoon",
                num_vehicles_target=len(vehicle_id_strs),
                description=(
                    "Real Phase 2 CARLA scene, ingested via "
                    "scripts/ingest_real_scene.py. source=carla_only "
                    "(no Sionna RT wireless data yet)."
                ),
            ),
        )
        print(f"  Created scene id={scene.id}")

        # Step 3: create Vehicle rows.
        vehicle_id_map: dict[str, object] = {}
        for vid_str in vehicle_id_strs:
            vehicle = add_vehicle(
                db,
                scene.id,
                VehicleCreate(vehicle_code=f"Vehicle{vid_str}", vehicle_type="car", is_ego=False),
            )
            vehicle_id_map[vid_str] = vehicle.id
        print(f"  Created {len(vehicle_id_map)} vehicle rows.")

        # Step 4: assemble FrameCreate payloads from real files.
        print("Assembling frame payloads from real scene files...")
        payloads = assemble_scene_payloads(SCENE_DIR, scene.id, vehicle_id_map)
        print(f"  Assembled {len(payloads)} frame payloads.")

        # Step 5: bulk-insert in batches of BULK_BATCH_SIZE.
        total_inserted = 0
        total_unsynced = 0
        for start in range(0, len(payloads), BULK_BATCH_SIZE):
            batch = payloads[start : start + BULK_BATCH_SIZE]
            inserted_frames = create_frames_bulk(db, batch)
            total_inserted += len(inserted_frames)
            total_unsynced += sum(1 for f in inserted_frames if not f.is_sync_valid)
            print(f"  Inserted batch: {start}-{start + len(batch)} "
                  f"({total_inserted}/{len(payloads)} total so far)")

        print()
        print("Done.")
        print(f"  Scene id: {scene.id}")
        print(f"  Vehicles created: {len(vehicle_id_map)}")
        print(f"  Frames inserted: {total_inserted}")
        print(f"  Frames flagged out-of-sync: {total_unsynced} "
              f"(expected: 0, since source=carla_only sets wireless_timestamp"
              f" == simulation_timestamp)")
    finally:
        db.close()


if __name__ == "__main__":
    main()
