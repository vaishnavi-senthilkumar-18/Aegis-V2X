from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import json
import os

SRC = Path(r"G:\My Drive\Aegis-V2X\datasets\raw\carla\roundabout_scene00_FULL\roundabout_sparse_night_Scene00")
DST = Path(r"G:\My Drive\Aegis-V2X\dashboard\public\camera_frames")
MANIFEST = DST.parent / "camera_frames_manifest_roundabout.json"

files = sorted(SRC.glob("frame_*/vehicle*_camera.npz"))

def convert(src):
    frame = src.parent.name.replace("frame_", "")
    vehicle = src.name.split("_")[0].replace("vehicle", "")
    out_dir = DST / vehicle
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"frame_{frame}.png"

    if out.exists():
        return "skip"

    with np.load(src) as data:
        arr = data["data"]

    Image.fromarray(arr).save(out, format="PNG", optimize=False)
    return "done"

if __name__ == "__main__":
    workers = max(2, min(8, (os.cpu_count() or 4) - 1))

    print(f"Source files : {len(files)}")
    print(f"Workers      : {workers}")
    print("Resuming existing conversion...")
    print()

    done = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(convert, files, chunksize=8), 1):
            if result == "done":
                done += 1
            else:
                skipped += 1

            if i % 100 == 0 or i == len(files):
                print(
                    f"\rProcessed: {i}/{len(files)} | "
                    f"New: {done} | Existing: {skipped}",
                    end="",
                    flush=True
                )

    print("\n\nBuilding manifest...")

    manifest = {}

    for png in DST.glob("*/*.png"):
        if not png.parent.name.isdigit():
            continue

        vehicle = png.parent.name
        frame = png.stem.replace("frame_", "")

        manifest.setdefault(vehicle, {})[frame] = (
            f"camera_frames/{vehicle}/{png.name}"
        )

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print(f"Manifest: {MANIFEST}")
    print("DONE")
