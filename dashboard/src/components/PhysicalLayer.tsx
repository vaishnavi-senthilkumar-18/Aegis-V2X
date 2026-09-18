/**
 * Renders the real CARLA onboard-camera feed for the selected ego vehicle.
 * Playback follows the same DB frame/tick timeline as the Digital Twin.
 *
 * If the exact DB frame is not available for the selected vehicle camera,
 * the nearest recorded real camera frame is displayed.
 */

import { useEffect, useMemo, useRef, useState } from "react";

interface Props {
  frameIndex: number | null;
  egoVehicleId: string;
  availableVehicleIds: string[];
  onChangeEgoVehicle: (vehicleId: string) => void;
  /** Active scene id. Selects which camera source PhysicalLayer resolves
   * against -- see EXTERNAL_CAMERA_SCENES below. Scene 1/2 (or any scene
   * not in that set) keep using the original /camera_frames/ static
   * route unchanged. */
  sceneId?: string;
}

type CameraFrameManifest = Record<string, number[]>;

/**
 * Scenes whose camera PNGs are served directly from their existing
 * extracted location on C:\AegisTemp (via the Checkpoint-1 Vite dev-server
 * middleware, vite-external-camera-plugin.ts) instead of from
 * dashboard/public/camera_frames/. Deliberately a small explicit set, not
 * derived from vehicle id or any other heuristic -- resolution is by
 * sceneId only, per Checkpoint 2's requirement.
 *
 * Dev-only: the external route this maps to only exists on the Vite dev
 * server (see that plugin's `apply: "serve"`), not in a production build.
 */
const EXTERNAL_CAMERA_SCENES = new Set<string>([
  "cyber_urban_night_multievent_safety_Scene00",
  "rainy_japan_mount_fuji_cherry_blossom_Scene00",
  "snowy_mountain_bluehour_fog_Scene00",
]);

/**
 * Returns the latest REAL camera frame that is <= requestedFrame -- i.e.
 * "hold the last real capture until a newer real capture exists", never
 * a future frame merely because it happens to be numerically closer.
 *
 * Exception: before the scene's very first real capture there is no past
 * frame to hold, so this falls back to the earliest available frame --
 * a one-time bootstrap, not a "look ahead" during playback.
 */
function findLatestFrameAtOrBefore(
  requestedFrame: number,
  availableFrames: number[]
): number | null {
  if (availableFrames.length === 0) return null;

  let left = 0;
  let right = availableFrames.length - 1;
  let result = -1;

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    const value = availableFrames[mid];

    if (value === requestedFrame) {
      return value;
    }

    if (value < requestedFrame) {
      result = mid;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  // No real capture at or before requestedFrame yet -- bootstrap on the
  // earliest one rather than showing nothing.
  return result >= 0 ? availableFrames[result] : availableFrames[0];
}

export function PhysicalLayer({
  frameIndex,
  egoVehicleId,
  availableVehicleIds,
  onChangeEgoVehicle,
  sceneId,
}: Props) {
  const preloadedRef = useRef<Set<string>>(new Set());
  const [manifest, setManifest] = useState<CameraFrameManifest | null>(null);

  // Resolver: sceneId (not vehicle id) picks the source. Scene 1/2 (and
  // any future scene not explicitly listed in EXTERNAL_CAMERA_SCENES)
  // keep the exact original static-route URL, unchanged. Scene 3/4/5 use
  // the dev-only external route from Checkpoint 1 -- no /camera_frames/
  // fallback for these, per Checkpoint 2's "no cross-scene fallback"
  // requirement: if the external route has no matching file, that image
  // request simply 404s, it does not resolve to a Scene 1/2 asset.
  const frameUrl = (vehicleId: string, idx: number) =>
    sceneId && EXTERNAL_CAMERA_SCENES.has(sceneId)
      ? `/camera_frames_ext/${sceneId}/${vehicleId}/${idx}.png`
      : `${import.meta.env.BASE_URL}camera_frames/${vehicleId}/frame_${idx}.png`;

  // Load the tiny camera availability manifest once.
  useEffect(() => {
    let cancelled = false;

    fetch(`${import.meta.env.BASE_URL}camera_frames_manifest.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Manifest request failed: ${response.status}`);
        }
        return response.json() as Promise<CameraFrameManifest>;
      })
      .then((data) => {
        if (!cancelled) {
          setManifest(data);
        }
      })
      .catch((error) => {
        console.warn("Camera frame manifest unavailable:", error);
        if (!cancelled) {
          setManifest({});
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const availableCameraFrames = useMemo(() => {
    return manifest?.[egoVehicleId] ?? [];
  }, [manifest, egoVehicleId]);

  const resolvedCameraFrame = useMemo(() => {
    if (frameIndex == null) return null;

    // While the manifest is loading, don't guess.
    if (manifest === null) return null;

    // If this vehicle has a manifest, hold the latest REAL camera frame
    // captured at or before the current simulation frame -- never a future
    // one, even if it is numerically closer.
    if (availableCameraFrames.length > 0) {
      return findLatestFrameAtOrBefore(frameIndex, availableCameraFrames);
    }

    // Fallback: attempt the exact DB frame if no manifest entry exists.
    return frameIndex;
  }, [frameIndex, manifest, availableCameraFrames]);

  // Preload only a small window around the RESOLVED camera frame.
  // Never preload the entire 4+ GB camera dataset.
  useEffect(() => {
    if (
      resolvedCameraFrame == null ||
      availableCameraFrames.length === 0
    ) {
      return;
    }

    const currentPosition = availableCameraFrames.indexOf(
      resolvedCameraFrame
    );

    const start = Math.max(0, currentPosition - 2);
    const end = Math.min(
      availableCameraFrames.length,
      currentPosition + 8
    );

    for (let i = start; i < end; i++) {
      const idx = availableCameraFrames[i];
      const url = frameUrl(egoVehicleId, idx);

      if (preloadedRef.current.has(url)) continue;

      const img = new window.Image();
      img.src = url;
      preloadedRef.current.add(url);
    }
  }, [egoVehicleId, resolvedCameraFrame, availableCameraFrames]);

  const imageSrc =
    resolvedCameraFrame != null
      ? frameUrl(egoVehicleId, resolvedCameraFrame)
      : null;

  const [loadedImageSrc, setLoadedImageSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!imageSrc) {
      setLoadedImageSrc(null);
      return;
    }

    const img = new window.Image();

    img.onload = () => {
      setLoadedImageSrc(imageSrc);
    };

    img.onerror = () => {
      console.warn(`Camera frame unavailable: ${imageSrc}`);
    };

    img.src = imageSrc;
  }, [imageSrc]);

  const isExactFrame =
    frameIndex != null &&
    resolvedCameraFrame != null &&
    frameIndex === resolvedCameraFrame;

  return (
    <div className="physical-layer">
      <div className="physical-layer__header">
        <span className="physical-layer__title">PHYSICAL LAYER</span>
        <span className="physical-layer__subtitle">
          SIMULATED ROADVIEW
        </span>

        <select
          value={egoVehicleId}
          onChange={(e) => onChangeEgoVehicle(e.target.value)}
          className="physical-layer__vehicle-select"
        >
          {availableVehicleIds.map((vid) => (
            <option key={vid} value={vid}>
              CAM-{vid}
            </option>
          ))}
        </select>
      </div>

      <div className="physical-layer__viewport">
        {loadedImageSrc ? (
          <img
            src={loadedImageSrc}
            alt={`Onboard camera, vehicle ${egoVehicleId}, camera frame ${resolvedCameraFrame}`}
            className="physical-layer__image"
          />
        ) : (
          <div className="physical-layer__empty">
            {frameIndex != null
              ? "Loading camera frame..."
              : "No camera frame for this tick"}
          </div>
        )}
      </div>

      <div className="physical-layer__footer">
        <span>Vehicle {egoVehicleId}</span>

        <span>
          {frameIndex != null && resolvedCameraFrame != null
            ? isExactFrame
              ? `Frame ${frameIndex}`
              : `DB ${frameIndex} • Camera ${resolvedCameraFrame}`
            : "No frame"}
        </span>
      </div>
    </div>
  );
}

