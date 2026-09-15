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
}

type CameraFrameManifest = Record<string, number[]>;

function findNearestFrame(
  requestedFrame: number,
  availableFrames: number[]
): number | null {
  if (availableFrames.length === 0) return null;

  if (availableFrames.includes(requestedFrame)) {
    return requestedFrame;
  }

  let left = 0;
  let right = availableFrames.length - 1;

  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    const value = availableFrames[mid];

    if (value === requestedFrame) {
      return value;
    }

    if (value < requestedFrame) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  const lower = right >= 0 ? availableFrames[right] : null;
  const upper = left < availableFrames.length ? availableFrames[left] : null;

  if (lower == null) return upper;
  if (upper == null) return lower;

  return requestedFrame - lower <= upper - requestedFrame
    ? lower
    : upper;
}

export function PhysicalLayer({
  frameIndex,
  egoVehicleId,
  availableVehicleIds,
  onChangeEgoVehicle,
}: Props) {
  const preloadedRef = useRef<Set<string>>(new Set());
  const [manifest, setManifest] = useState<CameraFrameManifest | null>(null);

  const frameUrl = (vehicleId: string, idx: number) =>
    `${import.meta.env.BASE_URL}camera_frames/${vehicleId}/frame_${idx}.png`;

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

    // If this vehicle has a manifest, use the nearest REAL camera frame.
    if (availableCameraFrames.length > 0) {
      return findNearestFrame(frameIndex, availableCameraFrames);
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
        {imageSrc ? (
          <img
            key={`${egoVehicleId}-${resolvedCameraFrame}`}
            src={imageSrc}
            alt={`Onboard camera, vehicle ${egoVehicleId}, camera frame ${resolvedCameraFrame}`}
            className="physical-layer__image"
            onError={(e) => {
              const img = e.currentTarget;
              img.style.display = "";
              img.alt = `Camera frame unavailable: vehicle ${egoVehicleId}, camera frame ${resolvedCameraFrame}`;
            }}
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

