/**
 * Renders the real CARLA onboard-camera feed for a user-selected "ego"
 * vehicle, synchronized to the same frame_index the Digital Twin
 * playback is currently showing. The person can switch which vehicle's
 * camera is shown via a dropdown, live, with no code changes -- every
 * vehicle in the scene was pre-converted to PNGs (see
 * convert_all_ego_candidates.py), organized as
 * public/camera_frames/<vehicle_id>/frame_<index>.png.
 *
 * Design-integrity note: this is real CARLA-rendered simulation output
 * (a forward-facing onboard camera), NOT a real photograph. Labeled
 * "Simulated Roadview" rather than "Live Roadview" for that reason.
 *
 * Preloading: at 10Hz playback, swapping <img src> on every tick without
 * preloading causes a visible flash/flicker each time the browser has to
 * fetch a not-yet-cached image. To avoid this, every frame for the
 * CURRENTLY SELECTED ego vehicle is preloaded into the browser's image
 * cache (via `new Image()`) whenever the vehicle or frame list changes,
 * BEFORE playback needs them. This does not fabricate anything -- it
 * only pre-fetches real files that already exist, so they're ready
 * instantly when their real tick comes up during playback.
 */

import { useEffect, useRef } from "react";

interface Props {
  frameIndex: number | null;
  egoVehicleId: string;
  availableVehicleIds: string[];
  allFrameIndices: number[];
  onChangeEgoVehicle: (vehicleId: string) => void;
}

export function PhysicalLayer({
  frameIndex,
  egoVehicleId,
  availableVehicleIds,
  allFrameIndices,
  onChangeEgoVehicle,
}: Props) {
  const preloadedRef = useRef<Set<string>>(new Set());

  const frameUrl = (vehicleId: string, idx: number) =>
    `${import.meta.env.BASE_URL}camera_frames/${vehicleId}/frame_${idx}.png`;

  // Preload every real frame for the current ego vehicle whenever it
  // (or the available frame list) changes, so scrubbing/playing through
  // them later hits the browser cache instantly instead of flashing on
  // each fetch.
  useEffect(() => {
    for (const idx of allFrameIndices) {
      const url = frameUrl(egoVehicleId, idx);
      if (preloadedRef.current.has(url)) continue;
      const img = new window.Image();
      img.src = url;
      preloadedRef.current.add(url);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [egoVehicleId, allFrameIndices]);

  const imageSrc = frameIndex != null ? frameUrl(egoVehicleId, frameIndex) : null;

  return (
    <div className="physical-layer">
      <div className="physical-layer__header">
        <span className="physical-layer__title">PHYSICAL LAYER</span>
        <span className="physical-layer__subtitle">SIMULATED ROADVIEW</span>
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
            src={imageSrc}
            alt={`Onboard camera, vehicle ${egoVehicleId}, frame ${frameIndex}`}
            className="physical-layer__image"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div className="physical-layer__empty">No camera frame for this tick</div>
        )}
      </div>

      <div className="physical-layer__footer">
        <span>Vehicle {egoVehicleId}</span>
        <span>{frameIndex != null ? `Frame ${frameIndex}` : "No frame"}</span>
      </div>
    </div>
  );
}
