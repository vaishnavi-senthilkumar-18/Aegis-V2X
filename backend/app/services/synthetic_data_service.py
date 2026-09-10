"""Synthetic data generator — stands in for the real CARLA + Sionna RT pipeline.

Phase 2 (Simulation & Dataset, Haridharani) has not started yet, so Phase 3
cannot wait on real multimodal data to exercise the schema, API, and
dashboard end-to-end. This generator produces a full synthetic scene
through the *exact same* CRUD functions the real ingestion path would use
— meaning every metric, sync-validity check, and downstream trust/
criticality/decision computation runs identically regardless of data
source. Every row this generator writes is tagged `source="synthetic"` so
it is trivially filterable/purgeable once Phase 2 delivers real data, and
no other code anywhere needs to change when that happens.

Motion model: rather than random per-frame positions (which produced a
meaningless radar map during initial Phase 3 development), each vehicle is
assigned a fixed approach direction (N/S/E/W), a lane offset, and a
constant speed, then walks toward the intersection center with position
integrated frame-by-frame — physically plausible enough for the Digital
Twin radar map to show real 4-way-intersection convergence.
"""

from __future__ import annotations

import math
import random

from sqlalchemy.orm import Session

from app.crud import criticality as criticality_crud
from app.crud import decision as decision_crud
from app.crud import frame as frame_crud
from app.crud import scene as scene_crud
from app.crud import trust as trust_crud
from app.models.decision import VALID_FSDP_ACTIONS, VALID_PREDICTION_HORIZONS
from app.schemas.criticality import CriticalityRecordCreate
from app.schemas.decision import DecisionCreate
from app.schemas.frame import FrameCreate
from app.schemas.scene import SceneCreate, VehicleCreate
from app.schemas.synthetic import SyntheticSceneRequest, SyntheticSceneResponse
from app.schemas.trust import TrustRecordCreate

#: Four cardinal approach directions for a 4-way intersection, as unit
#: vectors pointing FROM the approach edge TOWARD the intersection center.
_APPROACH_DIRECTIONS: list[tuple[float, float]] = [
    (0.0, -1.0),  # from North, heading South
    (0.0, 1.0),  # from South, heading North
    (-1.0, 0.0),  # from East, heading West
    (1.0, 0.0),  # from West, heading East
]

_INTERSECTION_HALF_EXTENT_M = 60.0  # vehicles start ~60m from the center
_FRAME_DT_S = 0.1  # 10 Hz synthetic sample rate


def _vehicle_motion_params(index: int) -> dict:
    """Deterministic-ish per-vehicle motion parameters (direction, lane, speed)."""
    rng = random.Random(index * 7919 + 17)  # stable seed per vehicle index
    dx, dy = _APPROACH_DIRECTIONS[index % len(_APPROACH_DIRECTIONS)]
    lane_offset = rng.choice([-3.5, 3.5])  # meters, one lane either side of centerline
    speed_mps = rng.uniform(8.0, 14.0)  # ~29-50 km/h urban approach speed
    lane_id = (index % len(_APPROACH_DIRECTIONS)) * 2 + (0 if lane_offset < 0 else 1)
    return {
        "dx": dx,
        "dy": dy,
        "lane_offset": lane_offset,
        "speed_mps": speed_mps,
        "lane_id": lane_id,
    }


def _position_at_frame(params: dict, frame_index: int) -> tuple[float, float, float]:
    """Integrate position for a vehicle at a given frame, converging on the intersection.

    Distance traveled is clamped to `_INTERSECTION_HALF_EXTENT_M` (the
    vehicle comes to rest at the intersection center) rather than growing
    unbounded with `frame_index`. Without this, a scene generated with a
    large `num_frames` would have every vehicle's *latest* frame sitting
    hundreds of meters past the intersection — off the Digital Twin radar
    map's visible range entirely, so the map would render empty regardless
    of how much data existed. Found during Phase 3 dashboard verification.
    """
    dx, dy = params["dx"], params["dy"]
    raw_distance = params["speed_mps"] * frame_index * _FRAME_DT_S
    distance_traveled = min(raw_distance, _INTERSECTION_HALF_EXTENT_M)
    start_x = -dx * _INTERSECTION_HALF_EXTENT_M
    start_y = -dy * _INTERSECTION_HALF_EXTENT_M
    # Perpendicular lane offset so vehicles don't all sit exactly on the centerline.
    perp_x, perp_y = -dy, dx
    x = start_x + dx * distance_traveled + perp_x * params["lane_offset"]
    y = start_y + dy * distance_traveled + perp_y * params["lane_offset"]
    return x, y, 0.0


def _synthetic_channel_state(rng: random.Random, distance_to_center: float) -> dict:
    """Generate physically-plausible-ish CSI/SNR/RSSI that degrade with distance."""
    path_loss_db = 40.0 + 20.0 * math.log10(max(distance_to_center, 1.0))
    snr_db = max(0.0, 35.0 - path_loss_db * 0.3 + rng.uniform(-2.0, 2.0))
    rssi_dbm = -30.0 - path_loss_db * 0.4 + rng.uniform(-2.0, 2.0)
    csi = {"amplitude": [round(rng.uniform(0.1, 1.0), 4) for _ in range(8)],
           "phase": [round(rng.uniform(-math.pi, math.pi), 4) for _ in range(8)]}
    beam_index = rng.randint(0, 63)
    return {
        "csi": csi,
        "snr_db": snr_db,
        "rssi_dbm": rssi_dbm,
        "path_loss_db": path_loss_db,
        "beam_index": beam_index,
    }


def generate_synthetic_scene(
    db: Session, payload: SyntheticSceneRequest
) -> SyntheticSceneResponse:
    """Generate a full synthetic scene end-to-end and persist it via the real CRUD layer.

    For each of `num_vehicles` vehicles, generates `num_frames` frames of
    physically plausible 4-way-intersection convergence, then a
    TrustRecord + CriticalityRecord + Decision chain per frame — using the
    same `app.crud` functions (and therefore the same metrics, sync
    validation, and calibration math) the real ingestion path uses.
    """
    rng = random.Random(hash(payload.scene_code) & 0xFFFFFFFF)

    scene = scene_crud.create_scene(
        db,
        SceneCreate(
            scene_code=payload.scene_code,
            map_name=payload.map_name,
            weather_preset=payload.weather_preset,
            num_vehicles_target=payload.num_vehicles,
            description="Synthetic scene generated as a Phase 2 stand-in (Phase 3).",
        ),
    )

    vehicles = []
    for i in range(payload.num_vehicles):
        vehicle = scene_crud.add_vehicle(
            db,
            scene.id,
            VehicleCreate(vehicle_code=f"Vehicle{i:02d}", vehicle_type="car", is_ego=(i == 0)),
        )
        vehicles.append(vehicle)

    frames_created = trust_created = criticality_created = decisions_created = 0

    for i, vehicle in enumerate(vehicles):
        params = _vehicle_motion_params(i)
        for frame_index in range(payload.num_frames):
            x, y, z = _position_at_frame(params, frame_index)
            distance_to_center = math.hypot(x, y)
            channel = _synthetic_channel_state(rng, distance_to_center)

            sim_ts = frame_index * _FRAME_DT_S
            # Wireless clock is nearly aligned with sim clock; occasionally drifts
            # beyond the 10ms tolerance to exercise the sync-health dashboard panel.
            jitter_s = (
                rng.uniform(-0.003, 0.003) if rng.random() > 0.05 else rng.uniform(0.012, 0.03)
            )
            wireless_ts = sim_ts + jitter_s

            frame = frame_crud.create_frame(
                db,
                FrameCreate(
                    scene_id=scene.id,
                    vehicle_id=vehicle.id,
                    frame_index=frame_index,
                    simulation_timestamp=sim_ts,
                    wireless_timestamp=wireless_ts,
                    position_x=x,
                    position_y=y,
                    position_z=z,
                    lane_id=params["lane_id"],
                    speed_mps=params["speed_mps"] + rng.uniform(-0.5, 0.5),
                    imu_data={
                        "ax": round(rng.uniform(-1, 1), 3),
                        "ay": round(rng.uniform(-1, 1), 3),
                    },
                    csi=channel["csi"],
                    snr_db=channel["snr_db"],
                    rssi_dbm=channel["rssi_dbm"],
                    path_loss_db=channel["path_loss_db"],
                    beam_index=channel["beam_index"],
                    traffic_density=min(1.0, len(vehicles) / 10.0),
                    weather=payload.weather_preset,
                    prediction_uncertainty=round(rng.uniform(0.0, 0.4), 4),
                    twin_age_ms=round(rng.uniform(0.0, 50.0), 2),
                    source="synthetic",
                ),
            )
            frames_created += 1

            # --- Trust ---
            # distance_factor grows as the vehicle nears then crosses the intersection
            # and moves away again, so prediction error/uncertainty (which realistically
            # rise with distance/occlusion) vary meaningfully across a scene instead of
            # producing a degenerate, all-identical decision distribution.
            distance_factor = min(1.0, distance_to_center / _INTERSECTION_HALF_EXTENT_M)
            prediction_error = round(min(1.0, rng.uniform(0.0, 0.4) + 0.3 * distance_factor), 4)
            prediction_uncertainty = round(
                min(1.0, rng.uniform(0.0, 0.4) + 0.25 * distance_factor), 4
            )
            sync_age_penalty = min(1.0, frame.sync_offset_ms / 50.0)
            comm_quality = max(0.0, min(1.0, channel["snr_db"] / 35.0))
            trust_crud.create_trust_record(
                db,
                TrustRecordCreate(
                    frame_id=frame.id,
                    prediction_error=prediction_error,
                    prediction_uncertainty=prediction_uncertainty,
                    sync_age_penalty=sync_age_penalty,
                    comm_quality=comm_quality,
                ),
            )
            trust_created += 1

            # --- Criticality ---
            relative_speed_score = min(1.0, params["speed_mps"] / 20.0)
            blockage_probability_score = round(
                min(1.0, rng.uniform(0.0, 0.3) + 0.4 * distance_factor), 4
            )
            channel_degradation_score = max(0.0, min(1.0, 1.0 - comm_quality))
            traffic_density_score = min(1.0, len(vehicles) / 10.0)
            criticality_crud.create_criticality_record(
                db,
                CriticalityRecordCreate(
                    frame_id=frame.id,
                    relative_speed_score=relative_speed_score,
                    blockage_probability_score=blockage_probability_score,
                    sync_age_score=sync_age_penalty,
                    channel_degradation_score=channel_degradation_score,
                    traffic_density_score=traffic_density_score,
                ),
            )
            criticality_created += 1

            # --- Decision (Phase 3 placeholder heuristic; Phase 5 replaces the policy_source) ---
            trust_record = trust_crud.get_trust_record_by_frame(db, frame.id)
            criticality_record = criticality_crud.get_criticality_record_by_frame(db, frame.id)
            horizon = _heuristic_horizon(trust_record.trust_probability)
            action = _heuristic_action(
                trust_record.trust_probability, criticality_record.criticality_score
            )
            decision_crud.create_decision(
                db,
                DecisionCreate(
                    frame_id=frame.id,
                    prediction_horizon=horizon,
                    fsdp_action=action,
                    trust_probability_used=trust_record.trust_probability,
                    criticality_score_used=criticality_record.criticality_score,
                    policy_source="synthetic",
                    rationale=(
                        f"Placeholder heuristic: T_t={trust_record.trust_probability:.3f}, "
                        f"C_t={criticality_record.criticality_score:.3f}"
                    ),
                ),
            )
            decisions_created += 1

    return SyntheticSceneResponse(
        scene_id=scene.id,
        scene_code=scene.scene_code,
        vehicles_created=len(vehicles),
        frames_created=frames_created,
        trust_records_created=trust_created,
        criticality_records_created=criticality_created,
        decisions_created=decisions_created,
    )


def _heuristic_horizon(trust_probability: float) -> int:
    """Placeholder TAHS heuristic: higher trust -> longer prediction horizon allowed.

    Real TAHS optimization arrives in Phase 5; this exists only so Phase
    3's demo data populates a plausible-looking `prediction_horizon`
    distribution on the dashboard.
    """
    sorted_horizons = sorted(VALID_PREDICTION_HORIZONS)
    index = min(len(sorted_horizons) - 1, int(trust_probability * len(sorted_horizons)))
    return sorted_horizons[index]


def _heuristic_action(trust_probability: float, criticality_score: float) -> str:
    """Placeholder FSDP heuristic policy for Phase 3 demo data only."""
    if trust_probability < 0.4:
        return "trigger_resync"
    if criticality_score > 0.6 and trust_probability < 0.65:
        return "downgrade_mode"
    if criticality_score > 0.5:
        return "reselect_beam"
    if trust_probability > 0.7 and criticality_score < 0.35:
        return "upgrade_mode"
    return "maintain_beam"


assert set(VALID_FSDP_ACTIONS) >= {
    "maintain_beam",
    "reselect_beam",
    "trigger_resync",
    "downgrade_mode",
    "upgrade_mode",
}
