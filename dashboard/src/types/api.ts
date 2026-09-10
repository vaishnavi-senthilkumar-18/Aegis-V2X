/**
 * TypeScript mirrors of the backend's Pydantic response schemas
 * (`backend/app/schemas/*.py`). Kept hand-written and deliberately close
 * to the Python field names so a reader can cross-reference the two
 * without a mental mapping step.
 */

export interface Scene {
  id: string;
  scene_code: string;
  map_name: string;
  weather_preset: string;
  num_vehicles_target: number;
  description: string | null;
  created_at: string;
}

export interface Vehicle {
  id: string;
  scene_id: string;
  vehicle_code: string;
  vehicle_type: string;
  is_ego: boolean;
  created_at: string;
}

export interface Frame {
  id: string;
  scene_id: string;
  vehicle_id: string;
  frame_index: number;
  simulation_timestamp: number;
  wireless_timestamp: number;
  sync_timestamp: number;
  sync_offset_ms: number;
  is_sync_valid: boolean;
  lidar_path: string | null;
  gps_lat: number | null;
  gps_lon: number | null;
  gps_alt: number | null;
  position_x: number | null;
  position_y: number | null;
  position_z: number | null;
  lane_id: number | null;
  imu_data: Record<string, number> | null;
  speed_mps: number | null;
  csi: { amplitude: number[]; phase: number[] } | null;
  snr_db: number | null;
  rssi_dbm: number | null;
  path_loss_db: number | null;
  beam_index: number | null;
  traffic_density: number | null;
  weather: string | null;
  mobility_state: Record<string, unknown> | null;
  environmental_context: Record<string, unknown> | null;
  prediction_uncertainty: number | null;
  twin_age_ms: number | null;
  gt_future_csi: Record<string, unknown> | null;
  gt_future_beam: number | null;
  source: "synthetic" | "carla_sionna" | string;
  created_at: string;
}

export interface LatestVehicleFrame extends Frame {
  vehicle_code: string;
}

export interface UnsyncedCountResponse {
  total_frames: number;
  unsynchronized_frames: number;
  unsynchronized_ratio: number;
}

export type TrustInterpretation =
  | "very_unreliable"
  | "unreliable"
  | "moderate"
  | "reliable"
  | "highly_reliable";

export interface TrustRecord {
  id: string;
  frame_id: string;
  prediction_error: number;
  prediction_uncertainty: number;
  sync_age_penalty: number;
  comm_quality: number;
  raw_score: number;
  trust_probability: number;
  interpretation: TrustInterpretation;
  created_at: string;
}

export interface CriticalityRecord {
  id: string;
  frame_id: string;
  relative_speed_score: number;
  blockage_probability_score: number;
  sync_age_score: number;
  channel_degradation_score: number;
  traffic_density_score: number;
  criticality_score: number;
  created_at: string;
}

export type FsdpAction =
  | "maintain_beam"
  | "reselect_beam"
  | "trigger_resync"
  | "downgrade_mode"
  | "upgrade_mode"
  | "handover";

export interface Decision {
  id: string;
  frame_id: string;
  prediction_horizon: number;
  fsdp_action: FsdpAction;
  trust_probability_used: number;
  criticality_score_used: number;
  policy_source: "synthetic" | "fsdp_table" | string;
  rationale: string | null;
  created_at: string;
}

export interface ActionDistributionEntry {
  fsdp_action: string;
  count: number;
}

export interface ActionDistributionResponse {
  total_decisions: number;
  distribution: ActionDistributionEntry[];
}

export type ExperimentStatus = "planned" | "running" | "completed" | "failed" | "archived";

export interface Experiment {
  id: string;
  name: string;
  description: string | null;
  config: Record<string, unknown> | null;
  status: ExperimentStatus;
  latency_ms: number | null;
  sync_overhead_ms: number | null;
  energy_j: number | null;
  reliability_score: number | null;
  created_at: string;
  updated_at: string;
}

export interface HealthResponse {
  status: string;
  service: string;
}
