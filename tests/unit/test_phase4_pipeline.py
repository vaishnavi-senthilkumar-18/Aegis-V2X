"""Phase 4 integration tests -- the BRANCHING architecture the Phase 4
Integration Audit established, not a single serial chain.

REAL:
    Data Loader -> PointPillars -> V2X-ViT
    (real LiDAR from straight_road_dense_clear_day_Scene00, a real shared
    frame with multiple vehicles; validated end-to-end on real data.)

INDEPENDENT WIRELESS BRANCH (SYNTHETIC -- no real CSI/SNR/beam_index
exists anywhere in the Phase 2 dataset):
    synthetic historical wireless sequence -> GRUPredictor -> uncertainty
    -> TwinTrustEstimator's existing uncertainty reduction
    -> compute_trust_from_evidence()

INDEPENDENT CONTEXT BRANCH (SYNTHETIC -- no real evidence for any of the
five Criticality terms exists in DigitalTwinState/the real dataset):
    explicit synthetic evidence -> compute_criticality_from_evidence()

DOWNSTREAM: Trust/Criticality floats are structurally compatible with
Phase 5's BaseTAHS/BaseFSDP (both take plain `trust: float, criticality:
float`), but real evidence to produce them from DigitalTwinState is not
currently available -- see the Phase 4 Integration Audit.

This file does NOT test, imply, or fabricate:
  - a V2X-ViT -> GRU connection (not supported by any project
    specification -- GRU is an independent wireless-track component, see
    ai/gru/base.py's docstring)
  - a successful `estimate(DigitalTwinState)` call for Trust or
    Criticality (both are designed to always raise today)
  - any single real end-to-end Data Loader -> ... -> Phase 5 pipeline
"""

from __future__ import annotations

import numpy as np
import torch

from ai.criticality.model import CriticalityEstimator
from ai.gru.model import GRUPredictor
from ai.perception.data_loader import Phase4DataLoader
from ai.pointpillars.model import MissingPointCloudError, PointPillars
from ai.trust_estimator.model import MissingTrustEvidenceError, TwinTrustEstimator, _reduce_uncertainty
from ai.v2x_vit.model import MissingVehicleFeaturesError, V2XViT
from digital_twin.state import ChannelState, DigitalTwinState, EnvironmentalContext, MobilityState

STRAIGHT_ROAD_SCENE_ID = "6b08be32-e041-4ab4-861c-99090df77ffb"
#: A real shared frame confirmed (Phase 4 audit / V2X-ViT validation) to
#: have 46 vehicles present, 45 with valid real LiDAR and exactly one
#: (Vehicle192) with a known genuinely corrupted .npz file.
SHARED_FRAME_INDEX = 122407


# =============================================================================
# A. REAL: Data Loader -> PointPillars -> V2X-ViT (real LiDAR, real frame)
# =============================================================================


def test_real_data_loader_pointpillars_v2x_vit_pipeline():
    """REAL DATA -- straight_road_dense_clear_day_Scene00, shared frame
    122407. No fabrication: vehicles with missing/corrupt LiDAR are
    skipped using the data loader's own existing behavior
    (lidar_points=None), never substituted."""
    loader = Phase4DataLoader()
    pp = PointPillars()
    v2x = V2XViT()

    samples = loader.load_scene_samples(STRAIGHT_ROAD_SCENE_ID)
    frame_group = [s for s in samples if s.frame_index == SHARED_FRAME_INDEX]
    assert len(frame_group) > 0, "expected real vehicles at the known shared frame"

    features = []
    included_vehicle_codes = []
    skipped_vehicle_codes = []
    for s in frame_group:
        if s.lidar_points is None:
            # Real, known gap (e.g. Vehicle192's corrupted .npz) -- skip,
            # do not fabricate a replacement point cloud.
            skipped_vehicle_codes.append(s.vehicle_code)
            continue
        try:
            feat = pp.extract_features(s.lidar_points)
        except MissingPointCloudError:
            skipped_vehicle_codes.append(s.vehicle_code)
            continue
        features.append(feat)
        included_vehicle_codes.append(s.vehicle_code)

    assert len(included_vehicle_codes) >= 2, (
        f"expected at least 2 valid real vehicles, got {len(included_vehicle_codes)}"
    )
    # The known-corrupt case must be excluded, never fabricated through.
    assert "Vehicle192" in skipped_vehicle_codes or "Vehicle192" not in [s.vehicle_code for s in frame_group]

    for feat in features:
        assert feat.shape == (1, 128, 90, 180)
        assert feat.dtype == torch.float32
        assert torch.isfinite(feat).all()

    fused = v2x.fuse(features)
    assert fused.shape == (1, 256)
    assert torch.isfinite(fused).all()

    # Deterministic repeat, on the same real inputs.
    fused_again = v2x.fuse(features)
    assert torch.equal(fused, fused_again)


def test_real_pipeline_never_fabricates_missing_lidar():
    """The known-corrupt Vehicle192/frame_122416 case must still raise
    MissingPointCloudError rather than silently produce a pseudo-image --
    re-confirmed here as part of the integration path, not just in
    isolation."""
    loader = Phase4DataLoader()
    pp = PointPillars()
    seq = loader.load_full_sequence(STRAIGHT_ROAD_SCENE_ID, "Vehicle192")
    corrupt = next((s for s in seq if s.lidar_points is None), None)
    assert corrupt is not None, "expected the known corrupt-LiDAR sample to still be present"
    try:
        pp.extract_features(corrupt.lidar_points)
        raise AssertionError("expected MissingPointCloudError, got a result instead")
    except MissingPointCloudError:
        pass


# =============================================================================
# B. SYNTHETIC: GRU -> uncertainty -> Trust evidence helper
# =============================================================================


def _synthetic_wireless_sequence(batch: int = 1, seq_len: int = 8, input_dim: int = 4, seed: int = 0) -> torch.Tensor:
    """SYNTHETIC (code-correctness only) -- stands in for a historical
    wireless/channel sequence. No real CSI/SNR/beam_index exists anywhere
    in the Phase 2 dataset (Phase 4 GRU audit), so this is never presented
    as real data."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(batch, seq_len, input_dim)).astype(np.float32)
    return torch.from_numpy(arr)


def test_synthetic_gru_to_trust_contract():
    """SYNTHETIC SOFTWARE-CONTRACT TEST ONLY. The resulting trust value is
    NOT a real-data trust measurement -- prediction_error, comm_quality,
    and sync_age_penalty below are deterministic synthetic fixtures, and
    prediction_uncertainty comes from a GRU run on a synthetic wireless
    sequence, not real telemetry."""
    gru = GRUPredictor(input_dim=4, seed=0)
    trust_est = TwinTrustEstimator()

    historical_sequence = _synthetic_wireless_sequence(batch=1, seq_len=8, input_dim=4)
    prediction = gru.predict(historical_sequence, horizon=5)

    assert prediction.predicted_channel_state.shape == (1, 4)
    assert prediction.prediction_uncertainty.shape == (1, 4)
    assert torch.isfinite(prediction.prediction_uncertainty).all()
    assert bool((prediction.prediction_uncertainty >= 0).all())

    # Existing, already-tested Trust uncertainty reduction accepts the
    # GRU's real tensor output shape directly.
    reduced_uncertainty = _reduce_uncertainty(prediction.prediction_uncertainty)
    assert isinstance(reduced_uncertainty, float)
    assert torch.isfinite(torch.tensor(reduced_uncertainty))

    # SYNTHETIC evidence for the three terms Trust cannot obtain from
    # DigitalTwinState (see Phase 4 Trust input-contract review) --
    # supplied explicitly here purely to validate the formula wiring.
    trust = trust_est.compute_trust_from_evidence(
        prediction_error=0.15,          # SYNTHETIC fixture, not real
        prediction_uncertainty=prediction.prediction_uncertainty,  # real GRU output (on synthetic input)
        sync_age_penalty=0.05,          # SYNTHETIC fixture, not real
        comm_quality=0.7,               # SYNTHETIC fixture, not real
    )
    assert isinstance(trust, float)
    assert 0.0 <= trust <= 1.0
    assert torch.isfinite(torch.tensor(trust))


def test_synthetic_gru_to_trust_contract_is_deterministic():
    """Same synthetic inputs -> identical GRU prediction -> identical
    trust value, end to end."""
    gru = GRUPredictor(input_dim=4, seed=0)
    trust_est = TwinTrustEstimator()
    historical_sequence = _synthetic_wireless_sequence(batch=1, seq_len=8, input_dim=4)

    def run_once() -> float:
        prediction = gru.predict(historical_sequence, horizon=3)
        return trust_est.compute_trust_from_evidence(
            prediction_error=0.15,
            prediction_uncertainty=prediction.prediction_uncertainty,
            sync_age_penalty=0.05,
            comm_quality=0.7,
        )

    assert run_once() == run_once()


# =============================================================================
# C. Criticality evidence helper contract (SYNTHETIC, software-only)
# =============================================================================


def test_synthetic_criticality_evidence_helper_contract():
    """SYNTHETIC SOFTWARE-CONTRACT TEST ONLY. These five evidence values
    are deterministic test fixtures -- not derived from any real
    DigitalTwinState or Phase 2 data (none of the five terms has a real
    source; see the Phase 4 Criticality audit)."""
    crit_est = CriticalityEstimator()
    criticality = crit_est.compute_criticality_from_evidence(
        relative_speed=0.4,
        blockage_probability=0.3,
        sync_age=0.2,
        channel_degradation=0.5,
        traffic_density=0.6,
    )
    assert isinstance(criticality, float)
    assert 0.0 <= criticality <= 1.0


# =============================================================================
# D. Explicit non-connection / architectural-boundary tests
# =============================================================================


def test_v2x_vit_output_is_not_accepted_as_gru_input():
    """V2X-ViT's (1, 256) fused representation is NOT compatible with
    GRUPredictor's (batch, seq_len, input_dim) sequence contract, and
    there is no code path anywhere that feeds one into the other. This
    test confirms the shapes are actually incompatible, not just
    unconnected by convention -- passing V2X-ViT's 2D output directly to
    GRU.predict() must fail the sequence-shape validation."""
    v2x = V2XViT()
    gru = GRUPredictor(input_dim=4, seed=0)

    feat = torch.from_numpy(np.random.default_rng(0).normal(size=(1, 128, 90, 180)).astype(np.float32))
    fused = v2x.fuse([feat])
    assert fused.shape == (1, 256)

    try:
        gru.predict(fused, horizon=1)  # (1, 256) is 2D, GRU requires 3D (batch, seq_len, input_dim)
        raise AssertionError("expected GRU to reject V2X-ViT output, but it was accepted")
    except Exception as exc:
        from ai.gru.model import InvalidSequenceError

        assert isinstance(exc, InvalidSequenceError)


def test_trust_estimate_from_state_still_raises_missing_evidence():
    """Re-confirms, as part of the integration surface, that Trust's
    state-based estimate() still refuses to fabricate prediction_error/
    comm_quality -- the architectural boundary the Phase 4 Integration
    Audit established must not be silently bypassed by integration work."""
    trust_est = TwinTrustEstimator()
    state = DigitalTwinState(
        timestamp=10.0,
        channel=ChannelState(csi=None, snr=20.0, beam_index=3, path_loss=90.0),
        mobility=MobilityState(position=(0.0, 0.0, 0.0), velocity=(15.0, 0.0, 0.0), heading=0.0, relative_speed=5.0),
        environment=EnvironmentalContext(weather="clear_day", traffic_density="sparse", blockage_probability=0.05),
        prediction_uncertainty=0.2,
        sync_age_seconds=0.05,
        metadata={},
    )
    try:
        trust_est.estimate(state)
        raise AssertionError("expected MissingTrustEvidenceError, got a result instead")
    except MissingTrustEvidenceError:
        pass


def test_criticality_estimate_from_state_still_raises_missing_evidence():
    """Same architectural-boundary confirmation for Criticality."""
    crit_est = CriticalityEstimator()
    state = DigitalTwinState(
        timestamp=10.0,
        channel=ChannelState(csi=None, snr=20.0, beam_index=3, path_loss=90.0),
        mobility=MobilityState(position=(0.0, 0.0, 0.0), velocity=(15.0, 0.0, 0.0), heading=0.0, relative_speed=5.0),
        environment=EnvironmentalContext(weather="clear_day", traffic_density="sparse", blockage_probability=0.05),
        prediction_uncertainty=0.2,
        sync_age_seconds=0.05,
        metadata={},
    )
    try:
        crit_est.estimate(state)
        raise AssertionError("expected MissingCriticalityEvidenceError, got a result instead")
    except Exception as exc:
        from ai.criticality.model import MissingCriticalityEvidenceError

        assert isinstance(exc, MissingCriticalityEvidenceError)
