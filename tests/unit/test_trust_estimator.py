"""Focused tests for `ai.trust_estimator.model.TwinTrustEstimator`.

SYNTHETIC / CODE-CORRECTNESS VALIDATION ONLY. All `DigitalTwinState`
instances, evidence values, predictions, and outcomes here are small,
deterministic, hand-constructed test fixtures -- NOT real Phase 2 trust
records or outcomes (there are currently zero of either in the project;
see the Phase 4 Trust Estimator audit and input-contract review). Nothing
here calls the live backend, writes to any database, reads real CSI/SNR/
RSSI data, or claims real Trust performance/calibration quality.

CONTRACT NOTE: `estimate(state)` ALWAYS raises `MissingTrustEvidenceError`
today, because `DigitalTwinState` cannot supply `prediction_error` or an
authoritative `comm_quality` (see ai/trust_estimator/model.py's module
docstring and the Phase 4 Trust input-contract review). The actual sigmoid
math is tested through `compute_trust_from_evidence(...)`, which takes all
four evidence terms as explicit arguments -- never through `estimate()`.
"""

from __future__ import annotations

import math

import pytest
import torch

from ai.trust_estimator.base import BaseTrustEstimator
from ai.trust_estimator.model import (
    DEFAULT_TAU,
    DEFAULT_WEIGHTS,
    MissingTrustEvidenceError,
    TwinTrustEstimator,
)
from digital_twin.state import ChannelState, DigitalTwinState, EnvironmentalContext, MobilityState


def _make_state(prediction_uncertainty=0.2, sync_age_seconds=0.05, metadata=None) -> DigitalTwinState:
    """SYNTHETIC fixture -- a small, deterministic DigitalTwinState."""
    return DigitalTwinState(
        timestamp=10.0,
        channel=ChannelState(csi=None, snr=20.0, beam_index=3, path_loss=90.0),
        mobility=MobilityState(position=(0.0, 0.0, 0.0), velocity=(15.0, 0.0, 0.0), heading=0.0, relative_speed=5.0),
        environment=EnvironmentalContext(weather="clear_day", traffic_density="sparse", blockage_probability=0.05),
        prediction_uncertainty=prediction_uncertainty,
        sync_age_seconds=sync_age_seconds,
        metadata=metadata if metadata is not None else {"scene_id": "synthetic_scene", "vehicle_id": "synthetic_veh"},
    )


def _hand_sigmoid(prediction_error, prediction_uncertainty, sync_age_penalty, comm_quality, weights, tau) -> float:
    w1, w2, w3, w4 = weights
    raw = w4 * comm_quality - w1 * prediction_error - w2 * prediction_uncertainty - w3 * sync_age_penalty
    return 1.0 / (1.0 + math.exp(-raw / tau))


# --- 1. BaseTrustEstimator interface compliance ---


def test_concrete_estimator_can_be_instantiated():
    est = TwinTrustEstimator()
    assert isinstance(est, BaseTrustEstimator)
    assert est.weights == DEFAULT_WEIGHTS
    assert est.tau == DEFAULT_TAU


def test_base_trust_estimator_still_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseTrustEstimator()  # type: ignore[abstract]


def test_estimate_has_the_exact_required_signature():
    """estimate(state: DigitalTwinState) -> float -- single positional
    argument, no extra required parameters."""
    est = TwinTrustEstimator()
    state = _make_state()
    with pytest.raises(MissingTrustEvidenceError):
        est.estimate(state)  # must be callable with ONLY `state`


# --- 2. estimate(state): must fail explicitly, never fabricate ---


def test_estimate_always_raises_missing_trust_evidence_error():
    est = TwinTrustEstimator()
    state = _make_state()
    with pytest.raises(MissingTrustEvidenceError):
        est.estimate(state)


def test_estimate_does_not_silently_use_zero_for_missing_evidence():
    """If estimate() ever silently defaulted prediction_error/comm_quality
    to 0.0, it would return a float instead of raising. Confirm it never
    does, across several different (still-incomplete) states."""
    est = TwinTrustEstimator()
    for state in (
        _make_state(prediction_uncertainty=0.0, sync_age_seconds=0.0),
        _make_state(prediction_uncertainty=0.9, sync_age_seconds=5.0),
        _make_state(prediction_uncertainty=torch.tensor(0.5)),
    ):
        with pytest.raises(MissingTrustEvidenceError):
            est.estimate(state)


def test_estimate_does_not_read_metadata_even_when_metadata_contains_evidence_like_keys():
    """estimate() must not silently pull prediction_error/comm_quality out
    of state.metadata -- confirm it still raises even when metadata
    happens to contain those exact key names."""
    est = TwinTrustEstimator()
    state = _make_state(metadata={"prediction_error": 0.1, "comm_quality": 0.9})
    with pytest.raises(MissingTrustEvidenceError):
        est.estimate(state)


def test_estimate_error_message_explains_the_real_gap():
    est = TwinTrustEstimator()
    state = _make_state()
    with pytest.raises(MissingTrustEvidenceError, match="prediction_error"):
        est.estimate(state)


# --- 3. Isolated mathematical trust calculation ---


def test_exact_sigmoid_calculation_against_hand_computed_value():
    est = TwinTrustEstimator()
    result = est.compute_trust_from_evidence(
        prediction_error=0.2, prediction_uncertainty=0.3, sync_age_penalty=0.1, comm_quality=0.8
    )
    expected = _hand_sigmoid(0.2, 0.3, 0.1, 0.8, DEFAULT_WEIGHTS, DEFAULT_TAU)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_output_is_bounded_in_unit_interval():
    est = TwinTrustEstimator()
    result = est.compute_trust_from_evidence(
        prediction_error=0.0, prediction_uncertainty=0.0, sync_age_penalty=0.0, comm_quality=1.0
    )
    assert 0.0 <= result <= 1.0


def test_increasing_prediction_error_lowers_trust():
    est = TwinTrustEstimator()
    low = est.compute_trust_from_evidence(prediction_error=0.1, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.5)
    high = est.compute_trust_from_evidence(prediction_error=0.9, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.5)
    assert high < low


def test_increasing_uncertainty_lowers_trust():
    est = TwinTrustEstimator()
    low = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.1, sync_age_penalty=0.05, comm_quality=0.5)
    high = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.9, sync_age_penalty=0.05, comm_quality=0.5)
    assert high < low


def test_increasing_sync_age_penalty_lowers_trust():
    est = TwinTrustEstimator()
    low = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=0.01, comm_quality=0.5)
    high = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=2.0, comm_quality=0.5)
    assert high < low


def test_increasing_comm_quality_raises_trust():
    est = TwinTrustEstimator()
    low = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.1)
    high = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.9)
    assert high > low


def test_non_finite_evidence_rejected():
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.compute_trust_from_evidence(prediction_error=float("nan"), prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.5)
    with pytest.raises(ValueError):
        est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=float("inf"))
    with pytest.raises(ValueError):
        est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=float("nan"), comm_quality=0.5)


# --- 4. Parameter validation ---


def test_custom_valid_weights_work():
    est = TwinTrustEstimator(weights=(0.4, 0.3, 0.2, 0.1))
    result = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.6)
    expected = _hand_sigmoid(0.2, 0.2, 0.05, 0.6, (0.4, 0.3, 0.2, 0.1), DEFAULT_TAU)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_custom_tau_works():
    est = TwinTrustEstimator(tau=2.0)
    assert est.tau == 2.0
    result = est.compute_trust_from_evidence(prediction_error=0.2, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.6)
    expected = _hand_sigmoid(0.2, 0.2, 0.05, 0.6, DEFAULT_WEIGHTS, 2.0)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_invalid_weight_count_rejected():
    with pytest.raises(ValueError):
        TwinTrustEstimator(weights=(0.5, 0.5, 0.0))  # type: ignore[arg-type]


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        TwinTrustEstimator(weights=(-0.1, 0.4, 0.4, 0.3))


def test_non_normalized_weights_rejected():
    with pytest.raises(ValueError):
        TwinTrustEstimator(weights=(0.5, 0.5, 0.5, 0.5))


def test_invalid_tau_rejected():
    with pytest.raises(ValueError):
        TwinTrustEstimator(tau=0.0)
    with pytest.raises(ValueError):
        TwinTrustEstimator(tau=-1.0)
    with pytest.raises(ValueError):
        TwinTrustEstimator(tau=float("nan"))


# --- 5. Uncertainty handling (via compute_trust_from_evidence) ---


def test_scalar_tensor_uncertainty_accepted():
    """SYNTHETIC -- a 0-dim tensor, standing in for a scalar GRU output.

    Tolerance is looser here (1e-6, not 1e-9) because torch's default
    float32 storage only has ~7 significant decimal digits -- this is
    float32 rounding, not a model defect.
    """
    est = TwinTrustEstimator()
    result = est.compute_trust_from_evidence(
        prediction_error=0.1, prediction_uncertainty=torch.tensor(0.3), sync_age_penalty=0.05, comm_quality=0.5
    )
    expected = _hand_sigmoid(0.1, 0.3, 0.05, 0.5, DEFAULT_WEIGHTS, DEFAULT_TAU)
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_multi_element_tensor_uncertainty_uses_documented_mean_reduction():
    """SYNTHETIC -- mirrors GRUPrediction.prediction_uncertainty's real
    shape (batch, output_dim), i.e. not a scalar. The documented
    reduction is torch.mean() over all elements."""
    est = TwinTrustEstimator()
    tensor_uncertainty = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # mean = 0.25
    result = est.compute_trust_from_evidence(
        prediction_error=0.1, prediction_uncertainty=tensor_uncertainty, sync_age_penalty=0.05, comm_quality=0.5
    )
    expected = _hand_sigmoid(0.1, 0.25, 0.05, 0.5, DEFAULT_WEIGHTS, DEFAULT_TAU)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_non_finite_tensor_uncertainty_rejected():
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.compute_trust_from_evidence(
            prediction_error=0.1, prediction_uncertainty=torch.tensor([0.1, float("nan")]),
            sync_age_penalty=0.05, comm_quality=0.5,
        )


def test_empty_tensor_uncertainty_rejected():
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.compute_trust_from_evidence(
            prediction_error=0.1, prediction_uncertainty=torch.tensor([]),
            sync_age_penalty=0.05, comm_quality=0.5,
        )


def test_non_numeric_non_tensor_uncertainty_rejected():
    """Never silently assume tensor == float: a plain string is neither
    and must raise."""
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.compute_trust_from_evidence(
            prediction_error=0.1, prediction_uncertainty="not a number",  # type: ignore[arg-type]
            sync_age_penalty=0.05, comm_quality=0.5,
        )


# --- 6. Calibration / ECE ---


def test_perfect_calibration_yields_zero_ece():
    """SYNTHETIC -- hand-constructed, well-calibrated predictions."""
    est = TwinTrustEstimator()
    predictions = [0.1] * 10
    outcomes = [True] + [False] * 9
    ece = est.calibration_error(predictions, outcomes)
    assert math.isclose(ece, 0.0, abs_tol=1e-9)


def test_known_imperfect_calibration_matches_hand_computed_ece():
    """SYNTHETIC -- hand-computed expected ECE for a small, fixed batch.

    5 bins of width 0.2. Bin 0 ([0,0.2)): preds=[0.1,0.1], outcomes=[F,F]
      -> conf=0.1, acc=0.0, |diff|=0.1, weight=2/4
    Bin 4 ([0.8,1.0]): preds=[0.9,0.9], outcomes=[T,T]
      -> conf=0.9, acc=1.0, |diff|=0.1, weight=2/4
    ECE = 0.5*0.1 + 0.5*0.1 = 0.1
    """
    est = TwinTrustEstimator()
    predictions = [0.1, 0.1, 0.9, 0.9]
    outcomes = [False, False, True, True]
    ece = est.calibration_error(predictions, outcomes)
    assert math.isclose(ece, 0.1, abs_tol=1e-9)


def test_ece_invalid_probability_rejected():
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.calibration_error([0.5, 1.5], [True, False])
    with pytest.raises(ValueError):
        est.calibration_error([0.5, -0.1], [True, False])
    with pytest.raises(ValueError):
        est.calibration_error([0.5, float("nan")], [True, False])


def test_ece_invalid_binary_targets_rejected():
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.calibration_error([0.5, 0.5], [1, 0])  # ints, not bool
    with pytest.raises(ValueError):
        est.calibration_error([0.5, 0.5], ["yes", "no"])  # type: ignore[list-item]


def test_ece_invalid_lengths_rejected():
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.calibration_error([0.1, 0.2], [True])


def test_ece_empty_input_rejected():
    est = TwinTrustEstimator()
    with pytest.raises(ValueError):
        est.calibration_error([], [])


# --- 7. No silent zero for missing evidence (formula-level) ---


def test_compute_trust_from_evidence_requires_all_four_arguments_explicitly():
    """compute_trust_from_evidence has no defaults for any of the four
    evidence terms -- omitting one must fail loudly, not silently use 0."""
    est = TwinTrustEstimator()
    with pytest.raises(TypeError):
        est.compute_trust_from_evidence(  # type: ignore[call-arg]
            prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.5
        )


# --- Determinism ---


def test_compute_trust_from_evidence_is_deterministic():
    est = TwinTrustEstimator()
    r1 = est.compute_trust_from_evidence(prediction_error=0.15, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.65)
    r2 = est.compute_trust_from_evidence(prediction_error=0.15, prediction_uncertainty=0.2, sync_age_penalty=0.05, comm_quality=0.65)
    assert r1 == r2


def test_calibration_error_is_deterministic():
    est = TwinTrustEstimator()
    predictions = [0.1, 0.4, 0.6, 0.9]
    outcomes = [False, True, False, True]
    r1 = est.calibration_error(predictions, outcomes)
    r2 = est.calibration_error(predictions, outcomes)
    assert r1 == r2
