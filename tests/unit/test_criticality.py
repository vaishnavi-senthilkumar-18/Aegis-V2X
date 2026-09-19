"""Focused tests for `ai.criticality.model.CriticalityEstimator`.

SYNTHETIC / CODE-CORRECTNESS VALIDATION ONLY. All `DigitalTwinState`
instances and evidence values here are small, deterministic, hand-
constructed test fixtures -- NOT real Phase 2 criticality records (there
are currently zero in the project; see the Phase 4 Criticality audit).
Nothing here calls the live backend, writes to any database, or claims
real-data, scientific-performance, or calibrated criticality results.

CONTRACT NOTE: `estimate(state)` ALWAYS raises
`MissingCriticalityEvidenceError` today, because none of the five required
evidence terms can currently be authoritatively derived from
`DigitalTwinState` together with the real Phase 2 dataset (see
ai/criticality/model.py's module docstring and the Phase 4 Criticality
audit). The actual weighted-sum math is tested through
`compute_criticality_from_evidence(...)`, which takes all five evidence
terms as explicit arguments -- never through `estimate()`.
"""

from __future__ import annotations

import math

import pytest

from ai.criticality.base import BaseCriticalityEstimator
from ai.criticality.model import (
    DEFAULT_WEIGHTS,
    CriticalityEstimator,
    MissingCriticalityEvidenceError,
)
from digital_twin.state import ChannelState, DigitalTwinState, EnvironmentalContext, MobilityState


def _make_state(metadata=None) -> DigitalTwinState:
    """SYNTHETIC fixture -- a small, deterministic DigitalTwinState."""
    return DigitalTwinState(
        timestamp=10.0,
        channel=ChannelState(csi=None, snr=20.0, beam_index=3, path_loss=90.0),
        mobility=MobilityState(position=(0.0, 0.0, 0.0), velocity=(15.0, 0.0, 0.0), heading=0.0, relative_speed=5.0),
        environment=EnvironmentalContext(weather="clear_day", traffic_density="sparse", blockage_probability=0.05),
        prediction_uncertainty=0.2,
        sync_age_seconds=0.05,
        metadata=metadata if metadata is not None else {"scene_id": "synthetic_scene", "vehicle_id": "synthetic_veh"},
    )


# --- A. Interface compliance ---


def test_concrete_estimator_satisfies_base_interface():
    est = CriticalityEstimator()
    assert isinstance(est, BaseCriticalityEstimator)


def test_base_criticality_estimator_still_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseCriticalityEstimator()  # type: ignore[abstract]


def test_estimate_has_the_exact_required_signature():
    """estimate(state: DigitalTwinState) -> float -- single positional
    argument, no extra required parameters."""
    est = CriticalityEstimator()
    state = _make_state()
    with pytest.raises(MissingCriticalityEvidenceError):
        est.estimate(state)  # must be callable with ONLY `state`


# --- B. Default weights sum to 1 ---


def test_default_weights_sum_to_one():
    assert math.isclose(sum(DEFAULT_WEIGHTS), 1.0, abs_tol=1e-9)
    est = CriticalityEstimator()
    assert est.weights == DEFAULT_WEIGHTS


# --- C-G. Weight validation ---


def test_valid_custom_weights_accepted():
    est = CriticalityEstimator(weights=(0.4, 0.2, 0.2, 0.1, 0.1))
    assert est.weights == (0.4, 0.2, 0.2, 0.1, 0.1)


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        CriticalityEstimator(weights=(-0.1, 0.35, 0.25, 0.25, 0.25))


def test_non_finite_weights_rejected():
    with pytest.raises(ValueError):
        CriticalityEstimator(weights=(float("nan"), 0.25, 0.25, 0.25, 0.25))
    with pytest.raises(ValueError):
        CriticalityEstimator(weights=(float("inf"), 0.25, 0.25, 0.25, 0.25))


def test_incorrect_weight_count_rejected():
    with pytest.raises(ValueError):
        CriticalityEstimator(weights=(0.5, 0.5, 0.0, 0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CriticalityEstimator(weights=(0.2, 0.2, 0.2, 0.2, 0.1, 0.1))  # type: ignore[arg-type]


def test_weights_not_summing_to_one_rejected():
    with pytest.raises(ValueError):
        CriticalityEstimator(weights=(0.3, 0.3, 0.3, 0.3, 0.3))


# --- H. Weighted-sum formula ---


def test_valid_evidence_matches_hand_computed_formula():
    est = CriticalityEstimator()
    result = est.compute_criticality_from_evidence(
        relative_speed=0.4,
        blockage_probability=0.6,
        sync_age=0.2,
        channel_degradation=0.8,
        traffic_density=0.5,
    )
    expected = 0.25 * 0.4 + 0.25 * 0.6 + 0.15 * 0.2 + 0.20 * 0.8 + 0.15 * 0.5
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_custom_weights_matches_hand_computed_formula():
    est = CriticalityEstimator(weights=(0.4, 0.2, 0.2, 0.1, 0.1))
    result = est.compute_criticality_from_evidence(
        relative_speed=0.5, blockage_probability=0.5, sync_age=0.5, channel_degradation=0.5, traffic_density=0.5
    )
    expected = (0.4 + 0.2 + 0.2 + 0.1 + 0.1) * 0.5
    assert math.isclose(result, expected, rel_tol=1e-9)


# --- I/J. Boundary cases ---


def test_all_zero_evidence_produces_zero():
    est = CriticalityEstimator()
    result = est.compute_criticality_from_evidence(
        relative_speed=0.0, blockage_probability=0.0, sync_age=0.0, channel_degradation=0.0, traffic_density=0.0
    )
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_all_one_evidence_produces_one():
    est = CriticalityEstimator()
    result = est.compute_criticality_from_evidence(
        relative_speed=1.0, blockage_probability=1.0, sync_age=1.0, channel_degradation=1.0, traffic_density=1.0
    )
    assert math.isclose(result, 1.0, abs_tol=1e-9)


# --- K. Invalid evidence rejected ---


def test_non_finite_evidence_rejected():
    est = CriticalityEstimator()
    with pytest.raises(ValueError):
        est.compute_criticality_from_evidence(
            relative_speed=float("nan"), blockage_probability=0.5, sync_age=0.5, channel_degradation=0.5, traffic_density=0.5
        )
    with pytest.raises(ValueError):
        est.compute_criticality_from_evidence(
            relative_speed=0.5, blockage_probability=float("inf"), sync_age=0.5, channel_degradation=0.5, traffic_density=0.5
        )


def test_non_scalar_evidence_rejected():
    est = CriticalityEstimator()
    with pytest.raises(ValueError):
        est.compute_criticality_from_evidence(
            relative_speed=[0.5], blockage_probability=0.5, sync_age=0.5, channel_degradation=0.5, traffic_density=0.5  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        est.compute_criticality_from_evidence(
            relative_speed=0.5, blockage_probability="0.5", sync_age=0.5, channel_degradation=0.5, traffic_density=0.5  # type: ignore[arg-type]
        )


def test_missing_evidence_argument_raises():
    """No defaults exist for any of the five terms -- omitting one must
    fail loudly, not silently use 0."""
    est = CriticalityEstimator()
    with pytest.raises(TypeError):
        est.compute_criticality_from_evidence(  # type: ignore[call-arg]
            relative_speed=0.5, blockage_probability=0.5, sync_age=0.5, channel_degradation=0.5
        )


# --- L. estimate(state) raises rather than fabricating ---


def test_estimate_always_raises_missing_criticality_evidence_error():
    est = CriticalityEstimator()
    state = _make_state()
    with pytest.raises(MissingCriticalityEvidenceError):
        est.estimate(state)


def test_estimate_error_message_explains_the_real_gap():
    est = CriticalityEstimator()
    state = _make_state()
    with pytest.raises(MissingCriticalityEvidenceError, match="relative_speed"):
        est.estimate(state)


# --- M. No silent substitution from metadata/unrelated fields ---


def test_estimate_does_not_use_metadata_as_silent_substitute():
    """estimate() must not silently pull evidence out of state.metadata --
    confirm it still raises even when metadata happens to contain all five
    evidence-like key names with plausible values."""
    est = CriticalityEstimator()
    state = _make_state(
        metadata={
            "relative_speed": 0.3,
            "blockage_probability": 0.4,
            "sync_age": 0.1,
            "channel_degradation": 0.6,
            "traffic_density": 0.5,
        }
    )
    with pytest.raises(MissingCriticalityEvidenceError):
        est.estimate(state)


def test_estimate_does_not_use_unrelated_state_fields_as_silent_substitute():
    """Even though state.mobility.relative_speed, state.environment.
    blockage_probability, and state.sync_age_seconds happen to exist on
    DigitalTwinState, estimate() must not quietly read them as if they
    were authoritative -- it must still raise every time."""
    est = CriticalityEstimator()
    state = _make_state()
    assert state.mobility.relative_speed == 5.0  # a real field, present but not used
    assert state.environment.blockage_probability == 0.05  # present but not used
    with pytest.raises(MissingCriticalityEvidenceError):
        est.estimate(state)


# --- N. Determinism ---


def test_compute_criticality_from_evidence_is_deterministic():
    est = CriticalityEstimator()
    r1 = est.compute_criticality_from_evidence(
        relative_speed=0.3, blockage_probability=0.4, sync_age=0.2, channel_degradation=0.6, traffic_density=0.5
    )
    r2 = est.compute_criticality_from_evidence(
        relative_speed=0.3, blockage_probability=0.4, sync_age=0.2, channel_degradation=0.6, traffic_density=0.5
    )
    assert r1 == r2
