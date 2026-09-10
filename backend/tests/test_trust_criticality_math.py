"""Unit tests for the trust and criticality calibration math (unit-level, no DB)."""

from __future__ import annotations

import math

import pytest

from app.crud.criticality import compute_criticality_score
from app.crud.trust import DEFAULT_TAU, DEFAULT_TRUST_WEIGHTS, compute_trust_probability
from app.models.trust import TRUST_BANDS


def test_trust_probability_is_bounded() -> None:
    _, trust_probability, _ = compute_trust_probability(
        prediction_error=0.0, prediction_uncertainty=0.0, sync_age_penalty=0.0, comm_quality=1.0
    )
    assert 0.0 <= trust_probability <= 1.0


def test_high_penalties_lower_trust_than_low_penalties() -> None:
    _, high_penalty_trust, _ = compute_trust_probability(
        prediction_error=0.9, prediction_uncertainty=0.9, sync_age_penalty=0.9, comm_quality=0.1
    )
    _, low_penalty_trust, _ = compute_trust_probability(
        prediction_error=0.05, prediction_uncertainty=0.05, sync_age_penalty=0.05, comm_quality=0.95
    )
    assert low_penalty_trust > high_penalty_trust


def test_trust_probability_matches_sigmoid_formula() -> None:
    e_t, u_t, a_t, q_t = 0.2, 0.3, 0.1, 0.8
    w1, w2, w3, w4 = DEFAULT_TRUST_WEIGHTS
    raw_score, trust_probability, _ = compute_trust_probability(e_t, u_t, a_t, q_t)
    expected_raw = w4 * q_t - w1 * e_t - w2 * u_t - w3 * a_t
    expected_trust = 1.0 / (1.0 + math.exp(-expected_raw / DEFAULT_TAU))
    assert math.isclose(raw_score, expected_raw, rel_tol=1e-9)
    assert math.isclose(trust_probability, expected_trust, rel_tol=1e-9)


@pytest.mark.parametrize(
    ("trust_probability", "expected_band"),
    [
        (0.05, "very_unreliable"),
        (0.35, "unreliable"),
        (0.55, "moderate"),
        (0.75, "reliable"),
        (0.95, "highly_reliable"),
    ],
)
def test_trust_bands_match_expected_labels(trust_probability: float, expected_band: str) -> None:
    for upper_bound, label in TRUST_BANDS:
        if trust_probability < upper_bound:
            assert label == expected_band
            return
    pytest.fail("trust_probability did not match any band")


def test_criticality_score_is_weighted_average_of_uniform_features() -> None:
    score = compute_criticality_score(
        relative_speed_score=0.5,
        blockage_probability_score=0.5,
        sync_age_score=0.5,
        channel_degradation_score=0.5,
        traffic_density_score=0.5,
    )
    assert math.isclose(score, 0.5, rel_tol=1e-9)


def test_criticality_score_bounded_when_features_bounded() -> None:
    score = compute_criticality_score(
        relative_speed_score=1.0,
        blockage_probability_score=1.0,
        sync_age_score=1.0,
        channel_degradation_score=1.0,
        traffic_density_score=1.0,
    )
    assert math.isclose(score, 1.0, rel_tol=1e-9)
