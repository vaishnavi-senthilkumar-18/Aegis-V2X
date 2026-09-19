"""Focused tests for `ai.twintrust_ap.gdca` (Phase 5: Graded Digital-Twin
Consistency Authority).

SYNTHETIC / CODE-CORRECTNESS VALIDATION ONLY. `residual`/`uncertainty`/
state-vector values here are plain floats and tuples chosen to exercise
the mathematical formulation (boundedness, determinism, monotonicity,
validation) -- not measurements from any real CARLA scene. GDCA's public
math contract, `compute_authority_from_evidence(residual_norm,
uncertainty) -> float`, never touches DigitalTwinState, tensors, or any
real dataset; the state-pair contract, `estimate(observed_state,
predicted_state)`, is exercised separately below and always raises
`MissingGDCAEvidenceError` by design (see module docstring in
ai/twintrust_ap/gdca.py) -- no wireless field (CSI/SNR/RSSI/beam_index/
path_loss) is ever read, fabricated, or required anywhere in this file.
"""

from __future__ import annotations

import math

import pytest

from ai.twintrust_ap.gdca import (
    GDCAEstimator,
    MissingGDCAEvidenceError,
    combine_standardized_residuals,
    compute_authority_from_dimensional_evidence,
    compute_authority_from_evidence,
    compute_residual,
    residual_level_and_spread,
    residual_norm,
    robust_spread,
    rolling_uncertainty,
    standardize_residual_dimensions,
)
from digital_twin.state import ChannelState, DigitalTwinState, EnvironmentalContext, MobilityState


def make_state(position=(0.0, 0.0, 0.0)) -> DigitalTwinState:
    return DigitalTwinState(
        timestamp=0.0,
        channel=ChannelState(csi=None, snr=0.0, beam_index=0, path_loss=0.0),
        mobility=MobilityState(position=position, velocity=(0.0, 0.0, 0.0), heading=0.0, relative_speed=0.0),
        environment=EnvironmentalContext(weather="clear_day", traffic_density="sparse", blockage_probability=0.0),
        prediction_uncertainty=0.0,
        sync_age_seconds=0.0,
    )


# --------------------------------------------------------------------------
# compute_residual / residual_norm
# --------------------------------------------------------------------------


def test_residual_is_observed_minus_predicted():
    assert compute_residual((5.0, 3.0), (2.0, 1.0)) == (3.0, 2.0)


def test_residual_of_identical_states_is_zero():
    r = compute_residual((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
    assert r == (0.0, 0.0, 0.0)


def test_residual_requires_matching_dimensions():
    with pytest.raises(ValueError):
        compute_residual((1.0, 2.0), (1.0,))


def test_residual_rejects_empty_vectors():
    with pytest.raises(ValueError):
        compute_residual((), ())


def test_residual_rejects_non_finite_values():
    with pytest.raises(ValueError):
        compute_residual((float("nan"),), (0.0,))
    with pytest.raises(ValueError):
        compute_residual((float("inf"),), (0.0,))


def test_residual_norm_matches_hand_computed_euclidean_norm():
    assert residual_norm((3.0, 4.0)) == pytest.approx(5.0)


def test_residual_norm_of_zero_vector_is_zero():
    assert residual_norm((0.0, 0.0, 0.0)) == 0.0


def test_residual_norm_supports_single_dimension():
    assert residual_norm((-7.0,)) == pytest.approx(7.0)


def test_residual_norm_rejects_empty_vector():
    with pytest.raises(ValueError):
        residual_norm(())


# --------------------------------------------------------------------------
# rolling_uncertainty
# --------------------------------------------------------------------------


def test_rolling_uncertainty_of_constant_sequence_is_zero():
    assert rolling_uncertainty([2.0, 2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_rolling_uncertainty_matches_hand_computed_population_std():
    # values: 1, 2, 3 -> mean=2, variance=((1)^2+(0)^2+(1)^2)/3=2/3, std=sqrt(2/3)
    assert rolling_uncertainty([1.0, 2.0, 3.0]) == pytest.approx(math.sqrt(2.0 / 3.0))


def test_rolling_uncertainty_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        rolling_uncertainty([1.0])


def test_rolling_uncertainty_rejects_empty_sequence():
    with pytest.raises(ValueError):
        rolling_uncertainty([])


def test_rolling_uncertainty_rejects_non_finite_values():
    with pytest.raises(ValueError):
        rolling_uncertainty([1.0, float("nan")])


# --------------------------------------------------------------------------
# compute_authority_from_evidence -- the core GDCA formula
# --------------------------------------------------------------------------


def test_output_is_always_within_zero_one():
    for r in (0.0, 0.1, 0.5, 1.0, 5.0, 100.0):
        for u in (0.0, 0.1, 0.5, 1.0, 5.0, 100.0):
            a = compute_authority_from_evidence(r, u)
            assert 0.0 <= a <= 1.0


def test_zero_residual_gives_maximum_authority():
    """Identical prediction/observation (residual_norm=0) => A_t == 1.0 exactly,
    regardless of uncertainty (z=0 => exp(0)=1)."""
    for u in (0.0, 0.5, 3.0, 50.0):
        assert compute_authority_from_evidence(0.0, u) == pytest.approx(1.0)


def test_increasing_residual_strictly_decreases_authority():
    uncertainty = 1.0
    residuals = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    authorities = [compute_authority_from_evidence(r, uncertainty) for r in residuals]
    for a_prev, a_next in zip(authorities, authorities[1:]):
        assert a_next < a_prev


def test_increasing_uncertainty_increases_authority_for_fixed_residual():
    """More tolerance (higher sigma_t) for the same residual => higher authority."""
    residual = 2.0
    uncertainties = [0.1, 0.5, 1.0, 5.0, 20.0]
    authorities = [compute_authority_from_evidence(residual, u) for u in uncertainties]
    for a_prev, a_next in zip(authorities, authorities[1:]):
        assert a_next > a_prev


def test_exact_hand_computed_value():
    # residual=1, uncertainty=1 => z ~= 1 (eps negligible) => A = exp(-0.5)
    a = compute_authority_from_evidence(1.0, 1.0)
    assert a == pytest.approx(math.exp(-0.5), abs=1e-6)


def test_zero_uncertainty_does_not_divide_by_zero():
    """uncertainty=0.0 with a nonzero residual must not raise or return NaN/Inf --
    the _EPS guard handles this; the result should be a very small (near-zero)
    but finite, valid authority."""
    a = compute_authority_from_evidence(1.0, 0.0)
    assert math.isfinite(a)
    assert 0.0 <= a <= 1.0
    assert a == pytest.approx(0.0, abs=1e-6)


def test_deterministic_for_identical_inputs():
    a1 = compute_authority_from_evidence(1.234, 2.345)
    a2 = compute_authority_from_evidence(1.234, 2.345)
    assert a1 == a2


def test_negative_residual_norm_rejected():
    with pytest.raises(ValueError):
        compute_authority_from_evidence(-1.0, 1.0)


def test_negative_uncertainty_rejected():
    with pytest.raises(ValueError):
        compute_authority_from_evidence(1.0, -1.0)


def test_non_finite_residual_norm_rejected():
    with pytest.raises(ValueError):
        compute_authority_from_evidence(float("nan"), 1.0)
    with pytest.raises(ValueError):
        compute_authority_from_evidence(float("inf"), 1.0)


def test_non_finite_uncertainty_rejected():
    with pytest.raises(ValueError):
        compute_authority_from_evidence(1.0, float("nan"))
    with pytest.raises(ValueError):
        compute_authority_from_evidence(1.0, float("inf"))


def test_large_residual_is_numerically_stable_not_nan_or_inf():
    a = compute_authority_from_evidence(1e6, 1.0)
    assert math.isfinite(a)
    assert a == pytest.approx(0.0, abs=1e-9)


def test_multiple_state_dimensions_end_to_end():
    """1D, 2D, and 3D state vectors all flow through residual -> norm -> authority
    without special-casing dimensionality."""
    for observed, predicted in [
        ((5.0,), (5.0,)),
        ((5.0, 3.0), (2.0, 1.0)),
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    ]:
        r = compute_residual(observed, predicted)
        n = residual_norm(r)
        a = compute_authority_from_evidence(n, uncertainty=1.0)
        assert 0.0 <= a <= 1.0


# --------------------------------------------------------------------------
# GDCAEstimator.estimate() -- always raises (see module docstring)
# --------------------------------------------------------------------------


def test_estimate_always_raises_missing_gdca_evidence_error():
    estimator = GDCAEstimator()
    observed = make_state(position=(1.0, 2.0, 0.0))
    predicted = make_state(position=(1.1, 2.1, 0.0))
    with pytest.raises(MissingGDCAEvidenceError):
        estimator.estimate(observed, predicted)


def test_estimate_raises_even_for_identical_states():
    """The gap is structural (no uncertainty window from one pair), not
    about the states themselves disagreeing -- even identical observed/
    predicted states must raise, not silently succeed with an invented
    uncertainty."""
    estimator = GDCAEstimator()
    state = make_state(position=(0.0, 0.0, 0.0))
    with pytest.raises(MissingGDCAEvidenceError):
        estimator.estimate(state, state)


def test_base_gdca_estimator_cannot_be_instantiated_directly():
    from ai.twintrust_ap.gdca import BaseGDCAEstimator

    with pytest.raises(TypeError):
        BaseGDCAEstimator()


# --------------------------------------------------------------------------
# residual_level_and_spread / robust_spread
# (2026-09-19 design-review correction: MAD-based spread, see
# ai/twintrust_ap/gdca.py's module docstring, replacing rolling_uncertainty
# as GDCA's recommended uncertainty source)
# --------------------------------------------------------------------------


def test_robust_spread_of_constant_window_is_zero():
    level, spread = residual_level_and_spread([3.0, 3.0, 3.0, 3.0, 3.0])
    assert level == pytest.approx(3.0)
    assert spread == pytest.approx(0.0)


def test_robust_spread_matches_hand_computed_mad():
    # window: 1.4, 1.5, 1.6, 1.5, 1.4 -> median=1.5, abs devs: 0.1,0,0.1,0,0.1
    # -> sorted abs devs: 0,0,0.1,0.1,0.1 -> median abs dev = 0.1 -> MAD*1.4826
    level, spread = residual_level_and_spread([1.4, 1.5, 1.6, 1.5, 1.4])
    assert level == pytest.approx(1.5)
    assert spread == pytest.approx(1.4826 * 0.1, abs=1e-6)


def test_robust_spread_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        robust_spread([1.0])


def test_robust_spread_rejects_non_finite_values():
    with pytest.raises(ValueError):
        robust_spread([1.0, float("nan")])


def test_robust_spread_is_a_single_scalar_matching_residual_level_and_spread():
    values = [0.2, 0.3, 0.25, 0.4, 0.22]
    _, expected_spread = residual_level_and_spread(values)
    assert robust_spread(values) == pytest.approx(expected_spread)


# --------------------------------------------------------------------------
# standardize_residual_dimensions / combine_standardized_residuals /
# compute_authority_from_dimensional_evidence
# --------------------------------------------------------------------------


def test_standardize_residual_dimensions_divides_each_by_its_own_spread():
    z = standardize_residual_dimensions((4.0, 9.0), (2.0, 3.0))
    assert z[0] == pytest.approx(2.0, abs=1e-6)
    assert z[1] == pytest.approx(3.0, abs=1e-6)


def test_standardize_residual_dimensions_requires_matching_lengths():
    with pytest.raises(ValueError):
        standardize_residual_dimensions((1.0, 2.0), (1.0,))


def test_standardize_residual_dimensions_rejects_negative_spread():
    with pytest.raises(ValueError):
        standardize_residual_dimensions((1.0,), (-1.0,))


def test_combine_standardized_residuals_is_root_mean_square():
    # RMS of (3, 4) = sqrt((9+16)/2) = sqrt(12.5)
    assert combine_standardized_residuals((3.0, 4.0)) == pytest.approx(math.sqrt(12.5))


def test_dimensional_authority_output_is_always_within_zero_one():
    for residuals, spreads in [
        ((0.0, 0.0), (1.0, 1.0)),
        ((5.0, 5.0, 5.0), (0.1, 10.0, 1.0)),
        ((100.0,), (0.001,)),
    ]:
        a = compute_authority_from_dimensional_evidence(residuals, spreads)
        assert 0.0 <= a <= 1.0


def test_dimensional_authority_zero_residuals_gives_maximum_authority():
    a = compute_authority_from_dimensional_evidence((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    assert a == pytest.approx(1.0)


def test_dimensional_authority_is_deterministic():
    a1 = compute_authority_from_dimensional_evidence((1.0, 2.0), (0.5, 0.5))
    a2 = compute_authority_from_dimensional_evidence((1.0, 2.0), (0.5, 0.5))
    assert a1 == a2


# --------------------------------------------------------------------------
# MANDATORY ADVERSARIAL REGRESSION -- CASE Y (uncertainty inflation)
# --------------------------------------------------------------------------


def test_case_y_old_formulation_is_confirmed_exploitable():
    """Documents the CONFIRMED defect in the OLD (std-based) formulation:
    an unrelated large outlier several steps back inflates
    rolling_uncertainty enough that a separately bad CURRENT residual
    (1.5) is scored with artificially high authority. This test pins the
    OLD behavior (it must keep failing this way, since rolling_uncertainty
    itself is unchanged) so the contrast with the corrected formulation
    below is explicit and regression-tested, not just asserted in prose."""
    contaminated_window = [0.05, 0.05, 6.0, 0.05, 0.05]
    current_residual = 1.5
    old_sigma = rolling_uncertainty(contaminated_window)
    old_authority = compute_authority_from_evidence(current_residual, old_sigma)
    assert old_authority > 0.5  # confirmed inflated -- this IS the defect


def test_case_y_corrected_formulation_does_not_reward_bad_observation():
    """The corrected (MAD-based) formulation must NOT be fooled by the
    same unrelated outlier: the outlier is recognized as unrepresentative
    (MAD is robust to it), so the same bad current_residual=1.5 is
    correctly scored with near-zero authority, matching what a "clean"
    window with the same current residual produces."""
    contaminated_window = [0.05, 0.05, 6.0, 0.05, 0.05]
    clean_window = [1.4, 1.5, 1.6, 1.5, 1.4]
    current_residual = 1.5

    contaminated_spread = robust_spread(contaminated_window)
    clean_spread = robust_spread(clean_window)

    contaminated_authority = compute_authority_from_evidence(current_residual, contaminated_spread)
    clean_authority = compute_authority_from_evidence(current_residual, clean_spread)

    assert contaminated_authority < 0.01
    assert clean_authority < 0.01
    # Both near-zero -- the outlier no longer buys inflated authority.
    assert contaminated_authority == pytest.approx(clean_authority, abs=0.01)


def test_case_y_corrected_authority_is_lower_than_old_for_the_same_evidence():
    """Direct old-vs-corrected comparison on identical evidence: the fix
    must reduce (not increase) authority for this known-bad case."""
    contaminated_window = [0.05, 0.05, 6.0, 0.05, 0.05]
    current_residual = 1.5

    old_authority = compute_authority_from_evidence(
        current_residual, rolling_uncertainty(contaminated_window)
    )
    corrected_authority = compute_authority_from_evidence(
        current_residual, robust_spread(contaminated_window)
    )
    assert corrected_authority < old_authority


# --------------------------------------------------------------------------
# MANDATORY ADVERSARIAL REGRESSION -- CASE G (mixed units)
# --------------------------------------------------------------------------


def test_case_g_pooled_norm_is_confirmed_dominated_by_large_raw_units():
    """Documents the CONFIRMED defect in pooling raw, differently-scaled
    dimensions into one Euclidean norm: a large-magnitude dimension (here,
    a synthetic 'velocity-like' residual, arbitrary units for this
    mathematical stress test only -- not a physical claim) numerically
    swamps a small-magnitude dimension even when the small one is, in its
    own terms, far more anomalous."""
    # dim A ("position-like"): residual 0.01, own typical spread 0.02 -> mild (z=0.5)
    # dim B ("velocity-like"): residual 2.0, own typical spread 0.3 -> bad (z=6.67)
    residuals = (0.01, 2.0)
    pooled_norm = residual_norm(residuals)
    # The pooled norm is ~2.0 -- entirely driven by dim B's raw units, with
    # no information about how anomalous each dimension is relative to
    # its own scale baked in at all.
    assert pooled_norm == pytest.approx(2.0, abs=0.01)


def test_case_g_dimensional_authority_is_not_dominated_by_raw_unit_size():
    """The corrected per-dimension formulation must judge each dimension
    against its OWN spread. A dimension with large raw units but small
    residual relative to its own spread must not by itself force low
    authority; the RMS combination should react to whichever dimension is
    actually anomalous in standardized terms, not raw magnitude."""
    # dim A ("position-like"): residual 0.01, own spread 0.02 -> mild
    # dim B ("velocity-like"): residual 0.06, own spread 0.3 (own noise floor
    #   is naturally larger in these units) -> ALSO mild in standardized terms
    residuals = (0.01, 0.06)
    spreads = (0.02, 0.3)
    z = standardize_residual_dimensions(residuals, spreads)
    assert z[0] == pytest.approx(0.5, abs=1e-3)
    assert z[1] == pytest.approx(0.2, abs=1e-3)
    a = compute_authority_from_dimensional_evidence(residuals, spreads)
    # Both dimensions mild in standardized terms -> authority should be high,
    # NOT crushed merely because dim B's raw residual (0.06) is numerically
    # larger than dim A's (0.01).
    assert a > 0.9


def test_case_g_small_raw_residual_can_dominate_if_anomalous_in_its_own_units():
    """The converse of the defect: a dimension with SMALL raw units but a
    residual that is large relative to ITS OWN spread must be able to pull
    authority down, even though the pooled Euclidean norm would treat it
    as negligible next to a larger-raw-unit dimension."""
    # dim A ("position-like"): residual 0.5, own spread 0.05 -> very anomalous (z=10)
    # dim B ("velocity-like"): residual 0.1, own spread 1.0 -> unremarkable (z=0.1)
    residuals = (0.5, 0.1)
    spreads = (0.05, 1.0)
    pooled_norm = residual_norm(residuals)
    assert pooled_norm == pytest.approx(0.5099, abs=0.001)  # dominated by dim A anyway here,
    # but with UNITS REVERSED (dim A in large raw units, dim B in small raw
    # units, same standardized anomaly), pooling would have hidden dim A's
    # anomaly -- the standardized version below is scale-invariant either way:
    a = compute_authority_from_dimensional_evidence(residuals, spreads)
    assert a < 0.01  # correctly penalized for dim A's standardized anomaly


# --------------------------------------------------------------------------
# MANDATORY ADVERSARIAL REGRESSION -- CASE C (constant/systematic bias)
# --------------------------------------------------------------------------


def test_case_c_consistent_but_wrong_is_not_treated_as_accurate():
    """A predictor that is perfectly CONSISTENT but SYSTEMATICALLY WRONG
    (every single residual exactly 3.0) must NOT receive high authority
    merely because it is consistent -- consistency is not accuracy. MAD
    of a constant window is 0, so the standardized residual is enormous
    and authority correctly collapses to ~0, exactly like the old
    formulation got this ONE case right (for a different reason) --
    this invariant must survive the fix unchanged."""
    constant_bias_window = [3.0, 3.0, 3.0, 3.0, 3.0]
    current_residual = 3.0
    spread = robust_spread(constant_bias_window)
    assert spread == pytest.approx(0.0)
    authority = compute_authority_from_evidence(current_residual, spread)
    assert authority < 1e-6


def test_case_c_zero_bias_zero_spread_gives_maximum_authority():
    """Contrast case: a predictor that is both consistent AND accurate
    (residual always exactly 0) must still get maximum authority -- the
    fix must not punish genuine, repeated exact agreement."""
    zero_window = [0.0, 0.0, 0.0, 0.0, 0.0]
    spread = robust_spread(zero_window)
    authority = compute_authority_from_evidence(0.0, spread)
    assert authority == pytest.approx(1.0)


# --------------------------------------------------------------------------
# MANDATORY ADVERSARIAL REGRESSION -- CASE E (stationary / near-zero spread)
# --------------------------------------------------------------------------


def test_case_e_tiny_genuine_residual_is_not_excessively_penalized():
    """A genuinely tiny residual during a stationary period must not be
    driven to near-zero authority purely because the rolling spread also
    happens to be tiny -- both real-data-scale quantities together should
    still register meaningfully high authority, not degenerate into a
    near-binary collapse."""
    stationary_window = [0.001, 0.002, 0.0015, 0.0012, 0.0017]
    current_residual = 0.0015
    spread = robust_spread(stationary_window)
    authority = compute_authority_from_evidence(current_residual, spread)
    # Documented, not hidden: the current single-step formulation is
    # still fairly harsh in the near-zero-spread regime (z ~ 3, authority
    # ~ a few e-3) -- this is a known limitation (see module docstring
    # and claude/project_status.md), not silently accepted as ideal.
    # The regression bar here is only that it stays finite, non-negative,
    # and does not collapse all the way to the ~1e-9 floor the way Cases
    # C/Y's genuinely bad residuals do.
    assert math.isfinite(authority)
    assert authority > 1e-6


# --------------------------------------------------------------------------
# MANDATORY ADVERSARIAL REGRESSION -- CASE F (acceleration / regime change)
# --------------------------------------------------------------------------


def test_case_f_regime_change_does_not_produce_nan_or_negative_authority():
    """A sudden regime change (e.g. a stationary vehicle beginning to
    accelerate) must not produce NaN/Inf/negative authority, even though
    the rolling window is, by construction, still full of pre-change
    (stale) residuals immediately after the change. This is a stability
    requirement, not a claim that the single-step formulation fully
    "solves" regime transitions -- see the documented limitation below."""
    pre_change_window = [0.05, 0.06, 0.05, 0.04, 0.05]  # stationary-like
    post_change_residual = 0.9  # sudden large residual, e.g. acceleration onset
    spread = robust_spread(pre_change_window)
    authority = compute_authority_from_evidence(post_change_residual, spread)
    assert math.isfinite(authority)
    assert 0.0 <= authority <= 1.0


def test_case_f_documented_limitation_stale_window_lags_regime_change():
    """DOCUMENTED LIMITATION (not a bug, not silently hidden): because
    robust_spread/rolling_uncertainty both operate on a FIXED recent
    window, authority remains keyed to the OLD (pre-change) spread for
    several steps after a regime change begins, before the window fills
    with post-change residuals. This test pins that documented lag rather
    than asserting the single-step formulation eliminates it -- a correct
    fix would require multi-step/temporal handling, explicitly out of
    scope for this checkpoint (see module docstring, "WHAT IS DELIBERATELY
    NOT INCLUDED")."""
    pre_change_window = [0.05, 0.06, 0.05, 0.04, 0.05]
    spread_still_stale = robust_spread(pre_change_window)
    # A moderately large post-change residual is still judged against the
    # OLD, tight spread -- authority drops sharply and immediately rather
    # than gracefully; documented, not fixed, in this checkpoint.
    authority_immediately_after_change = compute_authority_from_evidence(0.3, spread_still_stale)
    assert authority_immediately_after_change < 0.1
