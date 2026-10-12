from simulation.fault_injection.fault_types.v2x_consistency import (
    apply_conflicting_info,
    apply_misreported_position,
    apply_stale_cooperative_info,
)


def test_conflicting_info_zero_severity_no_conflicts(synthetic_frame, rng):
    result = apply_conflicting_info(synthetic_frame, severity=0.0, rng=rng)
    assert len(result["cooperative_info"]) == len(synthetic_frame["cooperative_info"])


def test_conflicting_info_full_severity_doubles_reports(synthetic_frame, rng):
    result = apply_conflicting_info(synthetic_frame, severity=1.0, rng=rng)
    assert len(result["cooperative_info"]) == 2 * len(synthetic_frame["cooperative_info"])


def test_stale_cooperative_info_ages_timestamps(synthetic_frame, rng):
    original_ts = synthetic_frame["cooperative_info"][0]["timestamp"]
    result = apply_stale_cooperative_info(synthetic_frame, severity=0.5, rng=rng)
    assert result["cooperative_info"][0]["timestamp"] < original_ts


def test_stale_cooperative_info_zero_severity_no_change(synthetic_frame, rng):
    original_ts = synthetic_frame["cooperative_info"][0]["timestamp"]
    result = apply_stale_cooperative_info(synthetic_frame, severity=0.0, rng=rng)
    assert result["cooperative_info"][0]["timestamp"] == original_ts


def test_misreported_position_zero_severity_no_change(synthetic_frame, rng):
    original_pos = synthetic_frame["cooperative_info"][0]["position"]
    result = apply_misreported_position(synthetic_frame, severity=0.0, rng=rng)
    assert result["cooperative_info"][0]["position"] == original_pos


def test_misreported_position_nonzero_severity_changes_position(synthetic_frame, rng):
    original_pos = synthetic_frame["cooperative_info"][0]["position"]
    result = apply_misreported_position(synthetic_frame, severity=0.7, rng=rng)
    assert result["cooperative_info"][0]["position"] != original_pos
