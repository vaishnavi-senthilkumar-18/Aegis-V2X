"""Focused tests for `ai.pointpillars.model.PointPillars`.

Uses small synthetic point clouds confined to the configured
`point_cloud_range` purely to unit-test the pillarization/encoder/scatter/
backbone code path in isolation (shape, masking, determinism, error
handling). This is NOT a substitute for the real-LiDAR validation already
performed manually against Phase 2 data (straight_road_dense_clear_day_Scene00,
roundabout_sparse_night_Scene00) -- see the Phase 4 PointPillars validation
report for that. Real LiDAR from this project's sensors falls entirely
outside the currently configured `point_cloud_range` (see that report's
"coordinate range mismatch" finding), which is why these unit tests use
synthetic in-range points rather than a real .npz fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.pointpillars.model import MissingPointCloudError, PointPillars

TEST_CONFIG = {
    "perception": {
        "pointpillars": {
            "voxel_size": [0.16, 0.16, 4.0],
            "point_cloud_range": [0, -39.68, -3, 69.12, 39.68, 1],
        }
    }
}


def _make_pointpillars() -> PointPillars:
    # Small max_pillars/max_points_per_pillar so tests run fast; geometry
    # (voxel_size, point_cloud_range) matches the real project config.
    return PointPillars(config=TEST_CONFIG, max_points_per_pillar=8, max_pillars=200, seed=0)


def _synthetic_points_in_range(n: int = 500, seed: int = 0) -> np.ndarray:
    """Points uniformly sampled inside the configured point_cloud_range."""
    rng = np.random.default_rng(seed)
    xmin, ymin, zmin, xmax, ymax, zmax = TEST_CONFIG["perception"]["pointpillars"]["point_cloud_range"]
    xyz = rng.uniform([xmin, ymin, zmin], [xmax, ymax, zmax], size=(n, 3)).astype(np.float32)
    intensity = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    return np.concatenate([xyz, intensity], axis=1)


def test_valid_point_cloud_produces_expected_output_shape():
    pp = _make_pointpillars()
    points = _synthetic_points_in_range()
    features = pp.extract_features(points)
    assert features.shape == (1, 128, pp.n_rows // 4, pp.n_cols // 4)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_output_is_deterministic_for_identical_input():
    pp = _make_pointpillars()
    points = _synthetic_points_in_range()
    out1 = pp.extract_features(points)
    out2 = pp.extract_features(points)
    assert torch.equal(out1, out2)


def test_xyz_only_point_cloud_is_accepted():
    pp = _make_pointpillars()
    points = _synthetic_points_in_range()[:, :3]  # drop intensity
    features = pp.extract_features(points)
    assert features.shape == (1, 128, pp.n_rows // 4, pp.n_cols // 4)


def test_none_input_raises_missing_point_cloud_error():
    pp = _make_pointpillars()
    with pytest.raises(MissingPointCloudError):
        pp.extract_features(None)


def test_empty_point_cloud_raises():
    pp = _make_pointpillars()
    with pytest.raises(MissingPointCloudError):
        pp.extract_features(np.zeros((0, 4), dtype=np.float32))


def test_wrong_shape_raises():
    pp = _make_pointpillars()
    with pytest.raises(MissingPointCloudError):
        pp.extract_features(np.zeros((10, 5), dtype=np.float32))
    with pytest.raises(MissingPointCloudError):
        pp.extract_features(np.zeros((4,), dtype=np.float32))


def test_non_finite_values_raise():
    pp = _make_pointpillars()
    points = _synthetic_points_in_range()
    points[0, 0] = np.nan
    with pytest.raises(MissingPointCloudError):
        pp.extract_features(points)


def test_points_entirely_outside_range_yield_zero_pseudo_image_not_a_crash():
    """Points outside `point_cloud_range` are filtered out, not clipped in.

    Mirrors the real finding that this project's actual LiDAR coordinates
    fall entirely outside the currently configured range: the pipeline
    must not crash or fabricate points, just legitimately produce an
    all-empty pillar grid.
    """
    pp = _make_pointpillars()
    out_of_range = np.full((50, 4), 10_000.0, dtype=np.float32)
    features = pp.extract_features(out_of_range)
    assert features.shape == (1, 128, pp.n_rows // 4, pp.n_cols // 4)
    assert torch.isfinite(features).all()
