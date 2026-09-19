"""Focused tests for `ai.v2x_vit.model.V2XViT`.

Uses small synthetic PointPillars-shaped feature tensors purely to unit-test
the fusion code path in isolation (shape, masking-free variable-N handling,
determinism, error handling, and that cross-vehicle attention is actually
exercised). This is NOT a substitute for the real-PointPillars-output
validation performed manually against real Phase 2 LiDAR (Vehicle179,
Vehicle147, Vehicle6742, and a real multi-vehicle shared frame) -- see the
Phase 4 V2X-ViT implementation report for that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.v2x_vit.model import MissingVehicleFeaturesError, V2XViT

TEST_CONFIG = {
    "perception": {
        "v2x_vit": {"embed_dim": 32, "num_heads": 4, "num_layers": 2},
    }
}

#: Matches ai/pointpillars/model.py's SpatialBackbone output channels.
IN_CHANNELS = 128


def _make_v2x_vit() -> V2XViT:
    # Small embed_dim/heads/layers so tests run fast; the *shape* of the
    # config (embed_dim/num_heads/num_layers keys) matches the real project
    # config exactly -- only the magnitudes are reduced for test speed.
    return V2XViT(config=TEST_CONFIG, in_channels=IN_CHANNELS, seed=0)


def _make_feature_map(seed: int = 0, h: int = 6, w: int = 5) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(1, IN_CHANNELS, h, w)).astype(np.float32)
    return torch.from_numpy(arr)


def test_concrete_class_can_be_instantiated():
    model = _make_v2x_vit()
    assert model.embed_dim == 32


def test_single_vehicle_input_produces_expected_shape():
    model = _make_v2x_vit()
    out = model.fuse([_make_feature_map(seed=1)])
    assert out.shape == (1, 32)
    assert out.dtype == torch.float32


def test_multiple_vehicle_input_produces_expected_shape():
    model = _make_v2x_vit()
    feats = [_make_feature_map(seed=i) for i in range(5)]
    out = model.fuse(feats)
    assert out.shape == (1, 32)


def test_variable_vehicle_counts_all_produce_fixed_output_shape():
    model = _make_v2x_vit()
    for n in (1, 2, 3, 7, 12):
        feats = [_make_feature_map(seed=i) for i in range(n)]
        out = model.fuse(feats)
        assert out.shape == (1, 32), f"failed for N={n}"


def test_output_is_finite():
    model = _make_v2x_vit()
    feats = [_make_feature_map(seed=i) for i in range(4)]
    out = model.fuse(feats)
    assert torch.isfinite(out).all()


def test_output_is_deterministic_in_eval_mode():
    model = _make_v2x_vit()
    feats = [_make_feature_map(seed=i) for i in range(4)]
    out1 = model.fuse(feats)
    out2 = model.fuse(feats)
    assert torch.equal(out1, out2)


def test_empty_input_raises_missing_vehicle_features_error():
    model = _make_v2x_vit()
    with pytest.raises(MissingVehicleFeaturesError):
        model.fuse([])


def test_invalid_tensor_shape_is_rejected_clearly():
    model = _make_v2x_vit()
    with pytest.raises(MissingVehicleFeaturesError):
        model.fuse([torch.zeros(1, IN_CHANNELS, 6, 5), torch.zeros(2, IN_CHANNELS, 6, 5)])  # bad batch dim
    with pytest.raises(MissingVehicleFeaturesError):
        model.fuse([torch.zeros(1, IN_CHANNELS + 1, 6, 5)])  # wrong channel count
    with pytest.raises(MissingVehicleFeaturesError):
        model.fuse([torch.zeros(IN_CHANNELS, 6, 5)])  # wrong ndim


def test_non_finite_input_raises():
    model = _make_v2x_vit()
    bad = _make_feature_map(seed=2)
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(MissingVehicleFeaturesError):
        model.fuse([_make_feature_map(seed=1), bad])


def test_non_tensor_input_raises():
    model = _make_v2x_vit()
    with pytest.raises(MissingVehicleFeaturesError):
        model.fuse([_make_feature_map(seed=1), "not a tensor"])


def test_attention_operates_across_the_vehicle_dimension():
    """The transformer must actually transform each vehicle's token using
    the other vehicles' tokens -- not just pass per-vehicle embeddings
    through unchanged before pooling.
    """
    model = _make_v2x_vit()
    feats = [_make_feature_map(seed=i) for i in range(4)]

    with torch.no_grad():
        raw_embeddings = torch.cat([model.vehicle_encoder(f) for f in feats], dim=0).unsqueeze(0)
        attended = model.transformer(raw_embeddings)

    # If attention (and the rest of the encoder layer: FFN/residual/norm)
    # were a no-op, attended would equal raw_embeddings exactly.
    assert not torch.allclose(attended, raw_embeddings)


def test_output_changes_when_any_single_vehicle_changes():
    """A real (not merely independent-per-vehicle) fusion should have the
    fused output respond to a change in any one vehicle's input.
    """
    model = _make_v2x_vit()
    base_feats = [_make_feature_map(seed=i) for i in range(3)]
    out_base = model.fuse(base_feats)

    changed_feats = list(base_feats)
    changed_feats[1] = _make_feature_map(seed=99)
    out_changed = model.fuse(changed_feats)

    assert not torch.allclose(out_base, out_changed)
