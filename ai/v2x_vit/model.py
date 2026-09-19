"""
Concrete V2X-ViT cooperative multimodal feature fusion (Phase 4).

Implements the encoder + cross-vehicle self-attention stage `ai/v2x_vit/base.py`
specifies: a list of per-vehicle PointPillars spatial feature maps -> one fused
cooperative representation, via a standard Transformer encoder applied across
the VEHICLE dimension (not spatial pixels).

Scope: isolated feature-fusion only, exactly like `ai/pointpillars/model.py`.
UNTRAINED: weights are randomly initialized (fixed seed), never trained on
real or synthetic data. No checkpoint exists. No GRU integration, no
trust/criticality weighting, no communication-aware weighting, and no
camera/CSI/SNR/V2X-message fusion -- none of that data is available or wired
into this pipeline yet (see the Phase 4 V2X-ViT audit). Do not report this
module's output as evidence of cooperative-perception accuracy or a trained
model.

`configs/model.yaml`'s `perception.v2x_vit` hyperparameters (embed_dim=256,
num_heads=8, num_layers=4) are treated as ordinary Transformer architecture
choices, not empirically-derived physical parameters (unlike PointPillars'
`point_cloud_range`) -- kept exactly as configured per the Phase 4 V2X-ViT
audit, not changed here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ai.utils.config_loader import load_config
from ai.v2x_vit.base import BaseV2XViT

#: PointPillars' documented backbone output channel count (see
#: ai/pointpillars/model.py::SpatialBackbone) -- the expected channel
#: dimension of every per-vehicle feature map this module consumes.
DEFAULT_IN_CHANNELS = 128


class MissingVehicleFeaturesError(ValueError):
    """Raised when `fuse` has no usable per-vehicle features to fuse.

    Mirrors `ai/pointpillars/model.py`'s `MissingPointCloudError` -- callers
    must catch this and skip/report the frame rather than receive a
    fabricated fused representation for zero (or invalid) real vehicles.
    """


class _VehicleEncoder(nn.Module):
    """Per-vehicle PointPillars spatial feature map `(1, C, H, W)` -> a
    single fixed-size embedding `(1, embed_dim)`.

    Global-average-pools the spatial dimensions first (so it tolerates any
    PointPillars grid size, e.g. a future range/voxel change, without a
    shape mismatch), then linearly projects the channel count to
    `embed_dim`.
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)  # (1, C, H, W) -> (1, C)
        return self.proj(pooled)  # (1, embed_dim)


class V2XViT(BaseV2XViT):
    """Concrete `BaseV2XViT`: fuses a variable-length list of per-vehicle
    PointPillars feature maps into one cooperative representation via
    self-attention across the vehicle dimension.

    `fuse(per_vehicle_features)` expects a non-empty list of real
    PointPillars outputs (each `(1, C, H, W)`, as produced by
    `ai.pointpillars.model.PointPillars.extract_features`), one entry per
    real vehicle actually present in one shared real frame -- never a
    fabricated or padded placeholder for a missing vehicle. Returns a
    single deterministic tensor of shape `(1, embed_dim)`: the mean-pooled,
    self-attended cooperative representation across all N input vehicles.

    Padding/masking is intentionally NOT used: each call processes exactly
    one frame's real vehicle count N as a single attention sequence (not a
    padded batch of multiple frames with differing N), so there is nothing
    to mask. If a future caller needs to batch multiple frames with
    different vehicle counts in one call, that would require adding
    padding/masking then -- not assumed here.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        in_channels: int = DEFAULT_IN_CHANNELS,
        seed: int = 0,
    ):
        cfg = (config or load_config("model"))["perception"]["v2x_vit"]
        self.embed_dim: int = cfg["embed_dim"]
        self.num_heads: int = cfg["num_heads"]
        self.num_layers: int = cfg["num_layers"]
        self.in_channels = in_channels

        torch.manual_seed(seed)
        self.vehicle_encoder = _VehicleEncoder(in_channels=in_channels, embed_dim=self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=self.num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Inference-only: no training loop exists yet, so these stay in
        # eval() permanently -- same rationale as ai/pointpillars/model.py
        # (deterministic repeated calls on the same input).
        self.vehicle_encoder.eval()
        self.transformer.eval()

    def fuse(self, per_vehicle_features: list[Any]) -> torch.Tensor:
        """Raw list of real per-vehicle `(1, C, H, W)` PointPillars feature
        maps -> `(1, embed_dim)` fused cooperative representation. Raises
        `MissingVehicleFeaturesError` on an empty list or any
        malformed/non-finite entry -- never substitutes a fabricated
        vehicle or silently drops one.
        """
        if not per_vehicle_features:
            raise MissingVehicleFeaturesError(
                "fuse received an empty list -- at least one real vehicle's "
                "PointPillars feature map is required; never fabricate one."
            )
        for i, feat in enumerate(per_vehicle_features):
            if not torch.is_tensor(feat):
                raise MissingVehicleFeaturesError(
                    f"per_vehicle_features[{i}] is not a tensor: {type(feat)!r}"
                )
            if feat.ndim != 4 or feat.shape[0] != 1 or feat.shape[1] != self.in_channels:
                raise MissingVehicleFeaturesError(
                    f"per_vehicle_features[{i}] has shape {tuple(feat.shape)}; "
                    f"expected (1, {self.in_channels}, H, W)"
                )
            if not torch.isfinite(feat).all():
                raise MissingVehicleFeaturesError(
                    f"per_vehicle_features[{i}] contains non-finite (NaN/Inf) values"
                )

        with torch.no_grad():
            embeddings = [self.vehicle_encoder(feat) for feat in per_vehicle_features]  # N x (1, embed_dim)
            sequence = torch.cat(embeddings, dim=0).unsqueeze(0)  # (1, N, embed_dim)
            attended = self.transformer(sequence)  # (1, N, embed_dim)
            fused = attended.mean(dim=1)  # (1, embed_dim)
        return fused
