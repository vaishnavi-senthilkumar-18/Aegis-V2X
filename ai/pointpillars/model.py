"""
Concrete PointPillars spatial feature extractor (Phase 4).

Implements the encoder + pseudo-image backbone stages of Zhou & Tuzel
(2019), "PointPillars: Fast Encoders for Object Detection from Point
Clouds", restricted to what `ai/pointpillars/base.py` actually specifies:
raw LiDAR point cloud -> pillarized spatial feature map, consumed by
V2X-ViT (see `ai/v2x_vit/base.py`). No detection head is implemented --
Phase 4's own interface contract describes PointPillars as a spatial
*feature extractor* feeding cooperative fusion, not a standalone detector,
and there is no bounding-box ground truth in the dataset to supervise one.

UNTRAINED: `PillarFeatureNet`/`SpatialBackbone` weights are randomly
initialized (PyTorch defaults, fixed seed for reproducibility) and have
never been trained on real or synthetic data. No checkpoint exists. This
module produces a correctly-shaped, deterministic spatial feature tensor
from real LiDAR -- nothing more. Do not report its output as evidence of
detection accuracy or a trained model.

Pillar geometry (voxel_size, point_cloud_range) is read from
`configs/model.yaml`'s `perception.pointpillars` section -- the only
PointPillars hyperparameters the project has actually specified.
`max_points_per_pillar` and `max_pillars` are NOT specified anywhere in
the repo's config; the values below are the standard defaults from the
original paper's KITTI configuration, used here as a documented
implementation choice, not a project requirement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from ai.pointpillars.base import BasePointPillars
from ai.utils.config_loader import load_config

#: Standard PointPillars (Zhou & Tuzel 2019) KITTI defaults -- not present
#: in configs/model.yaml, so pinned here explicitly rather than guessed
#: per-call.
DEFAULT_MAX_POINTS_PER_PILLAR = 32
DEFAULT_MAX_PILLARS = 12000

#: [x, y, z, intensity] + [xc, yc, zc] (offset from pillar's point-mean) +
#: [xp, yp] (offset from pillar's geometric center) -- the standard
#: PointPillars per-point augmented feature vector.
AUGMENTED_POINT_DIM = 9


class MissingPointCloudError(ValueError):
    """Raised when `extract_features` has no usable point cloud to run on.

    Callers must catch this and skip/report the frame -- this module never
    fabricates a substitute point cloud or a zero-filled feature map for a
    genuinely missing/corrupt LiDAR read. `ai/perception/data_loader.py`
    already returns `lidar_points=None` for exactly this case (e.g.
    vehicle192/frame_122416 in straight_road_dense_clear_day_Scene00, a
    real corrupted .npz file) -- this exception is what a caller passing
    that `None` straight through will hit.
    """


def _pillarize(
    points: np.ndarray,
    point_cloud_range: tuple[float, float, float, float, float, float],
    voxel_size: tuple[float, float, float],
    max_points_per_pillar: int,
    max_pillars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin raw points into (x, y) pillars and build the augmented per-point
    features the PointNet-style pillar encoder expects.

    Returns
    -------
    pillar_features : (max_pillars, max_points_per_pillar, 9) float32
        Per-point [x, y, z, intensity, xc, yc, zc, xp, yp], zero-padded.
    pillar_mask : (max_pillars, max_points_per_pillar) float32
        1.0 for a real point, 0.0 for padding -- used to exclude padding
        from the max-pool over points-in-a-pillar.
    pillar_coords : (max_pillars, 2) int64
        (row, col) grid index of each occupied pillar. Unused rows (beyond
        however many pillars were actually occupied, up to `max_pillars`)
        are (-1, -1) and must be skipped when scattering.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = point_cloud_range
    vx, vy, _vz = voxel_size

    in_range = (
        (points[:, 0] >= xmin) & (points[:, 0] < xmax)
        & (points[:, 1] >= ymin) & (points[:, 1] < ymax)
        & (points[:, 2] >= zmin) & (points[:, 2] < zmax)
    )
    pts = points[in_range]

    n_cols = int(round((xmax - xmin) / vx))
    n_rows = int(round((ymax - ymin) / vy))

    pillar_features = np.zeros((max_pillars, max_points_per_pillar, AUGMENTED_POINT_DIM), dtype=np.float32)
    pillar_mask = np.zeros((max_pillars, max_points_per_pillar), dtype=np.float32)
    pillar_coords = np.full((max_pillars, 2), -1, dtype=np.int64)

    if pts.shape[0] == 0:
        return pillar_features, pillar_mask, pillar_coords

    col_idx = np.clip(((pts[:, 0] - xmin) / vx).astype(np.int64), 0, n_cols - 1)
    row_idx = np.clip(((pts[:, 1] - ymin) / vy).astype(np.int64), 0, n_rows - 1)

    pillar_key = row_idx * n_cols + col_idx
    order = np.argsort(pillar_key, kind="stable")
    pts_sorted = pts[order]
    row_sorted = row_idx[order]
    col_sorted = col_idx[order]
    key_sorted = pillar_key[order]

    _unique_keys, start_idx, counts = np.unique(key_sorted, return_index=True, return_counts=True)
    num_pillars = min(len(start_idx), max_pillars)

    for p_i in range(num_pillars):
        s = start_idx[p_i]
        c = counts[p_i]
        take = min(c, max_points_per_pillar)
        group = pts_sorted[s : s + take]

        row = int(row_sorted[s])
        col = int(col_sorted[s])
        x_center = xmin + (col + 0.5) * vx
        y_center = ymin + (row + 0.5) * vy

        xyz_mean = group[:, :3].mean(axis=0)
        xc = group[:, :3] - xyz_mean
        xp = group[:, 0:1] - x_center
        yp = group[:, 1:2] - y_center

        pillar_features[p_i, :take, :] = np.concatenate([group, xc, xp, yp], axis=1)
        pillar_mask[p_i, :take] = 1.0
        pillar_coords[p_i] = (row, col)

    return pillar_features, pillar_mask, pillar_coords


class PillarFeatureNet(nn.Module):
    """Simplified PointNet: per-point Linear+BN+ReLU, then max-pool over
    the (masked) points within each pillar -> one feature vector/pillar.
    """

    def __init__(self, in_channels: int = AUGMENTED_POINT_DIM, out_channels: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, pillar_features: torch.Tensor, pillar_mask: torch.Tensor) -> torch.Tensor:
        num_pillars, num_points, dim = pillar_features.shape
        x = self.linear(pillar_features.reshape(num_pillars * num_points, dim))
        x = self.bn(x)
        x = torch.relu(x)
        x = x.reshape(num_pillars, num_points, -1)
        x = x * pillar_mask.unsqueeze(-1)  # zero out padding before pooling
        pooled, _ = x.max(dim=1)  # (num_pillars, out_channels)
        return pooled


class PointPillarsScatter(nn.Module):
    """Scatters per-pillar feature vectors back into a (C, H, W)
    pseudo-image at their real (row, col) grid position; empty grid cells
    stay zero.
    """

    def __init__(self, channels: int, n_rows: int, n_cols: int):
        super().__init__()
        self.channels = channels
        self.n_rows = n_rows
        self.n_cols = n_cols

    def forward(self, pillar_features: torch.Tensor, pillar_coords: np.ndarray) -> torch.Tensor:
        canvas = torch.zeros(self.channels, self.n_rows, self.n_cols, dtype=pillar_features.dtype)
        valid = pillar_coords[:, 0] >= 0
        rows = pillar_coords[valid, 0]
        cols = pillar_coords[valid, 1]
        canvas[:, rows, cols] = pillar_features[valid].t()
        return canvas.unsqueeze(0)  # (1, C, H, W)


class SpatialBackbone(nn.Module):
    """Small 2D CNN over the pseudo-image, producing the downstream spatial
    feature representation V2X-ViT consumes. Two stride-2 conv blocks --
    deliberately small since this is Phase 4's feature-extraction stage,
    not a full detection backbone.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointPillars(BasePointPillars):
    """Concrete `BasePointPillars`: raw LiDAR point cloud -> spatial
    feature map. See module docstring for scope (encoder + backbone only,
    no detection head; untrained weights).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        max_points_per_pillar: int = DEFAULT_MAX_POINTS_PER_PILLAR,
        max_pillars: int = DEFAULT_MAX_PILLARS,
        seed: int = 0,
    ):
        cfg = (config or load_config("model"))["perception"]["pointpillars"]
        self.point_cloud_range: tuple[float, float, float, float, float, float] = tuple(cfg["point_cloud_range"])
        self.voxel_size: tuple[float, float, float] = tuple(cfg["voxel_size"])
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        xmin, ymin, _zmin, xmax, ymax, _zmax = self.point_cloud_range
        vx, vy, _vz = self.voxel_size
        self.n_cols = int(round((xmax - xmin) / vx))
        self.n_rows = int(round((ymax - ymin) / vy))

        torch.manual_seed(seed)
        self.pillar_feature_net = PillarFeatureNet(in_channels=AUGMENTED_POINT_DIM, out_channels=64)
        self.scatter = PointPillarsScatter(channels=64, n_rows=self.n_rows, n_cols=self.n_cols)
        self.backbone = SpatialBackbone(in_channels=64, out_channels=128)
        # Inference-only: no training loop exists yet, so these stay in
        # eval() permanently -- BatchNorm uses its (untrained, default)
        # running stats rather than per-call batch statistics, which is
        # what makes repeated calls on the same input deterministic.
        self.pillar_feature_net.eval()
        self.backbone.eval()

    def extract_features(self, lidar_point_cloud: Any) -> torch.Tensor:
        """Raw (N, 3|4) point cloud -> (1, 128, H/4, W/4) spatial feature
        tensor. Raises `MissingPointCloudError` on `None`, empty, wrong-shape,
        or non-finite input -- never fabricates a substitute.
        """
        if lidar_point_cloud is None:
            raise MissingPointCloudError(
                "extract_features received None -- the caller must skip this "
                "frame rather than pass a missing/corrupt LiDAR read through."
            )
        points = np.asarray(lidar_point_cloud)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] not in (3, 4):
            raise MissingPointCloudError(
                f"invalid point cloud shape {points.shape}; expected (N, 3) or (N, 4) with N > 0"
            )
        if not np.all(np.isfinite(points)):
            raise MissingPointCloudError("point cloud contains non-finite (NaN/Inf) values")
        if points.shape[1] == 3:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=points.dtype)], axis=1)

        pillar_features_np, pillar_mask_np, pillar_coords = _pillarize(
            points.astype(np.float32),
            self.point_cloud_range,
            self.voxel_size,
            self.max_points_per_pillar,
            self.max_pillars,
        )
        pillar_features = torch.from_numpy(pillar_features_np)
        pillar_mask = torch.from_numpy(pillar_mask_np)

        with torch.no_grad():
            pooled = self.pillar_feature_net(pillar_features, pillar_mask)
            pseudo_image = self.scatter(pooled, pillar_coords)
            spatial_features = self.backbone(pseudo_image)
        return spatial_features
