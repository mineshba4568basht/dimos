# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoiCleanupConfig:
    """3D cleanup config for cluttered scenes.

    Strategy:
    - Downsample to keep AnyGrasp fast and stable.
    - Remove statistical + radius outliers.
    - DBSCAN cluster and keep the largest cluster (often removes leakage into neighbors/background).
    """

    voxel_size: float = 0.005
    enable_statistical: bool = True
    statistical_nb_neighbors: int = 30
    statistical_std_ratio: float = 2.0
    enable_radius: bool = True
    radius_nb_points: int = 20
    radius_radius: float = 0.02

    enable_dbscan_main_cluster: bool = True
    dbscan_eps: float = 0.02
    dbscan_min_points: int = 50


def _safe_len_points(pcd) -> int:  # type: ignore[no-untyped-def]
    try:
        return int(len(np.asarray(pcd.points)))
    except Exception:
        return 0


def keep_largest_dbscan_cluster(
    pcd, eps: float, min_points: int  # type: ignore[no-untyped-def]
):
    import open3d as o3d  # type: ignore[import-untyped]

    if _safe_len_points(pcd) == 0:
        return o3d.geometry.PointCloud()

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.size == 0:
        return pcd

    # ignore noise (-1)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return pcd

    # pick largest label
    largest = int(np.bincount(valid).argmax())
    idx = np.where(labels == largest)[0]
    if idx.size == 0:
        return pcd
    return pcd.select_by_index(idx.tolist())


def cleanup_roi_pointcloud(
    pcd, cfg: RoiCleanupConfig = RoiCleanupConfig()  # type: ignore[no-untyped-def]
):
    import open3d as o3d  # type: ignore[import-untyped]

    if _safe_len_points(pcd) == 0:
        return o3d.geometry.PointCloud()

    cur = o3d.geometry.PointCloud(pcd)

    if cfg.voxel_size > 0:
        cur = cur.voxel_down_sample(cfg.voxel_size)

    if cfg.enable_statistical and _safe_len_points(cur) > 0:
        cur, _ = cur.remove_statistical_outlier(
            nb_neighbors=cfg.statistical_nb_neighbors, std_ratio=cfg.statistical_std_ratio
        )

    if cfg.enable_radius and _safe_len_points(cur) > 0:
        cur, _ = cur.remove_radius_outlier(nb_points=cfg.radius_nb_points, radius=cfg.radius_radius)

    if cfg.enable_dbscan_main_cluster and _safe_len_points(cur) > 0:
        cur = keep_largest_dbscan_cluster(cur, eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points)

    return cur


