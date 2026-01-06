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

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.perception.grasp_generation.grasp_filters import grasp_collision_free_against_pointcloud
from dimos.perception.grasp_generation.roi_cleanup import RoiCleanupConfig, cleanup_roi_pointcloud
from dimos.perception.grasp_generation.roi_grasp_pipeline import RoiGraspPipeline
from dimos.perception.grasp_generation.segmenters import BBoxOnlySegmenter


def test_bbox_only_segmenter_mask_shape_and_area() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    seg = BBoxOnlySegmenter()
    mask = seg.predict_mask(rgb, (10, 20, 60, 70))
    assert mask.shape == (100, 200)
    assert mask.dtype == bool
    assert int(mask.sum()) == (70 - 20) * (60 - 10)


def test_cleanup_keeps_largest_cluster() -> None:
    # two clusters: one big, one small
    big = np.random.randn(500, 3).astype(np.float32) * 0.002 + np.array([0.0, 0.0, 0.5])
    small = np.random.randn(50, 3).astype(np.float32) * 0.002 + np.array([0.1, 0.1, 0.5])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([big, small]))

    cfg = RoiCleanupConfig(
        voxel_size=0.0,
        enable_statistical=False,
        enable_radius=False,
        enable_dbscan_main_cluster=True,
        dbscan_eps=0.01,
        dbscan_min_points=10,
    )
    out = cleanup_roi_pointcloud(pcd, cfg)
    assert len(np.asarray(out.points)) >= 400


def test_grasp_collision_proxy_detects_collision() -> None:
    # a grasp at origin with identity rotation, width 0.08.
    grasp = {
        "translation": [0.0, 0.0, 0.0],
        "rotation_matrix": np.eye(3).tolist(),
        "width": 0.08,
    }

    # Put points inside left finger volume (x ~ +width/2, y in [-finger_length,0], z ~ 0)
    pts = np.array([[0.04, -0.02, 0.0], [0.04, -0.03, 0.001]], dtype=np.float32)
    assert grasp_collision_free_against_pointcloud(grasp, pts) is False

    # Put points far away
    pts2 = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    assert grasp_collision_free_against_pointcloud(grasp, pts2) is True


def test_pipeline_runs_without_backend_and_returns_empty_grasps() -> None:
    h, w = 60, 80
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.ones((h, w), dtype=np.float32) * 0.5

    intr = [100.0, 100.0, w / 2, h / 2]
    pipe = RoiGraspPipeline(camera_intrinsics=intr, segmenter=BBoxOnlySegmenter(), grasp_backend=None)
    out = pipe.process_frame(rgb, depth, selection=(10, 10, 30, 30))
    assert out["mask"] is not None
    assert isinstance(out["grasps_raw"], list)
    assert isinstance(out["grasps_topk"], list)
    assert len(out["grasps_raw"]) == 0
    assert len(out["grasps_topk"]) == 0



