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

"""ROI grasp pipeline: bbox → (Mobile)SAM mask → ROI pointcloud → cleanup → grasps.

This is designed to plug into Dimos the same way `ManipulationPipeline` does:
- Pure-python "pipeline" object
- Camera-agnostic as long as you provide aligned RGB + depth + intrinsics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dimos.perception.grasp_generation.backends import GraspGeneratorBackend
from dimos.perception.grasp_generation.grasp_filters import ParallelJawGripperModel, filter_grasps
from dimos.perception.grasp_generation.roi_cleanup import RoiCleanupConfig, cleanup_roi_pointcloud
from dimos.perception.grasp_generation.roi_types import BBoxXYXY, UserBBoxSelection
from dimos.perception.grasp_generation.segmenters import BBoxMaskSegmenter, BBoxOnlySegmenter
from dimos.perception.grasp_generation.utils import parse_grasp_results
from dimos.perception.pointcloud.utils import load_camera_matrix_from_yaml, create_point_cloud_and_extract_masks


@dataclass
class RoiGraspPipelineConfig:
    # Depth options
    depth_scale: float = 1.0
    depth_trunc: float = 2.0

    # Cleanup
    cleanup: RoiCleanupConfig = RoiCleanupConfig()

    # Filtering
    top_k: int = 10
    collision_check: bool = True
    collision_margin: float = 0.002
    gripper: ParallelJawGripperModel = ParallelJawGripperModel()


class RoiGraspPipeline:
    def __init__(
        self,
        camera_intrinsics: list[float] | np.ndarray | dict,  # type: ignore[type-arg]
        segmenter: BBoxMaskSegmenter | None = None,
        grasp_backend: GraspGeneratorBackend | None = None,
        config: RoiGraspPipelineConfig = RoiGraspPipelineConfig(),
        kinematic_feasibility: Callable[[dict], bool] | None = None,
    ) -> None:
        self.config = config
        self.segmenter = segmenter or BBoxOnlySegmenter()
        self.backend = grasp_backend
        self.kinematic_feasibility = kinematic_feasibility

        self.camera_matrix = load_camera_matrix_from_yaml(camera_intrinsics)
        if self.camera_matrix is None:
            raise ValueError("camera_intrinsics must be provided")

        # last selection (for stream usage)
        self._selection: UserBBoxSelection | None = None

    def set_selection(self, selection: UserBBoxSelection | BBoxXYXY | None) -> None:
        if selection is None:
            self._selection = None
            return
        if isinstance(selection, UserBBoxSelection):
            self._selection = selection
        else:
            self._selection = UserBBoxSelection(bbox_xyxy=selection, source="rpc")

    def process_frame(
        self,
        rgb: np.ndarray,  # type: ignore[type-arg]
        depth_m: np.ndarray,  # type: ignore[type-arg]
        selection: UserBBoxSelection | BBoxXYXY | None = None,
    ) -> dict:
        """Run the full ROI grasp pipeline on a single RGB-D frame."""
        try:
            import open3d as o3d  # type: ignore[import-untyped]
        except Exception as e:
            raise ImportError(
                "ROI grasp pipeline requires `open3d` installed (used for pointcloud ops). "
                "Install `open3d` in your runtime environment to use this pipeline."
            ) from e

        if selection is None:
            selection = self._selection
        elif not isinstance(selection, UserBBoxSelection):
            selection = UserBBoxSelection(bbox_xyxy=selection, source="rpc")

        if selection is None:
            return {
                "mask": None,
                "roi_pcd": o3d.geometry.PointCloud(),
                "roi_pcd_clean": o3d.geometry.PointCloud(),
                "grasps_raw": [],
                "grasps_topk": [],
                "full_pcd": o3d.geometry.PointCloud(),
            }

        bbox = selection.bbox_xyxy

        # 1) Segment with bbox prompt → mask
        mask = self.segmenter.predict_mask(rgb, bbox)

        # 2) Masked point cloud
        full_pcd, masked_pcds = create_point_cloud_and_extract_masks(
            color_img=rgb,
            depth_img=depth_m,
            masks=[mask],
            intrinsic=self.camera_matrix,  # type: ignore[arg-type]
            depth_scale=self.config.depth_scale,
            depth_trunc=self.config.depth_trunc,
        )
        roi_pcd = masked_pcds[0] if masked_pcds else o3d.geometry.PointCloud()

        # 3) Cleanup (remove leakage/outliers + keep main cluster)
        roi_pcd_clean = cleanup_roi_pointcloud(roi_pcd, self.config.cleanup)

        # 4) AnyGrasp (backend)
        grasps_raw: list[dict] = []
        if self.backend is not None and len(np.asarray(roi_pcd_clean.points)) > 0:
            pts = np.asarray(roi_pcd_clean.points).astype(np.float32)
            cols = (
                np.asarray(roi_pcd_clean.colors).astype(np.float32)
                if roi_pcd_clean.has_colors()
                else None
            )
            grasps = self.backend.generate_grasps(points_xyz=pts, colors_rgb=cols)
            # normalize into Dimos grasp format (id/score/width/translation/rotation_matrix)
            grasps_raw = parse_grasp_results(grasps)
            grasps_raw.sort(key=lambda g: float(g.get("score", 0.0)), reverse=True)

        # 5) Filter collision/kinematics and keep top-K
        grasps_topk = filter_grasps(
            grasps_raw,
            full_scene_pcd=full_pcd,
            collision_check=self.config.collision_check,
            kinematic_feasibility=self.kinematic_feasibility,
            gripper=self.config.gripper,
            collision_margin=self.config.collision_margin,
            top_k=self.config.top_k,
        )

        return {
            "mask": mask,
            "roi_pcd": roi_pcd,
            "roi_pcd_clean": roi_pcd_clean,
            "grasps_raw": grasps_raw,
            "grasps_topk": grasps_topk,
            "full_pcd": full_pcd,
        }


