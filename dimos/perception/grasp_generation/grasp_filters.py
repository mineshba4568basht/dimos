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
from typing import Callable

import numpy as np


@dataclass
class ParallelJawGripperModel:
    """Very simple geometric proxy for collision checks.

    Coordinate convention matches `dimos.perception.grasp_generation.utils`:
    - X: gripper width direction (opening/closing)
    - Y: finger length direction (fingers extend along +Y, but fingertips are at origin)
    - Z: approach direction (toward object), handle extends along -Z

    In that convention, the fingers occupy a volume behind the fingertips (negative Y).
    """

    finger_length: float = 0.08
    finger_thickness: float = 0.004
    finger_width: float = 0.006

    # simple palm/base behind the fingers
    base_thickness: float = 0.006
    base_back: float = 0.01


def _transform_points_to_gripper_frame(points_world: np.ndarray, grasp: dict) -> np.ndarray:
    R = np.array(grasp.get("rotation_matrix", np.eye(3)), dtype=np.float32)
    t = np.array(grasp.get("translation", [0, 0, 0]), dtype=np.float32).reshape(3)
    # world -> gripper: p_g = R^T (p_w - t)
    return (R.T @ (points_world - t).T).T


def _points_in_aabb(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all((points >= lo) & (points <= hi), axis=1)


def grasp_collision_free_against_pointcloud(
    grasp: dict,
    scene_points_world: np.ndarray,
    gripper: ParallelJawGripperModel = ParallelJawGripperModel(),
    collision_margin: float = 0.002,
) -> bool:
    """Conservative collision check.

    This checks whether any scene points fall inside the *solid volumes* of either finger or
    the base/palm region. It intentionally ignores points between the fingers near the tips
    (where the object should be).
    """

    if scene_points_world.size == 0:
        return True

    p_g = _transform_points_to_gripper_frame(scene_points_world, grasp)

    width = float(grasp.get("width", 0.08))

    # finger boxes in gripper frame
    # left finger near +X, right near -X
    fw = gripper.finger_width / 2 + collision_margin
    ft = gripper.finger_thickness / 2 + collision_margin
    fl = gripper.finger_length + collision_margin

    # Finger spans Y in [-finger_length, 0] (behind fingertips)
    y_lo = -fl
    y_hi = 0.0 + collision_margin

    # Left finger AABB
    lx = width / 2.0
    left_lo = np.array([lx - fw, y_lo, -ft], dtype=np.float32)
    left_hi = np.array([lx + fw, y_hi, ft], dtype=np.float32)

    # Right finger AABB
    rx = -width / 2.0
    right_lo = np.array([rx - fw, y_lo, -ft], dtype=np.float32)
    right_hi = np.array([rx + fw, y_hi, ft], dtype=np.float32)

    # Base/palm region: a thin slab connecting fingers, slightly behind them
    base_y_lo = -fl - gripper.base_back
    base_y_hi = -fl + gripper.base_thickness
    base_lo = np.array([rx - fw, base_y_lo, -ft], dtype=np.float32)
    base_hi = np.array([lx + fw, base_y_hi, ft], dtype=np.float32)

    coll = _points_in_aabb(p_g, left_lo, left_hi) | _points_in_aabb(p_g, right_lo, right_hi) | _points_in_aabb(
        p_g, base_lo, base_hi
    )

    return not bool(np.any(coll))


def filter_grasps(
    grasps: list[dict],
    full_scene_pcd=None,  # type: ignore[no-untyped-def]
    collision_check: bool = True,
    kinematic_feasibility: Callable[[dict], bool] | None = None,
    gripper: ParallelJawGripperModel = ParallelJawGripperModel(),
    collision_margin: float = 0.002,
    top_k: int = 10,
) -> list[dict]:
    """Filter and keep top-K grasps (already assumed sorted high→low score)."""

    out: list[dict] = []

    scene_points = (
        np.asarray(full_scene_pcd.points).astype(np.float32)
        if (collision_check and full_scene_pcd is not None and hasattr(full_scene_pcd, "points"))
        else np.empty((0, 3), dtype=np.float32)
    )

    for g in grasps:
        if kinematic_feasibility is not None and not kinematic_feasibility(g):
            continue

        if collision_check and full_scene_pcd is not None:
            if not grasp_collision_free_against_pointcloud(
                g, scene_points_world=scene_points, gripper=gripper, collision_margin=collision_margin
            ):
                continue

        out.append(g)
        if top_k > 0 and len(out) >= top_k:
            break

    return out


