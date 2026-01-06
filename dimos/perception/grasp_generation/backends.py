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
from typing import Protocol

import numpy as np


class GraspGeneratorBackend(Protocol):
    """Backend interface that returns grasps in Dimos' grasp dict format."""

    def generate_grasps(
        self,
        points_xyz: np.ndarray,  # type: ignore[type-arg]
        colors_rgb: np.ndarray | None = None,  # type: ignore[type-arg]
    ) -> list[dict]:  # type: ignore[type-arg]
        """Return grasps sorted by score descending.

        Expected grasp dict keys used across Dimos:
        - translation: [x, y, z] in meters (world or camera frame—consistent with your pipeline)
        - rotation_matrix: 3x3
        - score: float
        - width: float (meters)
        """


@dataclass
class AnyGraspBackend:
    """Local AnyGrasp backend (optional dependency).

    This is intentionally thin and only wires up AnyGrasp if the user has it installed.
    """

    checkpoint: str | None = None
    device: str = "cuda"
    collision_detection: bool = True

    def __post_init__(self) -> None:
        # Validate dependency at construction time for clearer errors.
        try:
            import anygrasp  # type: ignore[import-not-found]  # noqa: F401
        except Exception as e:
            raise ImportError(
                "AnyGrasp backend requires an `anygrasp` python package installed and importable."
            ) from e

    def generate_grasps(
        self, points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None
    ) -> list[dict]:  # type: ignore[type-arg]
        # NOTE: AnyGrasp APIs vary by fork. We keep this as a placeholder adapter.
        # You can patch this to match your AnyGrasp install (weights path + inference call).
        raise NotImplementedError(
            "AnyGrasp adapter not yet wired to a concrete AnyGrasp API in this repo. "
            "Install AnyGrasp and update `AnyGraspBackend.generate_grasps()` to call it."
        )



