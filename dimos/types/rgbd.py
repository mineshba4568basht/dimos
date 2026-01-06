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
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RgbdFrame:
    """Camera-agnostic RGB-D frame for perception pipelines.

    Conventions:
    - `rgb` is RGB uint8 (H,W,3) aligned to `depth`.
    - `depth_m` is float32 depth in meters (H,W) aligned to `rgb`.
    - `intrinsics` is either [fx, fy, cx, cy] or a 3x3 matrix.
    """

    rgb: np.ndarray  # type: ignore[type-arg]
    depth_m: np.ndarray  # type: ignore[type-arg]
    intrinsics: Any  # [fx, fy, cx, cy] | 3x3
    frame_id: str = "camera_optical"
    ts: float | None = None

    def intrinsics_fx_fy_cx_cy(self) -> list[float]:
        if isinstance(self.intrinsics, list) and len(self.intrinsics) == 4:
            return [float(x) for x in self.intrinsics]
        k = np.array(self.intrinsics, dtype=np.float32)
        return [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])]



