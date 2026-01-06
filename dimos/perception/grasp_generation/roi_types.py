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
from typing import Literal, Tuple

import numpy as np

BBoxXYXY = Tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class UserBBoxSelection:
    """A user selection on an image.

    Notes:
    - `bbox_xyxy` is in pixel coordinates on the RGB image: (x1, y1, x2, y2).
    - `ts` is optional; if omitted the pipeline will treat it as "latest".
    """

    bbox_xyxy: BBoxXYXY
    ts: float | None = None
    source: Literal["ui", "rpc", "test"] = "ui"


def clamp_bbox_xyxy(bbox: BBoxXYXY, width: int, height: int) -> BBoxXYXY:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, max(0, width - 1)))
    x2 = float(np.clip(x2, 0, max(0, width - 1)))
    y1 = float(np.clip(y1, 0, max(0, height - 1)))
    y2 = float(np.clip(y2, 0, max(0, height - 1)))
    # enforce ordering
    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
    return (x_lo, y_lo, x_hi, y_hi)


def bbox_to_int_xyxy(bbox: BBoxXYXY) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


