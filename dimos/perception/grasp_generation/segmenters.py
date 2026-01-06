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

from dimos.perception.grasp_generation.roi_types import BBoxXYXY, bbox_to_int_xyxy, clamp_bbox_xyxy


class BBoxMaskSegmenter(Protocol):
    """Segment an object from an RGB image given a bbox prompt."""

    def predict_mask(self, rgb: np.ndarray, bbox_xyxy: BBoxXYXY) -> np.ndarray:  # type: ignore[type-arg]
        """Return a boolean mask (H,W) in the input image coordinates."""


@dataclass
class BBoxOnlySegmenter:
    """Fallback segmenter: returns the bbox rectangle as the mask.

    This keeps the pipeline working even when SAM is not installed.
    """

    def predict_mask(self, rgb: np.ndarray, bbox_xyxy: BBoxXYXY) -> np.ndarray:  # type: ignore[type-arg]
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = bbox_to_int_xyxy(clamp_bbox_xyxy(bbox_xyxy, w, h))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x2 = min(w, max(0, x2))
        y2 = min(h, max(0, y2))
        x1 = min(w, max(0, x1))
        y1 = min(h, max(0, y1))
        mask = np.zeros((h, w), dtype=bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
        return mask


@dataclass
class SamBBoxSegmenter:
    """SAM/MobileSAM bbox-prompt segmenter (optional dependency).

    This tries MobileSAM first, then falls back to Segment-Anything.

    Requirements (one of):
    - `mobile_sam` package providing `sam_model_registry` and `SamPredictor`
    - `segment_anything` package providing `sam_model_registry` and `SamPredictor`
    """

    checkpoint_path: str
    model_type: str = "vit_t"
    device: str = "cuda"

    _predictor: object | None = None
    _last_image_shape: tuple[int, int] | None = None

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return

        # Try MobileSAM then SAM
        sam_model_registry = None
        SamPredictor = None

        try:
            from mobile_sam import SamPredictor as _MSamPredictor  # type: ignore[import-not-found]
            from mobile_sam import sam_model_registry as _msam_model_registry  # type: ignore[import-not-found]

            SamPredictor = _MSamPredictor
            sam_model_registry = _msam_model_registry
        except Exception:
            from segment_anything import SamPredictor as _SamPredictor  # type: ignore[import-not-found]
            from segment_anything import sam_model_registry as _sam_model_registry  # type: ignore[import-not-found]

            SamPredictor = _SamPredictor
            sam_model_registry = _sam_model_registry

        if sam_model_registry is None or SamPredictor is None:
            raise RuntimeError("SAM backend unavailable")

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)

    def predict_mask(self, rgb: np.ndarray, bbox_xyxy: BBoxXYXY) -> np.ndarray:  # type: ignore[type-arg]
        self._ensure_loaded()
        assert self._predictor is not None

        h, w = rgb.shape[:2]
        bbox_xyxy = clamp_bbox_xyxy(bbox_xyxy, w, h)
        x1, y1, x2, y2 = bbox_xyxy

        # predictor expects RGB uint8
        if rgb.dtype != np.uint8:
            rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb_u8 = rgb

        # Cache set_image calls if image shape is stable (helps throughput)
        if self._last_image_shape != (h, w):
            self._predictor.set_image(rgb_u8)
            self._last_image_shape = (h, w)
        else:
            # Still set_image because content changed; SAM predictor caches embeddings.
            self._predictor.set_image(rgb_u8)

        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        masks, scores, _logits = self._predictor.predict(  # type: ignore[attr-defined]
            box=box[None, :],
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            return np.zeros((h, w), dtype=bool)

        best_idx = int(np.argmax(scores)) if scores is not None else 0
        mask = masks[best_idx]
        if mask.shape != (h, w):
            mask = mask.reshape(h, w)
        return mask.astype(bool)



