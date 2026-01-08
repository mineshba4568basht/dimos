# Copyright 2025-2026 Dimensional Inc.
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
from typing import Any

from dimos_lcm.vision_msgs.Detection2DArray import (
    Detection2DArray as LCMDetection2DArray,
)

from dimos.types.timestamped import to_timestamp


class Detection2DArray(LCMDetection2DArray):  # type: ignore[misc]
    msg_name = "vision_msgs.Detection2DArray"

    # for _get_field_type() to work when decoding in _decode_one()
    __annotations__ = LCMDetection2DArray.__annotations__

    @property
    def ts(self) -> float:
        return to_timestamp(self.header.stamp)

    def to_rerun(self) -> Any:
        """Convert to Rerun Boxes2D archetype for visualization.

        Returns:
            rr.Boxes2D archetype with labeled 2D bounding boxes
        """
        import rerun as rr

        if self.detections_length == 0:
            return rr.Boxes2D(centers=[], sizes=[])

        centers = []
        sizes = []
        labels = []
        class_ids = []

        for det in self.detections:
            # bbox has center.position (x, y) and size_x, size_y
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            centers.append([cx, cy])
            sizes.append([w, h])

            # Get label and confidence from results
            if det.results_length > 0 and len(det.results) > 0:
                hyp = det.results[0].hypothesis
                score = hyp.score
                class_id = hyp.class_id
                labels.append(f"{class_id} ({score:.0%})")
                class_ids.append(hash(class_id) % 256)
            else:
                labels.append(det.id if det.id else "unknown")
                class_ids.append(0)

        return rr.Boxes2D(
            centers=centers,
            sizes=sizes,
            labels=labels,
            class_ids=class_ids,
        )
