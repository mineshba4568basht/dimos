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

from dimos_lcm.vision_msgs.Detection3DArray import (
    Detection3DArray as LCMDetection3DArray,
)


class Detection3DArray(LCMDetection3DArray):  # type: ignore[misc]
    msg_name = "vision_msgs.Detection3DArray"

    @property
    def frame_id(self) -> str:
        """Get frame_id from header for Rerun transform association."""
        return getattr(self.header, "frame_id", "map")

    def to_rerun(self) -> Any:
        """Convert to Rerun Boxes3D, one per detection with unique entity paths.

        Returns:
            List of (entity_path, rr.Boxes3D) tuples for each detection
        """
        import rerun as rr

        if self.detections_length == 0:
            return []

        results = []
        for det in self.detections:
            pos = det.bbox.center.position
            orient = det.bbox.center.orientation
            size = det.bbox.size

            # Get label from results
            label = "unknown"
            if det.results_length > 0 and len(det.results) > 0:
                hyp = det.results[0].hypothesis
                label = f"{hyp.class_id} ({hyp.score:.0%})"
            elif det.id:
                label = det.id

            entity_path = det.id if det.id else f"det_{id(det)}"

            half = [size.x * 0.5, size.y * 0.5, size.z * 0.5]
            box = rr.Boxes3D(
                centers=[[pos.x, pos.y, pos.z]],
                half_sizes=[half],
                quaternions=[[orient.x, orient.y, orient.z, orient.w]],
                labels=[label],
            )
            results.append((entity_path, box))

        return results
