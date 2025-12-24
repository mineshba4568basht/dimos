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

import logging
from typing import Tuple

from reactivex import operators as ops
from reactivex.observable import Observable

from dimos.core import In, Module, Out, rpc
from dimos.msgs.geometry_msgs import PoseStamped, Transform, Vector3
from dimos.msgs.sensor_msgs import CameraInfo, Image
from dimos.msgs.vision_msgs import Detection2DArray
from dimos.perception.detection.type import ImageDetections2D
from dimos.perception.detection.type.detection2d.bbox import Detection2DBBox
from dimos.types.timestamped import align_timestamped
from dimos.utils.reactive import backpressure
from dimos.utils.logging_config import setup_logger
from dimos.msgs.nav_msgs.Path import Path
from dimos.protocol.skill import skill
from dimos.protocol.tf import TF

logger = setup_logger("dimos.perception.detection.person_tracker", level=logging.INFO)


class PersonTracker(Module):
    detections: In[Detection2DArray] = None  # type: ignore
    image: In[Image] = None  # type: ignore
    target: Out[Path] = None  # type: ignore

    camera_info: CameraInfo

    def __init__(self, cameraInfo: CameraInfo, arrival_threshold: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.camera_info = cameraInfo
        self.tf = TF()
        self._sub = None
        self._is_tracking = False
        self._arrival_threshold = arrival_threshold  # bbox bottom must be in bottom 30% of frame

    def center_to_3d(
        self,
        pixel: Tuple[int, int],
        camera_info: CameraInfo,
        assumed_depth: float = 1.0,
    ) -> Vector3:
        """Unproject 2D pixel coordinates to 3D position in camera_link frame.

        Args:
            camera_info: Camera calibration information
            assumed_depth: Assumed depth in meters (default 1.0m from camera)

        Returns:
            Vector3 position in camera_link frame coordinates (Z up, X forward)
        """
        # Extract camera intrinsics
        fx, fy = camera_info.K[0], camera_info.K[4]
        cx, cy = camera_info.K[2], camera_info.K[5]

        # Unproject pixel to normalized camera coordinates
        x_norm = (pixel[0] - cx) / fx
        y_norm = (pixel[1] - cy) / fy

        # Create 3D point at assumed depth in camera optical frame
        # Camera optical frame: X right, Y down, Z forward
        x_optical = x_norm * assumed_depth
        y_optical = y_norm * assumed_depth
        z_optical = assumed_depth

        # Transform from camera optical frame to camera_link frame
        # Optical: X right, Y down, Z forward
        # Link: X forward, Y left, Z up
        # Transformation: x_link = z_optical, y_link = -x_optical, z_link = -y_optical
        return Vector3(z_optical, -x_optical, -y_optical)

    def detections_stream(self) -> Observable[ImageDetections2D]:
        return backpressure(
            align_timestamped(
                self.image.pure_observable(),
                self.detections.pure_observable().pipe(
                    ops.filter(lambda d: d.detections_length > 0)  # type: ignore[attr-defined]
                ),
                match_tolerance=0.15,
                buffer_size=0.3,
            ).pipe(ops.map(lambda pair: ImageDetections2D.from_ros_detection2d_array(*pair)))
        )

    def check_arrival(self, detection: Detection2DBBox) -> bool:
        """Check if person bbox is near bottom of screen (arrived).

        Args:
            detection: Detection2DBBox with bbox and image

        Returns:
            True if bbox bottom edge is in bottom threshold% of frame
        """
        x1, y1, x2, y2 = detection.bbox
        image_height = self.camera_info.height

        # Check if bottom of bbox is in bottom portion of frame
        bottom_position = y2 / image_height
        is_arrived = bottom_position >= self._arrival_threshold

        if is_arrived:
            logger.info(f"ARRIVAL DETECTED: bbox bottom at {bottom_position:.2%} of frame height")

        return is_arrived

    @skill()
    def start_tracking(self):
        """Start person tracking."""
        if not self._is_tracking:
            self._sub = self.detections_stream().subscribe(self.track)
            self._is_tracking = True
            logger.info("PersonTracker: Tracking started")
        return "Person tracking started"

    @skill()
    def stop_tracking(self):
        """Stop person tracking."""
        if self._sub:
            self._sub.dispose()
            self._is_tracking = False
            logger.info("PersonTracker: Tracking stopped")
        return "Person tracking stopped"

    @rpc
    def stop(self):
        super().stop()

    def is_tracking(self) -> bool:
        """Check if currently tracking a person."""
        return self._is_tracking

    def track(self, detections2D: ImageDetections2D):
        logger.info(
            f"PersonTracker.track() called with {len(detections2D)} detections at ts={detections2D.ts:.3f}"
        )
        print(detections2D)

        if len(detections2D) == 0:
            logger.warning("PersonTracker: No detections, skipping")
            return

        target = max(detections2D.detections, key=lambda det: det.bbox_2d_volume())
        logger.info(
            f"PersonTracker: Selected target person - center={target.center_bbox}, "
            f"bbox_volume={target.bbox_2d_volume():.1f}px"
        )

        # Check if arrived at person
        if self.check_arrival(target):
            logger.info("Person reached, stopping tracker")
            self.stop_tracking()
            return

        vector = self.center_to_3d(target.center_bbox, self.camera_info, 3.0)
        logger.info(
            f"PersonTracker: 3D position in camera_link: x={vector.x:.3f}, y={vector.y:.3f}, z={vector.z:.3f}"
        )

        pose_in_camera = PoseStamped(
            ts=detections2D.ts,
            position=vector,
            frame_id="camera_link",
        )

        logger.info(f"PersonTracker: Looking up TF world->camera_link at ts={detections2D.ts:.3f}")
        tf_world_to_camera = self.tf.get("world", "camera_link", detections2D.ts, 1.0)
        if not tf_world_to_camera:
            logger.error(
                f"PersonTracker: TF lookup FAILED! world->camera_link at ts={detections2D.ts:.3f} "
                f"(tolerance=1.0s) - NO GOAL PUBLISHED"
            )
            return

        logger.info(
            f"PersonTracker: TF lookup succeeded - "
            f"translation=({tf_world_to_camera.translation.x:.2f}, "
            f"{tf_world_to_camera.translation.y:.2f}, "
            f"{tf_world_to_camera.translation.z:.2f})"
        )

        tf_camera_to_target = Transform.from_pose("target", pose_in_camera)
        tf_world_to_target = tf_world_to_camera + tf_camera_to_target
        pose_in_world = tf_world_to_target.to_pose(ts=detections2D.ts)

        logger.info(
            f"PersonTracker: PUBLISHING GOAL - world position=("
            f"{pose_in_world.position.x:.2f}, "
            f"{pose_in_world.position.y:.2f}, "
            f"{pose_in_world.position.z:.2f})"
        )

        path = Path(poses=[pose_in_world])

        self.target.publish(path)
