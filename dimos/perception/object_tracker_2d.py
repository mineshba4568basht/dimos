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

import time

import cv2
import numpy as np
from reactivex.disposable import Disposable

from dimos.core import In, Module, Out, rpc
from dimos.models.segmentation.edge_tam import EdgeTAMProcessor
from dimos.msgs.sensor_msgs import Image, ImageFormat
from dimos.msgs.std_msgs import Header
from dimos.msgs.vision_msgs import Detection2DArray
from dimos.perception.detection.type.detection2d.seg import Detection2DSeg
from dimos.utils.gpu_utils import is_cuda_available
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger(__name__)


class ObjectTracker2D(Module):
    """
    2D Object Tracker using EdgeTAM for segmentation/tracking.
    Replaces deprecated OpenCV tracker with robust neural tracking.
    """

    color_image: In[Image] = None  # type: ignore
    detection2darray: Out[Detection2DArray] = None  # type: ignore
    tracked_overlay: Out[Image] = None  # Visualization output # type: ignore

    def __init__(self, frame_id: str = "camera_link") -> None:
        super().__init__()
        self.frame_id = frame_id
        self.processor: EdgeTAMProcessor | None = None
        self.is_tracking = False
        self.pending_bbox: list[float] | None = None

    @rpc
    def start(self) -> None:
        super().start()

        try:
            self.processor = EdgeTAMProcessor(
                device="cuda" if is_cuda_available() else "cpu"
            )
            logger.info(f"EdgeTAM initialized on {self.processor.device}")
        except Exception as e:
            logger.error(f"Failed to initialize EdgeTAM: {e}")
            return

        # Process stream with backpressure to avoid lagging behind
        stream = backpressure(self.color_image.observable())
        self.disposables.add(stream.subscribe(self._process_frame))

        logger.info("ObjectTracker2D module started")

    @rpc
    def stop(self) -> None:
        self.is_tracking = False
        self.pending_bbox = None
        if self.processor:
            self.processor.stop()
        super().stop()

    @rpc
    def track(self, bbox: list[float]) -> dict:
        """
        Initialize tracking with a bounding box.

        Args:
            bbox: Bounding box in format [x1, y1, x2, y2]

        Returns:
            Dict containing tracking status
        """
        if not self.processor:
            return {"status": "not_initialized"}

        # Store bbox to be used on next frame
        self.pending_bbox = bbox
        self.is_tracking = False  # Pause tracking until re-initialization
        
        return {"status": "tracking_scheduled", "bbox": bbox}

    @rpc
    def stop_track(self) -> bool:
        """Stop tracking."""
        self.is_tracking = False
        self.pending_bbox = None
        return True

    @rpc
    def is_tracking(self) -> bool:
        return self.is_tracking

    def _process_frame(self, image: Image) -> None:
        if not self.processor:
            return

        # Handle initialization if pending
        if self.pending_bbox is not None:
            try:
                box = np.array(self.pending_bbox, dtype=np.float32)
                self.processor.init_track(image, box=box, obj_id=1)
                self.is_tracking = True
                self.pending_bbox = None
                logger.info(f"Initialized EdgeTAM tracking with bbox {box}")
            except Exception as e:
                logger.error(f"Failed to init tracking: {e}")
                self.pending_bbox = None
                return

        if not self.is_tracking:
            return

        try:
            # 1. Track
            detections_2d = self.processor.process_image(image)

            # 2. Publish Detection2DArray
            header = Header(image.ts, self.frame_id)
            
            ros_detections = []
            if detections_2d and detections_2d.detections:
                for det in detections_2d.detections:
                    if isinstance(det, Detection2DSeg):
                        ros_detections.append(det.to_ros_detection2d())
            
            det_array = Detection2DArray(
                detections_length=len(ros_detections),
                header=header,
                detections=ros_detections
            )
            self.detection2darray.publish(det_array)

            # 3. Publish Visualization
            self._publish_visualization(image, detections_2d)

        except Exception as e:
            logger.error(f"Tracking error: {e}")
            self.is_tracking = False

    def _publish_visualization(self, image: Image, detections) -> None:
        if not self.tracked_overlay.has_subscribers():
            return

        # Create visualization
        viz = image.to_opencv().copy()

        if detections and detections.detections:
            for det in detections.detections:
                if isinstance(det, Detection2DSeg):
                    # Draw mask
                    mask = det.mask.astype(bool)
                    if mask.any():
                        # Create colored mask (green)
                        color = np.array([0, 255, 0], dtype=np.uint8)
                        
                        # Blend mask
                        roi = viz[mask]
                        blended = (roi * 0.6 + color * 0.4).astype(np.uint8)
                        viz[mask] = blended

                        # Draw contour
                        contours, _ = cv2.findContours(
                            det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(viz, contours, -1, (0, 255, 0), 2)

                    # Draw bbox
                    x1, y1, x2, y2 = map(int, det.bbox)
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label
                    label = f"Track ID {det.track_id} ({det.confidence:.2f})"
                    cv2.putText(
                        viz, 
                        label, 
                        (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0), 
                        2
                    )

        viz_msg = Image.from_numpy(viz, format=ImageFormat.RGB)
        self.tracked_overlay.publish(viz_msg)
