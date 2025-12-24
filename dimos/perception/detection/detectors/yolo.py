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

from typing import Optional

from ultralytics import YOLO

from dimos.msgs.sensor_msgs import Image
from dimos.perception.detection.detectors.types import Detector
from dimos.perception.detection.type import ImageDetections2D
from dimos.utils.data import get_data
from dimos.utils.gpu_utils import is_cuda_available
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.perception.detection.yolo_2d_det")


class Yolo2DDetector(Detector):
    def __init__(
        self,
        model_path: str = "models_yolo",
        model_name: str = "yolo11n.pt",
        device: Optional[str] = None,
        conf: float = 0.5,
        iou: float = 0.6,
    ):
        """Initialize YOLO detector.

        Args:
            model_path: Path to model directory (default: "models_yolo")
            model_name: Model filename (default: "yolo11n.pt")
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            conf: Confidence threshold for detections (default: 0.5)
            iou: IoU threshold for NMS (default: 0.6)
        """
        super().__init__()

        self.model = YOLO(
            get_data(model_path) / model_name,
            task="detect",
        )

        if device:
            self.device = device
        elif is_cuda_available():
            self.device = "cuda"
            logger.debug("Using CUDA for YOLO 2d detector")
        else:
            self.device = "cpu"
            logger.debug("Using CPU for YOLO 2d detector")

        self.conf = conf
        self.iou = iou

    def process_image(self, image: Image) -> ImageDetections2D:
        return ImageDetections2D.from_ultralytics_result(
            image,
            self.model.track(
                source=image.to_opencv(),
                device=self.device,
                conf=self.conf,
                iou=self.iou,
                persist=True,
                verbose=False,
            ),
        )

    def stop(self):
        """Clean up resources used by the detector, including tracker threads."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            predictor = self.model.predictor
            if hasattr(predictor, "trackers") and predictor.trackers:
                for tracker in predictor.trackers:
                    if hasattr(tracker, "tracker") and hasattr(tracker.tracker, "gmc"):
                        gmc = tracker.tracker.gmc
                        if hasattr(gmc, "executor") and gmc.executor is not None:
                            gmc.executor.shutdown(wait=True)
            self.model.predictor = None
