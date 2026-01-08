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

from abc import ABC, abstractmethod
from typing import Callable, Optional

import reactivex as rx
from reactivex import Observable
from reactivex import operators as ops

from dimos.core.resource import Resource
from dimos.msgs.sensor_msgs import Image
from dimos.perception.detection.type import Detection2D, ImageDetections2D


class Detector(Resource):
    """Base class for detectors that transform image streams to detection streams.

    Detectors are RxPY operators: they take Observable[Image] and return Observable[ImageDetections2D].
    They handle model initialization, resource management, and cleanup automatically.

    Example usage:
        ```python
        detector = Yolo2DDetector()

        # Use as operator
        detections = image_stream.pipe(detector)

        # Or call directly
        detections = detector(image_stream)

        # Compose with backpressure operators
        from dimos.stream.video_operators import Operators
        detections = backpressure(image_stream).pipe(detector)
        ```
    """

    @abstractmethod
    def process_image(self, image: Image) -> ImageDetections2D: ...

    def start(self):
        pass

    def stop(self):
        pass

    def __call__(self, source: Observable) -> Observable:
        """Apply detector to image stream (makes detector callable as operator).

        Args:
            source: Observable[Image] stream

        Returns:
            Observable[ImageDetections2D] stream with automatic cleanup
        """

        def subscribe(observer, scheduler=None):
            """Custom subscription with cleanup handling."""

            # Start detector when stream begins
            self.start()

            def on_error(error):
                try:
                    self.stop()
                finally:
                    observer.on_error(error)

            def on_completed():
                try:
                    self.stop()
                finally:
                    observer.on_completed()

            # Simple map with lifecycle management
            stream = source.pipe(ops.map(self.process_image))
            return stream.subscribe(observer.on_next, on_error, on_completed)

        return rx.create(subscribe)
