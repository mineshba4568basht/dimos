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
from functools import cache
import threading
import time
from typing import Literal

import cv2
from dimos_lcm.sensor_msgs import CameraInfo  # type: ignore[import-untyped]
import numpy as np
from reactivex import create  # type: ignore[import-not-found]
from reactivex.observable import Observable  # type: ignore[import-not-found]

from dimos.hardware.sensors.camera.spec import CameraConfig, StereoCameraHardware
from dimos.msgs.sensor_msgs import Image
from dimos.msgs.sensor_msgs.Image import ImageFormat
from dimos.utils.reactive import backpressure
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class DepthAIConfig(CameraConfig):
    """DepthAI/OAK-D Pro configuration.

    Notes:
    - Depth is aligned to RGB by default (best for ROI pipelines).
    - OAK-D Pro supports active stereo (IR dot projector) and IR flood for low light.
    """

    fps: int = 30
    rgb_resolution: Literal["1080p", "4k", "720p"] = "1080p"
    mono_resolution: Literal["400p", "720p", "800p"] = "720p"

    align_depth_to_rgb: bool = True
    lr_check: bool = True
    subpixel: bool = True
    extended_disparity: bool = False

    # IR controls (OAK-D Pro has IR dot projector + flood illuminator)
    enable_ir: bool = False
    ir_dot_projector_ma: int = 0
    ir_flood_ma: int = 0

    frame_id_prefix: str | None = None


class DepthAI(StereoCameraHardware[DepthAIConfig]):
    """DepthAI/OAK-D Pro stereo RGB-D camera (no implemented without ROS).

    Provides:
    - `image_stream()` emitting RGB frames
    - `depth_stream()` emitting DEPTH16 frames (uint16 millimeters) aligned to RGB
    - `camera_info` for the RGB stream intrinsics
    """

    default_config = DepthAIConfig

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._device = None
        self._pipeline = None
        self._rgb_out = None
        self._depth_out = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._rgb_observer = None
        self._depth_observer = None

        self._camera_info = CameraInfo()

    def _frame(self, frame: str) -> str:
        if not self.config.frame_id_prefix:
            return frame
        return f"{self.config.frame_id_prefix}/{frame}"

    def _make_pipeline(self):  # type: ignore[no-untyped-def]
        try:
            import depthai as dai  # type: ignore[import-not-found]
        except Exception as e:
            raise ImportError(
                "DepthAI camera requires `depthai` installed. Install with `pip install depthai`."
            ) from e

        pipeline = dai.Pipeline()

        # Nodes
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        # RGB config
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        if self.config.rgb_resolution == "4k":
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        elif self.config.rgb_resolution == "720p":
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        else:
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        cam_rgb.setFps(self.config.fps)

        # Mono config
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setFps(self.config.fps)
        mono_right.setFps(self.config.fps)

        if self.config.mono_resolution == "400p":
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        elif self.config.mono_resolution == "800p":
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        else:
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Stereo depth config
        stereo.setLeftRightCheck(self.config.lr_check)
        stereo.setSubpixel(self.config.subpixel)
        stereo.setExtendedDisparity(self.config.extended_disparity)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        if self.config.align_depth_to_rgb:
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        # DepthAI v3: host queues are created directly from outputs (no XLinkOut nodes).
        self._rgb_out = cam_rgb.video
        self._depth_out = stereo.depth

        return pipeline

    def _init_camera_info(self) -> None:
        """Populate `camera_info` from DepthAI calibration (RGB intrinsics)."""
        import depthai as dai  # type: ignore[import-not-found]

        if self._device is None:
            return

        cal = self._device.readCalibration()
        # Use RGB resolution that matches cam_rgb.video output size
        # DepthAI wants (width,height)
        w, h = self._rgb_wh
        K = cal.getCameraIntrinsics(dai.CameraBoardSocket.RGB, w, h)  # 3x3
        K = np.array(K, dtype=np.float32)

        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info = CameraInfo()
        info.width = int(w)
        info.height = int(h)
        info.distortion_model = ""
        info.D = []
        info.D_length = 0
        info.K = K.reshape(-1).tolist()
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = P
        info.header.frame_id = self._frame("camera_optical")

        now = time.time()
        info.header.stamp.sec = int(now)
        info.header.stamp.nsec = int((now - int(now)) * 1e9)

        self._camera_info = info

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        import depthai as dai  # type: ignore[import-not-found]

        try:
            self._pipeline = self._make_pipeline()
            # Create output queues BEFORE starting the pipeline (DepthAI v3 API).
            if self._rgb_out is None or self._depth_out is None:
                raise RuntimeError("DepthAI pipeline did not expose RGB/depth outputs")

            self._q_rgb = self._rgb_out.createOutputQueue(maxSize=1, blocking=False)
            self._q_depth = self._depth_out.createOutputQueue(maxSize=1, blocking=False)

            self._pipeline.start()
            # Access the underlying device for calibration/IR controls.
            self._device = self._pipeline.getDefaultDevice()
        except Exception:
            # Ensure we don't leave the device in-use after a failed start.
            self.stop()
            raise

        # Enable IR if requested (OAK-D Pro supports both dot projector and flood)
        if self.config.enable_ir:
            try:
                # DepthAI v3 uses normalized intensity [0..1] instead of mA directly.
                dot_intensity = float(self.config.ir_dot_projector_ma) / 1200.0
                flood_intensity = float(self.config.ir_flood_ma) / 1500.0
                dot_intensity = max(0.0, min(1.0, dot_intensity))
                flood_intensity = max(0.0, min(1.0, flood_intensity))

                self._device.setIrLaserDotProjectorIntensity(dot_intensity)
                self._device.setIrFloodLightIntensity(flood_intensity)
            except Exception as e:
                logger.warning(f"Failed to set IR brightness: {e}")

        # Determine RGB output size from first frame (or config)
        self._rgb_wh = (0, 0)
        try:
            pkt = self._q_rgb.get()
            if pkt is not None:
                frame = pkt.getCvFrame()
                self._rgb_wh = (int(frame.shape[1]), int(frame.shape[0]))
        except Exception:
            self._rgb_wh = (0, 0)

        # If not available yet, approximate from resolution setting
        if self._rgb_wh == (0, 0):
            if self.config.rgb_resolution == "4k":
                self._rgb_wh = (3840, 2160)
            elif self.config.rgb_resolution == "720p":
                self._rgb_wh = (1280, 720)
            else:
                self._rgb_wh = (1920, 1080)

        self._init_camera_info()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            if self._device is not None:
                # Device is owned by pipeline in DepthAI v3; just close pipeline.
                pass
        except Exception:
            pass
        self._device = None
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        except Exception:
            pass
        self._pipeline = None
        self._rgb_out = None
        self._depth_out = None

    def _loop(self) -> None:
        frame_interval = 1.0 / max(1, int(self.config.fps))
        next_time = time.time()

        while not self._stop_event.is_set():
            try:
                rgb_pkt = self._q_rgb.tryGet()
                depth_pkt = self._q_depth.tryGet()

                ts = time.time()

                if rgb_pkt is not None and self._rgb_observer is not None:
                    rgb = rgb_pkt.getCvFrame()  # already RGB when colorOrder=RGB
                    if rgb is not None and rgb.ndim == 3:
                        # Ensure uint8 RGB
                        rgb_u8 = rgb.astype(np.uint8, copy=False)
                        msg = Image(
                            data=rgb_u8,
                            format=ImageFormat.RGB,
                            frame_id=self._frame("camera_optical"),
                            ts=ts,
                        )
                        self._rgb_observer.on_next(msg)

                if depth_pkt is not None and self._depth_observer is not None:
                    depth_mm = depth_pkt.getFrame()  # uint16 millimeters
                    depth_mm = depth_mm.astype(np.uint16, copy=False)
                    msg = Image(
                        data=depth_mm,
                        format=ImageFormat.DEPTH16,
                        frame_id=self._frame("camera_optical"),
                        ts=ts,
                    )
                    self._depth_observer.on_next(msg)

            except Exception as e:
                # Emit errors to observers (if any), then keep running
                if self._rgb_observer is not None:
                    self._rgb_observer.on_error(e)
                if self._depth_observer is not None:
                    self._depth_observer.on_error(e)

            next_time += frame_interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)
            else:
                next_time = time.time()

    @cache
    def image_stream(self) -> Observable[Image]:
        """RGB stream."""

        def subscribe(observer, scheduler=None):  # type: ignore[no-untyped-def]
            self._rgb_observer = observer
            try:
                self.start()
            except Exception as e:
                observer.on_error(e)
                return None

            def dispose() -> None:
                self._rgb_observer = None
                # Do not stop the device if depth still has a subscriber.
                if self._depth_observer is None:
                    self.stop()

            return dispose

        return backpressure(create(subscribe))

    @cache
    def depth_stream(self) -> Observable[Image]:
        """Aligned depth stream (DEPTH16 in mm)."""

        def subscribe(observer, scheduler=None):  # type: ignore[no-untyped-def]
            self._depth_observer = observer
            try:
                self.start()
            except Exception as e:
                observer.on_error(e)
                return None

            def dispose() -> None:
                self._depth_observer = None
                if self._rgb_observer is None:
                    self.stop()

            return dispose

        return backpressure(create(subscribe))

    @property
    def camera_info(self) -> CameraInfo:
        # Update timestamp for consumers that treat it as "latest"
        now = time.time()
        self._camera_info.header.stamp.sec = int(now)
        self._camera_info.header.stamp.nsec = int((now - int(now)) * 1e9)
        return self._camera_info



