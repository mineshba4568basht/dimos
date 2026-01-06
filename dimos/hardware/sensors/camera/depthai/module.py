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

from collections.abc import Callable
from dataclasses import dataclass, field
import time

import reactivex as rx
from reactivex import operators as ops
from reactivex.disposable import Disposable
from reactivex.observable import Observable

from dimos.core import Module, Out, rpc
from dimos.core.module import ModuleConfig
from dimos.hardware.sensors.camera.depthai.camera import DepthAI
from dimos.hardware.sensors.camera.spec import StereoCameraHardware
from dimos.msgs.sensor_msgs import CameraInfo, Image
from dimos.msgs.geometry_msgs import Quaternion, Transform, Vector3


def default_transform():
    return Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
        child_frame_id="camera_link",
    )


@dataclass
class DepthAICameraModuleConfig(ModuleConfig):
    frame_id: str = "camera_link"
    transform: Transform | None = field(default_factory=default_transform)
    hardware: Callable[[], StereoCameraHardware] | StereoCameraHardware = DepthAI  # type: ignore[type-arg]
    camera_info_frequency: float = 2.0


class DepthAICameraModule(Module):
    """RGB-D camera module for DepthAI / OAK-D Pro.

    Outputs:
    - `color_image`: RGB image
    - `depth_image`: aligned DEPTH16 (uint16 millimeters)
    - `camera_info`: intrinsics matching the RGB stream resolution
    """

    color_image: Out[Image]
    depth_image: Out[Image]
    camera_info: Out[CameraInfo]

    hardware: StereoCameraHardware = None
    _rgb_sub: Disposable | None = None
    _depth_sub: Disposable | None = None
    _info_sub: Disposable | None = None

    default_config = DepthAICameraModuleConfig

    @rpc
    def start(self):  # type: ignore[no-untyped-def]
        if callable(self.config.hardware):
            self.hardware = self.config.hardware()
        else:
            self.hardware = self.config.hardware

        if self._rgb_sub or self._depth_sub:
            return "already started"

        # RGB + Depth streaming
        self._rgb_sub = self.hardware.image_stream().subscribe(self.color_image.publish)
        self._depth_sub = self.hardware.depth_stream().subscribe(self.depth_image.publish)
        self._disposables.add(self._rgb_sub)
        self._disposables.add(self._depth_sub)

        # Camera info publishing + TF
        def info_tick(_):  # type: ignore[no-untyped-def]
            info = self.hardware.camera_info
            info.ts = time.time()
            self.camera_info.publish(info)

            if self.config.transform is None:
                return

            camera_link = self.config.transform
            camera_link.ts = info.ts
            camera_optical = Transform(
                translation=Vector3(0.0, 0.0, 0.0),
                rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
                frame_id="camera_link",
                child_frame_id="camera_optical",
                ts=camera_link.ts,
            )
            self.tf.publish(camera_link, camera_optical)

        self._info_sub = rx.interval(1.0 / max(0.5, float(self.config.camera_info_frequency))).pipe(
            ops.map(info_tick)
        ).subscribe(lambda _: None)
        self._disposables.add(self._info_sub)

        return "started"

    @rpc
    def stop(self) -> None:
        if self._rgb_sub:
            self._rgb_sub.dispose()
            self._rgb_sub = None
        if self._depth_sub:
            self._depth_sub.dispose()
            self._depth_sub = None
        if self._info_sub:
            self._info_sub.dispose()
            self._info_sub = None
        if self.hardware and hasattr(self.hardware, "stop"):
            self.hardware.stop()
        super().stop()


depthai_camera_module = DepthAICameraModule.blueprint

__all__ = ["DepthAICameraModule", "DepthAICameraModuleConfig", "depthai_camera_module"]



