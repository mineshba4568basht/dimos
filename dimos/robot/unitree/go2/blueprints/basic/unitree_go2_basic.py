#!/usr/bin/env python3

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

from pathlib import Path
import platform

from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE
from dimos.core.blueprints import autoconnect
from dimos.core.transport import pSHMTransport
from dimos.dashboard.tf_rerun_module import tf_rerun
from dimos.msgs.sensor_msgs import Image
from dimos.robot.foxglove_bridge import foxglove_bridge
from dimos.robot.unitree.go2.connection import GO2Connection, go2_connection
from dimos.web.websocket_vis.websocket_vis_module import websocket_vis

_GO2_URDF = Path(__file__).parent.parent / "go2.urdf"

# Mac has some issue with high bandwidth UDP
#
# so we use pSHMTransport for color_image
# (Could we address this on the system config layer? Is this fixable on mac?)
_mac = autoconnect(
    foxglove_bridge(
        shm_channels=[
            "/color_image#sensor_msgs.Image",
        ]
    ),
).transports(
    {
        ("color_image", Image): pSHMTransport(
            "color_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
        ),
    }
)

_linux = autoconnect(foxglove_bridge())

unitree_go2_basic = autoconnect(
    go2_connection(),
    _linux if platform.system() == "Linux" else _mac,
    websocket_vis(),
    tf_rerun(
        urdf_path=str(_GO2_URDF),
        cameras=[
            ("world/robot/camera", "camera_optical", GO2Connection.camera_info_static),
        ],
    ),
).global_config(n_dask_workers=4, robot_model="unitree_go2")

__all__ = [
    "_linux",
    "_mac",
    "unitree_go2_basic",
]
