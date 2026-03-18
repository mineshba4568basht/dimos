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

"""Unitree Go2 RL locomotion policy via ControlCoordinator.

Runs a trained ONNX locomotion policy at 50 Hz inside the coordinator
tick loop.  Keyboard teleop provides velocity commands (vx, vy, yaw_rate)
as the policy input.

Controls (via keyboard_teleop):
    W/S: Forward/backward (linear.x)
    Q/E: Strafe left/right (linear.y)
    A/D: Turn left/right (angular.z)
    Shift: 2x boost
    Ctrl: 0.5x slow
    Space: Emergency stop
    ESC: Quit

Usage:
    dimos run unitree-go2-rl-policy
"""

from __future__ import annotations

from dimos.control.components import HardwareComponent, HardwareType, make_quadruped_joints
from dimos.control.coordinator import TaskConfig, control_coordinator
from dimos.core.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs import Twist
from dimos.msgs.sensor_msgs import JointState
from dimos.robot.unitree.keyboard_teleop import keyboard_teleop

_go2_joints = make_quadruped_joints("go2")

_go2 = control_coordinator(
    tick_rate=100.0,
    hardware=[
        HardwareComponent(
            hardware_id="go2",
            hardware_type=HardwareType.QUADRUPED,
            joints=_go2_joints,
            adapter_type="unitree_go2",
            kp=25.0,
            kd=0.5,
        ),
    ],
    tasks=[
        TaskConfig(
            name="rl_go2",
            type="rl_policy",
            joint_names=_go2_joints,
            priority=50,
            policy_path="unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/"
            "2026-03-17_16-18-21/exported/policy.onnx",
            hardware_id="go2",
            default_positions=[
                0.1,
                -0.1,
                0.1,
                -0.1,
                0.8,
                0.8,
                1.0,
                1.0,
                -1.5,
                -1.5,
                -1.5,
                -1.5,
            ],
            action_scale=0.25,
            decimation=2,
        ),
    ],
).transports(
    {
        ("joint_state", JointState): LCMTransport("/coordinator/joint_state", JointState),
        ("twist_command", Twist): LCMTransport("/cmd_vel", Twist),
    }
)

_teleop = keyboard_teleop().transports(
    {
        ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
    }
)

unitree_go2_rl_policy = autoconnect(_go2, _teleop)

__all__ = ["unitree_go2_rl_policy"]
