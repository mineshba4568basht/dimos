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

import threading
import time
from typing import TYPE_CHECKING

from dimos_lcm.geometry_msgs import Pose as LCMPose
import numpy as np
import pytest

from dimos.core import In, Module, rpc
from dimos.hardware.so101_utils.so101_interface import SO101Interface
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos_lcm.geometry_msgs import Twist

    from dimos.msgs.geometry_msgs import Pose

logger = setup_logger(__file__)


class SO101Arm:
    def __init__(self, arm_name: str = "so101", port: str = "/dev/ttyACM0") -> None:
        self.arm_name = arm_name
        self.arm = SO101Interface(port=port)

        # Connect + configure motors (OperatingMode, PID, etc.)
        self.arm.connect()

        # Allow time for connection
        time.sleep(0.5)
        self.enable()

        # Go to a known configuration
        self.gotoZero()
        time.sleep(1)

        # Init velocity controller (Jacobians etc.)
        self.init_vel_controller()

        # Track last commanded gripper effort for detection heuristic
        self._last_gripper_effort_cmd = 0.5

    # ---------------- Basic arm management ----------------

    def enable(self) -> None:
        while not self.arm.enable():
            time.sleep(0.01)
        logger.info("Arm enabled")

    def disable(self) -> None:
        """Soft-stop and fully disable motors + bus."""
        self.softStop()
        self.arm.disconnect()

    def gotoZero(self, duration: float | None = None) -> None:
        """Move to home position (all joints at 0 rad)."""
        logger.info("Going to zero")
        q_zero = np.zeros(5, dtype=float)
        self.arm.move_joint_ptp(q_zero, duration=duration)
        self.release_gripper()
        time.sleep(1.0)
        self.close_gripper()
        time.sleep(0.5)

    def gotoObserve(self, duration: float | None = None) -> None:
        """Move to an 'observe' pose with simple joint interpolation."""
        logger.info("Going to observe")
        observe_angles = np.array(
            [-0.062909, -1.396263, 0.208672, 1.793661, -1.614142], dtype=float
        )
        self.arm.move_joint_ptp(observe_angles, duration=duration)
        self.release_gripper()
        time.sleep(1.0)
        self.close_gripper()
        time.sleep(0.5)

    def goToRest(self, duration: float | None = None) -> None:
        """Move to a resting pose."""

        q_rest = np.array(
            [-0.03989324, -1.81284089, 1.69085964, 1.28578981, -0.00613742], dtype=float
        )
        self.arm.move_joint_ptp(q_rest, duration=duration)
        self.release_gripper()
        time.sleep(1.0)
        self.close_gripper()
        time.sleep(0.5)

    def softStop(self) -> None:
        """Move to zero and then disable torque (no hard 'kill')."""
        self.goToRest()
        time.sleep(1.0)
        self.arm.disable()

    # ---------------- Cartesian pose control ----------------

    def cmd_ee_pose(
        self,
        pose: Pose,
        line_mode: bool = False,
        duration: float | None = None,
    ) -> None:
        """Command end-effector to target pose using Pose message."""
        self.arm.set_ee_pose(pose, linear=line_mode, duration=duration)

    def get_ee_pose(self) -> Pose:
        """Get current end-effector pose."""
        return self.arm.get_ee_pose()

    # ---------------- Gripper control ----------------

    def cmd_gripper_ctrl(self, position: float, effort: float = 0.25) -> None:
        """
        Command gripper position (in meters).

        Args:
            position: 0.0 (closed) to 0.1 (open) in meters.
            effort: logical effort in [0,1]; not sent to hardware, but used
                    for detection thresholds.
        """
        self.arm.set_gripper_position(position)
        self._last_gripper_effort_cmd = effort

    def release_gripper(self) -> None:
        """Open gripper to ~max opening."""
        self.cmd_gripper_ctrl(0.07)

    def get_gripper_feedback(self) -> tuple[float, float]:
        """
        Get gripper position (meters) and normalized effort (0..1).
        """
        position_m, raw_load = self.arm.get_gripper_state()
        effort_mag = min(1000.0, abs(raw_load))

        norm_effort = effort_mag / 1000.0
        return position_m, norm_effort

    def close_gripper(self, commanded_effort: float = 0.5) -> None:
        """
        Close gripper.
        """
        self._last_gripper_effort_cmd = commanded_effort
        self.cmd_gripper_ctrl(0.0, effort=commanded_effort)

    def gripper_object_detected(self, commanded_effort: float | None = None) -> bool:
        """
        Heuristic object detection based on Present_Load.

        Args:
            commanded_effort: nominal effort you were trying to use [0..1].
                              If None, uses last commanded effort.

        Returns:
            True if measured effort exceeds 0.8 * commanded_effort.
        """
        if commanded_effort is None:
            commanded_effort = self._last_gripper_effort_cmd

        _position_m, actual_effort = self.get_gripper_feedback()
        effort_threshold = 0.8 * commanded_effort
        return actual_effort > effort_threshold

    # ---------------- Velocity control (task-space) ----------------

    def init_vel_controller(self) -> None:
        """Initialize velocity controller with kinematics."""
        self.kinematics = self.arm.kinematics
        self.dt = 0.01  # 100 Hz

    def cmd_vel(
        self,
        x_dot: float,
        y_dot: float,
        z_dot: float,
        R_dot: float,
        P_dot: float,
        Y_dot: float,
    ) -> None:
        """
        Command end-effector twist (Cartesian velocity).

        Units:
            - x_dot, y_dot, z_dot: m/s
            - R_dot, P_dot, Y_dot: rad/s (about X, Y, Z in EE frame)
        """
        if not self.kinematics:
            return

        # Joint angles in radians
        q = self.arm.get_joint_angles()

        twist = np.array(
            [x_dot, y_dot, z_dot, R_dot, P_dot, Y_dot],
            dtype=float,
        )
        dq = self.kinematics.joint_velocity(q, twist)

        new_q = q + dq * self.dt

        # Send new joint angles (radians)
        self.arm.set_joint_angles(new_q)
        time.sleep(self.dt)

    def cmd_vel_ee(
        self,
        x_dot: float,
        y_dot: float,
        z_dot: float,
        RX_dot: float,
        PY_dot: float,
        YZ_dot: float,
    ) -> None:
        """Alias for cmd_vel with slightly different naming."""
        self.cmd_vel(x_dot, y_dot, z_dot, RX_dot, PY_dot, YZ_dot)


# ---------------- Dimos VelocityController skeleton ----------------


class VelocityController(Module):
    cmd_vel: In[Twist] = None

    def __init__(
        self,
        arm: SO101Arm,
        period: float = 0.01,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.arm = arm
        self.period = period
        self.latest_cmd: Twist | None = None
        self.last_cmd_time: float | None = None
        self._thread: threading.Thread | None = None
        self._stop_flag = False

    @rpc
    def start(self) -> None:
        """Start background velocity control loop."""
        if self._thread is not None:
            return

        self._stop_flag = False

        def loop() -> None:
            while not self._stop_flag:
                if self.latest_cmd is not None:
                    cmd = self.latest_cmd
                    self.arm.cmd_vel(
                        cmd.linear.x,
                        cmd.linear.y,
                        cmd.linear.z,
                        cmd.angular.x,
                        cmd.angular.y,
                        cmd.angular.z,
                    )
                else:
                    time.sleep(self.period)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        """Stop background velocity control loop."""
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def handle_cmd_vel(self, cmd_vel: Twist) -> None:
        """Callback from Dimos wiring to update latest twist command."""
        self.latest_cmd = cmd_vel
        self.last_cmd_time = time.time()


@pytest.mark.tool
def run_velocity_controller() -> None:
    """Simple tool entrypoint; wire into Dimos as needed."""
    arm = SO101Arm()
    vc = VelocityController(arm=arm)
    vc.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        vc.stop()
        arm.disable()


if __name__ == "__main__":
    arm = SO101Arm()
    run_velocity_controller()
