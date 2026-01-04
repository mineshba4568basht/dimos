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

import json
import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from dimos.hardware.lerobot_kinematics import LerobotKinematics
from dimos.msgs.geometry_msgs import Pose, Quaternion, Vector3

logger = logging.getLogger(__name__)


class SO101Interface:
    """
    Interface class for SO-101 robotic arm.

    Responsibilities:
      - Feetech bus setup (IDs, calibration, operating modes, PID).
      - Kinematics via LerobotKinematics (FK / IK / Jacobians).
      - Joint-space get/set and joint PTP.
      - Cartesian-space PTP + linear interpolation.
      - Gripper position + load feedback.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        urdf_path: str = "dimos/hardware/so101_utils/urdf/so101_new_calib.urdf",
        ee_link_name: str = "gripper_frame_link",
        calibration_path: str = "dimos/hardware/so101_utils/calibration/so101_arm.json",
    ) -> None:
        self.port = port
        self.urdf_path = urdf_path
        self.ee_link_name = ee_link_name
        self.calibration_path = calibration_path

        self.bus: FeetechMotorsBus | None = None
        self.kinematics: LerobotKinematics | None = None

        # Motor configuration: 5 DOF arm + 1 gripper
        self.motor_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        self.gripper_name = "gripper"
        # Joint angle offsets in degrees (motor frame → robot frame)
        # Measured offsets when the arm is mechanically at zero.
        self.joint_offsets_deg = np.array(
            [0, 0.26373626, 2.59340659, 0.65934066, 0.21978022], dtype=float
        )

        self.motor_ids: dict[str, int] = {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }

        # Default Cartesian motion speed (0–100)
        self.move_speed: int = 100

        # Initialize kinematics
        try:
            self.kinematics = LerobotKinematics(
                self.urdf_path,
                self.ee_link_name,
                joint_names=self.motor_names,
            )
            logger.info(
                "Initialized LerobotKinematics with URDF %s, EE link %s",
                self.urdf_path,
                self.ee_link_name,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize kinematics: {e}")
            self.kinematics = None

    # -------------------------------------------------------------------------
    #   Bus / motor setup
    # -------------------------------------------------------------------------

    def _load_calibration(self) -> dict[str, MotorCalibration]:
        """Load motor calibration from JSON file."""
        try:
            with open(self.calibration_path) as f:
                calib_data = json.load(f)
            calibration: dict[str, MotorCalibration] = {}
            for name, data in calib_data.items():
                calibration[name] = MotorCalibration(**data)
            return calibration
        except Exception as e:
            logger.warning(f"Failed to load calibration from {self.calibration_path}: {e}")
            return {}

    def connect(self) -> None:
        """Connect to the SO-101 arm and configure motors."""
        logger.info("Connecting to SO-101 arm on port %s", self.port)

        norm_mode_body = MotorNormMode.DEGREES
        motors: dict[str, Motor] = {}

        # Arm joints: normalized in degrees
        for name in self.motor_names:
            motors[name] = Motor(self.motor_ids[name], "sts3215", norm_mode_body)

        # Gripper: normalized in [0, 100]
        motors[self.gripper_name] = Motor(
            self.motor_ids[self.gripper_name],
            "sts3215",
            MotorNormMode.RANGE_0_100,
        )

        calibration = self._load_calibration()

        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=motors,
            calibration=calibration,
        )
        self.bus.connect()
        logger.info("FeetechMotorsBus connected")

        # Configure modes, PID, etc.
        self.configure_motors()

    def configure_motors(self) -> None:
        """
        Configure motors: operating mode + PID.

        - Puts all motors (joints + gripper) into POSITION mode.
        - Sets PID coefficients tuned to reduce shakiness.
        """
        if not self.bus:
            raise RuntimeError("Bus not connected.")

        logger.info("Configuring motors (OperatingMode + PID gains)")

        # Disable torque while tweaking settings
        with self.bus.torque_disabled():
            # Let LeRobot configure registers (limits, accel, etc.)
            self.bus.configure_motors()

            for motor in self.bus.motors:
                # Position control for all motors
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

                # PID gains – tuned for smoother motion
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

        logger.info("Motor configuration complete")

    def enable(self) -> bool:
        """Enable motor torque."""
        if self.bus:
            self.bus.enable_torque()
            logger.info("SO-101 torque enabled")
            return True
        else:
            return False

    def disable(self) -> None:
        """Disable motor torque."""
        if self.bus:
            self.bus.disable_torque()
            logger.info("SO-101 torque disabled")

    def disconnect(self) -> None:
        """Disconnect from the arm."""
        if self.bus:
            self.bus.disconnect()
            logger.info("SO-101 bus disconnected")

    # -------------------------------------------------------------------------
    #   Joint-space API
    # -------------------------------------------------------------------------

    def get_joint_angles(self, degree: bool = False) -> np.ndarray:
        """
        Get current joint angles with joint offsets applied.

        Args:
            degree: If True, return in degrees; otherwise radians.

        Returns:
            np.ndarray of shape (5,) - true joint angles (raw - offset)
        """
        if not self.bus:
            return np.zeros(len(self.motor_names), dtype=float)

        values = self.bus.sync_read("Present_Position")
        q_deg_raw = np.array([values[name] for name in self.motor_names], dtype=float)
        q_deg = q_deg_raw - self.joint_offsets_deg
        return q_deg if degree else np.radians(q_deg)

    def set_joint_angles(self, q: np.ndarray, degree: bool = False) -> None:
        """

        TODO. make private func
        Set joint angles.

        Args:
            q: Angles for 5 joints (radians if degree=False, else degrees).
            degree: True if `q` is already in degrees.
        """
        if not self.bus:
            raise RuntimeError("Bus not connected.")

        q = np.asarray(q, dtype=float)
        if q.shape[0] != len(self.motor_names):
            raise ValueError(f"Expected {len(self.motor_names)} joint values, got {q.shape[0]}")

        q_deg = q if degree else np.degrees(q)
        cmd = {name: float(q_deg[i]) for i, name in enumerate(self.motor_names)}
        self.bus.sync_write("Goal_Position", cmd)

    def move_joint_ptp(self, q_target: np.ndarray, duration: float | None = None) -> None:
        """
        Joint-space PTP interpolation. TODO check logic

        Args:
            q_target: target joint angles [rad], shape (5,)
            duration: total motion time (s). If None, derived from self.move_speed.
        """
        if not self.bus:
            raise RuntimeError("Bus not connected.")

        q_target = np.asarray(q_target, dtype=float)
        if q_target.shape[0] != len(self.motor_names):
            raise ValueError(f"Expected {len(self.motor_names)} joints, got {q_target.shape[0]}")

        q_start = self.get_joint_angles()
        dq = q_target - q_start
        max_delta = float(np.max(np.abs(dq)))
        dt = 0.02  # 50 Hz

        if max_delta < 1e-6:
            self.set_joint_angles(q_target)
            return

        if duration is None:
            # Derive from move_speed: 0..100 → 0.2..1.0 rad/s (tunable)
            speed_scale = float(self.move_speed) / 100.0
            min_speed = 0.2
            max_speed = 1.0
            joint_speed = min_speed + (max_speed - min_speed) * speed_scale
            if joint_speed < 1e-3:
                joint_speed = min_speed

            duration = max_delta / joint_speed
            if duration < dt:
                duration = dt

        steps = max(int(duration / dt), 1)

        for i in range(1, steps + 1):
            alpha = i / steps
            q_interp = q_start + alpha * dq
            self.set_joint_angles(q_interp)
            time.sleep(dt)

    # -------------------------------------------------------------------------
    #   Cartesian-space API
    # -------------------------------------------------------------------------

    def _move_ptp(
        self,
        current_q: np.ndarray,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        duration: float | None = None,
    ) -> None:
        """
        Cartesian point-to-point motion using IK + joint-space interpolation.

        Args:
            current_q: current joint angles [rad] (motor angles)
            target_pos: desired position [x, y, z] in meters
            target_quat_wxyz: desired orientation [w, x, y, z]
            duration: total motion time (s). If None, derived from self.move_speed.
        """
        if not self.kinematics:
            raise RuntimeError("Kinematics not initialized.")

        q_start = np.asarray(current_q, dtype=float)

        # LerobotKinematics uses degrees, so convert
        q_start_deg = np.degrees(q_start)
        q_target_kin_deg = self.kinematics.ik(q_start_deg, target_pos, target_quat_wxyz)
        q_target = np.radians(q_target_kin_deg)

        # Reuse the generic joint-space PTP helper
        self.move_joint_ptp(q_target, duration=duration)

    def _move_linear(
        self,
        current_q: np.ndarray,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        duration: float | None = None,
    ) -> None:
        """
        Linear Cartesian motion with orientation slerp.

        - Interpolates in task-space along a straight line.
        - Uses Slerp for orientation interpolation.
        - Solves IK at each step and sends joint positions.

        Args:
            current_q: current joint angles [rad] (motor angles)
            target_pos: desired position [x, y, z] in meters
            target_quat_wxyz: desired orientation [w, x, y, z]
            duration: total motion time (s). If None, derived from self.move_speed.
        """
        if not self.kinematics:
            raise RuntimeError("Kinematics not initialized.")

        # LerobotKinematics uses degrees, so convert
        current_q_deg = np.degrees(current_q)
        start_pos, start_quat_wxyz = self.kinematics.fk(current_q_deg)
        dt = 0.02  # 50 Hz

        dist = np.linalg.norm(target_pos - start_pos)
        if dist < 1e-6:
            # Same position; just do a PTP in joints
            self._move_ptp(current_q, target_pos, target_quat_wxyz, duration=duration)
            return

        if duration is None:
            # self.move_speed in [0, 100] → [0.01, 0.1] m/s
            speed_m_s = (self.move_speed / 100.0) * 0.1
            if speed_m_s < 0.01:
                speed_m_s = 0.01
            duration = dist / speed_m_s
            if duration < dt:
                duration = dt

        steps = max(int(duration / dt), 1)

        # Orientation interpolation (Kinpy: wxyz, scipy: xyzw)
        start_xyzw = [
            start_quat_wxyz[1],
            start_quat_wxyz[2],
            start_quat_wxyz[3],
            start_quat_wxyz[0],
        ]
        target_xyzw = [
            target_quat_wxyz[1],
            target_quat_wxyz[2],
            target_quat_wxyz[3],
            target_quat_wxyz[0],
        ]
        key_rots = R.from_quat(np.array([start_xyzw, target_xyzw], dtype=float))
        slerp = Slerp([0.0, 1.0], key_rots)

        # Track current joint angles for IK seed
        current_q_kin_deg = current_q_deg

        for i in range(1, steps + 1):
            t = i / steps

            # Linear position
            interp_pos = start_pos + (target_pos - start_pos) * t

            # Slerp orientation
            interp_rot_xyzw = slerp(t).as_quat()
            interp_quat_wxyz = np.array(
                [
                    interp_rot_xyzw[3],  # w
                    interp_rot_xyzw[0],  # x
                    interp_rot_xyzw[1],  # y
                    interp_rot_xyzw[2],  # z
                ],
                dtype=float,
            )

            # LerobotKinematics uses degrees
            q_sol_kin_deg = self.kinematics.ik(current_q_kin_deg, interp_pos, interp_quat_wxyz)
            q_sol = np.radians(q_sol_kin_deg)
            self.set_joint_angles(q_sol)
            current_q_kin_deg = q_sol_kin_deg  # Update for next iteration

            time.sleep(dt)

    def get_ee_pose(self) -> Pose:
        """
        Get current end-effector pose.

        Returns:
            Pose with position in meters and orientation as quaternion (x,y,z,w).
        """
        if not self.kinematics:
            raise RuntimeError("Kinematics not initialized.")

        # LerobotKinematics uses degrees
        q = self.get_joint_angles(degree=True)
        pos, quat_wxyz = self.kinematics.fk(q)

        position = Vector3(pos[0], pos[1], pos[2])
        orientation = Quaternion(
            quat_wxyz[1],  # x
            quat_wxyz[2],  # y
            quat_wxyz[3],  # z
            quat_wxyz[0],  # w
        )
        return Pose(position, orientation)

    def set_ee_pose(
        self,
        pose: Pose,
        linear: bool = False,
        duration: float | None = None,
    ) -> None:
        """
        Move end-effector to target pose.

        Args:
            pose: Target pose (position in meters, quaternion orientation).
            linear: If True, use linear Cartesian interpolation; else PTP.
            duration: total motion time (s). If None, derived from self.move_speed.
        """
        if not self.kinematics:
            raise RuntimeError("Kinematics not initialized.")

        target_pos = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=float,
        )
        target_quat_wxyz = np.array(
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
            dtype=float,
        )
        current_q = self.get_joint_angles()

        if linear:
            self._move_linear(current_q, target_pos, target_quat_wxyz, duration=duration)
        else:
            self._move_ptp(current_q, target_pos, target_quat_wxyz, duration=duration)

    # -------------------------------------------------------------------------
    #   Gripper API
    # -------------------------------------------------------------------------

    def set_gripper_position(self, position: float) -> None:
        """
        Move gripper to target position.

        Args:
            position: 0.0 (closed) to 0.1 (open) in meters.
        """
        if not self.bus:
            raise RuntimeError("Bus not connected.")

        # Map 0–0.1 m → 0–100 normalized range
        val = (position / 0.1) * 100.0
        val = max(0.0, min(100.0, val))
        self.bus.write("Goal_Position", self.gripper_name, val)

    def get_gripper_state(self) -> tuple[float, float]:
        """
        Get current gripper state.

        Returns:
            (position_m, load) where:
                position_m ∈ [0.0, 0.1] meters (approx)
                load is raw Present_Load value from the motor.
        """
        if not self.bus:
            return 0.0, 0.0

        try:
            vals = self.bus.read("Present_Position", self.gripper_name)
            efforts = self.bus.read("Present_Load", self.gripper_name)

            raw_pos = float(vals)
            # Inverse of set_gripper_position mapping:
            # val (0–100) → approx 0–0.1 m
            position_m = max(0.0, min(0.1, raw_pos * 0.001))

            load = float(efforts)
            return position_m, load
        except Exception as e:
            msg = str(e)
            if "Overload error" in msg or "Overload" in msg:
                # If overloaded, we likely gripped something hard.
                # Return max load (1000) so the controller thinks we have contact.
                return 0.0, 1000.0

            logger.warning(f"Failed to read gripper state: {e}")
            return 0.0, 0.0

    # -------------------------------------------------------------------------
    #   Misc
    # -------------------------------------------------------------------------

    def set_speed(self, speed: int) -> None:
        """
        Set Cartesian motion speed scaling (0–100).

        0 → ~0.01 m/s, 100 → ~0.1 m/s (see _move_linear).
        """
        speed_int = int(speed)
        self.move_speed = max(0, min(100, speed_int))
