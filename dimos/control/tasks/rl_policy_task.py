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

"""RL locomotion policy task for quadruped robots.

Runs an ONNX-exported RL policy inside the coordinator tick loop.
The policy reads joint state from CoordinatorState, IMU from the
QuadrupedAdapter, and outputs joint position targets each tick.

Velocity commands (vx, vy, yaw_rate) are the policy's control input
— what direction the robot should walk.

CRITICAL: Uses t_now from CoordinatorState, never calls time.time()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from dimos.control.task import (
    BaseControlTask,
    ControlMode,
    CoordinatorState,
    JointCommandOutput,
    ResourceClaim,
)
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.hardware.quadrupeds.spec import QuadrupedAdapter

logger = setup_logger()

# Isaac Lab → Robot motor index mapping.
# JOINT_MAP[isaac_idx] = robot_motor_idx
_DEFAULT_ISAAC_TO_ROBOT_MAP = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]

_DEFAULT_GO2_POSITIONS = [
    0.1,
    -0.1,
    0.1,
    -0.1,  # hips: FL, FR, RL, RR
    0.8,
    0.8,
    1.0,
    1.0,  # thighs: FL, FR, RL, RR
    -1.5,
    -1.5,
    -1.5,
    -1.5,  # calves: FL, FR, RL, RR
]

_NUM_MOTORS = 12
_OBS_DIM = 45


@dataclass
class RLPolicyTaskConfig:
    """Configuration for the RL policy task.

    Attributes:
        policy_path: Path to ONNX model file.
        joint_names: 12 coordinator joint names in robot motor order
                     (e.g. go2_FR_0, go2_FR_1, ...).
        default_positions: 12 default joint positions in Isaac Lab order.
        action_scale: Multiplier on raw policy output before adding defaults.
        obs_ang_vel_scale: Scale for angular velocity observation.
        obs_joint_vel_scale: Scale for joint velocity observation.
        isaac_to_robot_map: Mapping from Isaac Lab joint index to robot motor index.
        priority: Priority for coordinator arbitration (higher wins).
        timeout: Seconds without velocity command before zeroing velocity input.
        decimation: Run inference every N ticks (e.g. 2 → 50Hz at 100Hz tick rate).
    """

    policy_path: str | Path
    joint_names: list[str] = field(default_factory=list)
    default_positions: list[float] = field(default_factory=lambda: list(_DEFAULT_GO2_POSITIONS))
    action_scale: float = 0.25
    obs_ang_vel_scale: float = 0.2
    obs_joint_vel_scale: float = 0.05
    isaac_to_robot_map: list[int] = field(default_factory=lambda: list(_DEFAULT_ISAAC_TO_ROBOT_MAP))
    priority: int = 50
    timeout: float = 1.0
    decimation: int = 2


class RLPolicyTask(BaseControlTask):
    """Runs an ONNX RL locomotion policy inside the coordinator tick loop.

    Observation vector (45 dims, built each inference tick):
        [0:3]   base angular velocity * scale (from IMU gyroscope)
        [3:6]   projected gravity (from IMU quaternion)
        [6:9]   velocity command [vx, vy, yaw_rate]
        [9:21]  joint_pos - default_pos (Isaac Lab order)
        [21:33] joint_vel * scale (Isaac Lab order)
        [33:45] last action (raw, before scaling)

    Action (12 dims):
        Joint position offsets in Isaac Lab order.
        target_pos = action * action_scale + default_positions

    The task reads joint state from CoordinatorState (populated by
    ConnectedQuadruped.read_state) and IMU from a direct QuadrupedAdapter
    reference.

    Example:
        >>> from dimos.hardware.quadrupeds.registry import quadruped_adapter_registry
        >>> adapter = quadruped_adapter_registry.create("unitree_go2")
        >>> task = RLPolicyTask(
        ...     "rl_go2",
        ...     RLPolicyTaskConfig(
        ...         policy_path="policy.onnx",
        ...         joint_names=make_quadruped_joints("go2"),
        ...     ),
        ...     adapter=adapter,
        ... )
        >>> coordinator.add_task(task)
        >>> task.start()
    """

    def __init__(
        self,
        name: str,
        config: RLPolicyTaskConfig,
        adapter: QuadrupedAdapter,
    ) -> None:
        if len(config.joint_names) != _NUM_MOTORS:
            raise ValueError(
                f"RLPolicyTask '{name}' requires exactly {_NUM_MOTORS} joint names, "
                f"got {len(config.joint_names)}"
            )
        if len(config.default_positions) != _NUM_MOTORS:
            raise ValueError(
                f"RLPolicyTask '{name}' requires exactly {_NUM_MOTORS} default positions, "
                f"got {len(config.default_positions)}"
            )

        self._name = name
        self._config = config
        self._adapter = adapter
        self._joint_names_list = list(config.joint_names)
        self._joint_names_set = frozenset(config.joint_names)

        # ONNX session
        self._session = ort.InferenceSession(
            str(config.policy_path),
            providers=ort.get_available_providers(),
        )
        logger.info(
            f"RLPolicyTask '{name}' loaded policy: {config.policy_path} "
            f"(providers: {self._session.get_providers()})"
        )

        # Numpy arrays for efficient computation
        self._default_pos = np.array(config.default_positions, dtype=np.float32)
        self._last_action = np.zeros(_NUM_MOTORS, dtype=np.float32)

        # Build inverse map: robot_motor_idx → isaac_idx
        self._robot_to_isaac = np.zeros(_NUM_MOTORS, dtype=np.intp)
        for isaac_idx, robot_idx in enumerate(config.isaac_to_robot_map):
            self._robot_to_isaac[robot_idx] = isaac_idx
        self._isaac_to_robot = np.array(config.isaac_to_robot_map, dtype=np.intp)

        # Velocity command input (thread-safe)
        self._lock = threading.Lock()
        self._velocity_cmd = np.zeros(3, dtype=np.float32)
        self._last_cmd_time: float = 0.0

        # State
        self._active = False
        self._tick_count = 0

        # Last computed targets for re-sending on decimation skip ticks
        self._last_targets_robot: list[float] | None = None

        logger.info(
            f"RLPolicyTask '{name}' initialized: "
            f"decimation={config.decimation}, priority={config.priority}"
        )

    @property
    def name(self) -> str:
        return self._name

    def claim(self) -> ResourceClaim:
        return ResourceClaim(
            joints=self._joint_names_set,
            priority=self._config.priority,
            mode=ControlMode.SERVO_POSITION,
        )

    def is_active(self) -> bool:
        return self._active

    def compute(self, state: CoordinatorState) -> JointCommandOutput | None:
        if not self._active:
            return None

        self._tick_count += 1

        # Decimation: only run inference every N ticks
        if self._tick_count % self._config.decimation != 0:
            # Re-send last computed targets to keep the robot commanded
            if self._last_targets_robot is not None:
                return JointCommandOutput(
                    joint_names=self._joint_names_list,
                    positions=self._last_targets_robot,
                    mode=ControlMode.SERVO_POSITION,
                )
            return None

        # Read joint state from CoordinatorState (robot order) ---
        joint_pos_robot = np.zeros(_NUM_MOTORS, dtype=np.float32)
        joint_vel_robot = np.zeros(_NUM_MOTORS, dtype=np.float32)
        for i, jname in enumerate(self._joint_names_list):
            pos = state.joints.get_position(jname)
            vel = state.joints.get_velocity(jname)
            joint_pos_robot[i] = pos if pos is not None else 0.0
            joint_vel_robot[i] = vel if vel is not None else 0.0

        # Read IMU from adapter ---
        imu = self._adapter.read_imu()

        # Remap robot order → Isaac Lab order ---
        joint_pos_isaac = joint_pos_robot[self._robot_to_isaac]
        joint_vel_isaac = joint_vel_robot[self._robot_to_isaac]

        # Compute projected gravity ---
        proj_grav = self._projected_gravity(imu.quaternion)

        # Get velocity command (with timeout) ---
        with self._lock:
            if (
                self._config.timeout > 0
                and self._last_cmd_time > 0
                and (state.t_now - self._last_cmd_time) > self._config.timeout
            ):
                vel_cmd = np.zeros(3, dtype=np.float32)
            else:
                vel_cmd = self._velocity_cmd.copy()

        # Build observation (45 dims) ---
        obs = np.concatenate(
            [
                np.array(imu.gyroscope, dtype=np.float32) * self._config.obs_ang_vel_scale,
                proj_grav,
                vel_cmd,
                joint_pos_isaac - self._default_pos,
                joint_vel_isaac * self._config.obs_joint_vel_scale,
                self._last_action,
            ]
        ).reshape(1, -1)

        # Run ONNX inference ---
        action = self._session.run(None, {"obs": obs})[0][0]
        self._last_action = action.copy()

        # Compute joint targets (Isaac Lab order) ---
        target_pos_isaac = action * self._config.action_scale + self._default_pos

        # Remap Isaac Lab → robot order ---
        target_pos_robot = np.zeros(_NUM_MOTORS, dtype=np.float32)
        for isaac_idx, robot_idx in enumerate(self._isaac_to_robot):
            target_pos_robot[robot_idx] = target_pos_isaac[isaac_idx]

        self._last_targets_robot = target_pos_robot.tolist()

        return JointCommandOutput(
            joint_names=self._joint_names_list,
            positions=self._last_targets_robot,
            mode=ControlMode.SERVO_POSITION,
        )

    def on_preempted(self, by_task: str, joints: frozenset[str]) -> None:
        if joints & self._joint_names_set:
            logger.warning(f"RLPolicyTask '{self._name}' preempted by {by_task} on {joints}")

    # Velocity command input ---

    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float, t_now: float) -> None:
        """Set the velocity command input to the policy.

        Args:
            vx: Forward/backward velocity (m/s)
            vy: Left/right lateral velocity (m/s)
            yaw_rate: Yaw rotation rate (rad/s)
            t_now: Current time (from coordinator or perf_counter)
        """
        with self._lock:
            self._velocity_cmd[:] = [vx, vy, yaw_rate]
            self._last_cmd_time = t_now

    # Lifecycle ---

    def start(self) -> None:
        """Activate the task."""
        self._active = True
        self._tick_count = 0
        self._last_action[:] = 0.0
        self._last_targets_robot = None
        logger.info(f"RLPolicyTask '{self._name}' started")

    def stop(self) -> None:
        """Deactivate the task."""
        self._active = False
        self._last_targets_robot = None
        logger.info(f"RLPolicyTask '{self._name}' stopped")

    # Internal helpers ---

    @staticmethod
    def _projected_gravity(quaternion: tuple[float, ...]) -> np.ndarray:
        """Compute projected gravity from IMU quaternion.

        Args:
            quaternion: (w, x, y, z) from IMU

        Returns:
            3D projected gravity vector in body frame
        """
        w, x, y, z = quaternion
        # Rotation matrix from quaternion (body → world), then R^T @ [0,0,-1]
        gx = -2.0 * (x * z - w * y)
        gy = -2.0 * (y * z + w * x)
        gz = -(w * w - x * x - y * y + z * z)
        return np.array([gx, gy, gz], dtype=np.float32)


__all__ = [
    "RLPolicyTask",
    "RLPolicyTaskConfig",
]
