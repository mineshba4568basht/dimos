# Copyright 2026 Dimensional Inc.
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

import os
import threading
import time

import numpy as np
from openpi_client import websocket_client_policy
from xarm.wrapper import XArmAPI

from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs import Image, JointState
from dimos.msgs.sensor_msgs.image_impls.AbstractImage import ImageFormat

ACTION_HORIZON = 15
ACTION_CHUNK = None
GRIPPER_CHUNK = None


def get_camera_image(timeout: float = 5.0, topic: str = "/camera/color") -> np.ndarray:
    event = threading.Event()
    image_data: dict[str, np.ndarray] = {}

    def on_img(msg: Image) -> None:
        if event.is_set():
            return
        image_data["image"] = msg.to_rgb().to_opencv()
        os.makedirs("captures", exist_ok=True)
        filename = f"camera_color_{time.time()}.png"
        Image.from_numpy(image_data["image"], format=ImageFormat.RGB).save(
            os.path.join("captures", filename)
        )
        event.set()

    transport = LCMTransport(topic, Image)
    transport.subscribe(on_img)

    if not event.wait(timeout=timeout):
        raise TimeoutError(f"No image received on {topic} within {timeout} seconds.")

    return image_data["image"]


def get_xarm_joint_positions(timeout: float = 5.0, topic: str = "/xarm/joint_states"):
    event = threading.Event()
    joint_positions: dict[str, np.ndarray] = {}

    def on_joint_state(msg: JointState) -> None:
        if event.is_set():
            return
        joint_positions["joint_positions"] = msg.position
        event.set()

    transport = LCMTransport(topic, JointState)
    transport.subscribe(on_joint_state)

    if not event.wait(timeout=timeout):
        raise TimeoutError(f"No joint states received on {topic} within {timeout} seconds.")

    return joint_positions["joint_positions"]


def get_observation():
    return {
        "observation/exterior_image_1_left": get_camera_image(),  # ADD SECOND CAMERA IN BLUEPRINT DEFINED WITH SERIAL NUMBER
        "observation/wrist_image_left": get_camera_image(),
        "observation/joint_position": get_xarm_joint_positions(),
        "observation/gripper_position": 0.0,
        "prompt": "move the arm slightly to the left",
    }


def franka_to_xarm(franka_joint_positions):
    offsets = np.array([0, 0, 0, 180, 0, 180, 0])
    return offsets - franka_joint_positions


def xarm_to_franka(xarm_joint_positions):
    offsets = np.array([0, 0, 0, 180, 0, 180, 0])
    return offsets + xarm_joint_positions


def run_inference():
    """
    Run inference loop until user interrupts
    """
    actions_from_chunk_completed = 0
    while True:
        if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= ACTION_HORIZON:
            actions_from_chunk_completed = 0
            observation = get_observation()

            result = policy.infer(observation)
            action_chunk = result["actions"]  # Shape: (15, 8) - these are VELOCITY COMMANDS

            dt = 1.0 / 15.0
            action_chunk = action_chunk.copy()
            action_chunk[:, :-1] *= dt
            action_chunk[:, :-1] = np.cumsum(
                action_chunk[:, :-1], axis=0
            )  # integrate to get delta position in radians
            current_joint_positions = np.array(arm.get_servo_angle()[1])
            action_chunk[:, :-1] += current_joint_positions
            action_chunk[:, :-1] *= 360 / (2 * np.pi)  # convert to degrees
            ACTION_CHUNK = action_chunk
            GRIPPER_CHUNK = action_chunk[:, 7]
            GRIPPER_CHUNK = np.where(GRIPPER_CHUNK > 0.5, 0.0, GRIPPER_CHUNK)

        action = ACTION_CHUNK[actions_from_chunk_completed]
        gripper_xarm = (1.0 - GRIPPER_CHUNK[actions_from_chunk_completed]) * 850
        actions_from_chunk_completed += 1

        print(f"Setting joint positions: {action[:7]} and gripper position: {gripper_xarm}")
        arm.set_servo_angle(angle=action[:7], speed=10, wait=False)
        # arm.set_gripper_position(pos=gripper_xarm, speed=50, wait=False)


if __name__ == "__main__":
    # connect to policy server
    policy = websocket_client_policy.WebsocketClientPolicy(
        host="localhost",
        port=8000,
    )

    # connect to xArm
    arm = XArmAPI("192.168.2.235")
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    time.sleep(1)
    arm.move_gohome(wait=True)

    observation = get_observation()

    run_inference()

    arm.disconnect()
