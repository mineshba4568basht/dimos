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

import asyncio

# Import LCM message types
from dimos_lcm.sensor_msgs import CameraInfo

from dimos import core
from dimos.hardware.camera.realsense import RealSenseModule
from dimos.manipulation.visual_servoing.manipulation_module import ManipulationModule
from dimos.msgs.sensor_msgs import Image
from dimos.protocol import pubsub
from dimos.robot.foxglove_bridge import FoxgloveBridge
from dimos.robot.robot import Robot
from dimos.skills.skills import SkillLibrary
from dimos.types.robot_capabilities import RobotCapability
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.robot.lerobot.so101_arm")


class SO101ArmRobot(Robot):
    """SO101 Arm robot with RGB camera and manipulation capabilities."""

    def __init__(self, robot_capabilities: list[RobotCapability] | None = None) -> None:
        super().__init__()
        self.dimos = None
        self.camera = None
        self.manipulation_interface = None
        self.skill_library = SkillLibrary()

        # Initialize capabilities
        self.capabilities = robot_capabilities or [
            RobotCapability.VISION,
            RobotCapability.MANIPULATION,
        ]

    async def start(self) -> None:
        """Start the robot modules."""
        self.dimos = core.start(2)
        self.foxglove_bridge = FoxgloveBridge()

        pubsub.lcm.autoconf()

        # Deploy Camera Module (Can replace with ZED module)
        logger.info("Deploying camera module...")
        self.camera = self.dimos.deploy(
            RealSenseModule,
            serial_number="215322078948",
            width=640,
            height=480,
            fps=30,
            enable_color=True,
            enable_depth=True,
            align_depth_to_color=True,
        )

        # Configure camera LCM
        self.camera.color_image.transport = core.LCMTransport("/camera/rgb", Image)
        self.camera.depth_image.transport = core.LCMTransport("/camera/depth", Image)
        self.camera.camera_info.transport = core.LCMTransport("/camera/info", CameraInfo)

        # Deploy manipulation module
        logger.info("Deploying manipulation module...")
        self.manipulation_interface = self.dimos.deploy(
            ManipulationModule,
            arm="so101",
        )

        # Connect modules
        self.manipulation_interface.rgb_image.connect(self.camera.color_image)
        self.manipulation_interface.depth_image.connect(self.camera.depth_image)
        self.manipulation_interface.camera_info.connect(self.camera.camera_info)

        # Configure manipulation output
        self.manipulation_interface.viz_image.transport = core.LCMTransport("/viz/output", Image)

        # Print module info
        logger.info("Modules configured:")
        print("\nCamera Module:")
        print(self.camera.io())
        print("\nManipulation Module:")
        print(self.manipulation_interface.io())

        # Start modules
        logger.info("Starting modules...")
        self.foxglove_bridge.start()
        self.camera.start()
        self.manipulation_interface.start()

        await asyncio.sleep(2)
        logger.info("SO101ArmRobot initialized and started")

    def pick_and_place(
        self, pick_x: int, pick_y: int, place_x: int | None = None, place_y: int | None = None
    ):
        """Execute pick and place task.

        Args:
            pick_x: X coordinate for pick location
            pick_y: Y coordinate for pick location
            place_x: X coordinate for place location (optional)
            place_y: Y coordinate for place location (optional)

        Returns:
            Result of the pick and place operation
        """
        if self.manipulation_interface:
            return self.manipulation_interface.pick_and_place(pick_x, pick_y, place_x, place_y)
        else:
            logger.error("Manipulation module not initialized")
            return False

    def handle_keyboard_command(self, key: str):
        """Pass keyboard commands to manipulation module.

        Args:
            key: Keyboard key pressed

        Returns:
            Action taken or None
        """
        if self.manipulation_interface:
            return self.manipulation_interface.handle_keyboard_command(key)
        else:
            logger.error("Manipulation module not initialized")
            return None

    def stop(self) -> None:
        """Stop all modules and clean up."""
        logger.info("Stopping SO101ArmRobot...")

        try:
            if self.manipulation_interface:
                self.manipulation_interface.stop()

            if self.camera:
                self.camera.stop()
        except Exception as e:
            logger.warning(f"Error stopping modules: {e}")

        if self.dimos:
            self.dimos.close()

        logger.info("SO101ArmRobot stopped")

    # ------------------------------------------------------------------
    # Required abstract methods from Robot base class
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up robot resources.

        Implements the abstract ``cleanup`` method from ``Robot`` by delegating
        to ``stop()``, which already performs the shutdown logic.
        """
        self.stop()


async def run_so101_arm() -> None:
    """Run the SO101 Arm robot."""
    robot = SO101ArmRobot()
    await robot.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        robot.stop()


if __name__ == "__main__":
    asyncio.run(run_so101_arm())
