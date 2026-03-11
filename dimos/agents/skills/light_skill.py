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

import requests

from dimos.agents.annotation import skill
from dimos.core.module import Module
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Sonoff S31 smart plug running ESPHome
_SONOFF_HOST = "http://10.0.0.201"
_RELAY_ENTITY = "switch/sonoff_s31_relay"


class LightSkill(Module):
    def start(self) -> None:
        super().start()

    def stop(self) -> None:
        super().stop()

    @skill
    def turn_on_lights(self) -> str:
        """Turn on the office lights.

        Example:

            turn_on_lights()
        """
        try:
            r = requests.post(
                f"{_SONOFF_HOST}/{_RELAY_ENTITY}/turn_on",
                timeout=5,
            )
            r.raise_for_status()
            return "Lights turned ON"
        except Exception as e:
            logger.warning(f"Failed to turn on lights: {e}")
            return f"Error turning on lights: {e}"

    @skill
    def turn_off_lights(self) -> str:
        """Turn off the office lights.

        Example:

            turn_off_lights()
        """
        try:
            r = requests.post(
                f"{_SONOFF_HOST}/{_RELAY_ENTITY}/turn_off",
                timeout=5,
            )
            r.raise_for_status()
            return "Lights turned OFF"
        except Exception as e:
            logger.warning(f"Failed to turn off lights: {e}")
            return f"Error turning off lights: {e}"


light_skill = LightSkill.blueprint

__all__ = ["LightSkill", "light_skill"]
