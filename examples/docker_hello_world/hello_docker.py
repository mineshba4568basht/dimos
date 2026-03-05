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

"""
Hello World Docker Module
==========================

Minimal example showing a DimOS module running inside Docker.

The module receives a string on its ``prompt`` input stream, runs it through
cowsay inside the container, and publishes the ASCII art on its ``greeting``
output stream.

NOTE: Requires Linux. Docker Desktop on macOS does not support host networking,
which is needed for LCM multicast between host and container.

Usage:
    python examples/docker_hello_world/hello_docker.py
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import time

from dimos.core.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.core.docker_runner import DockerModuleConfig
from dimos.core.module import Module
from dimos.core.stream import In, Out

# ---------------------------------------------------------------------------
# Docker module (runs inside container)
# ---------------------------------------------------------------------------


class HelloDockerConfig(DockerModuleConfig):
    docker_image: str = "dimos-hello-docker:latest"
    docker_file: Path | None = Path(__file__).parent / "Dockerfile"
    docker_build_context: Path | None = Path(__file__).parents[2]  # repo root
    docker_gpus: str | None = None  # no GPU needed
    docker_rm: bool = True
    docker_restart_policy: str = "no"
    docker_env: dict[str, str] = {"CI": "1"}  # skip interactive system configurator


class HelloDockerModule(Module["HelloDockerConfig"]):
    """A trivial module that runs inside Docker and echoes greetings."""

    default_config = HelloDockerConfig

    prompt: In[str]
    greeting: Out[str]

    @rpc
    def start(self) -> None:
        super().start()
        self.prompt.subscribe(self._on_prompt)

    def _cowsay(self, text: str) -> str:
        """Run cowsay inside the container and return the ASCII art."""
        result = subprocess.run(
            ["/usr/games/cowsay", text],
            capture_output=True,
            text=True,
        )
        return result.stdout

    def _on_prompt(self, text: str) -> None:
        art = self._cowsay(text)
        print(f"[HelloDockerModule]\n{art}")
        self.greeting.publish(art)

    @rpc
    def greet(self, name: str) -> str:
        """RPC method that can be called directly."""
        return self._cowsay(f"Hello, {name}!")


# ---------------------------------------------------------------------------
# Host-side module (sends prompts and prints greetings)
# ---------------------------------------------------------------------------


class PromptModule(Module):
    """Publishes prompts and listens to greetings."""

    prompt: Out[str]
    greeting: In[str]

    @rpc
    def start(self) -> None:
        super().start()
        self.greeting.subscribe(self._on_greeting)

    def _on_greeting(self, text: str) -> None:
        print(f"[PromptModule] Received: {text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    coordinator = autoconnect(
        PromptModule.blueprint(),
        HelloDockerModule.blueprint(),
    ).build()

    # Get module proxies
    prompt_mod = coordinator.get_instance(PromptModule)
    docker_mod = coordinator.get_instance(HelloDockerModule)

    # Test RPC
    print(docker_mod.greet("World"))

    # Test stream
    prompt_mod.prompt.publish("stream test")
    time.sleep(2)

    coordinator.close_all()
    print("Done!")
