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

"""Configuration for the eval generator and runner modules."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from dimos.agents.spec import Model, Provider
from dimos.core.module import ModuleConfig


class MatchMode(str, Enum):
    """How to compare tool calls for evaluation."""

    EXACT = "exact"  # Tool name and arguments must match exactly
    SEMANTIC = "semantic"  # Use LLM to judge semantic equivalence


@dataclass
class EvalGeneratorConfig(ModuleConfig):
    """Configuration for the EvalGenerator module."""

    model: str = Model.GPT_4O.value
    provider: Provider = Provider.OPENAI

    # Generation settings
    num_evals: int = 100
    include_single_turn: bool = True
    include_multi_turn: bool = True
    max_turns_per_conversation: int = 5

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./evals"))
    output_format: Literal["jsonl", "json", "both"] = "both"

    # LLM settings
    generation_prompt: str | None = None
    temperature: float = 0.7
    batch_size: int = 10


@dataclass
class EvalRunnerConfig(ModuleConfig):
    """Configuration for the EvalRunner module."""

    # Model to evaluate
    model: str = Model.GPT_4O.value
    provider: Provider = Provider.OPENAI

    # Evaluation settings
    match_mode: MatchMode = MatchMode.EXACT
    temperature: float = 0.0  # Use deterministic outputs for evaluation

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./eval_results"))
    save_results: bool = True

    # Execution settings
    max_concurrent: int = 5  # Max concurrent eval runs
    timeout_seconds: float = 30.0  # Timeout per eval

    # Semantic matching settings (only used when match_mode is SEMANTIC)
    judge_model: str = Model.GPT_4O.value
    judge_provider: Provider = Provider.OPENAI
