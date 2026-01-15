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

"""Eval runner module for evaluating generated evals against a model."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from dimos.agents.evals.config import EvalRunnerConfig, MatchMode
from dimos.agents.spec import ToolSchemaList
from dimos.core.module import Module
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class ToolCallResult:
    """Result of comparing a single tool call."""

    expected_name: str
    expected_args: dict[str, Any]
    actual_name: str | None
    actual_args: dict[str, Any] | None
    name_match: bool
    args_match: bool
    is_match: bool


@dataclass
class EvalResult:
    """Result of running a single eval."""

    eval_id: str
    user_query: str
    expected_tool_calls: list[dict[str, Any]]
    actual_tool_calls: list[dict[str, Any]]
    tool_call_results: list[ToolCallResult]
    passed: bool
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class EvalRunSummary:
    """Summary of an evaluation run."""

    model: str = ""
    provider: str = ""

    total_evals: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0

    # Tool call metrics
    total_expected_calls: int = 0
    total_actual_calls: int = 0
    correct_tool_names: int = 0
    correct_tool_args: int = 0
    exact_matches: int = 0

    # Timing
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # Results
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate."""
        if self.total_evals == 0:
            return 0.0
        return self.passed / self.total_evals

    @property
    def tool_name_accuracy(self) -> float:
        """Calculate tool name matching accuracy."""
        if self.total_expected_calls == 0:
            return 0.0
        return self.correct_tool_names / self.total_expected_calls

    @property
    def tool_args_accuracy(self) -> float:
        """Calculate tool arguments matching accuracy."""
        if self.total_expected_calls == 0:
            return 0.0
        return self.correct_tool_args / self.total_expected_calls

    @property
    def exact_match_accuracy(self) -> float:
        """Calculate exact match accuracy (name + args)."""
        if self.total_expected_calls == 0:
            return 0.0
        return self.exact_matches / self.total_expected_calls

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "model": self.model,
            "provider": self.provider,
            "total_evals": self.total_evals,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "total_expected_calls": self.total_expected_calls,
            "total_actual_calls": self.total_actual_calls,
            "correct_tool_names": self.correct_tool_names,
            "correct_tool_args": self.correct_tool_args,
            "exact_matches": self.exact_matches,
            "tool_name_accuracy": self.tool_name_accuracy,
            "tool_args_accuracy": self.tool_args_accuracy,
            "exact_match_accuracy": self.exact_match_accuracy,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class ModelSpec:
    """Specification for a model to evaluate."""

    model: str
    provider: str
    temperature: float = 0.0


@dataclass
class ModelComparisonResult:
    """Result of comparing multiple models on the same eval dataset."""

    summaries: list[EvalRunSummary] = field(default_factory=list)
    eval_file: str = ""
    compared_at: str = ""

    def get_ranking(self, metric: str = "pass_rate") -> list[tuple[str, float]]:
        """Get models ranked by a metric.

        Args:
            metric: One of 'pass_rate', 'tool_name_accuracy', 'tool_args_accuracy',
                   'exact_match_accuracy', 'avg_latency_ms'.

        Returns:
            List of (model_name, metric_value) sorted by metric descending.
        """
        results = []
        for summary in self.summaries:
            model_name = f"{summary.provider}/{summary.model}"
            value = getattr(summary, metric, 0.0)
            results.append((model_name, value))

        # Sort descending for accuracy metrics, ascending for latency
        reverse = metric != "avg_latency_ms"
        return sorted(results, key=lambda x: x[1], reverse=reverse)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "eval_file": self.eval_file,
            "compared_at": self.compared_at,
            "summaries": [s.to_dict() for s in self.summaries],
            "rankings": {
                "by_pass_rate": self.get_ranking("pass_rate"),
                "by_exact_match": self.get_ranking("exact_match_accuracy"),
                "by_latency": self.get_ranking("avg_latency_ms"),
            },
        }


class EvalRunner(Module):
    """Module for running evaluations on generated evals.

    Example:
        ```python
        from dimos.agents.evals import EvalRunner, MatchMode

        runner = EvalRunner(model="gpt-4o", match_mode=MatchMode.EXACT)
        summary = runner.run_from_file("evals/evals_20240101_120000.json")
        print(f"Pass rate: {summary.pass_rate:.2%}")
        ```
    """

    config: EvalRunnerConfig
    default_config: type[EvalRunnerConfig] = EvalRunnerConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._llm: BaseChatModel | None = None
        self._judge_llm: BaseChatModel | None = None

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize the LLM being evaluated."""
        if self._llm is None:
            self._llm = init_chat_model(
                model_provider=self.config.provider.value,
                model=self.config.model,
                temperature=self.config.temperature,
            )
        return self._llm

    @property
    def judge_llm(self) -> BaseChatModel:
        """Lazily initialize the judge LLM for semantic matching."""
        if self._judge_llm is None:
            self._judge_llm = init_chat_model(
                model_provider=self.config.judge_provider.value,
                model=self.config.judge_model,
                temperature=0.0,
            )
        return self._judge_llm

    def run_from_file(
        self,
        eval_file: str | Path,
        tools: ToolSchemaList | None = None,
    ) -> EvalRunSummary:
        """Run evaluations from a file.

        Args:
            eval_file: Path to JSON or JSONL file with evals.
            tools: Optional tool definitions. If not provided, uses tools from file.

        Returns:
            Summary of evaluation results.
        """
        eval_file = Path(eval_file)

        if eval_file.suffix == ".jsonl":
            evals = self._load_jsonl(eval_file)
        elif eval_file.suffix == ".json":
            evals, file_tools = self._load_json(eval_file)
            if tools is None:
                tools = file_tools
        else:
            raise ValueError(f"Unsupported file format: {eval_file.suffix}")

        return self.run_evals(evals, tools)

    def compare_models(
        self,
        eval_file: str | Path,
        models: list[ModelSpec],
        tools: ToolSchemaList | None = None,
    ) -> ModelComparisonResult:
        """Compare multiple models on the same eval dataset.

        Args:
            eval_file: Path to JSON or JSONL file with evals.
            models: List of model specifications to evaluate.
            tools: Optional tool definitions. If not provided, uses tools from file.

        Returns:
            Comparison result with summaries for each model.

        Example:
            ```python
            from dimos.agents.evals import EvalRunner, ModelSpec

            runner = EvalRunner()
            result = runner.compare_models(
                "evals/evals_20240101.json",
                models=[
                    ModelSpec(model="gpt-4o", provider="openai"),
                    ModelSpec(model="gpt-4o-mini", provider="openai"),
                    ModelSpec(model="claude-3-5-sonnet-latest", provider="anthropic"),
                ],
            )
            print(result.get_ranking("pass_rate"))
            ```
        """
        eval_file = Path(eval_file)

        # Load evals once
        if eval_file.suffix == ".jsonl":
            evals = self._load_jsonl(eval_file)
            file_tools = None
        elif eval_file.suffix == ".json":
            evals, file_tools = self._load_json(eval_file)
        else:
            raise ValueError(f"Unsupported file format: {eval_file.suffix}")

        if tools is None:
            tools = file_tools

        # Run evals for each model
        comparison = ModelComparisonResult(
            eval_file=str(eval_file),
            compared_at=datetime.now().isoformat(),
        )

        for model_spec in models:
            logger.info(f"Evaluating model: {model_spec.provider}/{model_spec.model}")
            summary = self._run_evals_with_model(evals, tools, model_spec)
            comparison.summaries.append(summary)

        # Save comparison results
        if self.config.save_results:
            self._save_comparison_results(comparison)

        return comparison

    def _run_evals_with_model(
        self,
        evals: list[dict[str, Any]],
        tools: ToolSchemaList | None,
        model_spec: ModelSpec,
    ) -> EvalRunSummary:
        """Run evals with a specific model.

        Args:
            evals: List of eval examples.
            tools: Tool definitions.
            model_spec: Model specification to use.

        Returns:
            Summary of evaluation results.
        """
        # Create a new LLM for this model
        llm = init_chat_model(
            model_provider=model_spec.provider,
            model=model_spec.model,
            temperature=model_spec.temperature,
        )

        summary = EvalRunSummary(
            model=model_spec.model,
            provider=model_spec.provider,
        )
        summary.total_evals = len(evals)

        for i, eval_item in enumerate(evals):
            eval_id = f"eval_{i}"
            messages = eval_item.get("messages", [])
            eval_tools = tools or eval_item.get("tools", [])

            try:
                result = self._run_single_eval_with_llm(eval_id, messages, eval_tools, llm)
                summary.results.append(result)

                if result.error:
                    summary.errors += 1
                elif result.passed:
                    summary.passed += 1
                else:
                    summary.failed += 1

                # Aggregate metrics
                summary.total_expected_calls += len(result.expected_tool_calls)
                summary.total_actual_calls += len(result.actual_tool_calls)
                summary.total_latency_ms += result.latency_ms

                for tc_result in result.tool_call_results:
                    if tc_result.name_match:
                        summary.correct_tool_names += 1
                    if tc_result.args_match:
                        summary.correct_tool_args += 1
                    if tc_result.is_match:
                        summary.exact_matches += 1

            except Exception as e:
                logger.error(f"Error running eval {eval_id}: {e}")
                summary.errors += 1
                summary.results.append(
                    EvalResult(
                        eval_id=eval_id,
                        user_query="",
                        expected_tool_calls=[],
                        actual_tool_calls=[],
                        tool_call_results=[],
                        passed=False,
                        error=str(e),
                    )
                )

        # Calculate average latency
        if summary.total_evals > 0:
            summary.avg_latency_ms = summary.total_latency_ms / summary.total_evals

        return summary

    def _run_single_eval_with_llm(
        self,
        eval_id: str,
        messages: list[dict[str, Any]],
        tools: ToolSchemaList,
        llm: BaseChatModel,
    ) -> EvalResult:
        """Run a single evaluation with a specific LLM.

        Args:
            eval_id: Identifier for this eval.
            messages: The conversation messages.
            tools: Available tool definitions.
            llm: The LLM to use for this eval.

        Returns:
            Result of the evaluation.
        """
        # Extract user query and expected tool calls from messages
        user_query = ""
        expected_tool_calls: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                user_query = msg.get("content", "")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    expected_tool_calls.append({
                        "name": func.get("name", ""),
                        "arguments": json.loads(func.get("arguments", "{}")),
                    })

        if not user_query:
            return EvalResult(
                eval_id=eval_id,
                user_query="",
                expected_tool_calls=expected_tool_calls,
                actual_tool_calls=[],
                tool_call_results=[],
                passed=False,
                error="No user query found in eval",
            )

        # Build messages for LLM
        llm_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                llm_messages.append(SystemMessage(content=content))
            elif role == "user":
                llm_messages.append(HumanMessage(content=content))
                break  # Only include up to the first user message for single-turn

        # Bind tools and invoke
        start_time = datetime.now()
        llm_with_tools = llm.bind_tools(tools)
        response: AIMessage = llm_with_tools.invoke(llm_messages)
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Extract actual tool calls
        actual_tool_calls: list[dict[str, Any]] = []
        for tc in response.tool_calls:
            actual_tool_calls.append({
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
            })

        # Compare tool calls
        tool_call_results = self._compare_tool_calls(
            expected_tool_calls, actual_tool_calls
        )

        # Determine if passed based on match mode
        passed = self._determine_pass(tool_call_results)

        return EvalResult(
            eval_id=eval_id,
            user_query=user_query,
            expected_tool_calls=expected_tool_calls,
            actual_tool_calls=actual_tool_calls,
            tool_call_results=tool_call_results,
            passed=passed,
            latency_ms=latency_ms,
        )

    def run_evals(
        self,
        evals: list[dict[str, Any]],
        tools: ToolSchemaList | None = None,
    ) -> EvalRunSummary:
        """Run evaluations on a list of eval examples.

        Args:
            evals: List of eval examples in OpenAI format.
            tools: Tool definitions for the model.

        Returns:
            Summary of evaluation results.
        """
        summary = EvalRunSummary(
            model=self.config.model,
            provider=self.config.provider.value,
        )
        summary.total_evals = len(evals)

        for i, eval_item in enumerate(evals):
            eval_id = f"eval_{i}"
            messages = eval_item.get("messages", [])
            eval_tools = tools or eval_item.get("tools", [])

            try:
                result = self._run_single_eval(eval_id, messages, eval_tools)
                summary.results.append(result)

                if result.error:
                    summary.errors += 1
                elif result.passed:
                    summary.passed += 1
                else:
                    summary.failed += 1

                # Aggregate metrics
                summary.total_expected_calls += len(result.expected_tool_calls)
                summary.total_actual_calls += len(result.actual_tool_calls)
                summary.total_latency_ms += result.latency_ms

                for tc_result in result.tool_call_results:
                    if tc_result.name_match:
                        summary.correct_tool_names += 1
                    if tc_result.args_match:
                        summary.correct_tool_args += 1
                    if tc_result.is_match:
                        summary.exact_matches += 1

            except Exception as e:
                logger.error(f"Error running eval {eval_id}: {e}")
                summary.errors += 1
                summary.results.append(
                    EvalResult(
                        eval_id=eval_id,
                        user_query="",
                        expected_tool_calls=[],
                        actual_tool_calls=[],
                        tool_call_results=[],
                        passed=False,
                        error=str(e),
                    )
                )

        # Calculate average latency
        if summary.total_evals > 0:
            summary.avg_latency_ms = summary.total_latency_ms / summary.total_evals

        # Save results if configured
        if self.config.save_results:
            self._save_results(summary)

        return summary

    def _run_single_eval(
        self,
        eval_id: str,
        messages: list[dict[str, Any]],
        tools: ToolSchemaList,
    ) -> EvalResult:
        """Run a single evaluation example.

        Args:
            eval_id: Identifier for this eval.
            messages: The conversation messages.
            tools: Available tool definitions.

        Returns:
            Result of the evaluation.
        """
        # Extract user query and expected tool calls from messages
        user_query = ""
        expected_tool_calls: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                user_query = msg.get("content", "")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    expected_tool_calls.append({
                        "name": func.get("name", ""),
                        "arguments": json.loads(func.get("arguments", "{}")),
                    })

        if not user_query:
            return EvalResult(
                eval_id=eval_id,
                user_query="",
                expected_tool_calls=expected_tool_calls,
                actual_tool_calls=[],
                tool_call_results=[],
                passed=False,
                error="No user query found in eval",
            )

        # Build messages for LLM
        llm_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                llm_messages.append(SystemMessage(content=content))
            elif role == "user":
                llm_messages.append(HumanMessage(content=content))
                break  # Only include up to the first user message for single-turn

        # Bind tools and invoke
        start_time = datetime.now()
        llm_with_tools = self.llm.bind_tools(tools)
        response: AIMessage = llm_with_tools.invoke(llm_messages)
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Extract actual tool calls
        actual_tool_calls: list[dict[str, Any]] = []
        for tc in response.tool_calls:
            actual_tool_calls.append({
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
            })

        # Compare tool calls
        tool_call_results = self._compare_tool_calls(
            expected_tool_calls, actual_tool_calls
        )

        # Determine if passed based on match mode
        passed = self._determine_pass(tool_call_results)

        return EvalResult(
            eval_id=eval_id,
            user_query=user_query,
            expected_tool_calls=expected_tool_calls,
            actual_tool_calls=actual_tool_calls,
            tool_call_results=tool_call_results,
            passed=passed,
            latency_ms=latency_ms,
        )

    def _compare_tool_calls(
        self,
        expected: list[dict[str, Any]],
        actual: list[dict[str, Any]],
    ) -> list[ToolCallResult]:
        """Compare expected and actual tool calls.

        Args:
            expected: Expected tool calls.
            actual: Actual tool calls from model.

        Returns:
            List of comparison results.
        """
        results: list[ToolCallResult] = []

        # Match expected to actual (order-sensitive for now)
        for i, exp in enumerate(expected):
            exp_name = exp.get("name", "")
            exp_args = exp.get("arguments", {})

            if i < len(actual):
                act = actual[i]
                act_name = act.get("name", "")
                act_args = act.get("arguments", {})

                name_match = exp_name == act_name
                args_match = self._compare_arguments(exp_args, act_args)

                if self.config.match_mode == MatchMode.SEMANTIC:
                    is_match = self._semantic_match(exp, act)
                else:  # EXACT
                    is_match = name_match and args_match

                results.append(ToolCallResult(
                    expected_name=exp_name,
                    expected_args=exp_args,
                    actual_name=act_name,
                    actual_args=act_args,
                    name_match=name_match,
                    args_match=args_match,
                    is_match=is_match,
                ))
            else:
                # Expected call not made
                results.append(ToolCallResult(
                    expected_name=exp_name,
                    expected_args=exp_args,
                    actual_name=None,
                    actual_args=None,
                    name_match=False,
                    args_match=False,
                    is_match=False,
                ))

        return results

    def _compare_arguments(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> bool:
        """Compare tool call arguments.

        Args:
            expected: Expected arguments.
            actual: Actual arguments.

        Returns:
            True if arguments match.
        """
        # Normalize and compare
        return self._normalize_args(expected) == self._normalize_args(actual)

    def _normalize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize arguments for comparison.

        Handles type coercion (e.g., int vs float) and sorting.
        """
        normalized = {}
        for k, v in args.items():
            if isinstance(v, float) and v.is_integer():
                normalized[k] = int(v)
            elif isinstance(v, dict):
                normalized[k] = self._normalize_args(v)
            elif isinstance(v, list):
                normalized[k] = sorted(v) if all(isinstance(x, (str, int, float)) for x in v) else v
            else:
                normalized[k] = v
        return normalized

    def _semantic_match(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> bool:
        """Use LLM judge for semantic matching.

        Args:
            expected: Expected tool call.
            actual: Actual tool call.

        Returns:
            True if semantically equivalent.
        """
        prompt = f"""Compare these two tool calls and determine if they are semantically equivalent.
They are equivalent if they would accomplish the same goal, even if the exact arguments differ slightly.

Expected tool call:
- Name: {expected.get('name')}
- Arguments: {json.dumps(expected.get('arguments', {}))}

Actual tool call:
- Name: {actual.get('name')}
- Arguments: {json.dumps(actual.get('arguments', {}))}

Respond with ONLY "YES" if they are semantically equivalent, or "NO" if they are not."""

        response = self.judge_llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        return "YES" in content.upper()

    def _determine_pass(self, results: list[ToolCallResult]) -> bool:
        """Determine if the eval passed based on tool call results.

        Args:
            results: List of tool call comparison results.

        Returns:
            True if the eval passed.
        """
        if not results:
            return False
        return all(r.is_match for r in results)

    def _load_jsonl(self, filepath: Path) -> list[dict[str, Any]]:
        """Load evals from JSONL file.

        Args:
            filepath: Path to JSONL file.

        Returns:
            List of eval examples.
        """
        evals = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    evals.append(json.loads(line))
        return evals

    def _load_json(
        self,
        filepath: Path,
    ) -> tuple[list[dict[str, Any]], ToolSchemaList | None]:
        """Load evals from JSON file.

        Args:
            filepath: Path to JSON file.

        Returns:
            Tuple of (evals, tools).
        """
        with open(filepath) as f:
            data = json.load(f)

        evals = data.get("evals", [])
        tools = data.get("metadata", {}).get("tools")
        return evals, tools

    def _save_results(self, summary: EvalRunSummary) -> Path:
        """Save evaluation results to file.

        Args:
            summary: Evaluation summary to save.

        Returns:
            Path to saved file.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_results_{timestamp}.json"
        filepath = self.config.output_dir / filename

        output = {
            "metadata": {
                "evaluated_at": datetime.now().isoformat(),
                "model": self.config.model,
                "provider": self.config.provider.value,
                "match_mode": self.config.match_mode.value,
            },
            "summary": summary.to_dict(),
            "results": [
                {
                    "eval_id": r.eval_id,
                    "user_query": r.user_query,
                    "expected_tool_calls": r.expected_tool_calls,
                    "actual_tool_calls": r.actual_tool_calls,
                    "passed": r.passed,
                    "error": r.error,
                    "latency_ms": r.latency_ms,
                }
                for r in summary.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved eval results to {filepath}")
        return filepath

    def _save_comparison_results(self, comparison: ModelComparisonResult) -> Path:
        """Save model comparison results to file.

        Args:
            comparison: Comparison result to save.

        Returns:
            Path to saved file.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}.json"
        filepath = self.config.output_dir / filename

        output = comparison.to_dict()

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved model comparison results to {filepath}")
        return filepath

    def start(self) -> None:
        """Start the module."""
        super().start()

    def stop(self) -> None:
        """Stop the module."""
        self._llm = None
        self._judge_llm = None
        super().stop()


# Blueprint for use in DIMOS pipelines
eval_runner = EvalRunner.blueprint

__all__ = [
    "EvalRunner",
    "eval_runner",
    "EvalResult",
    "EvalRunSummary",
    "ToolCallResult",
    "ModelSpec",
    "ModelComparisonResult",
]
