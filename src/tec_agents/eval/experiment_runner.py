"""
Experiment runner.

This module runs evaluation tasks through:

1. deterministic gold runner;
2. agent under evaluation;
3. metric comparison;
4. structured experiment records.

It is the first reusable experiment loop for comparing orchestration variants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tec_agents.eval.gold_runner import GoldResult, GoldRunner
from tec_agents.eval.metrics import MetricResult, compare_agent_to_gold, summarize_metric_results
from tec_agents.eval.task_set import EvalTask, task_to_dict


class AgentProtocol(Protocol):
    """Minimal interface required from an agent."""

    def run(self, query: str) -> Any:
        """Run agent on a natural-language query."""


@dataclass
class ExperimentRecord:
    """One task-level experiment record."""

    task: dict[str, Any]
    gold: dict[str, Any]
    agent: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable record."""

        return {
            "task": self.task,
            "gold": self.gold,
            "agent": self.agent,
            "metrics": self.metrics,
        }


@dataclass
class ExperimentResult:
    """Full experiment result."""

    experiment_id: str
    architecture: str
    model_name: str
    records: list[ExperimentRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable experiment result."""

        return {
            "experiment_id": self.experiment_id,
            "architecture": self.architecture,
            "model_name": self.model_name,
            "summary": self.summary,
            "records": [record.to_dict() for record in self.records],
        }

    def save_json(self, path: str | Path) -> None:
        """Save experiment result as formatted JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class ExperimentRunner:
    """
    Run evaluation tasks for one agent.

    The runner assumes that the agent result object has these fields:
    - tool_results
    - trace
    - answer

    This matches RuleBasedSingleAgent. Later, LLM-based agents should return
    the same result shape or be adapted before metric calculation.
    """

    def __init__(
        self,
        agent: AgentProtocol,
        *,
        architecture: str,
        model_name: str,
        experiment_id: str,
        gold_runner: GoldRunner | None = None,
    ) -> None:
        self.agent = agent
        self.architecture = architecture
        self.model_name = model_name
        self.experiment_id = experiment_id
        self.gold_runner = gold_runner or GoldRunner()

    def run_tasks(self, tasks: list[EvalTask]) -> ExperimentResult:
        """Run all tasks and return full experiment result."""

        records: list[ExperimentRecord] = []
        metric_results: list[MetricResult] = []

        for task in tasks:
            record, metric_result = self.run_one(task)
            records.append(record)
            metric_results.append(metric_result)

        summary = summarize_metric_results(metric_results)
        summary.update(
            {
                "experiment_id": self.experiment_id,
                "architecture": self.architecture,
                "model_name": self.model_name,
            }
        )

        return ExperimentResult(
            experiment_id=self.experiment_id,
            architecture=self.architecture,
            model_name=self.model_name,
            records=records,
            summary=summary,
        )

    def run_one(self, task: EvalTask) -> tuple[ExperimentRecord, MetricResult]:
        """Run one task through gold runner, agent and metrics."""

        gold = self.gold_runner.run(task)

        if gold.status != "success" or gold.result is None:
            metric = MetricResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=False,
                metrics={},
                errors=[f"Gold runner failed: {gold.error}"],
            )

            record = ExperimentRecord(
                task=task_to_dict(task),
                gold=gold.to_dict(),
                agent={
                    "status": "skipped",
                    "answer": "",
                    "tool_results": None,
                    "trace": None,
                    "error": "gold_runner_failed",
                },
                metrics=metric.to_dict(),
            )

            return record, metric

        try:
            agent_output = self.agent.run(task.query)

            agent_payload = {
                "status": "success",
                "answer": getattr(agent_output, "answer", ""),
                "parsed_task": _safe_model_or_dataclass_to_dict(
                    getattr(agent_output, "parsed_task", None)
                ),
                "tool_results": getattr(agent_output, "tool_results", None),
                "trace": getattr(agent_output, "trace", None),
                "error": None,
            }

            metric = compare_agent_to_gold(
                task_id=task.task_id,
                task_type=task.task_type,
                agent_result=agent_payload["tool_results"],
                gold_result=gold.result,
                agent_trace=agent_payload["trace"],
            )

        except Exception as exc:
            agent_payload = {
                "status": "error",
                "answer": "",
                "parsed_task": None,
                "tool_results": None,
                "trace": None,
                "error": str(exc),
            }

            metric = MetricResult(
                task_id=task.task_id,
                task_type=task.task_type,
                success=False,
                metrics={},
                errors=[f"Agent failed: {exc}"],
            )

        record = ExperimentRecord(
            task=task_to_dict(task),
            gold=gold.to_dict(),
            agent=agent_payload,
            metrics=metric.to_dict(),
        )

        return record, metric


def _safe_model_or_dataclass_to_dict(obj: Any) -> dict[str, Any] | None:
    """Convert common structured objects to dictionaries."""

    if obj is None:
        return None

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "__dataclass_fields__"):
        return {
            field_name: getattr(obj, field_name)
            for field_name in obj.__dataclass_fields__
        }

    if isinstance(obj, dict):
        return obj

    return {"repr": repr(obj)}