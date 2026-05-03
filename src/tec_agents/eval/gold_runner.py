"""
Gold runner for evaluation tasks.

The gold runner computes deterministic reference results for evaluation tasks.
It does not use an LLM and does not parse natural language. It relies on the
structured fields of EvalTask.

The purpose is to provide a reproducible baseline for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tec_agents.eval.task_set import EvalTask
from tec_agents.tools.executor import ToolExecutor, build_default_executor


GoldStatus = Literal["success", "error"]


@dataclass
class GoldResult:
    """Reference result for one evaluation task."""

    task_id: str
    task_type: str
    status: GoldStatus
    result: dict[str, Any] | None
    error: str | None
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable gold result."""

        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "trace": self.trace,
        }


class GoldRunner:
    """
    Deterministic reference runner for evaluation tasks.

    Each GoldRunner owns a ToolExecutor. For fully isolated evaluation, create
    a new GoldRunner per task or call reset() between tasks.
    """

    def __init__(self, executor: ToolExecutor | None = None) -> None:
        self.executor = executor or build_default_executor(run_id="gold_runner")

    def run(self, task: EvalTask) -> GoldResult:
        """Run one evaluation task and return a reference result."""

        self.reset(run_id=f"gold_{task.task_id}")

        try:
            if task.task_type == "high_tec":
                result = self._run_high_tec(task)
            elif task.task_type == "compare_regions":
                result = self._run_compare_regions(task)
            else:
                raise ValueError(
                    f"GoldRunner does not support task_type={task.task_type!r} yet"
                )

            return GoldResult(
                task_id=task.task_id,
                task_type=task.task_type,
                status="success",
                result=result,
                error=None,
                trace=self.executor.get_trace(),
            )

        except Exception as exc:
            return GoldResult(
                task_id=task.task_id,
                task_type=task.task_type,
                status="error",
                result=None,
                error=str(exc),
                trace=self.executor.get_trace(),
            )

    def run_many(self, tasks: list[EvalTask]) -> list[GoldResult]:
        """Run several tasks and return gold results."""

        return [self.run(task) for task in tasks]

    def reset(self, run_id: str | None = None) -> None:
        """Reset executor store and trace."""

        self.executor.reset_store()
        self.executor.reset_trace(run_id=run_id)

    def _run_high_tec(self, task: EvalTask) -> dict[str, Any]:
        """Compute reference result for a high-TEC interval task."""

        if task.region_id is None:
            raise ValueError(f"Task {task.task_id!r} requires region_id")

        ts_result = self.executor.call(
            "tec_get_timeseries",
            {
                "dataset_ref": task.dataset_ref,
                "region_id": task.region_id,
                "start": task.start,
                "end": task.end,
            },
            agent_name="gold_runner",
            step=1,
        )

        series_id = ts_result["series_id"]

        threshold_result = self.executor.call(
            "tec_compute_high_threshold",
            {
                "series_id": series_id,
                "method": "quantile",
                "q": task.q,
            },
            agent_name="gold_runner",
            step=2,
        )

        threshold_id = threshold_result["threshold_id"]

        intervals_result = self.executor.call(
            "tec_detect_high_intervals",
            {
                "series_id": series_id,
                "threshold_id": threshold_id,
                "min_duration_minutes": 0,
                "merge_gap_minutes": 60,
            },
            agent_name="gold_runner",
            step=3,
        )

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dataset_ref": task.dataset_ref,
            "region_id": task.region_id,
            "start": task.start,
            "end": task.end,
            "q": task.q,
            "series": ts_result,
            "threshold": threshold_result,
            "intervals": intervals_result,
        }

    def _run_compare_regions(self, task: EvalTask) -> dict[str, Any]:
        """Compute reference result for a region-comparison task."""

        if len(task.region_ids) < 2:
            raise ValueError(
                f"Task {task.task_id!r} requires at least two region_ids"
            )

        compare_result = self.executor.call(
            "tec_compare_regions",
            {
                "dataset_ref": task.dataset_ref,
                "region_ids": list(task.region_ids),
                "start": task.start,
                "end": task.end,
            },
            agent_name="gold_runner",
            step=1,
        )

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dataset_ref": task.dataset_ref,
            "region_ids": list(task.region_ids),
            "start": task.start,
            "end": task.end,
            "comparison": compare_result,
        }


def run_gold(task: EvalTask) -> GoldResult:
    """Convenience function for running one task with a fresh GoldRunner."""

    return GoldRunner().run(task)


def run_gold_many(tasks: list[EvalTask]) -> list[GoldResult]:
    """Convenience function for running several tasks with a fresh GoldRunner."""

    return GoldRunner().run_many(tasks)