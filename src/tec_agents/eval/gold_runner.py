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
            elif task.task_type == "stable_intervals":
                result = self._run_stable_intervals(task)
            elif task.task_type == "report":
                result = self._run_report(task)
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
        """Compute reference result for a region-comparison task via primitives."""

        if len(task.region_ids) < 2:
            raise ValueError(
                f"Task {task.task_id!r} requires at least two region_ids"
            )

        metrics = _compare_metrics_from_task(task)
        series_results: list[dict[str, Any]] = []
        stats_results: list[dict[str, Any]] = []
        stats_ids: list[str] = []

        step = 1
        for region_id in task.region_ids:
            ts_result = self.executor.call(
                "tec_get_timeseries",
                {
                    "dataset_ref": task.dataset_ref,
                    "region_id": region_id,
                    "start": task.start,
                    "end": task.end,
                },
                agent_name="gold_runner",
                step=step,
            )
            series_results.append(ts_result)
            step += 1

        for ts_result in series_results:
            stats_result = self.executor.call(
                "tec_compute_series_stats",
                {
                    "series_id": ts_result["series_id"],
                    "metrics": metrics,
                },
                agent_name="gold_runner",
                step=step,
            )
            stats_results.append(stats_result)
            stats_ids.append(stats_result["stats_id"])
            step += 1

        compare_result = self.executor.call(
            "tec_compare_stats",
            {
                "stats_ids": stats_ids,
                "metrics": metrics,
            },
            agent_name="gold_runner",
            step=step,
        )

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dataset_ref": task.dataset_ref,
            "region_ids": list(task.region_ids),
            "regions": list(task.region_ids),
            "start": task.start,
            "end": task.end,
            "series": series_results,
            "stats": stats_results,
            "comparison": compare_result,
        }

    def _run_stable_intervals(self, task: EvalTask) -> dict[str, Any]:
        """Compute reference result for a stable-interval task."""

        if task.region_id is None:
            raise ValueError(f"Task {task.task_id!r} requires region_id")

        params = task.params or {}
        window_minutes = int(params.get("window_minutes", 180))
        q_delta = float(params.get("q_delta", 0.60))
        q_std = float(params.get("q_std", 0.60))

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

        thresholds_result = self.executor.call(
            "tec_compute_stability_thresholds",
            {
                "series_id": series_id,
                "window_minutes": window_minutes,
                "method": "quantile",
                "q_delta": q_delta,
                "q_std": q_std,
            },
            agent_name="gold_runner",
            step=2,
        )

        intervals_result = self.executor.call(
            "tec_detect_stable_intervals",
            {
                "series_id": series_id,
                "threshold_id": thresholds_result["threshold_id"],
                "min_duration_minutes": window_minutes,
                "merge_gap_minutes": 60,
            },
            agent_name="gold_runner",
            step=3,
        )

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dataset_ref": task.dataset_ref,
            "region": task.region_id,
            "region_id": task.region_id,
            "start": task.start,
            "end": task.end,
            "series": ts_result,
            "thresholds": thresholds_result,
            "intervals": intervals_result.get("intervals", []),
            "n_intervals": intervals_result["n_intervals"],
            "interval_result": intervals_result,
        }

    def _run_report(self, task: EvalTask) -> dict[str, Any]:
        """Compute reference result for a structured report task via primitives."""

        if not task.region_ids:
            raise ValueError(f"Task {task.task_id!r} requires region_ids")

        include = _report_include_from_task(task)
        metrics = _compare_metrics_from_task(task)
        series_by_region: dict[str, dict[str, Any]] = {}
        report_inputs: dict[str, Any] = {}
        step = 1

        for region_id in task.region_ids:
            ts_result = self.executor.call(
                "tec_get_timeseries",
                {
                    "dataset_ref": task.dataset_ref,
                    "region_id": region_id,
                    "start": task.start,
                    "end": task.end,
                },
                agent_name="gold_runner",
                step=step,
            )
            step += 1
            series_by_region[region_id] = {
                "series_id": ts_result["series_id"],
                "tool_result": ts_result,
                "metadata": ts_result.get("metadata"),
            }

        if "basic_stats" in include:
            by_region: dict[str, dict[str, Any]] = {}
            stats_ids: list[str] = []

            for region_id in task.region_ids:
                series_id = series_by_region[region_id]["series_id"]
                stats_result = self.executor.call(
                    "tec_compute_series_stats",
                    {
                        "series_id": series_id,
                        "metrics": metrics,
                    },
                    agent_name="gold_runner",
                    step=step,
                )
                step += 1

                by_region[region_id] = {
                    "stats_id": stats_result["stats_id"],
                    "series_id": series_id,
                    "stats": stats_result,
                    "metrics": stats_result.get("metrics", {}),
                }
                stats_ids.append(stats_result["stats_id"])

            comparison = None
            if len(stats_ids) >= 2:
                comparison = self.executor.call(
                    "tec_compare_stats",
                    {
                        "stats_ids": stats_ids,
                        "metrics": metrics,
                    },
                    agent_name="gold_runner",
                    step=step,
                )
                step += 1

            report_inputs["basic_stats"] = {
                "by_region": by_region,
                "comparison": comparison,
            }

        if "high_tec" in include:
            by_region = {}

            for region_id in task.region_ids:
                series_id = series_by_region[region_id]["series_id"]
                threshold_result = self.executor.call(
                    "tec_compute_high_threshold",
                    {
                        "series_id": series_id,
                        "method": "quantile",
                        "q": task.q,
                    },
                    agent_name="gold_runner",
                    step=step,
                )
                step += 1

                intervals_result = self.executor.call(
                    "tec_detect_high_intervals",
                    {
                        "series_id": series_id,
                        "threshold_id": threshold_result["threshold_id"],
                        "min_duration_minutes": 0,
                        "merge_gap_minutes": 60,
                    },
                    agent_name="gold_runner",
                    step=step,
                )
                step += 1

                by_region[region_id] = {
                    "threshold": threshold_result,
                    "intervals": intervals_result,
                }

            report_inputs["high_tec"] = {"by_region": by_region}

        if "stable_intervals" in include:
            params = task.params or {}
            window_minutes = int(params.get("window_minutes", 180))
            q_delta = float(params.get("q_delta", 0.60))
            q_std = float(params.get("q_std", 0.60))
            by_region = {}

            for region_id in task.region_ids:
                series_id = series_by_region[region_id]["series_id"]
                thresholds_result = self.executor.call(
                    "tec_compute_stability_thresholds",
                    {
                        "series_id": series_id,
                        "window_minutes": window_minutes,
                        "method": "quantile",
                        "q_delta": q_delta,
                        "q_std": q_std,
                    },
                    agent_name="gold_runner",
                    step=step,
                )
                step += 1

                intervals_result = self.executor.call(
                    "tec_detect_stable_intervals",
                    {
                        "series_id": series_id,
                        "threshold_id": thresholds_result["threshold_id"],
                        "min_duration_minutes": window_minutes,
                        "merge_gap_minutes": 60,
                    },
                    agent_name="gold_runner",
                    step=step,
                )
                step += 1

                by_region[region_id] = {
                    "thresholds": thresholds_result,
                    "intervals": intervals_result,
                }

            report_inputs["stable_intervals"] = {"by_region": by_region}

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "dataset_ref": task.dataset_ref,
            "regions": list(task.region_ids),
            "region_ids": list(task.region_ids),
            "start": task.start,
            "end": task.end,
            "data": {
                "series_by_region": series_by_region,
                "regions": list(task.region_ids),
                "dataset_ref": task.dataset_ref,
                "start": task.start,
                "end": task.end,
            },
            "math": {
                "report_inputs": report_inputs,
            },
        }


def run_gold(task: EvalTask) -> GoldResult:
    """Convenience function for running one task with a fresh GoldRunner."""

    return GoldRunner().run(task)


def run_gold_many(tasks: list[EvalTask]) -> list[GoldResult]:
    """Convenience function for running several tasks with a fresh GoldRunner."""

    return GoldRunner().run_many(tasks)


def _compare_metrics_from_task(task: EvalTask) -> list[str]:
    """Return deterministic compare metrics from task params or defaults."""

    params = task.params or {}
    metrics = params.get("metrics")
    if metrics:
        return list(metrics)

    return ["mean", "median", "min", "max", "std", "p90", "p95"]


def _report_include_from_task(task: EvalTask) -> list[str]:
    """Return deterministic report sections from task params or defaults."""

    params = task.params or {}
    include = params.get("include")
    if include:
        return list(include)

    return ["basic_stats", "high_tec", "stable_intervals"]
