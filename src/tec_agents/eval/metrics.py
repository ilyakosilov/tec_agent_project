"""
Evaluation metrics for agent-vs-gold comparison.

The metrics module compares agent outputs with deterministic gold results.
It also computes orchestration and tool-calling metrics that are important for
comparing single-agent and multi-agent architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Metric result for one task."""

    task_id: str
    task_type: str
    success: bool
    metrics: dict[str, float | int | bool | str | list[str] | None] = field(
        default_factory=dict
    )
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metric result."""

        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "success": self.success,
            "metrics": self.metrics,
            "errors": self.errors,
        }


def compare_agent_to_gold(
    *,
    task_id: str,
    task_type: str,
    agent_result: dict[str, Any],
    gold_result: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    parsed_task: dict[str, Any] | None = None,
    orchestration_steps: list[dict[str, Any]] | None = None,
) -> MetricResult:
    """
    Compare one agent result with one gold result.
    """

    if task_type == "high_tec":
        return compare_high_tec(
            task_id=task_id,
            agent_result=agent_result,
            gold_result=gold_result,
            agent_trace=agent_trace,
            task=task,
            parsed_task=parsed_task,
            orchestration_steps=orchestration_steps,
        )

    if task_type == "compare_regions":
        return compare_region_comparison(
            task_id=task_id,
            agent_result=agent_result,
            gold_result=gold_result,
            agent_trace=agent_trace,
            task=task,
            parsed_task=parsed_task,
            orchestration_steps=orchestration_steps,
        )

    return MetricResult(
        task_id=task_id,
        task_type=task_type,
        success=False,
        metrics={},
        errors=[f"Unsupported task_type={task_type!r}"],
    )


def compare_high_tec(
    *,
    task_id: str,
    agent_result: dict[str, Any],
    gold_result: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    parsed_task: dict[str, Any] | None = None,
    orchestration_steps: list[dict[str, Any]] | None = None,
) -> MetricResult:
    """Compare high-TEC interval detection results."""

    errors: list[str] = []
    metrics: dict[str, float | int | bool | str | list[str] | None] = {}

    try:
        agent_threshold = agent_result["threshold"]
        agent_intervals = agent_result["intervals"]

        gold_threshold = gold_result["threshold"]
        gold_intervals = gold_result["intervals"]

        agent_threshold_value = _as_float(agent_threshold["value"])
        gold_threshold_value = _as_float(gold_threshold["value"])

        agent_n_intervals = int(agent_intervals["n_intervals"])
        gold_n_intervals = int(gold_intervals["n_intervals"])

        metrics["threshold_abs_error"] = abs(
            agent_threshold_value - gold_threshold_value
        )
        metrics["threshold_exact_match"] = (
            agent_threshold_value == gold_threshold_value
        )
        metrics["agent_n_intervals"] = agent_n_intervals
        metrics["gold_n_intervals"] = gold_n_intervals
        metrics["interval_count_error"] = abs(agent_n_intervals - gold_n_intervals)
        metrics["interval_count_match"] = agent_n_intervals == gold_n_intervals

        agent_peak = _max_peak(agent_intervals.get("intervals", []))
        gold_peak = _max_peak(gold_intervals.get("intervals", []))

        metrics["agent_global_peak_value"] = agent_peak
        metrics["gold_global_peak_value"] = gold_peak

        if agent_peak is None or gold_peak is None:
            metrics["global_peak_abs_error"] = None
            metrics["global_peak_match"] = False
        else:
            metrics["global_peak_abs_error"] = abs(agent_peak - gold_peak)
            metrics["global_peak_match"] = agent_peak == gold_peak

        metrics["top_interval_start_match"] = _first_interval_field_match(
            agent_intervals.get("intervals", []),
            gold_intervals.get("intervals", []),
            field_name="start",
        )
        metrics["top_interval_end_match"] = _first_interval_field_match(
            agent_intervals.get("intervals", []),
            gold_intervals.get("intervals", []),
            field_name="end",
        )

        metrics.update(_parse_metrics(task=task, parsed_task=parsed_task))
        metrics.update(
            _orchestration_metrics(
                orchestration_steps=orchestration_steps,
                task_type="high_tec",
            )
        )

        if agent_trace is not None:
            metrics.update(_trace_metrics(agent_trace, task_type="high_tec"))

    except Exception as exc:
        errors.append(str(exc))

    success = (
        not errors
        and metrics.get("threshold_abs_error") == 0
        and metrics.get("interval_count_error") == 0
    )

    return MetricResult(
        task_id=task_id,
        task_type="high_tec",
        success=bool(success),
        metrics=metrics,
        errors=errors,
    )


def compare_region_comparison(
    *,
    task_id: str,
    agent_result: dict[str, Any],
    gold_result: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    parsed_task: dict[str, Any] | None = None,
    orchestration_steps: list[dict[str, Any]] | None = None,
) -> MetricResult:
    """Compare region statistics results."""

    errors: list[str] = []
    metrics: dict[str, float | int | bool | str | list[str] | None] = {}

    try:
        agent_comparison = _extract_comparison(agent_result)
        gold_comparison = gold_result["comparison"]

        agent_stats = _stats_by_region(agent_comparison["stats"])
        gold_stats = _stats_by_region(gold_comparison["stats"])

        agent_regions = set(agent_stats)
        gold_regions = set(gold_stats)

        metrics["region_set_match"] = agent_regions == gold_regions
        metrics["agent_region_count"] = len(agent_regions)
        metrics["gold_region_count"] = len(gold_regions)

        shared_regions = sorted(agent_regions.intersection(gold_regions))
        metrics["shared_region_count"] = len(shared_regions)

        mean_errors: list[float] = []
        max_errors: list[float] = []
        p90_errors: list[float] = []

        for region_id in shared_regions:
            a = agent_stats[region_id]
            g = gold_stats[region_id]

            mean_errors.append(abs(_as_float(a["mean"]) - _as_float(g["mean"])))
            max_errors.append(abs(_as_float(a["max"]) - _as_float(g["max"])))
            p90_errors.append(abs(_as_float(a["p90"]) - _as_float(g["p90"])))

        metrics["mean_abs_error_avg"] = _avg(mean_errors)
        metrics["max_abs_error_avg"] = _avg(max_errors)
        metrics["p90_abs_error_avg"] = _avg(p90_errors)

        metrics["mean_abs_error_max"] = max(mean_errors) if mean_errors else None
        metrics["max_abs_error_max"] = max(max_errors) if max_errors else None
        metrics["p90_abs_error_max"] = max(p90_errors) if p90_errors else None

        metrics.update(_parse_metrics(task=task, parsed_task=parsed_task))
        metrics.update(
            _orchestration_metrics(
                orchestration_steps=orchestration_steps,
                task_type="compare_regions",
            )
        )

        if agent_trace is not None:
            metrics.update(_trace_metrics(agent_trace, task_type="compare_regions"))

    except Exception as exc:
        errors.append(str(exc))

    success = (
        not errors
        and metrics.get("region_set_match") is True
        and metrics.get("mean_abs_error_max") == 0
        and metrics.get("max_abs_error_max") == 0
        and metrics.get("p90_abs_error_max") == 0
    )

    return MetricResult(
        task_id=task_id,
        task_type="compare_regions",
        success=bool(success),
        metrics=metrics,
        errors=errors,
    )


def summarize_metric_results(results: list[MetricResult]) -> dict[str, Any]:
    """Summarize several metric results."""

    n = len(results)
    n_success = sum(1 for result in results if result.success)
    n_failed = n - n_success

    tool_calls = [
        result.metrics.get("tool_call_count")
        for result in results
        if isinstance(result.metrics.get("tool_call_count"), int)
    ]

    invalid_tool_calls = [
        result.metrics.get("tool_error_count")
        for result in results
        if isinstance(result.metrics.get("tool_error_count"), int)
    ]

    orchestration_steps = [
        result.metrics.get("orchestration_step_count")
        for result in results
        if isinstance(result.metrics.get("orchestration_step_count"), int)
    ]

    route_correct_values = [
        result.metrics.get("route_correct")
        for result in results
        if isinstance(result.metrics.get("route_correct"), bool)
    ]

    tool_sequence_match_values = [
        result.metrics.get("tool_sequence_match")
        for result in results
        if isinstance(result.metrics.get("tool_sequence_match"), bool)
    ]

    return {
        "n_tasks": n,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": n_success / n if n else 0.0,
        "avg_tool_call_count": _avg([float(x) for x in tool_calls]),
        "avg_tool_error_count": _avg([float(x) for x in invalid_tool_calls]),
        "avg_orchestration_step_count": _avg(
            [float(x) for x in orchestration_steps]
        ),
        "route_correct_rate": _bool_rate(route_correct_values),
        "tool_sequence_match_rate": _bool_rate(tool_sequence_match_values),
    }


def _parse_metrics(
    *,
    task: dict[str, Any] | None,
    parsed_task: dict[str, Any] | None,
) -> dict[str, float | int | bool | str | list[str] | None]:
    """Compare structured task fields with parsed agent task."""

    metrics: dict[str, float | int | bool | str | list[str] | None] = {}

    if task is None or parsed_task is None:
        metrics["task_type_match"] = None
        metrics["region_parse_match"] = None
        metrics["date_parse_match"] = None
        metrics["q_abs_error"] = None
        return metrics

    metrics["task_type_match"] = parsed_task.get("task_type") == task.get("task_type")
    metrics["start_date_match"] = parsed_task.get("start") == task.get("start")
    metrics["end_date_match"] = parsed_task.get("end") == task.get("end")
    metrics["date_parse_match"] = (
        metrics["start_date_match"] is True and metrics["end_date_match"] is True
    )

    if task.get("task_type") == "high_tec":
        metrics["region_parse_match"] = parsed_task.get("region_id") == task.get(
            "region_id"
        )
    else:
        parsed_regions = set(parsed_task.get("region_ids") or [])
        task_regions = set(task.get("region_ids") or [])
        metrics["region_parse_match"] = parsed_regions == task_regions

    try:
        metrics["q_abs_error"] = abs(float(parsed_task.get("q")) - float(task.get("q")))
    except Exception:
        metrics["q_abs_error"] = None

    return metrics


def _orchestration_metrics(
    *,
    orchestration_steps: list[dict[str, Any]] | None,
    task_type: str,
) -> dict[str, float | int | bool | str | list[str] | None]:
    """Compute architecture-level orchestration metrics."""

    steps = orchestration_steps or []
    nodes = [str(step.get("node", "")) for step in steps]

    orchestrator_count = sum(1 for node in nodes if "orchestrator" in node)
    reporter_count = sum(1 for node in nodes if "reporter" in node)
    worker_count = sum(
        1
        for node in nodes
        if node not in {"", "rule_based_single_agent"}
        and "orchestrator" not in node
        and "reporter" not in node
    )

    selected_worker = _selected_worker_from_steps(steps)
    expected_worker = _expected_worker_for_task_type(task_type)

    return {
        "orchestration_step_count": len(steps),
        "orchestrator_step_count": orchestrator_count,
        "worker_step_count": worker_count,
        "reporter_step_count": reporter_count,
        "handoff_count": max(0, len(steps) - 1),
        "selected_worker": selected_worker,
        "expected_worker": expected_worker,
        "route_correct": _route_correct(
            selected_worker=selected_worker,
            expected_worker=expected_worker,
            task_type=task_type,
        ),
        "orchestration_nodes": nodes,
    }


def _selected_worker_from_steps(steps: list[dict[str, Any]]) -> str | None:
    """Extract selected worker from orchestration steps."""

    for step in steps:
        details = step.get("details") or {}
        worker = details.get("worker") or details.get("selected_worker")
        if worker:
            return str(worker)

    for step in steps:
        node = str(step.get("node", ""))
        if node in {"high_tec_agent", "compare_agent", "single_agent"}:
            return node

    return None


def _expected_worker_for_task_type(task_type: str) -> str | None:
    """Return expected worker for a task type."""

    if task_type == "high_tec":
        return "high_tec_agent"

    if task_type == "compare_regions":
        return "compare_agent"

    return None


def _route_correct(
    *,
    selected_worker: str | None,
    expected_worker: str | None,
    task_type: str,
) -> bool | None:
    """
    Evaluate routing correctness.

    For single-agent baseline selected_worker may be "single_agent". This is
    treated as correct because there is no specialized worker route.
    """

    if selected_worker == "single_agent":
        return True

    if expected_worker is None or selected_worker is None:
        return None

    return selected_worker == expected_worker


def _trace_metrics(
    trace: dict[str, Any],
    *,
    task_type: str,
) -> dict[str, float | int | bool | str | list[str] | None]:
    """Compute tool-calling metrics from execution trace."""

    calls = trace.get("calls", [])

    tool_sequence = [str(call.get("tool_name")) for call in calls]
    expected_sequence = _expected_tool_sequence(task_type)

    tool_call_count = len(calls)
    tool_error_count = sum(1 for call in calls if call.get("status") != "ok")
    total_latency_sec = sum(float(call.get("latency_sec", 0.0)) for call in calls)

    invalid_tool_name_count = sum(
        1
        for call in calls
        if call.get("status") != "ok"
        and "Unknown tool" in str(call.get("error_message", ""))
    )

    schema_validation_error_count = sum(
        1
        for call in calls
        if call.get("status") != "ok"
        and str(call.get("error_type")) == "validation_error"
    )

    tool_runtime_error_count = sum(
        1
        for call in calls
        if call.get("status") != "ok"
        and str(call.get("error_type")) not in {"validation_error"}
        and "Unknown tool" not in str(call.get("error_message", ""))
    )

    return {
        "tool_call_count": tool_call_count,
        "tool_error_count": tool_error_count,
        "total_tool_latency_sec": total_latency_sec,
        "tool_sequence": tool_sequence,
        "expected_tool_sequence": expected_sequence,
        "tool_sequence_match": tool_sequence == expected_sequence,
        "unnecessary_tool_call_count": _unnecessary_tool_call_count(
            actual=tool_sequence,
            expected=expected_sequence,
        ),
        "invalid_tool_name_count": invalid_tool_name_count,
        "schema_validation_error_count": schema_validation_error_count,
        "tool_runtime_error_count": tool_runtime_error_count,
        "tool_success_rate": (
            (tool_call_count - tool_error_count) / tool_call_count
            if tool_call_count
            else None
        ),
    }


def _expected_tool_sequence(task_type: str) -> list[str]:
    """Return expected tool sequence for supported task types."""

    if task_type == "high_tec":
        return [
            "tec_get_timeseries",
            "tec_compute_high_threshold",
            "tec_detect_high_intervals",
        ]

    if task_type == "compare_regions":
        return ["tec_compare_regions"]

    return []


def _unnecessary_tool_call_count(actual: list[str], expected: list[str]) -> int:
    """
    Count extra tool calls relative to expected sequence.

    This simple metric is enough for early experiments. Later it can be replaced
    with edit distance for more nuanced sequence comparison.
    """

    if len(actual) <= len(expected):
        return 0

    return len(actual) - len(expected)


def _extract_comparison(result: dict[str, Any]) -> dict[str, Any]:
    """
    Extract comparison payload from different result shapes.

    Gold shape:
      {"comparison": {...}}

    Direct tool shape:
      {"stats": [...]}
    """

    if "comparison" in result:
        return result["comparison"]

    if "stats" in result:
        return result

    raise KeyError("Could not find comparison result. Expected 'comparison' or 'stats'.")


def _stats_by_region(stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index region stats by region_id."""

    return {str(item["region_id"]): item for item in stats}


def _as_float(value: Any) -> float:
    """Convert a value to float with a clear error."""

    if value is None:
        raise ValueError("Expected numeric value, got None")
    return float(value)


def _avg(values: list[float]) -> float | None:
    """Return average or None for an empty list."""

    if not values:
        return None
    return sum(values) / len(values)


def _bool_rate(values: list[bool]) -> float | None:
    """Return share of True values or None for an empty list."""

    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _max_peak(intervals: list[dict[str, Any]]) -> float | None:
    """Return max peak value from interval records."""

    peaks: list[float] = []

    for item in intervals:
        peak = item.get("peak_value")
        if peak is not None:
            peaks.append(float(peak))

    if not peaks:
        return None

    return max(peaks)


def _first_interval_field_match(
    agent_intervals: list[dict[str, Any]],
    gold_intervals: list[dict[str, Any]],
    *,
    field_name: str,
) -> bool:
    """Check whether the first interval has matching selected field."""

    if not agent_intervals and not gold_intervals:
        return True

    if not agent_intervals or not gold_intervals:
        return False

    return agent_intervals[0].get(field_name) == gold_intervals[0].get(field_name)