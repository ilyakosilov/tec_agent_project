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

    if task_type == "stable_intervals":
        return compare_stable_intervals(
            task_id=task_id,
            agent_result=agent_result,
            gold_result=gold_result,
            agent_trace=agent_trace,
            task=task,
            parsed_task=parsed_task,
            orchestration_steps=orchestration_steps,
        )

    if task_type == "report":
        return compare_report(
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
        agent_high = _extract_high_tec_payload(agent_result)
        gold_high = _extract_high_tec_payload(gold_result)

        agent_threshold = agent_high["threshold"]
        agent_intervals = agent_high["intervals"]

        gold_threshold = gold_high["threshold"]
        gold_intervals = gold_high["intervals"]

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
                agent_result=agent_result,
                parsed_task=parsed_task,
            )
        )

        if agent_trace is not None:
            metrics.update(_trace_metrics(agent_trace, task_type="high_tec", task=task))

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

        agent_stats = _comparison_stats_by_region(agent_result)
        gold_stats = _comparison_stats_by_region(gold_result)

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

        agent_pairwise = agent_comparison.get("pairwise_deltas", [])
        gold_pairwise = gold_comparison.get("pairwise_deltas", [])

        metrics["stats_tool_call_count"] = _stats_tool_call_count(agent_trace)
        metrics["compare_stats_present"] = "items" in agent_comparison
        metrics["compare_stats_region_count"] = len(
            agent_comparison.get("regions", [])
        )
        metrics["pairwise_delta_count"] = len(agent_pairwise)
        metrics["expected_pairwise_delta_count"] = len(gold_pairwise)
        metrics["pairwise_delta_count_match"] = (
            len(agent_pairwise) == len(gold_pairwise)
        )

        metrics.update(_parse_metrics(task=task, parsed_task=parsed_task))
        metrics.update(
            _orchestration_metrics(
                orchestration_steps=orchestration_steps,
                task_type="compare_regions",
                agent_result=agent_result,
                parsed_task=parsed_task,
            )
        )

        if agent_trace is not None:
            metrics.update(
                _trace_metrics(
                    agent_trace,
                    task_type="compare_regions",
                    task=task,
                )
            )

    except Exception as exc:
        errors.append(str(exc))

    success = (
        not errors
        and metrics.get("region_set_match") is True
        and metrics.get("mean_abs_error_max") == 0
        and metrics.get("max_abs_error_max") == 0
        and metrics.get("p90_abs_error_max") == 0
        and metrics.get("pairwise_delta_count_match") is not False
    )

    return MetricResult(
        task_id=task_id,
        task_type="compare_regions",
        success=bool(success),
        metrics=metrics,
        errors=errors,
    )


def compare_stable_intervals(
    *,
    task_id: str,
    agent_result: dict[str, Any],
    gold_result: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    parsed_task: dict[str, Any] | None = None,
    orchestration_steps: list[dict[str, Any]] | None = None,
) -> MetricResult:
    """Compare stable/low-variability interval results."""

    errors: list[str] = []
    metrics: dict[str, float | int | bool | str | list[str] | None] = {}

    try:
        agent_stable = _extract_stable_payload(agent_result)
        gold_stable = _extract_stable_payload(gold_result)

        agent_intervals = agent_stable["intervals"]
        gold_intervals = gold_stable["intervals"]

        agent_n_intervals = int(agent_stable["n_intervals"])
        gold_n_intervals = int(gold_stable["n_intervals"])

        metrics["agent_n_stable_intervals"] = agent_n_intervals
        metrics["gold_n_stable_intervals"] = gold_n_intervals
        metrics["stable_interval_count_error"] = abs(
            agent_n_intervals - gold_n_intervals
        )
        metrics["stable_interval_count_match"] = agent_n_intervals == gold_n_intervals

        if not agent_intervals and not gold_intervals:
            metrics["stable_top_interval_start_match"] = None
            metrics["stable_top_interval_end_match"] = None
            metrics["stable_duration_abs_error_top"] = None
            metrics["stable_mean_abs_error_top"] = None
            metrics["stable_std_abs_error_top"] = None
        elif not agent_intervals or not gold_intervals:
            metrics["stable_top_interval_start_match"] = False
            metrics["stable_top_interval_end_match"] = False
            metrics["stable_duration_abs_error_top"] = None
            metrics["stable_mean_abs_error_top"] = None
            metrics["stable_std_abs_error_top"] = None
        else:
            agent_top = agent_intervals[0]
            gold_top = gold_intervals[0]

            metrics["stable_top_interval_start_match"] = (
                agent_top.get("start") == gold_top.get("start")
            )
            metrics["stable_top_interval_end_match"] = (
                agent_top.get("end") == gold_top.get("end")
            )
            metrics["stable_duration_abs_error_top"] = _field_abs_error(
                agent_top,
                gold_top,
                "duration_minutes",
            )
            metrics["stable_mean_abs_error_top"] = _field_abs_error(
                agent_top,
                gold_top,
                "mean_value",
            )
            metrics["stable_std_abs_error_top"] = _field_abs_error(
                agent_top,
                gold_top,
                "std_value",
            )

        metrics.update(_parse_metrics(task=task, parsed_task=parsed_task))
        metrics.update(
            _orchestration_metrics(
                orchestration_steps=orchestration_steps,
                task_type="stable_intervals",
                agent_result=agent_result,
                parsed_task=parsed_task,
            )
        )

        if agent_trace is not None:
            metrics.update(
                _trace_metrics(
                    agent_trace,
                    task_type="stable_intervals",
                    task=task,
                )
            )

    except Exception as exc:
        errors.append(str(exc))

    top_start = metrics.get("stable_top_interval_start_match")
    top_end = metrics.get("stable_top_interval_end_match")

    success = (
        not errors
        and metrics.get("stable_interval_count_error") == 0
        and top_start is not False
        and top_end is not False
        and _zero_or_none(metrics.get("stable_duration_abs_error_top"))
        and _zero_or_none(metrics.get("stable_mean_abs_error_top"))
        and _zero_or_none(metrics.get("stable_std_abs_error_top"))
    )

    return MetricResult(
        task_id=task_id,
        task_type="stable_intervals",
        success=bool(success),
        metrics=metrics,
        errors=errors,
    )


def compare_report(
    *,
    task_id: str,
    agent_result: dict[str, Any],
    gold_result: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    parsed_task: dict[str, Any] | None = None,
    orchestration_steps: list[dict[str, Any]] | None = None,
) -> MetricResult:
    """Compare structured deterministic TEC reports."""

    errors: list[str] = []
    metrics: dict[str, float | int | bool | str | list[str] | None] = {}

    try:
        agent_report = _extract_report_payload(agent_result)
        gold_report = _extract_report_payload(gold_result)

        metrics["report_present"] = bool(agent_report)

        agent_regions = set(agent_report.get("regions") or agent_report.get("region_ids") or [])
        gold_regions = set(gold_report.get("regions") or gold_report.get("region_ids") or [])
        shared_regions = sorted(agent_regions.intersection(gold_regions))

        metrics["report_region_set_match"] = agent_regions == gold_regions
        metrics["report_region_count_agent"] = len(agent_regions)
        metrics["report_region_count_gold"] = len(gold_regions)

        agent_sections = agent_report.get("sections") or {}
        gold_sections = gold_report.get("sections") or {}
        required_sections = {"basic_stats", "high_tec", "stable_intervals"}

        metrics["report_required_sections_present"] = required_sections.issubset(
            set(agent_sections)
        )
        metrics["report_basic_stats_present"] = "basic_stats" in agent_sections
        metrics["report_high_tec_present"] = "high_tec" in agent_sections
        metrics["report_stable_intervals_present"] = "stable_intervals" in agent_sections

        basic_mean_errors: list[float] = []
        basic_max_errors: list[float] = []
        basic_p90_errors: list[float] = []

        agent_basic = _section_by_region(agent_sections.get("basic_stats"))
        gold_basic = _section_by_region(gold_sections.get("basic_stats"))

        for region_id in shared_regions:
            if region_id not in agent_basic or region_id not in gold_basic:
                continue
            basic_mean_errors.append(
                abs(_as_float(agent_basic[region_id]["mean"]) - _as_float(gold_basic[region_id]["mean"]))
            )
            basic_max_errors.append(
                abs(_as_float(agent_basic[region_id]["max"]) - _as_float(gold_basic[region_id]["max"]))
            )
            basic_p90_errors.append(
                abs(_as_float(agent_basic[region_id]["p90"]) - _as_float(gold_basic[region_id]["p90"]))
            )

        metrics["report_mean_abs_error_avg"] = _avg(basic_mean_errors)
        metrics["report_max_abs_error_avg"] = _avg(basic_max_errors)
        metrics["report_p90_abs_error_avg"] = _avg(basic_p90_errors)

        agent_high = _section_by_region(agent_sections.get("high_tec"))
        gold_high = _section_by_region(gold_sections.get("high_tec"))
        high_threshold_errors: list[float] = []
        high_count_errors: list[float] = []

        for region_id in shared_regions:
            if region_id not in agent_high or region_id not in gold_high:
                continue
            high_threshold_errors.append(
                abs(
                    _as_float(_threshold_value(agent_high[region_id]))
                    - _as_float(_threshold_value(gold_high[region_id]))
                )
            )
            high_count_errors.append(
                abs(
                    float(agent_high[region_id].get("n_intervals", 0))
                    - float(gold_high[region_id].get("n_intervals", 0))
                )
            )

        metrics["report_high_tec_threshold_abs_error_avg"] = _avg(
            high_threshold_errors
        )
        metrics["report_high_tec_interval_count_error_avg"] = _avg(
            high_count_errors
        )

        agent_stable = _section_by_region(agent_sections.get("stable_intervals"))
        gold_stable = _section_by_region(gold_sections.get("stable_intervals"))
        stable_count_errors: list[float] = []

        for region_id in shared_regions:
            if region_id not in agent_stable or region_id not in gold_stable:
                continue
            stable_count_errors.append(
                abs(
                    float(agent_stable[region_id].get("n_intervals", 0))
                    - float(gold_stable[region_id].get("n_intervals", 0))
                )
            )

        metrics["report_stable_interval_count_error_avg"] = _avg(
            stable_count_errors
        )

        metrics.update(_parse_metrics(task=task, parsed_task=parsed_task))
        metrics.update(
            _orchestration_metrics(
                orchestration_steps=orchestration_steps,
                task_type="report",
                agent_result=agent_result,
                parsed_task=parsed_task,
            )
        )

        if agent_trace is not None:
            metrics.update(_trace_metrics(agent_trace, task_type="report", task=task))

    except Exception as exc:
        errors.append(str(exc))

    success = (
        not errors
        and metrics.get("report_present") is True
        and metrics.get("report_region_set_match") is True
        and metrics.get("report_required_sections_present") is True
        and _zero_or_none(metrics.get("report_mean_abs_error_avg"))
        and _zero_or_none(metrics.get("report_max_abs_error_avg"))
        and _zero_or_none(metrics.get("report_p90_abs_error_avg"))
        and _zero_or_none(metrics.get("report_high_tec_threshold_abs_error_avg"))
        and _zero_or_none(metrics.get("report_high_tec_interval_count_error_avg"))
        and _zero_or_none(metrics.get("report_stable_interval_count_error_avg"))
    )

    return MetricResult(
        task_id=task_id,
        task_type="report",
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

    role_agent_order_match_values = [
        result.metrics.get("role_agent_order_match")
        for result in results
        if isinstance(result.metrics.get("role_agent_order_match"), bool)
    ]

    artifact_flow_valid_values = [
        result.metrics.get("artifact_flow_valid")
        for result in results
        if isinstance(result.metrics.get("artifact_flow_valid"), bool)
    ]

    required_role_agents_called_values = [
        result.metrics.get("required_role_agents_called")
        for result in results
        if isinstance(result.metrics.get("required_role_agents_called"), bool)
    ]

    legacy_report_tool_used_values = [
        result.metrics.get("legacy_report_tool_used")
        for result in results
        if isinstance(result.metrics.get("legacy_report_tool_used"), bool)
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
        "role_agent_order_match_rate": _bool_rate(role_agent_order_match_values),
        "artifact_flow_valid_rate": _bool_rate(artifact_flow_valid_values),
        "required_role_agents_called_rate": _bool_rate(
            required_role_agents_called_values
        ),
        "legacy_report_tool_used_rate": _bool_rate(legacy_report_tool_used_values),
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
    agent_result: dict[str, Any] | None = None,
    parsed_task: dict[str, Any] | None = None,
) -> dict[str, float | int | bool | str | list[str] | None]:
    """Compute architecture-level orchestration metrics."""

    steps = orchestration_steps or []
    nodes = [str(step.get("node", "")) for step in steps]
    expected_role_order = [
        "orchestrator",
        "data_agent",
        "math_agent",
        "analysis_agent",
        "report_agent",
    ]

    orchestrator_count = sum(1 for node in nodes if "orchestrator" in node)
    reporter_count = sum(
        1
        for node in nodes
        if node in {"report_agent", "reporter_agent"} or "reporter" in node
    )
    worker_count = sum(
        1
        for node in nodes
        if node not in {"", "rule_based_single_agent"}
        and "orchestrator" not in node
        and node not in {"report_agent", "reporter_agent"}
        and "reporter" not in node
    )

    selected_worker = _selected_worker_from_steps(steps)
    role_agent_order_match = nodes == expected_role_order
    role_workflow_seen = any(
        node in {"data_agent", "math_agent", "analysis_agent", "report_agent"}
        for node in nodes
    )
    expected_worker = (
        "role_based_workflow"
        if role_workflow_seen
        else _expected_worker_for_task_type(task_type)
    )

    data_agent_called = "data_agent" in nodes
    math_agent_called = "math_agent" in nodes
    analysis_agent_called = "analysis_agent" in nodes
    report_agent_called = "report_agent" in nodes
    required_role_agents_called = (
        data_agent_called
        and math_agent_called
        and analysis_agent_called
        and report_agent_called
    )

    data_artifact_available = _data_artifact_available(agent_result)
    math_artifact_available = _math_artifact_available(agent_result)
    analysis_artifact_available = _analysis_artifact_available(agent_result)
    report_grounded_in_artifacts = (
        data_artifact_available and math_artifact_available
    )
    artifact_flow_valid = _artifact_flow_valid(
        agent_result=agent_result,
        parsed_task=parsed_task,
        nodes=nodes,
    )

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
            parsed_task=parsed_task,
            role_agent_order_match=role_agent_order_match,
        ),
        "orchestration_nodes": nodes,
        "data_agent_called": data_agent_called,
        "math_agent_called": math_agent_called,
        "analysis_agent_called": analysis_agent_called,
        "report_agent_called": report_agent_called,
        "required_role_agents_called": required_role_agents_called,
        "role_agent_order": nodes,
        "expected_role_agent_order": expected_role_order,
        "role_agent_order_match": role_agent_order_match,
        "artifact_flow_valid": artifact_flow_valid,
        "data_artifact_available": data_artifact_available,
        "math_artifact_available": math_artifact_available,
        "analysis_artifact_available": analysis_artifact_available,
        "report_grounded_in_artifacts": report_grounded_in_artifacts,
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
        if node in {
            "high_tec_worker",
            "compare_worker",
            "stable_worker",
            "report_worker",
            "high_tec_agent",
            "compare_agent",
            "data_agent",
            "math_agent",
            "analysis_agent",
            "report_agent",
            "single_agent",
        }:
            if node in {"data_agent", "math_agent", "analysis_agent", "report_agent"}:
                return "role_based_workflow"
            return node

    return None


def _expected_worker_for_task_type(task_type: str) -> str | None:
    """Return expected worker for a task type."""

    if task_type == "high_tec":
        return "high_tec_worker"

    if task_type == "compare_regions":
        return "compare_worker"

    if task_type == "stable_intervals":
        return "stable_worker"

    if task_type == "report":
        return "report_worker"

    return None


def _route_correct(
    *,
    selected_worker: str | None,
    expected_worker: str | None,
    task_type: str,
    parsed_task: dict[str, Any] | None = None,
    role_agent_order_match: bool | None = None,
) -> bool | None:
    """
    Evaluate routing correctness.

    For single-agent baseline selected_worker may be "single_agent". This is
    treated as correct because there is no specialized worker route.
    """

    if selected_worker == "single_agent":
        return True

    if selected_worker == "role_based_workflow":
        parsed_task_type = parsed_task.get("task_type") if parsed_task else task_type
        return parsed_task_type == task_type and role_agent_order_match is True

    if expected_worker is None or selected_worker is None:
        return None

    return selected_worker == expected_worker


def _data_artifact_available(agent_result: dict[str, Any] | None) -> bool:
    """Return True if role data artifacts are present."""

    data = (agent_result or {}).get("data") or {}
    series_by_region = data.get("series_by_region")
    return isinstance(series_by_region, dict) and bool(series_by_region)


def _math_artifact_available(agent_result: dict[str, Any] | None) -> bool:
    """Return True if role math artifacts are present."""

    math = (agent_result or {}).get("math")
    return isinstance(math, dict) and bool(math)


def _analysis_artifact_available(agent_result: dict[str, Any] | None) -> bool:
    """Return True if role analysis artifacts are present."""

    analysis = (agent_result or {}).get("analysis") or {}
    return isinstance(analysis.get("findings"), list)


def _artifact_flow_valid(
    *,
    agent_result: dict[str, Any] | None,
    parsed_task: dict[str, Any] | None,
    nodes: list[str],
) -> bool:
    """Validate the role artifact flow at a structural level."""

    expected_order = [
        "orchestrator",
        "data_agent",
        "math_agent",
        "analysis_agent",
        "report_agent",
    ]
    if nodes != expected_order:
        return False

    result = agent_result or {}
    data = result.get("data") or {}
    math = result.get("math") or {}
    analysis = result.get("analysis") or {}

    series_by_region = data.get("series_by_region")
    if not isinstance(series_by_region, dict) or not series_by_region:
        return False

    expected_regions = _parsed_regions(parsed_task)
    if expected_regions and set(series_by_region) != set(expected_regions):
        return False

    if not isinstance(math, dict) or not math:
        return False

    if not isinstance(analysis, dict) or "findings" not in analysis:
        return False

    return True


def _parsed_regions(parsed_task: dict[str, Any] | None) -> list[str]:
    """Return parsed regions from a serialized parsed task."""

    if not parsed_task:
        return []

    region_ids = parsed_task.get("region_ids") or []
    if region_ids:
        return [str(region_id) for region_id in region_ids]

    region_id = parsed_task.get("region_id")
    if region_id:
        return [str(region_id)]

    return []


def _trace_metrics(
    trace: dict[str, Any],
    *,
    task_type: str,
    task: dict[str, Any] | None = None,
) -> dict[str, float | int | bool | str | list[str] | None]:
    """Compute tool-calling metrics from execution trace."""

    calls = trace.get("calls", [])

    tool_sequence = [str(call.get("tool_name")) for call in calls]
    expected_sequence = _expected_tool_sequence(task_type, task=task)
    legacy_report_tool_used = "tec_build_report" in tool_sequence

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
        "legacy_report_tool_used": legacy_report_tool_used,
        "legacy_report_tool_absent": not legacy_report_tool_used,
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


def _expected_tool_sequence(
    task_type: str,
    *,
    task: dict[str, Any] | None = None,
) -> list[str]:
    """Return expected tool sequence for supported task types."""

    if task is not None and task.get("expected_tool_sequence"):
        return [str(name) for name in task["expected_tool_sequence"]]

    if task_type == "high_tec":
        return [
            "tec_get_timeseries",
            "tec_compute_high_threshold",
            "tec_detect_high_intervals",
        ]

    if task_type == "compare_regions":
        region_count = 2
        if task is not None:
            region_count = len(task.get("region_ids") or []) or region_count

        sequence: list[str] = []
        sequence.extend(["tec_get_timeseries"] * region_count)
        sequence.extend(["tec_compute_series_stats"] * region_count)
        sequence.append("tec_compare_stats")
        return sequence

    if task_type == "stable_intervals":
        return [
            "tec_get_timeseries",
            "tec_compute_stability_thresholds",
            "tec_detect_stable_intervals",
        ]

    if task_type == "report":
        region_count = 1
        include = ["basic_stats", "high_tec", "stable_intervals"]
        if task is not None:
            region_count = len(task.get("region_ids") or []) or region_count
            params = task.get("params") or {}
            include = list(params.get("include") or include)

        sequence = ["tec_get_timeseries"] * region_count

        if "basic_stats" in include:
            sequence.extend(["tec_compute_series_stats"] * region_count)
            if region_count >= 2:
                sequence.append("tec_compare_stats")

        if "high_tec" in include:
            for _ in range(region_count):
                sequence.extend(
                    [
                        "tec_compute_high_threshold",
                        "tec_detect_high_intervals",
                    ]
                )

        if "stable_intervals" in include:
            for _ in range(region_count):
                sequence.extend(
                    [
                        "tec_compute_stability_thresholds",
                        "tec_detect_stable_intervals",
                    ]
                )

        return sequence

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

    math = result.get("math") or {}
    if "comparison" in math:
        return math["comparison"]

    report_inputs = math.get("report_inputs") or {}
    basic_stats = report_inputs.get("basic_stats") or {}
    if basic_stats.get("comparison") is not None:
        return basic_stats["comparison"]

    if "comparison" in result:
        return result["comparison"]

    if "items" in result:
        return result

    if "stats" in result:
        return result

    raise KeyError("Could not find comparison result. Expected 'comparison' or 'stats'.")


def _comparison_stats_by_region(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract region-keyed stats from primitive or legacy comparison results."""

    math = result.get("math") or {}

    if isinstance(math.get("stats_by_region"), dict):
        out: dict[str, dict[str, Any]] = {}
        for region_id, item in math["stats_by_region"].items():
            stats = item.get("stats") or item
            metrics = item.get("metrics") or stats.get("metrics") or {}
            out[str(region_id)] = {
                "region_id": stats.get("region_id") or region_id,
                "stats_id": item.get("stats_id") or stats.get("stats_id"),
                "series_id": item.get("series_id") or stats.get("series_id"),
                **metrics,
            }
        return out

    report_inputs = math.get("report_inputs") or {}
    basic_stats = report_inputs.get("basic_stats") or {}
    by_region = basic_stats.get("by_region")
    if isinstance(by_region, dict):
        out = {}
        for region_id, item in by_region.items():
            stats = item.get("stats") or item
            metrics = item.get("metrics") or stats.get("metrics") or {}
            out[str(region_id)] = {
                "region_id": stats.get("region_id") or region_id,
                "stats_id": item.get("stats_id") or stats.get("stats_id"),
                "series_id": item.get("series_id") or stats.get("series_id"),
                **metrics,
            }
        return out

    comparison = _extract_comparison(result)

    if "items" in comparison:
        out: dict[str, dict[str, Any]] = {}
        for item in comparison["items"]:
            region_id = item.get("region_id")
            if region_id is None:
                continue
            metrics = item.get("metrics") or {}
            out[str(region_id)] = {
                "region_id": region_id,
                "stats_id": item.get("stats_id"),
                "series_id": item.get("series_id"),
                **metrics,
            }
        return out

    if "stats" in comparison:
        return _stats_by_region(comparison["stats"])

    if "stats" in result and isinstance(result["stats"], list):
        # Primitive agent/gold wrapper shape: {"stats": [...], "comparison": {...}}.
        out = {}
        for item in result["stats"]:
            region_id = item.get("region_id")
            if region_id is None:
                continue
            metrics = item.get("metrics") or {}
            out[str(region_id)] = {
                "region_id": region_id,
                "stats_id": item.get("stats_id"),
                "series_id": item.get("series_id"),
                **metrics,
            }
        return out

    raise KeyError("Could not find comparison statistics")


def _stats_tool_call_count(trace: dict[str, Any] | None) -> int | None:
    """Count primitive stats tool calls in an execution trace."""

    if trace is None:
        return None

    return sum(
        1
        for call in trace.get("calls", [])
        if call.get("tool_name") == "tec_compute_series_stats"
    )


def _extract_high_tec_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Extract high-TEC payload from flat or role-based result shape."""

    if "threshold" in result and "intervals" in result:
        return {
            "threshold": result["threshold"],
            "intervals": result["intervals"],
        }

    math = result.get("math") or {}
    high_by_region = math.get("high_tec")

    if not high_by_region:
        report_inputs = math.get("report_inputs") or {}
        high = report_inputs.get("high_tec") or {}
        high_by_region = high.get("by_region")

    if isinstance(high_by_region, dict) and high_by_region:
        first_item = next(iter(high_by_region.values()))
        return {
            "threshold": first_item["threshold"],
            "intervals": first_item["intervals"],
        }

    raise KeyError("Could not find high-TEC result")


def _extract_stable_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Extract stable interval payload from agent, gold, or direct tool shape."""

    math = result.get("math") or {}
    stable_by_region = math.get("stable_intervals")

    if not stable_by_region:
        report_inputs = math.get("report_inputs") or {}
        stable = report_inputs.get("stable_intervals") or {}
        stable_by_region = stable.get("by_region")

    if isinstance(stable_by_region, dict) and stable_by_region:
        first_item = next(iter(stable_by_region.values()))
        interval_result = first_item["intervals"]
        return {
            "intervals": interval_result.get("intervals", []),
            "n_intervals": int(interval_result.get("n_intervals", 0)),
        }

    if "interval_result" in result:
        interval_result = result["interval_result"]
        return {
            "intervals": interval_result.get("intervals", []),
            "n_intervals": int(interval_result.get("n_intervals", 0)),
        }

    if "intervals" in result and isinstance(result["intervals"], dict):
        interval_result = result["intervals"]
        return {
            "intervals": interval_result.get("intervals", []),
            "n_intervals": int(interval_result.get("n_intervals", 0)),
        }

    if "intervals" in result and isinstance(result["intervals"], list):
        return {
            "intervals": result["intervals"],
            "n_intervals": int(result.get("n_intervals", len(result["intervals"]))),
        }

    raise KeyError("Could not find stable interval result")


def _extract_report_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Extract structured report from legacy or role-based result shape."""

    if "report" in result:
        return result["report"]

    if "sections" in result:
        return result

    math = result.get("math") or {}
    report_inputs = math.get("report_inputs")
    if isinstance(report_inputs, dict):
        data = result.get("data") or {}
        regions = (
            result.get("regions")
            or result.get("region_ids")
            or data.get("regions")
            or list((data.get("series_by_region") or {}).keys())
        )
        sections: dict[str, Any] = {}

        basic = report_inputs.get("basic_stats") or {}
        if basic:
            sections["basic_stats"] = _report_basic_stats_section(basic)

        high = report_inputs.get("high_tec") or {}
        if high:
            sections["high_tec"] = _report_high_tec_section(high)

        stable = report_inputs.get("stable_intervals") or {}
        if stable:
            sections["stable_intervals"] = _report_stable_section(stable)

        return {
            "regions": list(regions),
            "region_ids": list(regions),
            "sections": sections,
        }

    raise KeyError("Could not find report payload")


def _report_basic_stats_section(section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize primitive report basic stats into report-section shape."""

    by_region = section.get("by_region") or {}
    out: dict[str, dict[str, Any]] = {}

    for region_id, item in by_region.items():
        stats = item.get("stats") or item
        metrics = item.get("metrics") or stats.get("metrics") or {}
        out[str(region_id)] = {
            "region_id": stats.get("region_id") or region_id,
            "stats_id": item.get("stats_id") or stats.get("stats_id"),
            "series_id": item.get("series_id") or stats.get("series_id"),
            "n_points": stats.get("n_points"),
            "finite_points": stats.get("finite_points"),
            **metrics,
        }

    return out


def _report_high_tec_section(section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize primitive report high-TEC artifacts into report-section shape."""

    by_region = section.get("by_region") or {}
    out: dict[str, dict[str, Any]] = {}

    for region_id, item in by_region.items():
        threshold = item.get("threshold") or {}
        intervals = item.get("intervals") or {}
        interval_records = intervals.get("intervals") or []
        peak_values = [
            float(record["peak_value"])
            for record in interval_records
            if record.get("peak_value") is not None
        ]
        top_intervals = sorted(
            interval_records,
            key=lambda record: (
                record.get("peak_value") is not None,
                float(record.get("peak_value") or float("-inf")),
            ),
            reverse=True,
        )[:5]
        out[str(region_id)] = {
            "region_id": region_id,
            "series_id": intervals.get("series_id") or threshold.get("series_id"),
            "threshold_id": threshold.get("threshold_id"),
            "threshold": threshold.get("value"),
            "threshold_value": threshold.get("value"),
            "q": threshold.get("q"),
            "n_intervals": intervals.get("n_intervals", 0),
            "global_peak_value": max(peak_values) if peak_values else None,
            "top_intervals": top_intervals,
        }

    return out


def _report_stable_section(section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize primitive report stable artifacts into report-section shape."""

    by_region = section.get("by_region") or {}
    out: dict[str, dict[str, Any]] = {}

    for region_id, item in by_region.items():
        thresholds = item.get("thresholds") or {}
        intervals = item.get("intervals") or {}
        interval_records = intervals.get("intervals") or []
        top_intervals = sorted(
            interval_records,
            key=lambda record: (
                float(record.get("duration_minutes") or 0.0),
                record.get("start") or "",
            ),
            reverse=True,
        )[:5]
        out[str(region_id)] = {
            "region_id": region_id,
            "series_id": intervals.get("series_id") or thresholds.get("series_id"),
            "threshold_id": thresholds.get("threshold_id"),
            "window_minutes": thresholds.get("window_minutes"),
            "q_delta": thresholds.get("q_delta"),
            "q_std": thresholds.get("q_std"),
            "max_delta_threshold": thresholds.get("max_delta_threshold"),
            "rolling_std_threshold": thresholds.get("rolling_std_threshold"),
            "estimated_step_minutes": thresholds.get("estimated_step_minutes"),
            "window_points": thresholds.get("window_points"),
            "n_intervals": intervals.get("n_intervals", 0),
            "top_intervals": top_intervals,
        }

    return out


def _section_by_region(section: Any) -> dict[str, dict[str, Any]]:
    """Normalize a report section into a region-keyed dictionary."""

    if section is None:
        return {}

    if isinstance(section, dict):
        return {
            str(region_id): value
            for region_id, value in section.items()
            if isinstance(value, dict)
        }

    if isinstance(section, list):
        out: dict[str, dict[str, Any]] = {}
        for item in section:
            if not isinstance(item, dict):
                continue
            region_id = item.get("region_id") or item.get("region")
            if region_id is not None:
                out[str(region_id)] = item
        return out

    return {}


def _threshold_value(section: dict[str, Any]) -> Any:
    """Return high-TEC threshold value from compatible report field names."""

    if "threshold" in section:
        return section["threshold"]

    return section.get("threshold_value")


def _stats_by_region(stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index region stats by region_id."""

    return {str(item["region_id"]): item for item in stats}


def _as_float(value: Any) -> float:
    """Convert a value to float with a clear error."""

    if value is None:
        raise ValueError("Expected numeric value, got None")
    return float(value)


def _field_abs_error(
    agent_item: dict[str, Any],
    gold_item: dict[str, Any],
    field_name: str,
) -> float | None:
    """Return absolute error for an optional numeric field."""

    if agent_item.get(field_name) is None or gold_item.get(field_name) is None:
        return None

    return abs(_as_float(agent_item[field_name]) - _as_float(gold_item[field_name]))


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


def _zero_or_none(value: Any) -> bool:
    """Return True when a metric is exactly zero or intentionally absent."""

    if value is None:
        return True

    try:
        return float(value) == 0.0
    except Exception:
        return False


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
