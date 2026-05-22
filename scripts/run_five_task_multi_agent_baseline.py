"""
Run the deterministic rule-based multi-agent baseline on the five Qwen smoke tasks.

This script does not run Qwen or any LLM. It uses the existing role-based
multi-agent architecture, GoldRunner, and metrics on the same five scenarios as
the Qwen single-agent Colab batch notebook.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.multi_agent import RuleBasedMultiAgent
from tec_agents.data.datasets import get_dataset_summary, register_dataset
from tec_agents.eval.five_task_configs import (
    DEFAULT_DATASET_REF,
    get_five_task_configs,
    build_five_task_eval_task,
    build_five_task_expected_sequence,
)
from tec_agents.eval.gold_runner import GoldRunner
from tec_agents.eval.metrics import (
    MetricResult,
    compare_agent_to_gold,
    summarize_metric_results,
)
from tec_agents.eval.task_set import EvalTask, task_to_dict
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
AGGREGATE_OUTPUT = OUTPUT_DIR / "multi_agent_rule_based_five_task_batch.json"


def main() -> None:
    dataset_path = _resolve_dataset_path()
    dataset_ref = DEFAULT_DATASET_REF

    register_dataset(
        dataset_ref=dataset_ref,
        path=dataset_path,
        file_format=_dataset_format(dataset_path),
    )

    print("Dataset:")
    print(f"  path={dataset_path}")
    print(f"  summary={get_dataset_summary(dataset_ref)}")

    configs = get_five_task_configs(dataset_ref=dataset_ref)
    records: list[dict[str, Any]] = []
    metric_results: list[MetricResult] = []

    for config in configs:
        task = build_five_task_eval_task(config)
        record, metric_result = run_one_task(config=config, task=task)
        records.append(record)
        metric_results.append(metric_result)

        per_task_path = OUTPUT_DIR / (
            f"multi_agent_rule_based_{config['preset_id']}_five_task.json"
        )
        _save_json(per_task_path, record)
        print(f"Saved per-task result: {per_task_path}")

    summary = summarize_metric_results(metric_results)
    summary.update(_batch_rates(records))
    summary.update(
        {
            "architecture": "multi_agent_rule_based",
            "model_name": "none",
            "dataset_ref": dataset_ref,
            "dataset_path": str(dataset_path),
        }
    )

    aggregate = {
        "experiment_id": "multi_agent_rule_based_five_task_batch",
        "architecture": "multi_agent_rule_based",
        "model_name": "none",
        "dataset_ref": dataset_ref,
        "dataset_path": str(dataset_path),
        "summary": summary,
        "results": records,
    }
    _save_json(AGGREGATE_OUTPUT, aggregate)

    print("\nSummary table:")
    _print_summary_table(records)

    print("\nAggregate summary:")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    print(f"\nSaved aggregate result: {AGGREGATE_OUTPUT}")


def run_one_task(
    *,
    config: dict[str, Any],
    task: EvalTask,
) -> tuple[dict[str, Any], MetricResult]:
    """Run one five-task config through multi-agent, gold, and metrics."""

    preset_id = str(config["preset_id"])
    print(f"\n=== {preset_id} ({task.task_type}) ===")
    print(task.query)

    gold = GoldRunner().run(task)
    if gold.status != "success" or gold.result is None:
        metric_result = MetricResult(
            task_id=task.task_id,
            task_type=task.task_type,
            success=False,
            metrics={},
            errors=[f"Gold runner failed: {gold.error}"],
        )
        record = _record_for_gold_failure(config=config, task=task, gold=gold)
        return record, metric_result

    try:
        server = build_local_mcp_server(run_id=f"multi_agent_rule_based_{preset_id}")
        client = LocalMCPClient(server)
        agent = RuleBasedMultiAgent(client=client, dataset_ref=task.dataset_ref)
        agent_output = agent.run(task.query)
        agent_payload = _agent_payload(agent_output)

        metric_agent_result = _agent_result_for_metrics(agent_payload)
        metric_result = compare_agent_to_gold(
            task_id=task.task_id,
            task_type=task.task_type,
            agent_result=metric_agent_result,
            gold_result=gold.result,
            agent_trace=agent_payload["trace"],
            task=task_to_dict(task),
            parsed_task=agent_payload["parsed_task"],
            orchestration_steps=agent_payload["orchestration_steps"],
        )
    except Exception as exc:
        agent_payload = {
            "status": "error",
            "answer": "",
            "parsed_task": None,
            "tool_results": None,
            "trace": None,
            "orchestration_steps": [],
            "error": str(exc),
        }
        metric_result = MetricResult(
            task_id=task.task_id,
            task_type=task.task_type,
            success=False,
            metrics={},
            errors=[f"Agent failed: {exc}"],
        )

    metrics = metric_result.metrics
    actual_tool_sequence = _actual_tool_sequence(agent_payload.get("trace"))
    expected_tool_sequence = list(build_five_task_expected_sequence(config))
    agent_success = agent_payload.get("status") == "success"
    final_answer = str(agent_payload.get("answer") or "")
    final_answer_present = bool(final_answer) and not final_answer.startswith("[ERROR]")
    verdict_checks = _build_verdict_checks(
        task_type=task.task_type,
        agent_success=agent_success,
        final_answer_present=final_answer_present,
        gold_success=gold.status == "success",
        metric_result=metric_result,
        metrics=metrics,
    )
    overall_ok = all(verdict_checks.values())

    record = {
        "selected_preset": preset_id,
        "task_config": _jsonable(config),
        "task": task_to_dict(task),
        "query": task.query,
        "expected_tool_sequence": expected_tool_sequence,
        "actual_tool_sequence": actual_tool_sequence,
        "agent_result": agent_payload,
        "gold_status": gold.status,
        "gold_error": gold.error,
        "gold_result": gold.result,
        "gold_trace": gold.trace,
        "metrics": metric_result.to_dict(),
        "verdict_checks": verdict_checks,
        "overall_ok": overall_ok,
        "success": overall_ok,
        "final_answer_preview": _preview(final_answer),
    }
    record.update(_role_fields(metrics))

    _print_task_result(record)
    return record, metric_result


def _resolve_dataset_path() -> Path:
    env_path = os.environ.get("TEC_DATASET_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            PROJECT_ROOT
            / "data"
            / "processed"
            / "tec_regions_2024_01_01_to_2024_04_01_hourly.parquet",
            PROJECT_ROOT / "data" / "processed" / "tec_regions_2024_03_hourly.parquet",
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Processed dataset not found. Expected one of:\n"
        + "\n".join(f"- {path}" for path in candidates)
        + "\nBuild/download the processed parquet before running this baseline."
    )


def _dataset_format(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return "csv"
    return "parquet"


def _agent_payload(agent_output: Any) -> dict[str, Any]:
    answer = str(getattr(agent_output, "answer", ""))
    status = "success" if answer and not answer.startswith("[ERROR]") else "error"
    return {
        "status": status,
        "answer": answer,
        "parsed_task": _jsonable(getattr(agent_output, "parsed_task", None)),
        "tool_results": _jsonable(getattr(agent_output, "tool_results", None)),
        "trace": _jsonable(getattr(agent_output, "trace", None)),
        "orchestration_steps": _steps_to_dicts(
            getattr(agent_output, "orchestration_steps", [])
        ),
        "error": None if status == "success" else answer,
    }


def _agent_result_for_metrics(agent_payload: dict[str, Any]) -> dict[str, Any]:
    tool_results = agent_payload.get("tool_results")
    if isinstance(tool_results, dict):
        result = dict(tool_results)
    else:
        result = {}
    result["answer"] = agent_payload.get("answer", "")
    result["final_answer"] = agent_payload.get("answer", "")
    return result


def _record_for_gold_failure(
    *,
    config: dict[str, Any],
    task: EvalTask,
    gold: Any,
) -> dict[str, Any]:
    metric = MetricResult(
        task_id=task.task_id,
        task_type=task.task_type,
        success=False,
        metrics={},
        errors=[f"Gold runner failed: {gold.error}"],
    )
    return {
        "selected_preset": config["preset_id"],
        "task_config": _jsonable(config),
        "task": task_to_dict(task),
        "query": task.query,
        "expected_tool_sequence": list(build_five_task_expected_sequence(config)),
        "actual_tool_sequence": [],
        "agent_result": {
            "status": "skipped",
            "answer": "",
            "parsed_task": None,
            "tool_results": None,
            "trace": None,
            "orchestration_steps": [],
            "error": "gold_runner_failed",
        },
        "gold_status": gold.status,
        "gold_error": gold.error,
        "gold_result": gold.result,
        "gold_trace": gold.trace,
        "metrics": metric.to_dict(),
        "verdict_checks": {"gold_success": False},
        "overall_ok": False,
        "success": False,
        "final_answer_preview": "",
    }


def _build_verdict_checks(
    *,
    task_type: str,
    agent_success: bool,
    final_answer_present: bool,
    gold_success: bool,
    metric_result: MetricResult,
    metrics: dict[str, Any],
) -> dict[str, bool]:
    checks = {
        "agent_success": agent_success,
        "gold_success": gold_success,
        "metric_success": metric_result.success,
        "final_answer_present": final_answer_present,
        "tool_sequence_match": metrics.get("tool_sequence_match") is True,
        "no_tool_errors": metrics.get("tool_error_count") == 0,
        "no_legacy_report_tool_used": metrics.get("legacy_report_tool_used") is not True,
        "role_agent_order_match": metrics.get("role_agent_order_match") is True,
        "required_role_agents_called": metrics.get("required_role_agents_called") is True,
        "artifact_flow_valid": metrics.get("artifact_flow_valid") is True,
        "start_date_match": metrics.get("start_date_match") is not False,
        "end_date_match": metrics.get("end_date_match") is not False,
        "date_parse_match": metrics.get("date_parse_match") is not False,
    }

    if task_type == "high_tec":
        checks.update(
            {
                "threshold_exact_match": metrics.get("threshold_exact_match") is True,
                "interval_count_match": metrics.get("interval_count_match") is True,
                "global_peak_match": metrics.get("global_peak_match") is not False,
            }
        )
    elif task_type == "stable_intervals":
        checks["stable_interval_count_match"] = (
            metrics.get("stable_interval_count_match") is True
        )
    elif task_type == "compare_regions":
        checks.update(
            {
                "region_set_match": metrics.get("region_set_match") is True,
                "pairwise_delta_count_match": (
                    metrics.get("pairwise_delta_count_match") is True
                ),
                "compare_stats_present": metrics.get("compare_stats_present") is True,
                "mean_abs_error_max_zero": _zero(metrics.get("mean_abs_error_max")),
                "p90_abs_error_max_zero": _zero(metrics.get("p90_abs_error_max")),
            }
        )
    elif task_type == "report":
        checks.update(
            {
                "report_present": metrics.get("report_present") is True,
                "required_artifacts_present": (
                    metrics.get("required_artifacts_present") is True
                ),
                "report_grounded_in_artifacts": (
                    metrics.get("report_grounded_in_artifacts") is True
                ),
                "report_high_counts_match": _zero(
                    metrics.get("report_high_tec_interval_count_error_avg")
                ),
                "report_stable_counts_match": _zero(
                    metrics.get("report_stable_interval_count_error_avg")
                ),
            }
        )

    return checks


def _role_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "role_agent_order",
        "expected_role_agent_order",
        "role_agent_order_match",
        "handoff_count",
        "data_agent_called",
        "math_agent_called",
        "analysis_agent_called",
        "report_agent_called",
        "required_role_agents_called",
        "artifact_flow_valid",
        "retry_count",
        "recovery_attempt_count",
        "recovery_success_count",
        "recovery_failure_count",
    ]
    return {key: metrics.get(key) for key in keys}


def _batch_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float | None:
        values = [record.get(key) for record in records if isinstance(record.get(key), bool)]
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    agent_success_values = [
        (record.get("agent_result") or {}).get("status") == "success"
        for record in records
    ]
    metrics = [
        record.get("metrics", {}).get("metrics", {})
        for record in records
        if isinstance(record.get("metrics"), dict)
    ]

    def metric_rate(name: str) -> float | None:
        values = [item.get(name) for item in metrics if isinstance(item.get(name), bool)]
        if not values:
            return None
        return sum(1 for value in values if value) / len(values)

    return {
        "n_overall_ok": sum(1 for record in records if record.get("overall_ok")),
        "overall_ok_rate": rate("overall_ok"),
        "agent_success_rate": (
            sum(1 for value in agent_success_values if value) / len(agent_success_values)
            if agent_success_values
            else None
        ),
        "tool_sequence_match_rate": metric_rate("tool_sequence_match"),
        "role_order_match_rate": metric_rate("role_agent_order_match"),
        "artifact_flow_valid_rate": metric_rate("artifact_flow_valid"),
    }


def _actual_tool_sequence(trace: Any) -> list[str]:
    if not isinstance(trace, dict):
        return []
    return [str(call.get("tool_name")) for call in trace.get("calls", [])]


def _key_metric_summary(task_type: str, metrics: dict[str, Any]) -> str:
    if task_type == "high_tec":
        return (
            f"thr_err={metrics.get('threshold_abs_error')}; "
            f"interval_err={metrics.get('interval_count_error')}; "
            f"peak_err={metrics.get('global_peak_abs_error')}"
        )
    if task_type == "stable_intervals":
        return (
            f"stable_count_err={metrics.get('stable_interval_count_error')}; "
            f"top_duration_err={metrics.get('top_stable_duration_abs_error')}"
        )
    if task_type == "compare_regions":
        return (
            f"mean_err_avg={metrics.get('mean_abs_error_avg')}; "
            f"p90_err_avg={metrics.get('p90_abs_error_avg')}; "
            f"pairwise_match={metrics.get('pairwise_delta_count_match')}"
        )
    if task_type == "report":
        return (
            f"required_artifacts={metrics.get('required_artifacts_present')}; "
            f"grounded={metrics.get('report_grounded_in_artifacts')}; "
            f"high_count_err={metrics.get('report_high_tec_interval_count_error_avg')}; "
            f"stable_count_err={metrics.get('report_stable_interval_count_error_avg')}"
        )
    return ""


def _print_task_result(record: dict[str, Any]) -> None:
    metrics = record.get("metrics", {}).get("metrics", {})
    print(
        "Result: "
        f"agent_success={record['agent_result']['status'] == 'success'}, "
        f"overall_ok={record['overall_ok']}, "
        f"tool_sequence_match={metrics.get('tool_sequence_match')}, "
        f"role_order_match={metrics.get('role_agent_order_match')}, "
        f"artifact_flow_valid={metrics.get('artifact_flow_valid')}"
    )
    if not record["overall_ok"]:
        failed = [
            name
            for name, passed in record.get("verdict_checks", {}).items()
            if passed is not True
        ]
        print(f"Failed checks: {failed}")


def _print_summary_table(records: list[dict[str, Any]]) -> None:
    headers = [
        "preset_id",
        "task_type",
        "agent_success",
        "overall_ok",
        "tool_sequence_match",
        "tool_call_count",
        "tool_error_count",
        "final_answer_present",
        "role_agent_order_match",
        "required_role_agents_called",
        "artifact_flow_valid",
        "handoff_count",
        "retry_count",
        "key_metric_summary",
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = record.get("metrics", {}).get("metrics", {})
        task_type = record.get("task", {}).get("task_type")
        rows.append(
            {
                "preset_id": record.get("selected_preset"),
                "task_type": task_type,
                "agent_success": record.get("agent_result", {}).get("status")
                == "success",
                "overall_ok": record.get("overall_ok"),
                "tool_sequence_match": metrics.get("tool_sequence_match"),
                "tool_call_count": metrics.get("tool_call_count"),
                "tool_error_count": metrics.get("tool_error_count"),
                "final_answer_present": record.get("verdict_checks", {}).get(
                    "final_answer_present"
                ),
                "role_agent_order_match": metrics.get("role_agent_order_match"),
                "required_role_agents_called": metrics.get(
                    "required_role_agents_called"
                ),
                "artifact_flow_valid": metrics.get("artifact_flow_valid"),
                "handoff_count": metrics.get("handoff_count"),
                "retry_count": metrics.get("retry_count"),
                "key_metric_summary": _key_metric_summary(str(task_type), metrics),
            }
        )

    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _steps_to_dicts(steps: Any) -> list[dict[str, Any]]:
    if steps is None:
        return []
    return [_jsonable(step) for step in steps]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)


def _preview(text: str, limit: int = 400) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _zero(value: Any, *, eps: float = 1e-9) -> bool:
    if value is None:
        return False
    try:
        return abs(float(value)) <= eps
    except Exception:
        return False


if __name__ == "__main__":
    main()
