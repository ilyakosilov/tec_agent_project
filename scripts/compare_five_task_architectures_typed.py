"""Compare five-task outputs including the typed full LLM multi-agent run.

This script only reads JSON files and prints a table. It does not run tools,
agents, or LLM inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

SOURCE_CANDIDATES = {
    "qwen_single": [
        METRICS_DIR / "qwen_single_agent_batch_colab.json",
        METRICS_DIR / "real_runs" / "single_agent" / "qwen_single_agent_batch_colab.json",
    ],
    "rule_multi": [
        METRICS_DIR / "multi_agent_rule_based_five_task_batch.json",
        METRICS_DIR
        / "real_runs"
        / "multi_agent"
        / "experiment_1"
        / "multi_agent_rule_based_five_task_batch.json",
    ],
    "qwen_multi_old": [
        METRICS_DIR / "qwen_multi_agent_batch_colab.json",
        METRICS_DIR
        / "real_runs"
        / "multi_agent"
        / "experiment_1"
        / "qwen_multi_agent_batch_colab.json",
    ],
    "typed_v1": [
        METRICS_DIR / "qwen_multi_agent_typed_batch_colab.json",
        METRICS_DIR
        / "real_runs"
        / "multi_agent"
        / "experiment_2"
        / "qwen_multi_agent_typed_batch_colab.json",
    ],
    "typed_v2": [
        METRICS_DIR / "qwen_multi_agent_typed_v2_batch_colab.json",
        METRICS_DIR / "experiment_3" / "qwen_multi_agent_typed_v2_batch_colab.json",
        METRICS_DIR
        / "real_runs"
        / "multi_agent"
        / "experiment_3"
        / "qwen_multi_agent_typed_v2_batch_colab.json",
    ],
    "typed_v3": [
        METRICS_DIR / "qwen_multi_agent_typed_v3_batch_colab.json",
        METRICS_DIR / "experiment_4" / "qwen_multi_agent_typed_v3_batch_colab.json",
        METRICS_DIR
        / "real_runs"
        / "multi_agent"
        / "experiment_4"
        / "qwen_multi_agent_typed_v3_batch_colab.json",
    ],
}


def main() -> None:
    sources = {
        name: _load_first(name, paths)
        for name, paths in SOURCE_CANDIDATES.items()
    }
    if sources["typed_v1"] is None:
        print(
            "Typed v1 Qwen multi-agent batch file not found. "
            "Run/export qwen_multi_agent_typed_batch_colab.json first."
        )
    if sources["typed_v2"] is None:
        print(
            "Typed v2 Qwen multi-agent batch file not found. "
            "Run/export qwen_multi_agent_typed_v2_batch_colab.json first."
        )
    if sources["typed_v3"] is None:
        print(
            "Typed v3 Qwen multi-agent batch file not found. "
            "Run/export outputs/metrics/experiment_4/qwen_multi_agent_typed_v3_batch_colab.json first."
        )

    records_by_source = {
        name: _records_by_preset(payload) if payload is not None else {}
        for name, payload in sources.items()
    }
    preset_ids = sorted(
        set().union(*(set(records) for records in records_by_source.values()))
    )
    if not preset_ids:
        print("No comparable five-task records found.")
        return

    rows = []
    for preset_id in preset_ids:
        single = records_by_source["qwen_single"].get(preset_id, {})
        rule = records_by_source["rule_multi"].get(preset_id, {})
        old = records_by_source["qwen_multi_old"].get(preset_id, {})
        typed_v1 = records_by_source["typed_v1"].get(preset_id, {})
        typed_v2 = records_by_source["typed_v2"].get(preset_id, {})
        typed_v3 = records_by_source["typed_v3"].get(preset_id, {})
        task_type = (
            _task_type(single)
            or _task_type(rule)
            or _task_type(old)
            or _task_type(typed_v1)
            or _task_type(typed_v2)
            or _task_type(typed_v3)
        )
        rows.append(
            {
                "preset_id": preset_id,
                "task_type": task_type,
                "qwen_single_success": _agent_success(single),
                "qwen_single_tool_sequence_match": _metric(single, "tool_sequence_match"),
                "rule_multi_success": _agent_success(rule),
                "rule_multi_tool_sequence_match": _metric(rule, "tool_sequence_match"),
                "rule_multi_role_order_match": _metric(rule, "role_agent_order_match"),
                "qwen_multi_old_success": _agent_success(old),
                "qwen_multi_old_stalled": _agent_field(old, "stalled_loop_detected"),
                "typed_v1_success": _agent_success(typed_v1),
                "typed_v1_stalled": _agent_field(typed_v1, "stalled_loop_detected"),
                "typed_v1_role_order": " -> ".join(
                    str(item) for item in _agent_field(typed_v1, "role_agent_order") or []
                ),
                "typed_v1_tool_sequence": " -> ".join(
                    str(item) for item in _agent_field(typed_v1, "actual_tool_sequence") or []
                ),
                "typed_v2_success": _agent_success(typed_v2),
                "typed_v2_overall_ok": typed_v2.get("overall_ok"),
                "typed_v2_tool_sequence_match": _metric(typed_v2, "tool_sequence_match"),
                "typed_v2_final_answer_present": bool(
                    (_agent_field(typed_v2, "answer") or typed_v2.get("final_answer_preview"))
                ),
                "typed_v2_role_order": " -> ".join(
                    str(item) for item in _agent_field(typed_v2, "role_agent_order") or []
                ),
                "typed_v2_tool_sequence": " -> ".join(
                    str(item) for item in _agent_field(typed_v2, "actual_tool_sequence") or []
                ),
                "typed_v2_premature_role_completion_count": _agent_field(
                    typed_v2,
                    "premature_role_completion_count",
                ),
                "typed_v2_empty_findings_done_count": _agent_field(
                    typed_v2,
                    "empty_findings_done_count",
                ),
                "typed_v2_repeated_equivalent_role_assignment_count": _agent_field(
                    typed_v2,
                    "repeated_equivalent_role_assignment_count",
                ),
                "typed_v2_tool_schema_validation_error_count": _agent_field(
                    typed_v2,
                    "tool_schema_validation_error_count",
                ),
                "typed_v2_stalled_loop_detected": _agent_field(
                    typed_v2,
                    "stalled_loop_detected",
                ),
                "typed_v3_success": _agent_success(typed_v3),
                "typed_v3_overall_ok": typed_v3.get("overall_ok"),
                "typed_v3_tool_sequence_match": _metric(typed_v3, "tool_sequence_match"),
                "typed_v3_final_answer_present": bool(
                    (_agent_field(typed_v3, "answer") or typed_v3.get("final_answer_preview"))
                ),
                "typed_v3_role_order": " -> ".join(
                    str(item) for item in _agent_field(typed_v3, "role_agent_order") or []
                ),
                "typed_v3_tool_sequence": " -> ".join(
                    str(item) for item in _agent_field(typed_v3, "actual_tool_sequence") or []
                ),
                "typed_v3_premature_role_completion_count": _agent_field(
                    typed_v3,
                    "premature_role_completion_count",
                ),
                "typed_v3_empty_findings_done_count": _agent_field(
                    typed_v3,
                    "empty_findings_done_count",
                ),
                "typed_v3_repeated_equivalent_role_assignment_count": _agent_field(
                    typed_v3,
                    "repeated_equivalent_role_assignment_count",
                ),
                "typed_v3_tool_schema_validation_error_count": _agent_field(
                    typed_v3,
                    "tool_schema_validation_error_count",
                ),
                "typed_v3_invalid_artifact_handle_count": _agent_field(
                    typed_v3,
                    "invalid_artifact_handle_count",
                ),
                "typed_v3_multiple_protocol_blocks_in_single_output_count": _agent_field(
                    typed_v3,
                    "multiple_protocol_blocks_in_single_output_count",
                ),
                "typed_v3_stalled_loop_detected": _agent_field(
                    typed_v3,
                    "stalled_loop_detected",
                ),
                "key_metric_summary": _key_metric_summary(
                    str(task_type),
                    single,
                    rule,
                    old,
                    typed_v1,
                    typed_v2,
                    typed_v3,
                ),
            }
        )

    _print_table(rows)


def _load_first(name: str, paths: list[Path]) -> dict[str, Any] | None:
    for path in paths:
        if path.exists():
            print(f"{name}: {path}")
            return json.loads(path.read_text(encoding="utf-8"))
    print(f"{name}: file not found")
    return None


def _records_by_preset(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record.get("selected_preset")): record
        for record in payload.get("results", [])
        if record.get("selected_preset")
    }


def _task_type(record: dict[str, Any]) -> str | None:
    return (
        record.get("task_type")
        or (record.get("task") or {}).get("task_type")
        or (record.get("task_config") or {}).get("task_type")
    )


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    if isinstance(metrics, dict) and isinstance(metrics.get("metrics"), dict):
        return metrics["metrics"]
    return metrics if isinstance(metrics, dict) else {}


def _metric(record: dict[str, Any], name: str) -> Any:
    return _metrics(record).get(name)


def _agent_field(record: dict[str, Any], name: str) -> Any:
    result = record.get("agent_result") or {}
    if name in result:
        return result[name]
    return record.get(name)


def _agent_success(record: dict[str, Any]) -> Any:
    result = record.get("agent_result") or {}
    if "success" in result:
        return result["success"]
    if "status" in result:
        return result["status"] == "success"
    return record.get("success")


def _key_metric_summary(
    task_type: str,
    single: dict[str, Any],
    rule: dict[str, Any],
    old: dict[str, Any],
    typed_v1: dict[str, Any],
    typed_v2: dict[str, Any],
    typed_v3: dict[str, Any],
) -> str:
    if task_type == "high_tec":
        return (
            "threshold_abs_error single/rule/old/typed_v1/typed_v2/typed_v3="
            f"{_metric(single, 'threshold_abs_error')}/"
            f"{_metric(rule, 'threshold_abs_error')}/"
            f"{_metric(old, 'threshold_abs_error')}/"
            f"{_metric(typed_v1, 'threshold_abs_error')}/"
            f"{_metric(typed_v2, 'threshold_abs_error')}/"
            f"{_metric(typed_v3, 'threshold_abs_error')}"
        )
    if task_type == "stable_intervals":
        return (
            "stable_interval_count_error single/rule/old/typed_v1/typed_v2/typed_v3="
            f"{_metric(single, 'stable_interval_count_error')}/"
            f"{_metric(rule, 'stable_interval_count_error')}/"
            f"{_metric(old, 'stable_interval_count_error')}/"
            f"{_metric(typed_v1, 'stable_interval_count_error')}/"
            f"{_metric(typed_v2, 'stable_interval_count_error')}/"
            f"{_metric(typed_v3, 'stable_interval_count_error')}"
        )
    if task_type == "compare_regions":
        return (
            "mean_abs_error_avg single/rule/old/typed_v1/typed_v2/typed_v3="
            f"{_metric(single, 'mean_abs_error_avg')}/"
            f"{_metric(rule, 'mean_abs_error_avg')}/"
            f"{_metric(old, 'mean_abs_error_avg')}/"
            f"{_metric(typed_v1, 'mean_abs_error_avg')}/"
            f"{_metric(typed_v2, 'mean_abs_error_avg')}/"
            f"{_metric(typed_v3, 'mean_abs_error_avg')}"
        )
    if task_type == "report":
        return (
            "required_artifacts_present single/rule/old/typed_v1/typed_v2/typed_v3="
            f"{_metric(single, 'required_artifacts_present')}/"
            f"{_metric(rule, 'required_artifacts_present')}/"
            f"{_metric(old, 'required_artifacts_present')}/"
            f"{_metric(typed_v1, 'required_artifacts_present')}/"
            f"{_metric(typed_v2, 'required_artifacts_present')}/"
            f"{_metric(typed_v3, 'required_artifacts_present')}"
        )
    return ""


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0])
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


if __name__ == "__main__":
    main()
