"""Compare five-task outputs across Qwen single, rule multi, and Qwen multi.

This script only reads JSON files and prints a table. It does not run tools,
agents, or LLM inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
QWEN_SINGLE_PATH = METRICS_DIR / "qwen_single_agent_batch_colab.json"
RULE_MULTI_PATH = METRICS_DIR / "multi_agent_rule_based_five_task_batch.json"
QWEN_MULTI_PATH = METRICS_DIR / "qwen_multi_agent_batch_colab.json"


def main() -> None:
    sources = {
        "qwen_single": _load_optional(QWEN_SINGLE_PATH),
        "rule_multi": _load_optional(RULE_MULTI_PATH),
        "qwen_multi": _load_optional(QWEN_MULTI_PATH),
    }
    if all(payload is None for payload in sources.values()):
        print("No five-task architecture result files found.")
        return

    records_by_source = {
        name: _records_by_preset(payload) if payload is not None else {}
        for name, payload in sources.items()
    }
    preset_ids = sorted(
        set().union(*(set(records) for records in records_by_source.values()))
    )
    rows = []
    for preset_id in preset_ids:
        single = records_by_source["qwen_single"].get(preset_id, {})
        rule = records_by_source["rule_multi"].get(preset_id, {})
        multi = records_by_source["qwen_multi"].get(preset_id, {})
        task_type = _task_type(single) or _task_type(rule) or _task_type(multi)
        rows.append(
            {
                "preset_id": preset_id,
                "task_type": task_type,
                "qwen_single_success": _agent_success(single),
                "qwen_single_overall_ok": _overall_ok(single),
                "qwen_single_tool_sequence_match": _metric(single, "tool_sequence_match"),
                "qwen_single_final_answer_present": _final_answer_present(single),
                "qwen_single_parse_error_count": _agent_field(
                    single, "parse_error_count"
                ),
                "qwen_single_stalled_loop_detected": _agent_field(
                    single, "stalled_loop_detected"
                ),
                "rule_multi_success": _agent_success(rule),
                "rule_multi_overall_ok": _overall_ok(rule),
                "rule_multi_tool_sequence_match": _metric(rule, "tool_sequence_match"),
                "rule_multi_role_order_match": _metric(rule, "role_agent_order_match"),
                "rule_multi_artifact_flow_valid": _metric(rule, "artifact_flow_valid"),
                "qwen_multi_success": _agent_success(multi),
                "qwen_multi_overall_ok": _overall_ok(multi),
                "qwen_multi_tool_sequence_match": _metric(multi, "tool_sequence_match"),
                "qwen_multi_role_order_match": _metric(
                    multi, "role_agent_order_match"
                ),
                "qwen_multi_artifact_flow_valid": _metric(
                    multi, "artifact_flow_valid"
                ),
                "qwen_multi_final_answer_present": _final_answer_present(multi),
                "qwen_multi_parse_error_count": _agent_field(
                    multi, "parse_error_count"
                ),
                "qwen_multi_stalled_loop_detected": _agent_field(
                    multi, "stalled_loop_detected"
                ),
                "key_metric_comparison": _key_metric_comparison(
                    str(task_type), single, rule, multi
                ),
            }
        )

    _print_table(rows)


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"Result file not found, skipping: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    return result.get(name)


def _agent_success(record: dict[str, Any]) -> Any:
    result = record.get("agent_result") or {}
    if "success" in result:
        return result["success"]
    if "status" in result:
        return result["status"] == "success"
    return None


def _overall_ok(record: dict[str, Any]) -> Any:
    if "overall_ok" in record:
        return record["overall_ok"]
    return record.get("success")


def _final_answer_present(record: dict[str, Any]) -> bool | None:
    if not record:
        return None
    result = record.get("agent_result") or {}
    answer = result.get("answer")
    if answer is None:
        answer = result.get("final_answer")
    return bool(answer)


def _key_metric_comparison(
    task_type: str,
    single: dict[str, Any],
    rule: dict[str, Any],
    multi: dict[str, Any],
) -> str:
    if task_type == "high_tec":
        return (
            f"thr_err single/rule/multi="
            f"{_metric(single, 'threshold_abs_error')}/"
            f"{_metric(rule, 'threshold_abs_error')}/"
            f"{_metric(multi, 'threshold_abs_error')}; "
            f"interval_err="
            f"{_metric(single, 'interval_count_error')}/"
            f"{_metric(rule, 'interval_count_error')}/"
            f"{_metric(multi, 'interval_count_error')}"
        )
    if task_type == "stable_intervals":
        return (
            "stable_count_err single/rule/multi="
            f"{_metric(single, 'stable_interval_count_error')}/"
            f"{_metric(rule, 'stable_interval_count_error')}/"
            f"{_metric(multi, 'stable_interval_count_error')}"
        )
    if task_type == "compare_regions":
        return (
            "mean_max_err single/rule/multi="
            f"{_metric(single, 'mean_abs_error_max')}/"
            f"{_metric(rule, 'mean_abs_error_max')}/"
            f"{_metric(multi, 'mean_abs_error_max')}; "
            "pairwise="
            f"{_metric(single, 'pairwise_delta_count_match')}/"
            f"{_metric(rule, 'pairwise_delta_count_match')}/"
            f"{_metric(multi, 'pairwise_delta_count_match')}"
        )
    if task_type == "report":
        return (
            "required_artifacts single/rule/multi="
            f"{_metric(single, 'required_artifacts_present')}/"
            f"{_metric(rule, 'required_artifacts_present')}/"
            f"{_metric(multi, 'required_artifacts_present')}; "
            "grounded="
            f"{_metric(single, 'report_grounded_in_artifacts')}/"
            f"{_metric(rule, 'report_grounded_in_artifacts')}/"
            f"{_metric(multi, 'report_grounded_in_artifacts')}"
        )
    return ""


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No comparable records found.")
        return
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
