"""
Compare exported Qwen single-agent batch results with rule-based multi-agent baseline.

This script reads existing JSON files only. It does not run Qwen, tools, or
agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_BATCH_PATH = PROJECT_ROOT / "outputs" / "metrics" / "qwen_single_agent_batch_colab.json"
MULTI_BATCH_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / "multi_agent_rule_based_five_task_batch.json"
)


def main() -> None:
    if not QWEN_BATCH_PATH.exists():
        print(
            "Qwen batch file not found, run/export "
            "qwen_single_agent_batch_colab.json first."
        )
        print(f"Expected path: {QWEN_BATCH_PATH}")
        return

    if not MULTI_BATCH_PATH.exists():
        raise FileNotFoundError(
            "Multi-agent baseline file not found. Run "
            "scripts/run_five_task_multi_agent_baseline.py first.\n"
            f"Expected path: {MULTI_BATCH_PATH}"
        )

    qwen = _load_json(QWEN_BATCH_PATH)
    multi = _load_json(MULTI_BATCH_PATH)
    qwen_by_preset = _records_by_preset(qwen)
    multi_by_preset = _records_by_preset(multi)

    all_presets = sorted(set(qwen_by_preset) | set(multi_by_preset))
    rows = []
    for preset_id in all_presets:
        qwen_record = qwen_by_preset.get(preset_id, {})
        multi_record = multi_by_preset.get(preset_id, {})
        task_type = _task_type(qwen_record) or _task_type(multi_record)
        rows.append(
            {
                "preset_id": preset_id,
                "task_type": task_type,
                "qwen_agent_success": _qwen_agent_success(qwen_record),
                "qwen_overall_ok": _overall_ok(qwen_record),
                "qwen_tool_sequence_match": _metric(qwen_record, "tool_sequence_match"),
                "qwen_final_answer_present": _qwen_final_answer_present(qwen_record),
                "qwen_parse_error_count": _qwen_field(qwen_record, "parse_error_count"),
                "qwen_stalled_loop_detected": _qwen_field(
                    qwen_record, "stalled_loop_detected"
                ),
                "multi_agent_success": _multi_agent_success(multi_record),
                "multi_agent_overall_ok": _overall_ok(multi_record),
                "multi_agent_tool_sequence_match": _metric(
                    multi_record, "tool_sequence_match"
                ),
                "multi_agent_role_order_match": _metric(
                    multi_record, "role_agent_order_match"
                ),
                "multi_agent_artifact_flow_valid": _metric(
                    multi_record, "artifact_flow_valid"
                ),
                "multi_agent_final_answer_present": multi_record.get(
                    "verdict_checks", {}
                ).get("final_answer_present"),
                "key_metric_comparison": _key_metric_comparison(
                    str(task_type), qwen_record, multi_record
                ),
            }
        )

    _print_table(rows)


def _load_json(path: Path) -> dict[str, Any]:
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


def _metric(record: dict[str, Any], name: str) -> Any:
    metrics = record.get("metrics") or {}
    if "metrics" in metrics:
        metrics = metrics["metrics"]
    return metrics.get(name)


def _qwen_field(record: dict[str, Any], name: str) -> Any:
    return (record.get("agent_result") or {}).get(name)


def _overall_ok(record: dict[str, Any]) -> Any:
    if "overall_ok" in record:
        return record["overall_ok"]
    return record.get("success")


def _qwen_agent_success(record: dict[str, Any]) -> Any:
    result = record.get("agent_result") or {}
    return result.get("success")


def _qwen_final_answer_present(record: dict[str, Any]) -> bool:
    answer = str((record.get("agent_result") or {}).get("answer") or "")
    return bool(answer)


def _multi_agent_success(record: dict[str, Any]) -> Any:
    return (record.get("agent_result") or {}).get("status") == "success"


def _key_metric_comparison(
    task_type: str,
    qwen_record: dict[str, Any],
    multi_record: dict[str, Any],
) -> str:
    if task_type == "high_tec":
        return (
            f"qwen_thr_err={_metric(qwen_record, 'threshold_abs_error')}, "
            f"multi_thr_err={_metric(multi_record, 'threshold_abs_error')}, "
            f"qwen_interval_err={_metric(qwen_record, 'interval_count_error')}, "
            f"multi_interval_err={_metric(multi_record, 'interval_count_error')}"
        )
    if task_type == "stable_intervals":
        return (
            f"qwen_stable_err={_metric(qwen_record, 'stable_interval_count_error')}, "
            f"multi_stable_err={_metric(multi_record, 'stable_interval_count_error')}"
        )
    if task_type == "compare_regions":
        return (
            f"qwen_mean_max_err={_metric(qwen_record, 'mean_abs_error_max')}, "
            f"multi_mean_max_err={_metric(multi_record, 'mean_abs_error_max')}, "
            f"qwen_pairwise={_metric(qwen_record, 'pairwise_delta_count_match')}, "
            f"multi_pairwise={_metric(multi_record, 'pairwise_delta_count_match')}"
        )
    if task_type == "report":
        return (
            f"qwen_required={_metric(qwen_record, 'required_artifacts_present')}, "
            f"multi_required={_metric(multi_record, 'required_artifacts_present')}, "
            f"qwen_grounded={_metric(qwen_record, 'report_grounded_in_artifacts')}, "
            f"multi_grounded={_metric(multi_record, 'report_grounded_in_artifacts')}"
        )
    return ""


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No records found.")
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
