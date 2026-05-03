"""
Compare real experiment JSON reports.

This script compares the rule-based single-agent and multi-agent experiment
results on the same real TEC dataset and prints compact summary tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Report not found: {path}\n"
            "Run the corresponding experiment script first."
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 3) -> str:
    """Format table values."""

    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def print_summary_table(reports: list[dict[str, Any]]) -> None:
    """Print compact architecture-level comparison table."""

    rows = []

    for report in reports:
        summary = report["summary"]

        rows.append(
            {
                "architecture": summary.get("architecture", report.get("architecture")),
                "model": summary.get("model_name", report.get("model_name")),
                "n_tasks": summary.get("n_tasks"),
                "success_rate": summary.get("success_rate"),
                "avg_tool_calls": summary.get("avg_tool_call_count"),
                "avg_tool_errors": summary.get("avg_tool_error_count"),
                "avg_orch_steps": summary.get("avg_orchestration_step_count"),
                "route_correct": summary.get("route_correct_rate"),
                "tool_seq_match": summary.get("tool_sequence_match_rate"),
            }
        )

    columns = [
        "architecture",
        "model",
        "n_tasks",
        "success_rate",
        "avg_tool_calls",
        "avg_tool_errors",
        "avg_orch_steps",
        "route_correct",
        "tool_seq_match",
    ]

    print("\nArchitecture summary")
    print("=" * 120)
    print_table(rows, columns)


def print_task_table(reports: list[dict[str, Any]]) -> None:
    """Print per-task comparison table."""

    rows = []

    for report in reports:
        architecture = report["summary"].get(
            "architecture",
            report.get("architecture"),
        )

        for record in report["records"]:
            task = record["task"]
            metrics = record["metrics"]
            metric_values = metrics.get("metrics", {})

            rows.append(
                {
                    "architecture": architecture,
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "success": metrics.get("success"),
                    "tool_calls": metric_values.get("tool_call_count"),
                    "tool_errors": metric_values.get("tool_error_count"),
                    "orch_steps": metric_values.get("orchestration_step_count"),
                    "route_correct": metric_values.get("route_correct"),
                    "tool_seq_match": metric_values.get("tool_sequence_match"),
                }
            )

    columns = [
        "architecture",
        "task_id",
        "task_type",
        "success",
        "tool_calls",
        "tool_errors",
        "orch_steps",
        "route_correct",
        "tool_seq_match",
    ]

    print("\nPer-task comparison")
    print("=" * 160)
    print_table(rows, columns)


def print_interpretation(reports: list[dict[str, Any]]) -> None:
    """Print short human-readable interpretation."""

    by_arch = {
        report["summary"].get("architecture", report.get("architecture")): report
        for report in reports
    }

    single = by_arch.get("single_agent_rule_based")
    multi = by_arch.get("multi_agent_rule_based")

    if not single or not multi:
        print("\nInterpretation skipped: expected both single and multi reports.")
        return

    s = single["summary"]
    m = multi["summary"]

    print("\nInterpretation")
    print("=" * 120)

    print(
        "- Both architectures use the same data, deterministic tools, "
        "gold runner and metrics."
    )

    print(
        "- Success rate: "
        f"single={fmt(s.get('success_rate'))}, "
        f"multi={fmt(m.get('success_rate'))}."
    )

    print(
        "- Average tool calls: "
        f"single={fmt(s.get('avg_tool_call_count'))}, "
        f"multi={fmt(m.get('avg_tool_call_count'))}."
    )

    print(
        "- Average orchestration steps: "
        f"single={fmt(s.get('avg_orchestration_step_count'))}, "
        f"multi={fmt(m.get('avg_orchestration_step_count'))}."
    )

    print(
        "- In the rule-based baseline, multi-agent orchestration adds explicit "
        "routing and reporting steps, but does not reduce the number of "
        "deterministic tool calls."
    )

    print(
        "- This baseline is useful before connecting Qwen: later experiments can "
        "test whether the additional orchestration reduces parsing, routing, "
        "tool-selection, or schema errors for LLM agents."
    )


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Print a simple fixed-width table."""

    if not rows:
        print("<no rows>")
        return

    formatted_rows = [
        {column: fmt(row.get(column)) for column in columns}
        for row in rows
    ]

    widths = {
        column: max(
            len(column),
            max(len(row[column]) for row in formatted_rows),
        )
        for column in columns
    }

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    sep = "-+-".join("-" * widths[column] for column in columns)

    print(header)
    print(sep)

    for row in formatted_rows:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    paths = [
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "real_single_agent_rule_based_march_2024.json",
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "real_multi_agent_rule_based_march_2024.json",
    ]

    reports = [load_json(path) for path in paths]

    print("Loaded reports:")
    for path in paths:
        print(f"  - {path}")

    print_summary_table(reports)
    print_task_table(reports)
    print_interpretation(reports)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise