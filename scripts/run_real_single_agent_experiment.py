"""
Run the rule-based single-agent experiment on real processed TEC data.

Expected processed dataset path:
data/processed/tec_regions_2024_03_hourly.parquet

The dataset file is intentionally not stored in Git. It should be created in
Colab and then downloaded locally into data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.single_agent import RuleBasedSingleAgent
from tec_agents.data.datasets import (
    get_dataset_summary,
    register_dataset,
)
from tec_agents.eval.experiment_runner import ExperimentRunner
from tec_agents.eval.task_set import build_default_research_tasks
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "processed" / "tec_regions_2024_03_hourly.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}\n"
            "Create it in Colab first, then download it into data/processed/."
        )

    register_dataset(
        dataset_ref="default",
        path=dataset_path,
        file_format="parquet",
    )

    print("Dataset summary:")
    summary = get_dataset_summary("default")
    print(summary)

    tasks = build_default_research_tasks(dataset_ref="default")

    print("\nResearch tasks:")
    for task in tasks:
        print(f"  - {task.task_id}: {task.task_type}")

    server = build_local_mcp_server(run_id="real_single_agent_rule_based")
    client = LocalMCPClient(server)

    agent = RuleBasedSingleAgent(
        client=client,
        dataset_ref="default",
    )

    runner = ExperimentRunner(
        agent=agent,
        architecture="single_agent_rule_based",
        model_name="none",
        experiment_id="real_single_agent_rule_based_march_2024",
    )

    result = runner.run_tasks(tasks)

    print("\nExperiment summary:")
    print(result.summary)

    print("\nTask metrics:")
    for record in result.records:
        task_id = record.task["task_id"]
        task_type = record.task["task_type"]
        agent_status = record.agent["status"]
        metric_success = record.metrics["success"]
        errors = record.metrics["errors"]

        print(
            f"  - {task_id}: "
            f"type={task_type}, "
            f"agent_status={agent_status}, "
            f"metric_success={metric_success}"
        )

        if errors:
            print(f"    errors={errors}")

        metric_values = record.metrics.get("metrics", {})
        if task_type == "high_tec":
            print(
                "    "
                f"threshold_abs_error={metric_values.get('threshold_abs_error')}, "
                f"interval_count_error={metric_values.get('interval_count_error')}, "
                f"tool_calls={metric_values.get('tool_call_count')}"
            )

        if task_type == "compare_regions":
            print(
                "    "
                f"region_set_match={metric_values.get('region_set_match')}, "
                f"mean_abs_error_max={metric_values.get('mean_abs_error_max')}, "
                f"tool_calls={metric_values.get('tool_call_count')}"
            )

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "real_single_agent_rule_based_march_2024.json"
    )
    result.save_json(output_path)

    print(f"\nSaved experiment result to: {output_path}")

    if result.summary["n_success"] != result.summary["n_tasks"]:
        print(
            "\nSome tasks failed. This is not necessarily a code error: "
            "the current rule-based agent supports only simple query patterns."
        )

    print("\nReal single-agent experiment finished.")


if __name__ == "__main__":
    main()