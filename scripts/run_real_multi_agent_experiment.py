"""
Run the rule-based multi-agent experiment on real processed TEC data.

Expected processed dataset path:
data/processed/tec_regions_2024_03_hourly.parquet

The dataset file is intentionally not stored in Git.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.multi_agent import RuleBasedMultiAgent
from tec_agents.data.datasets import get_dataset_summary, register_dataset
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

    server = build_local_mcp_server(run_id="real_multi_agent_rule_based")
    client = LocalMCPClient(server)

    agent = RuleBasedMultiAgent(
        client=client,
        dataset_ref="default",
    )

    runner = ExperimentRunner(
        agent=agent,
        architecture="multi_agent_rule_based",
        model_name="none",
        experiment_id="real_multi_agent_rule_based_march_2024",
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

        parsed_task = record.agent.get("parsed_task")
        if parsed_task:
            print(f"    parsed_task_type={parsed_task.get('task_type')}")

        metric_values = record.metrics.get("metrics", {})
        print(
            "    "
            f"tool_calls={metric_values.get('tool_call_count')}, "
            f"tool_errors={metric_values.get('tool_error_count')}"
        )

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "real_multi_agent_rule_based_march_2024.json"
    )
    result.save_json(output_path)

    print(f"\nSaved experiment result to: {output_path}")
    print("\nReal multi-agent experiment finished.")


if __name__ == "__main__":
    main()