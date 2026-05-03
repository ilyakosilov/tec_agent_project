"""
Run a small smoke experiment.

This script creates a synthetic TEC dataset, registers it, builds smoke tasks,
runs the rule-based single-agent baseline, compares it with gold results, and
saves a structured JSON report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.single_agent import RuleBasedSingleAgent
from tec_agents.data.datasets import register_dataset
from tec_agents.eval.experiment_runner import ExperimentRunner
from tec_agents.eval.task_set import build_smoke_tasks
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


def build_tiny_dataset(path: Path) -> None:
    """Create a synthetic regional TEC dataset for smoke experiments."""

    path.parent.mkdir(parents=True, exist_ok=True)

    time_index = pd.date_range(
        start="2024-03-01 00:00:00",
        end="2024-03-31 23:00:00",
        freq="1h",
    )

    hours = np.arange(len(time_index))

    df = pd.DataFrame(
        {
            "time": time_index,
            "midlat_europe": 25.0 + 8.0 * np.sin(hours / 6.0),
            "highlat_north": 12.0 + 3.0 * np.sin(hours / 8.0),
            "equatorial_atlantic": 35.0 + 10.0 * np.sin(hours / 5.0),
        }
    )

    # Artificial high-TEC events.
    df.loc[120:130, "midlat_europe"] += 20.0
    df.loc[400:408, "midlat_europe"] += 15.0
    df.loc[200:205, "highlat_north"] += 8.0

    df.to_csv(path, index=False)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_tec_regions_month.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    tasks = build_smoke_tasks(dataset_ref="smoke")

    server = build_local_mcp_server(run_id="smoke_experiment_agent")
    client = LocalMCPClient(server)
    agent = RuleBasedSingleAgent(
        client=client,
        dataset_ref="smoke",
    )

    runner = ExperimentRunner(
        agent=agent,
        architecture="single_agent_rule_based",
        model_name="none",
        experiment_id="smoke_single_agent_rule_based",
    )

    result = runner.run_tasks(tasks)

    print("Experiment summary:")
    print(result.summary)

    print("\nTask metrics:")
    for record in result.records:
        task_id = record.task["task_id"]
        status = record.agent["status"]
        success = record.metrics["success"]
        errors = record.metrics["errors"]
        print(f"  - {task_id}: agent_status={status}, metric_success={success}")
        if errors:
            print(f"    errors={errors}")

    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "smoke_single_agent_rule_based.json"
    )
    result.save_json(output_path)

    print(f"\nSaved experiment result to: {output_path}")

    # The current rule-based agent supports high_tec only, so the compare task
    # is expected to fail until we extend the baseline.
    assert result.summary["n_tasks"] == 3
    assert result.summary["n_success"] >= 2

    print("\nSmoke experiment finished successfully.")


if __name__ == "__main__":
    main()