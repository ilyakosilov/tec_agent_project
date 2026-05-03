"""
Smoke test for evaluation metrics.

This test compares the rule-based single-agent result with the deterministic
gold result on the same synthetic task.
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
from tec_agents.eval.gold_runner import GoldRunner
from tec_agents.eval.metrics import compare_agent_to_gold, summarize_metric_results
from tec_agents.eval.task_set import build_smoke_tasks
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


def build_tiny_dataset(path: Path) -> None:
    """Create a small synthetic regional TEC dataset for local testing."""

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
    task = tasks[0]

    gold = GoldRunner().run(task)
    assert gold.status == "success"
    assert gold.result is not None

    server = build_local_mcp_server(run_id=f"agent_{task.task_id}")
    client = LocalMCPClient(server)
    agent = RuleBasedSingleAgent(client=client, dataset_ref="smoke")

    agent_output = agent.run(task.query)

    metric_result = compare_agent_to_gold(
        task_id=task.task_id,
        task_type=task.task_type,
        agent_result=agent_output.tool_results,
        gold_result=gold.result,
        agent_trace=agent_output.trace,
    )

    print("Metric result:")
    print(metric_result.to_dict())

    summary = summarize_metric_results([metric_result])
    print("\nSummary:")
    print(summary)

    assert metric_result.success is True
    assert metric_result.metrics["threshold_abs_error"] == 0
    assert metric_result.metrics["interval_count_error"] == 0
    assert metric_result.metrics["tool_call_count"] == 3

    print("\nMetrics smoke test finished successfully.")


if __name__ == "__main__":
    main()