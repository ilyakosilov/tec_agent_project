"""
Smoke test for the rule-based single-agent baseline.

This test runs a simple high-TEC task through the MCP-like client.
No LLM, GPU, or real IONEX data is required.
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

    # Add artificial high-TEC events.
    df.loc[120:130, "midlat_europe"] += 20.0
    df.loc[400:408, "midlat_europe"] += 15.0

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

    server = build_local_mcp_server(run_id="smoke_single_agent")
    client = LocalMCPClient(server)

    agent = RuleBasedSingleAgent(
        client=client,
        dataset_ref="smoke",
    )

    query = "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    result = agent.run(query)

    print("Parsed task:")
    print(result.parsed_task)

    print("\nFinal answer:")
    print(result.answer)

    print("\nTrace:")
    print(f"run_id: {result.trace['run_id']}")
    print(f"n_calls: {result.trace['n_calls']}")

    for call in result.trace["calls"]:
        print(
            f"  step={call['step']} agent={call['agent_name']} "
            f"tool={call['tool_name']} status={call['status']} "
            f"latency={call['latency_sec']:.4f}s"
        )

    assert result.parsed_task.region_id == "midlat_europe"
    assert result.parsed_task.start == "2024-03-01"
    assert result.parsed_task.end == "2024-04-01"
    assert result.tool_results["intervals"]["n_intervals"] >= 1

    print("\nSingle-agent smoke test finished successfully.")


if __name__ == "__main__":
    main()