"""
Smoke test for the role-based multi-agent workflow.

This verifies that the deterministic multi-agent baseline uses the role order:
orchestrator -> data_agent -> math_agent -> analysis_agent -> report_agent.
It also checks grouped data loading before math for compare/report tasks.
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


from tec_agents.agents.multi_agent import RuleBasedMultiAgent
from tec_agents.data.datasets import register_dataset
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


EXPECTED_ROLE_NODES = [
    "orchestrator",
    "data_agent",
    "math_agent",
    "analysis_agent",
    "report_agent",
]

EXPECTED_COMPARE_SEQUENCE = [
    "tec_get_timeseries",
    "tec_get_timeseries",
    "tec_compute_series_stats",
    "tec_compute_series_stats",
    "tec_compare_stats",
]

EXPECTED_REPORT_SEQUENCE = [
    "tec_get_timeseries",
    "tec_get_timeseries",
    "tec_compute_series_stats",
    "tec_compute_series_stats",
    "tec_compare_stats",
    "tec_compute_high_threshold",
    "tec_detect_high_intervals",
    "tec_compute_high_threshold",
    "tec_detect_high_intervals",
    "tec_compute_stability_thresholds",
    "tec_detect_stable_intervals",
    "tec_compute_stability_thresholds",
    "tec_detect_stable_intervals",
]


def build_tiny_dataset(path: Path) -> None:
    """Create a small synthetic regional TEC dataset."""

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
    df.loc[200:205, "highlat_north"] += 8.0

    df.to_csv(path, index=False)


def tool_sequence(result) -> list[str]:
    """Return tool names from an agent result trace."""

    return [call["tool_name"] for call in result.trace["calls"]]


def role_nodes(result) -> list[str]:
    """Return orchestration nodes from an agent result."""

    return [step.node for step in result.orchestration_steps]


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_role_workflow.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    server = build_local_mcp_server(run_id="smoke_multi_agent_role_workflow")
    client = LocalMCPClient(server)
    agent = RuleBasedMultiAgent(client=client, dataset_ref="smoke")

    high_result = agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )
    assert role_nodes(high_result) == EXPECTED_ROLE_NODES
    assert tool_sequence(high_result) == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert set(high_result.tool_results) == {"data", "math", "analysis"}

    agent.reset()
    compare_result = agent.run(
        "Compare TEC statistics for midlat_europe and highlat_north in March 2024"
    )
    assert role_nodes(compare_result) == EXPECTED_ROLE_NODES
    assert tool_sequence(compare_result) == EXPECTED_COMPARE_SEQUENCE

    agent.reset()
    report_result = agent.run(
        "Build a TEC report for midlat_europe and highlat_north in March 2024"
    )
    report_sequence = tool_sequence(report_result)

    assert role_nodes(report_result) == EXPECTED_ROLE_NODES
    assert "tec_build_report" not in report_sequence
    assert report_sequence[:2] == ["tec_get_timeseries", "tec_get_timeseries"]
    assert report_sequence == EXPECTED_REPORT_SEQUENCE
    assert "tec_compute_series_stats" in report_sequence
    assert "tec_compute_high_threshold" in report_sequence
    assert "tec_detect_high_intervals" in report_sequence
    assert "tec_compute_stability_thresholds" in report_sequence
    assert "tec_detect_stable_intervals" in report_sequence

    print("Multi-agent role workflow smoke test finished successfully.")


if __name__ == "__main__":
    main()
