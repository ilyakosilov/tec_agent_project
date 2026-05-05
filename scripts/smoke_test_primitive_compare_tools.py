"""
Smoke test for primitive compare tools.

This test verifies the agentic compare chain:
all get_timeseries calls -> all compute_series_stats calls -> compare_stats.
No LLM or raw data download is used.
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
from tec_agents.tools.executor import build_default_executor


EXPECTED_COMPARE_SEQUENCE = [
    "tec_get_timeseries",
    "tec_get_timeseries",
    "tec_compute_series_stats",
    "tec_compute_series_stats",
    "tec_compare_stats",
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
        }
    )

    df.to_csv(path, index=False)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_primitive_compare.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    executor = build_default_executor(run_id="smoke_primitive_compare_executor")

    stats_ids = []
    for region_id in ["midlat_europe", "highlat_north"]:
        ts_result = executor.call(
            "tec_get_timeseries",
            {
                "dataset_ref": "smoke",
                "region_id": region_id,
                "start": "2024-03-01",
                "end": "2024-04-01",
            },
        )
        stats_result = executor.call(
            "tec_compute_series_stats",
            {
                "series_id": ts_result["series_id"],
            },
        )

        assert stats_result["stats_id"]
        assert stats_result["region_id"] == region_id
        assert {"mean", "max", "p90"}.issubset(stats_result["metrics"])
        stats_ids.append(stats_result["stats_id"])

    comparison = executor.call(
        "tec_compare_stats",
        {
            "stats_ids": stats_ids,
        },
    )

    assert comparison["comparison_id"]
    assert len(comparison["items"]) == 2
    assert len(comparison["pairwise_deltas"]) == 1
    assert {"mean", "max", "p90"}.issubset(
        comparison["pairwise_deltas"][0]["delta"]
    )

    server = build_local_mcp_server(run_id="smoke_primitive_compare_mcp")
    client = LocalMCPClient(server)
    tool_names = client.list_tool_names()
    assert "tec_compute_series_stats" in tool_names
    assert "tec_compare_stats" in tool_names

    agent = RuleBasedSingleAgent(client=client, dataset_ref="smoke")
    result = agent.run(
        "Compare TEC statistics for midlat_europe and highlat_north in March 2024"
    )

    tool_sequence = [call["tool_name"] for call in result.trace["calls"]]
    assert tool_sequence == EXPECTED_COMPARE_SEQUENCE
    assert result.tool_results["comparison"]["comparison_id"]
    assert result.tool_results["comparison"]["pairwise_deltas"]

    print("Primitive compare smoke test finished successfully.")


if __name__ == "__main__":
    main()
