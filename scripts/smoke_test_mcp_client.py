"""
Smoke test for the local MCP-like server/client layer.

This test checks that agents can access TEC tools through the MCP-like client
instead of calling ToolExecutor directly.
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


from tec_agents.data.datasets import register_dataset
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


def build_tiny_dataset(path: Path) -> None:
    """Create a small synthetic regional TEC dataset for local testing."""

    path.parent.mkdir(parents=True, exist_ok=True)

    time_index = pd.date_range(
        start="2024-03-01 00:00:00",
        end="2024-03-03 23:00:00",
        freq="1h",
    )

    hours = np.arange(len(time_index))

    df = pd.DataFrame(
        {
            "time": time_index,
            "midlat_europe": 25.0 + 8.0 * np.sin(hours / 6.0) + hours * 0.03,
            "highlat_north": 12.0 + 3.0 * np.sin(hours / 8.0),
            "equatorial_atlantic": 35.0 + 10.0 * np.sin(hours / 5.0),
        }
    )

    df.loc[30:36, "midlat_europe"] += 18.0

    df.to_csv(path, index=False)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_tec_regions.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    server = build_local_mcp_server(run_id="smoke_mcp_test")
    client = LocalMCPClient(server)

    print("Available MCP-like tools:")
    for name in client.list_tool_names():
        print(f"  - {name}")

    print("\nCalling tec_get_timeseries through MCP-like client...")
    ts_result = client.call_tool_result(
        "tec_get_timeseries",
        {
            "dataset_ref": "smoke",
            "region_id": "midlat_europe",
            "start": "2024-03-01",
            "end": "2024-03-04",
        },
        agent_name="smoke_agent",
        step=1,
    )
    print(ts_result)

    series_id = ts_result["series_id"]

    print("\nCalling tec_compute_high_threshold through MCP-like client...")
    threshold_result = client.call_tool_result(
        "tec_compute_high_threshold",
        {
            "series_id": series_id,
            "method": "quantile",
            "q": 0.9,
        },
        agent_name="smoke_agent",
        step=2,
    )
    print(threshold_result)

    threshold_id = threshold_result["threshold_id"]

    print("\nCalling tec_detect_high_intervals through MCP-like client...")
    intervals_response = client.call_tool(
        "tec_detect_high_intervals",
        {
            "series_id": series_id,
            "threshold_id": threshold_id,
            "min_duration_minutes": 0,
            "merge_gap_minutes": 60,
        },
        agent_name="smoke_agent",
        step=3,
    )

    print(intervals_response.to_dict())

    print("\nChecking structured error response...")
    error_response = client.call_tool(
        "tec_unknown_tool",
        {},
        agent_name="smoke_agent",
        step=4,
    )
    print(error_response.to_dict())

    assert error_response.status == "error"

    print("\nMCP-like trace:")
    trace = client.get_trace()
    print(f"run_id: {trace['run_id']}")
    print(f"n_calls: {trace['n_calls']}")

    for call in trace["calls"]:
        print(
            f"  step={call['step']} agent={call['agent_name']} "
            f"tool={call['tool_name']} status={call['status']} "
            f"latency={call['latency_sec']:.4f}s"
        )

    print("\nMCP-like smoke test finished successfully.")


if __name__ == "__main__":
    main()