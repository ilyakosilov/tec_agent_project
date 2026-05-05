"""
Smoke test for TEC tool registry and executor.

This test does not require GPU, Colab, Qwen, or real IONEX data.
It creates a tiny synthetic TEC dataset, registers it, and calls several tools.
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
from tec_agents.tools.executor import build_default_executor


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

    # Add an artificial high-TEC bump for testing interval detection.
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

    executor = build_default_executor(run_id="smoke_test")

    print("Available tools:")
    tool_names = [tool["name"] for tool in executor.list_tools()]
    assert "tec_build_report" not in tool_names

    for tool in executor.list_tools():
        print(f"  - {tool['name']}")

    print("\nCalling tec_get_timeseries...")
    ts_result = executor.call(
        "tec_get_timeseries",
        {
            "dataset_ref": "smoke",
            "region_id": "midlat_europe",
            "start": "2024-03-01",
            "end": "2024-03-04",
        },
    )
    print(ts_result)

    series_id = ts_result["series_id"]

    print("\nCalling tec_series_profile...")
    profile_result = executor.call(
        "tec_series_profile",
        {
            "series_id": series_id,
        },
    )
    print(profile_result)

    print("\nCalling tec_compute_high_threshold...")
    threshold_result = executor.call(
        "tec_compute_high_threshold",
        {
            "series_id": series_id,
            "method": "quantile",
            "q": 0.9,
        },
    )
    print(threshold_result)

    threshold_id = threshold_result["threshold_id"]

    print("\nCalling tec_detect_high_intervals...")
    intervals_result = executor.call(
        "tec_detect_high_intervals",
        {
            "series_id": series_id,
            "threshold_id": threshold_id,
            "min_duration_minutes": 0,
            "merge_gap_minutes": 60,
        },
    )
    print(intervals_result)

    print("\nExecutor trace:")
    trace = executor.get_trace()
    print(f"run_id: {trace['run_id']}")
    print(f"n_calls: {trace['n_calls']}")
    for call in trace["calls"]:
        print(
            f"  step={call['step']} tool={call['tool_name']} "
            f"status={call['status']} latency={call['latency_sec']:.4f}s"
        )

    print("\nSmoke test finished successfully.")


if __name__ == "__main__":
    main()
