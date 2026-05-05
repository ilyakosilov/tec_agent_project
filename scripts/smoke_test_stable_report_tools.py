"""
Smoke test for stable-interval and structured-report scenarios.

This test uses a synthetic processed TEC dataset. It does not download raw
IONEX data and does not call any LLM.
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
from tec_agents.agents.single_agent import RuleBasedSingleAgent
from tec_agents.data.datasets import register_dataset
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server
from tec_agents.tools.executor import build_default_executor


def build_tiny_dataset(path: Path) -> None:
    """Create a synthetic regional TEC dataset with calm and disturbed periods."""

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
            "midlat_europe": 25.0 + 2.0 * np.sin(hours / 24.0),
            "highlat_north": 12.0 + 1.0 * np.sin(hours / 18.0),
            "equatorial_atlantic": 35.0 + 4.0 * np.sin(hours / 12.0),
            "equatorial_africa": 38.0 + 4.5 * np.sin(hours / 10.0),
            "equatorial_pacific": 32.0 + 3.5 * np.sin(hours / 15.0),
        }
    )

    # Add a few deterministic disturbed periods so reports include high-TEC events.
    df.loc[120:130, "midlat_europe"] += 20.0
    df.loc[300:306, "highlat_north"] += 8.0
    df.loc[420:430, "equatorial_atlantic"] += 16.0

    df.to_csv(path, index=False)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_stable_report.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    executor = build_default_executor(run_id="smoke_stable_report_executor")

    ts_result = executor.call(
        "tec_get_timeseries",
        {
            "dataset_ref": "smoke",
            "region_id": "midlat_europe",
            "start": "2024-03-01",
            "end": "2024-04-01",
        },
    )
    thresholds = executor.call(
        "tec_compute_stability_thresholds",
        {
            "series_id": ts_result["series_id"],
            "window_minutes": 180,
            "method": "quantile",
            "q_delta": 0.60,
            "q_std": 0.60,
        },
    )
    intervals = executor.call(
        "tec_detect_stable_intervals",
        {
            "series_id": ts_result["series_id"],
            "threshold_id": thresholds["threshold_id"],
            "min_duration_minutes": 180,
            "merge_gap_minutes": 60,
        },
    )

    assert "max_delta_threshold" in thresholds
    assert "rolling_std_threshold" in thresholds
    assert intervals["n_intervals"] >= 1

    report = executor.call(
        "tec_build_report",
        {
            "dataset_ref": "smoke",
            "regions": ["midlat_europe", "highlat_north"],
            "start": "2024-03-01",
            "end": "2024-04-01",
            "include": ["basic_stats", "high_tec", "stable_intervals"],
        },
    )

    assert set(report["sections"]) == {
        "basic_stats",
        "high_tec",
        "stable_intervals",
    }

    server = build_local_mcp_server(run_id="smoke_stable_report_mcp")
    client = LocalMCPClient(server)

    mcp_report = client.call_tool_result(
        "tec_build_report",
        {
            "dataset_ref": "smoke",
            "regions": ["midlat_europe", "highlat_north"],
            "start": "2024-03-01",
            "end": "2024-04-01",
        },
        agent_name="smoke_agent",
        step=1,
    )
    assert "stable_intervals" in mcp_report["sections"]

    single_agent = RuleBasedSingleAgent(client=client, dataset_ref="smoke")
    single_agent.reset()

    stable_single = single_agent.run(
        "Find stable TEC intervals for midlat_europe in March 2024"
    )
    assert stable_single.parsed_task.task_type == "stable_intervals"
    assert stable_single.trace["n_calls"] == 3

    single_agent.reset()
    report_single = single_agent.run(
        "Build a TEC report for midlat_europe and highlat_north in March 2024"
    )
    assert report_single.parsed_task.task_type == "report"
    assert report_single.trace["n_calls"] == 13
    assert all(
        call["tool_name"] != "tec_build_report"
        for call in report_single.trace["calls"]
    )

    multi_agent = RuleBasedMultiAgent(client=client, dataset_ref="smoke")
    multi_agent.reset()

    stable_multi = multi_agent.run(
        "Find low variability TEC periods for highlat_north in March 2024"
    )
    assert stable_multi.parsed_task.task_type == "stable_intervals"
    assert [step.node for step in stable_multi.orchestration_steps] == [
        "orchestrator",
        "data_agent",
        "math_agent",
        "analysis_agent",
        "report_agent",
    ]
    assert stable_multi.trace["n_calls"] == 3

    multi_agent.reset()
    report_multi = multi_agent.run(
        "Create a summary report for equatorial_atlantic, equatorial_africa "
        "and equatorial_pacific in March 2024"
    )
    assert report_multi.parsed_task.task_type == "report"
    assert [step.node for step in report_multi.orchestration_steps] == [
        "orchestrator",
        "data_agent",
        "math_agent",
        "analysis_agent",
        "report_agent",
    ]
    assert report_multi.trace["n_calls"] == 19
    assert all(
        call["tool_name"] != "tec_build_report"
        for call in report_multi.trace["calls"]
    )

    print("Stable/report smoke test finished successfully.")


if __name__ == "__main__":
    main()
