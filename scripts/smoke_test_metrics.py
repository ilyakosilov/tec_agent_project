"""
Smoke test for evaluation metrics.

This test compares the rule-based single-agent result with the deterministic
gold result on the same synthetic task.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict
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
from tec_agents.eval.gold_runner import GoldRunner
from tec_agents.eval.metrics import compare_agent_to_gold, summarize_metric_results
from tec_agents.eval.task_set import EvalTask, build_smoke_tasks, task_to_dict
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
        task=task_to_dict(task),
        parsed_task=asdict(agent_output.parsed_task),
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
    assert metric_result.metrics["start_date_match"] is True
    assert metric_result.metrics["end_date_match"] is True
    assert metric_result.metrics["date_parse_match"] is True
    assert metric_result.metrics["expected_n_points"] == 744
    assert metric_result.metrics["agent_timeseries_n_points"] == 744
    assert metric_result.metrics["gold_timeseries_n_points"] == 744
    assert metric_result.metrics["timeseries_n_points_match"] is True

    parsed_without_region = asdict(agent_output.parsed_task)
    parsed_without_region["region_id"] = None
    parsed_without_region["region_ids"] = ()
    region_fallback_metric = compare_agent_to_gold(
        task_id=task.task_id,
        task_type=task.task_type,
        agent_result=agent_output.tool_results,
        gold_result=gold.result,
        agent_trace=agent_output.trace,
        task=task_to_dict(task),
        parsed_task=parsed_without_region,
    )

    assert region_fallback_metric.metrics["region_parse_match"] is True
    assert region_fallback_metric.metrics["expected_region_id"] == "midlat_europe"
    assert region_fallback_metric.metrics["actual_region_id"] == "midlat_europe"
    assert (
        region_fallback_metric.metrics["actual_region_source"]
        == "first_timeseries_tool_call"
    )

    wrong_date_trace = deepcopy(agent_output.trace)
    first_call = wrong_date_trace["calls"][0]
    first_call["arguments"]["end"] = "2024-03-31"
    first_call["output"]["metadata"]["requested_end"] = "2024-03-31"
    first_call["output"]["metadata"]["end"] = "2024-03-31"
    first_call["output"]["metadata"]["n_points"] = 720

    wrong_date_metric = compare_agent_to_gold(
        task_id=task.task_id,
        task_type=task.task_type,
        agent_result=agent_output.tool_results,
        gold_result=gold.result,
        agent_trace=wrong_date_trace,
        task=task_to_dict(task),
        parsed_task=asdict(agent_output.parsed_task),
    )

    assert wrong_date_metric.success is False
    assert wrong_date_metric.metrics["end_date_match"] is False
    assert wrong_date_metric.metrics["date_parse_match"] is False
    assert wrong_date_metric.metrics["timeseries_n_points_match"] is False

    stable_task = EvalTask(
        task_id="smoke_stable_midlat_europe_march_2024",
        query="Find stable TEC intervals for midlat_europe in March 2024",
        task_type="stable_intervals",
        dataset_ref="smoke",
        region_id="midlat_europe",
        region_ids=("midlat_europe",),
        start="2024-03-01",
        end="2024-04-01",
        expected_tool_sequence=(
            "tec_get_timeseries",
            "tec_compute_stability_thresholds",
            "tec_detect_stable_intervals",
        ),
    )

    stable_gold = GoldRunner().run(stable_task)
    assert stable_gold.status == "success"
    assert stable_gold.result is not None

    stable_server = build_local_mcp_server(run_id=f"agent_{stable_task.task_id}")
    stable_client = LocalMCPClient(stable_server)
    stable_agent = RuleBasedSingleAgent(client=stable_client, dataset_ref="smoke")
    stable_output = stable_agent.run(stable_task.query)

    stable_metric = compare_agent_to_gold(
        task_id=stable_task.task_id,
        task_type=stable_task.task_type,
        agent_result=stable_output.tool_results,
        gold_result=stable_gold.result,
        agent_trace=stable_output.trace,
        task=task_to_dict(stable_task),
        parsed_task=asdict(stable_output.parsed_task),
    )

    assert stable_metric.success is True
    assert stable_metric.metrics["gold_timeseries_n_points"] == 744
    assert stable_metric.metrics["agent_timeseries_n_points"] == 744
    assert stable_metric.metrics["expected_n_points"] == 744
    assert stable_metric.metrics["timeseries_n_points_match"] is True

    compare_task = tasks[2]
    compare_gold = GoldRunner().run(compare_task)
    assert compare_gold.status == "success"
    assert compare_gold.result is not None

    compare_server = build_local_mcp_server(run_id=f"agent_{compare_task.task_id}")
    compare_client = LocalMCPClient(compare_server)
    compare_agent = RuleBasedSingleAgent(client=compare_client, dataset_ref="smoke")
    compare_output = compare_agent.run(compare_task.query)

    compare_metric = compare_agent_to_gold(
        task_id=compare_task.task_id,
        task_type=compare_task.task_type,
        agent_result=compare_output.tool_results,
        gold_result=compare_gold.result,
        agent_trace=compare_output.trace,
        task=task_to_dict(compare_task),
        parsed_task=asdict(compare_output.parsed_task),
    )

    assert compare_metric.success is True
    assert compare_metric.metrics["region_set_match"] is True
    assert compare_metric.metrics["compare_stats_present"] is True
    assert compare_metric.metrics["pairwise_delta_count_match"] is True
    assert compare_metric.metrics["mean_abs_error_max"] == 0
    assert compare_metric.metrics["max_abs_error_max"] == 0
    assert compare_metric.metrics["p90_abs_error_max"] == 0
    assert compare_metric.metrics["mean_delta_abs_error_max"] == 0
    assert compare_metric.metrics["max_delta_abs_error_max"] == 0
    assert compare_metric.metrics["p90_delta_abs_error_max"] == 0

    multi_server = build_local_mcp_server(run_id=f"multi_agent_{task.task_id}")
    multi_client = LocalMCPClient(multi_server)
    multi_agent = RuleBasedMultiAgent(client=multi_client, dataset_ref="smoke")
    multi_output = multi_agent.run(task.query)

    multi_metric = compare_agent_to_gold(
        task_id=task.task_id,
        task_type=task.task_type,
        agent_result=multi_output.tool_results,
        gold_result=gold.result,
        agent_trace=multi_output.trace,
        task=task_to_dict(task),
        parsed_task=asdict(multi_output.parsed_task),
        orchestration_steps=[
            asdict(step)
            for step in multi_output.orchestration_steps
        ],
    )

    print("\nMulti-agent role metric result:")
    print(multi_metric.to_dict())

    assert multi_metric.success is True
    assert multi_metric.metrics["role_agent_order_match"] is True
    assert multi_metric.metrics["artifact_flow_valid"] is True
    assert multi_metric.metrics["data_agent_called"] is True
    assert multi_metric.metrics["math_agent_called"] is True
    assert multi_metric.metrics["analysis_agent_called"] is True
    assert multi_metric.metrics["report_agent_called"] is True

    print("\nMetrics smoke test finished successfully.")


if __name__ == "__main__":
    main()
