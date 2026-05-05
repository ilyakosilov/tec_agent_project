"""
Smoke test for AgentResponse protocol on the successful multi-agent path.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.multi_agent import RuleBasedMultiAgent
from tec_agents.data.datasets import register_dataset
from tec_agents.eval.gold_runner import GoldRunner
from tec_agents.eval.metrics import compare_agent_to_gold
from tec_agents.eval.task_set import EvalTask
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


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
    df.loc[120:130, "midlat_europe"] += 20.0
    df.to_csv(path, index=False)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_agent_protocol.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    task = EvalTask(
        task_id="smoke_protocol_high_tec",
        query="Find high TEC intervals for midlat_europe in March 2024 with q=0.9",
        task_type="high_tec",
        dataset_ref="smoke",
        region_id="midlat_europe",
        region_ids=("midlat_europe",),
        start="2024-03-01",
        end="2024-04-01",
        q=0.9,
        expected_tool_sequence=(
            "tec_get_timeseries",
            "tec_compute_high_threshold",
            "tec_detect_high_intervals",
        ),
        expected_worker="role_based_workflow",
    )

    server = build_local_mcp_server(run_id="smoke_agent_response_protocol")
    client = LocalMCPClient(server)
    agent = RuleBasedMultiAgent(client=client, dataset_ref="smoke")
    result = agent.run(task.query)

    steps = [asdict(step) for step in result.orchestration_steps]
    statuses = [step["status"] for step in steps]

    assert statuses == ["ok", "ok", "ok", "ok", "final"]
    assert all("agent_response" in step["details"] for step in steps[1:])
    assert all(step["attempt"] == 1 for step in steps)
    assert all(not step["missing_artifacts"] for step in steps)

    gold = GoldRunner().run(task)
    assert gold.status == "success"
    assert gold.result is not None

    metric = compare_agent_to_gold(
        task_id=task.task_id,
        task_type=task.task_type,
        agent_result=result.tool_results,
        gold_result=gold.result,
        agent_trace=result.trace,
        orchestration_steps=steps,
        parsed_task=asdict(result.parsed_task),
    )

    assert metric.success is True
    assert metric.metrics["agent_response_protocol_used"] is True
    assert metric.metrics["missing_artifact_event_count"] == 0
    assert metric.metrics["tool_error_event_count"] == 0
    assert metric.metrics["partial_event_count"] == 0
    assert metric.metrics["retry_count"] == 0

    print("AgentResponse protocol smoke test finished successfully.")


if __name__ == "__main__":
    main()
