"""
Smoke test that the legacy aggregate report tool is no longer exposed.

Report tasks must continue to work, but only through primitive tool chains.
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
from tec_agents.eval.gold_runner import GoldRunner
from tec_agents.eval.task_set import EvalTask, primitive_report_tool_sequence
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server
from tec_agents.tools.registry import build_tool_registry


LEGACY_REPORT_TOOL = "tec_build_report"


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
    df.loc[200:205, "highlat_north"] += 8.0

    df.to_csv(path, index=False)


def tool_sequence(trace: dict) -> list[str]:
    """Return tool names from a trace."""

    return [call["tool_name"] for call in trace["calls"]]


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_no_legacy_report.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    registry = build_tool_registry()
    assert LEGACY_REPORT_TOOL not in registry.names()

    task = EvalTask(
        task_id="smoke_no_legacy_report",
        query=(
            "Build a TEC report for midlat_europe and highlat_north "
            "in March 2024"
        ),
        task_type="report",
        dataset_ref="smoke",
        region_id=None,
        region_ids=("midlat_europe", "highlat_north"),
        start="2024-03-01",
        end="2024-04-01",
        params={
            "include": [
                "basic_stats",
                "high_tec",
                "stable_intervals",
            ]
        },
        expected_tool_sequence=primitive_report_tool_sequence(2),
        expected_worker="role_based_workflow",
        description="Report task must use primitive tools only.",
    )

    single_server = build_local_mcp_server(run_id="smoke_no_legacy_single")
    single_client = LocalMCPClient(single_server)
    assert LEGACY_REPORT_TOOL not in single_client.list_tool_names()
    single_agent = RuleBasedSingleAgent(single_client, dataset_ref="smoke")
    single_result = single_agent.run(task.query)
    single_sequence = tool_sequence(single_result.trace)
    assert LEGACY_REPORT_TOOL not in single_sequence
    assert single_sequence == list(task.expected_tool_sequence)

    multi_server = build_local_mcp_server(run_id="smoke_no_legacy_multi")
    multi_client = LocalMCPClient(multi_server)
    assert LEGACY_REPORT_TOOL not in multi_client.list_tool_names()
    multi_agent = RuleBasedMultiAgent(multi_client, dataset_ref="smoke")
    multi_result = multi_agent.run(task.query)
    multi_sequence = tool_sequence(multi_result.trace)
    assert LEGACY_REPORT_TOOL not in multi_sequence
    assert multi_sequence == list(task.expected_tool_sequence)

    gold = GoldRunner().run(task)
    assert gold.status == "success"
    gold_sequence = tool_sequence(gold.trace)
    assert LEGACY_REPORT_TOOL not in gold_sequence
    assert gold_sequence == list(task.expected_tool_sequence)

    print("No-legacy report tool smoke test finished successfully.")


if __name__ == "__main__":
    main()
