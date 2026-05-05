"""
Smoke test for deterministic multi-agent recovery decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.multi_agent import (
    MathAgent,
    ParsedMultiAgentTask,
    ReportAgent,
    RuleBasedOrchestrator,
)
from tec_agents.agents.protocol import (
    AgentStatus,
    RequestedNextAction,
    agent_missing_artifacts,
)
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


def parsed_report_task() -> ParsedMultiAgentTask:
    """Return a minimal parsed report task."""

    return ParsedMultiAgentTask(
        task_type="report",
        dataset_ref="smoke",
        region_id=None,
        region_ids=["midlat_europe", "highlat_north"],
        regions=["midlat_europe", "highlat_north"],
        start="2024-03-01",
        end="2024-04-01",
        include=["basic_stats", "high_tec", "stable_intervals"],
        metrics=["mean", "median", "min", "max", "std", "p90", "p95"],
    )


def main() -> None:
    server = build_local_mcp_server(run_id="smoke_multi_agent_recovery")
    client = LocalMCPClient(server)

    parsed = parsed_report_task()
    math_agent = MathAgent(client=client)
    report_agent = ReportAgent()
    orchestrator = RuleBasedOrchestrator(dataset_ref="smoke")

    math_response = math_agent.compute_for_task(
        parsed,
        data_artifacts={},
    )
    assert math_response.status == AgentStatus.MISSING_ARTIFACTS
    assert math_response.requested_next_action is not None
    assert math_response.requested_next_action.target_agent == "data_agent"

    report_response = report_agent.format_answer(
        parsed,
        data_artifacts={"series_by_region": {}, "regions": parsed.region_ids},
        math_artifacts={},
        analysis_artifacts={},
    )
    assert report_response.status == AgentStatus.MISSING_ARTIFACTS
    assert report_response.requested_next_action is not None
    assert report_response.requested_next_action.target_agent in {
        "data_agent",
        "math_agent",
        "analysis_agent",
    }

    missing_response = agent_missing_artifacts(
        agent="math_agent",
        missing_artifacts=["data.series_by_region.midlat_europe.series_id"],
        requested_next_action=RequestedNextAction(
            target_agent="data_agent",
            task="load_missing_series",
            reason="MathAgent cannot compute without series_id",
            required_artifacts=["data.series_by_region.midlat_europe.series_id"],
        ),
        message="Missing series_id.",
        attempt=1,
        max_attempts=2,
    )
    decision = orchestrator.decide_next_action(
        missing_response,
        {"attempt": 1, "max_attempts": 2},
    )
    assert decision.decision == "call_requested_agent"
    assert decision.target_agent == "data_agent"

    exhausted_response = agent_missing_artifacts(
        agent="math_agent",
        missing_artifacts=["data.series_by_region.midlat_europe.series_id"],
        requested_next_action=missing_response.requested_next_action,
        message="Missing series_id.",
        attempt=2,
        max_attempts=2,
    )
    exhausted_decision = orchestrator.decide_next_action(
        exhausted_response,
        {"attempt": 2, "max_attempts": 2},
    )
    assert exhausted_decision.decision == "fail"

    print("Multi-agent recovery smoke test finished successfully.")


if __name__ == "__main__":
    main()
