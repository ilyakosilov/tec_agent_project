"""Parser-only smoke tests for the full LLM multi-agent runner.

This test uses fake models and a tiny local dataset. It does not load Qwen.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.llm_multi_agent import (
    LLMAnalysisAgent,
    LLMDataAgent,
    LLMFullMultiAgent,
    parse_role_action,
    validate_role_action,
)
from tec_agents.data.datasets import register_dataset
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
            "equatorial_atlantic": 35.0 + 10.0 * np.sin(hours / 5.0),
        }
    )
    df.loc[120:130, "midlat_europe"] += 20.0
    df.to_csv(path, index=False)


class FullNetworkFakeModel:
    """Fake model that drives one high_tec full multi-agent run."""

    def __init__(self) -> None:
        self.orchestrator_calls = 0
        self.seen_messages: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.seen_messages.append(messages)
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if "LLM OrchestratorAgent" in system:
            self.orchestrator_calls += 1
            actions = [
                '<role_action>{"action":"call_role","role":"data_agent","message":"retrieve data"}</role_action>',
                '<role_action>{"action":"call_role","role":"math_agent","message":"compute numerical artifacts"}</role_action>',
                '<role_action>{"action":"call_role","role":"analysis_agent","message":"summarize artifacts"}</role_action>',
                '<role_action>{"action":"call_role","role":"report_agent","message":"write final answer"}</role_action>',
                '<role_action>{"action":"finish","reason":"report final answer exists"}</role_action>',
            ]
            return actions[self.orchestrator_calls - 1]

        if "LLM DataAgent" in system:
            if "<tool_result>" not in user:
                return (
                    "<tool_call>"
                    '{"name":"tec_get_timeseries","arguments":{"dataset_ref":"default","region_id":"midlat_europe","start":"2024-03-01","end":"2024-04-01"}}'
                    "</tool_call>"
                )
            return '<role_response>{"status":"ok","message":"data ready","artifacts":{},"findings":[]}</role_response>'

        if "LLM MathAgent" in system:
            if "high_intervals_by_region" in user:
                return '<role_response>{"status":"ok","message":"math ready","artifacts":{},"findings":[]}</role_response>'
            if "high_threshold_by_region" in user:
                series_id = _first_match(user, r"series_[a-f0-9]{10}")
                threshold_id = _first_match(user, r"thr_[a-f0-9]{10}")
                return (
                    "<tool_call>"
                    f'{{"name":"tec_detect_high_intervals","arguments":{{"series_id":"{series_id}","threshold_id":"{threshold_id}","min_duration_minutes":0,"merge_gap_minutes":60}}}}'
                    "</tool_call>"
                )
            series_id = _first_match(user, r"series_[a-f0-9]{10}")
            return (
                "<tool_call>"
                f'{{"name":"tec_compute_high_threshold","arguments":{{"series_id":"{series_id}","method":"quantile","q":0.9}}}}'
                "</tool_call>"
            )

        if "LLM AnalysisAgent" in system:
            return (
                "<role_response>"
                '{"status":"ok","message":"findings ready","artifacts":{},"findings":[{"type":"high_tec_summary"}]}'
                "</role_response>"
            )

        if "LLM ReportAgent" in system:
            return "<final_answer>High TEC report grounded in computed artifacts.</final_answer>"

        raise AssertionError("Unknown prompt")


class DataTwoRegionsFakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.calls += 1
        if self.calls == 1:
            region = "midlat_europe"
        elif self.calls == 2:
            region = "highlat_north"
        else:
            return '<role_response>{"status":"ok","message":"done","artifacts":{},"findings":[]}</role_response>'
        return (
            "<tool_call>"
            f'{{"name":"tec_get_timeseries","arguments":{{"dataset_ref":"default","region_id":"{region}","start":"2024-03-01","end":"2024-04-01"}}}}'
            "</tool_call>"
        )


class DataDuplicateFakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.calls += 1
        if self.calls <= 2:
            return (
                "<tool_call>"
                '{"name":"tec_get_timeseries","arguments":{"dataset_ref":"default","region_id":"midlat_europe","start":"2024-03-01","end":"2024-04-01"}}'
                "</tool_call>"
            )
        return '<role_response>{"status":"ok","message":"done","artifacts":{},"findings":[]}</role_response>'


class ForbiddenToolFakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.calls += 1
        if self.calls == 1:
            return (
                "<tool_call>"
                '{"name":"tec_get_timeseries","arguments":{"dataset_ref":"default","region_id":"midlat_europe","start":"2024-03-01","end":"2024-04-01"}}'
                "</tool_call>"
            )
        return '<role_response>{"status":"ok","message":"findings","artifacts":{},"findings":[]}</role_response>'


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    assert match, f"Pattern not found: {pattern}"
    return match.group(0)


def _new_client(run_id: str) -> LocalMCPClient:
    return LocalMCPClient(build_local_mcp_server(run_id=run_id))


def test_full_network_success() -> None:
    model = FullNetworkFakeModel()
    client = _new_client("smoke_llm_multi_agent_full")
    runner = LLMFullMultiAgent(
        model=model,
        client=client,
        max_orchestration_steps=8,
        max_role_steps=5,
        max_tool_calls_per_role=5,
    )
    result = runner.run(
        "Find high TEC intervals for midlat_europe from 2024-03-01 to 2024-04-01 using q=0.90 threshold."
    )

    assert result.success is True
    assert result.role_agent_order == [
        "data_agent",
        "math_agent",
        "analysis_agent",
        "report_agent",
    ]
    assert result.actual_tool_sequence == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert result.parse_error_count == 0
    assert result.forbidden_tool_call_count == 0
    assert "High TEC report" in result.answer

    all_prompt_text = "\n".join(
        message["content"]
        for prompt_messages in model.seen_messages
        for message in prompt_messages
    )
    forbidden_prompt_fragments = [
        "expected_tool_sequence",
        "expected_role_agent_order",
        "GoldRunner",
        "missing_goal_artifacts",
        "remaining goals",
        "deterministic trace",
    ]
    for fragment in forbidden_prompt_fragments:
        assert fragment not in all_prompt_text


def test_invalid_role_action_rejected() -> None:
    action, _, _ = parse_role_action(
        '<role_action>{"action":"call_role","role":"unknown_agent","message":"x"}</role_action>'
    )
    assert action is not None
    assert validate_role_action(action, context={}) == "Unknown role 'unknown_agent'."


def test_same_tool_different_args_allowed() -> None:
    model = DataTwoRegionsFakeModel()
    client = _new_client("smoke_llm_multi_agent_different_args")
    agent = LLMDataAgent(model, client, max_role_steps=4, max_tool_calls=4)
    context = {"available_artifacts": {}, "completed_tool_calls": []}
    output = agent.run(
        user_query="Compare TEC statistics for midlat_europe and highlat_north.",
        role_message="retrieve data",
        context=context,
    )
    assert output.status == "ok"
    assert output.repeated_tool_call_count == 0
    assert output.tool_sequence == ["tec_get_timeseries", "tec_get_timeseries"]
    assert len(client.get_trace()["calls"]) == 2


def test_identical_successful_call_rejected() -> None:
    model = DataDuplicateFakeModel()
    client = _new_client("smoke_llm_multi_agent_duplicate")
    agent = LLMDataAgent(model, client, max_role_steps=4, max_tool_calls=4)
    context = {"available_artifacts": {}, "completed_tool_calls": []}
    output = agent.run(
        user_query="Find high TEC intervals for midlat_europe.",
        role_message="retrieve data",
        context=context,
    )
    assert output.status == "ok"
    assert output.repeated_tool_call_count == 1
    assert output.tool_sequence == ["tec_get_timeseries"]
    assert len(client.get_trace()["calls"]) == 1


def test_forbidden_tool_by_role_rejected() -> None:
    model = ForbiddenToolFakeModel()
    client = _new_client("smoke_llm_multi_agent_forbidden")
    agent = LLMAnalysisAgent(model, client, max_role_steps=3, max_tool_calls=2)
    context = {"available_artifacts": {}, "completed_tool_calls": []}
    output = agent.run(
        user_query="Find high TEC intervals for midlat_europe.",
        role_message="analyze",
        context=context,
    )
    assert output.status == "ok"
    assert output.forbidden_tool_call_count == 1
    assert client.get_trace()["calls"] == []


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_llm_multi_agent.csv"
    build_tiny_dataset(dataset_path)
    register_dataset(
        dataset_ref="default",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    test_full_network_success()
    test_invalid_role_action_rejected()
    test_same_tool_different_args_allowed()
    test_identical_successful_call_rejected()
    test_forbidden_tool_by_role_rejected()

    print("LLM multi-agent parser-only smoke test finished successfully.")


if __name__ == "__main__":
    main()
