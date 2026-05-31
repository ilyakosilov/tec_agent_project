"""End-to-end fake-model smoke test for typed full LLM multi-agent wiring.

This script does not load Qwen and does not require GPU. It uses the real
MCP/tool executor with a deterministic fake chat model that emits typed blocks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.llm_multi_agent_typed import LLMFullTypedMultiAgent
from tec_agents.agents.llm_multi_agent_typed import build_typed_state_packet
from tec_agents.agents.llm_multi_agent_typed import invalid_artifact_handle_error
from tec_agents.agents.llm_multi_agent_typed_protocol import RoleAssignment, RoleScope, TypedToolCall
from tec_agents.data.datasets import register_dataset
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


class FakeTypedChatModel:
    """Typed-block fake model for one high-TEC flow."""

    def __init__(self) -> None:
        self.orchestrator_calls = 0
        self.role_calls: dict[str, int] = {
            "data_agent": 0,
            "math_agent": 0,
            "analysis_agent": 0,
            "report_agent": 0,
        }

    def generate(self, messages, **_: object) -> str:
        system = messages[0]["content"]
        visible_text = "\n".join(str(message.get("content") or "") for message in messages)
        if "OrchestratorAgent" in system:
            return self._orchestrator()
        if "You are LLM DataAgent." in system:
            return self._data_agent()
        if "You are LLM MathAgent." in system:
            return self._math_agent(visible_text)
        if "You are LLM AnalysisAgent." in system:
            return (
                "<role_response>"
                '{"status":"done","message":"findings ready",'
                '"findings":["A high interval artifact is available for midlat_europe."]}'
                "</role_response>"
            )
        if "You are LLM ReportAgent." in system:
            return '<final_answer>{"answer":"High TEC analysis completed for midlat_europe."}</final_answer>'
        raise AssertionError("Unknown fake prompt")

    def _orchestrator(self) -> str:
        self.orchestrator_calls += 1
        if self.orchestrator_calls == 1:
            return (
                "<role_action>"
                '{"action":"call_role","role":"data_agent","assignment":'
                '{"objective":"prepare_data_artifacts","task_summary":"Prepare TEC series handles.",'
                '"scope":{"dataset_ref":"smoke","regions":["midlat_europe"],"start":"2024-03-01","end":"2024-04-01","task_intent":"high_tec"},'
                '"deliverables_to_produce":["series_id"],"expected_output_type":"data_artifacts",'
                '"completion_criteria":"Return done when series_id handles are visible.",'
                '"constraints":["Use only DataAgent tools."]},'
                '"reason":"prepare data"}'
                "</role_action>"
            )
        if self.orchestrator_calls == 2:
            return (
                "<role_action>"
                '{"action":"call_role","role":"math_agent","assignment":'
                '{"objective":"prepare_high_interval_artifacts","task_summary":"Prepare numerical high TEC artifacts.",'
                '"scope":{"dataset_ref":"smoke","regions":["midlat_europe"],"start":"2024-03-01","end":"2024-04-01","task_intent":"high_tec"},'
                '"deliverables_to_produce":["high_intervals"],"expected_output_type":"computed_artifacts",'
                '"completion_criteria":"Return done only when required computed artifact types are visible.",'
                '"constraints":["Use only MathAgent tools."]},'
                '"reason":"prepare computed artifacts"}'
                "</role_action>"
            )
        if self.orchestrator_calls == 3:
            return (
                "<role_action>"
                '{"action":"call_role","role":"analysis_agent","assignment":'
                '{"objective":"interpret_computed_artifacts","task_summary":"Interpret visible high TEC artifacts.",'
                '"scope":{"dataset_ref":"smoke","regions":["midlat_europe"],"start":"2024-03-01","end":"2024-04-01","task_intent":"high_tec"},'
                '"deliverables_to_produce":["findings"],"expected_output_type":"findings",'
                '"completion_criteria":"Return done with non-empty findings.",'
                '"constraints":["Do not call tools."]},'
                '"reason":"interpret artifacts"}'
                "</role_action>"
            )
        if self.orchestrator_calls == 4:
            return (
                "<role_action>"
                '{"action":"call_role","role":"report_agent","assignment":'
                '{"objective":"produce_final_answer","task_summary":"Write final answer from visible artifacts and findings.",'
                '"scope":{"dataset_ref":"smoke","regions":["midlat_europe"],"start":"2024-03-01","end":"2024-04-01","task_intent":"high_tec"},'
                '"deliverables_to_produce":["final_answer"],"expected_output_type":"final_answer",'
                '"completion_criteria":"Return final_answer.",'
                '"constraints":["Do not call tools."]},'
                '"reason":"finalize answer"}'
                "</role_action>"
            )
        return '<role_action>{"action":"finish","role":null,"assignment":null,"reason":"final_answer is present"}</role_action>'

    def _data_agent(self) -> str:
        self.role_calls["data_agent"] += 1
        if self.role_calls["data_agent"] == 1:
            return (
                '<tool_call>{"name":"tec_get_timeseries","arguments":'
                '{"dataset_ref":"smoke","region_id":"midlat_europe","start":"2024-03-01","end":"2024-04-01"}}'
                "</tool_call>"
            )
        return '<role_response>{"status":"done","message":"done"}</role_response>'

    def _math_agent(self, visible_text: str) -> str:
        self.role_calls["math_agent"] += 1
        series_id = _last_match(r"series_[A-Za-z0-9_]+", visible_text)
        if self.role_calls["math_agent"] == 1:
            return (
                '<tool_call>{"name":"tec_compute_high_threshold","arguments":'
                f'{{"series_id":"{series_id}","q":0.9}}'
                "}</tool_call>"
            )
        if self.role_calls["math_agent"] == 2:
            threshold_id = _last_match(r"thr_[A-Za-z0-9_]+", visible_text)
            return (
                '<tool_call>{"name":"tec_detect_high_intervals","arguments":'
                f'{{"series_id":"{series_id}","threshold_id":"{threshold_id}"}}'
                "}</tool_call>"
            )
        return '<role_response>{"status":"done","message":"done"}</role_response>'


def _last_match(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text)
    assert matches, f"No match for {pattern}"
    return matches[-1]


def build_tiny_dataset(path: Path) -> None:
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


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_typed_fake_run.csv"
    try:
        build_tiny_dataset(dataset_path)
        register_dataset(
            dataset_ref="smoke",
            path=dataset_path,
            file_format="csv",
            time_column="time",
        )
        test_fake_high_tec_success()
        test_fake_invented_handle_failure()
        test_fake_multi_region_completion_state()
    finally:
        if dataset_path.exists():
            dataset_path.unlink()
    print("Typed fake-run smoke test finished successfully.")


def test_fake_high_tec_success() -> None:
    server = build_local_mcp_server(run_id="smoke_typed_fake_run")
    agent = LLMFullTypedMultiAgent(
        model=FakeTypedChatModel(),
        client=LocalMCPClient(server),
        max_orchestration_steps=8,
        max_role_steps=5,
        max_parse_retries=1,
    )
    result = agent.run(
        "Find high TEC intervals for midlat_europe from 2024-03-01 to 2024-04-01 using q=0.90 threshold."
    )
    assert result.success, result.to_dict()
    assert result.answer
    assert result.actual_tool_sequence == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert result.premature_role_completion_count == 0
    assert result.empty_findings_done_count == 0
    assert result.repeated_equivalent_role_assignment_count == 0
    assert result.tool_schema_validation_error_count == 0
    assert result.tool_error_count == 0
    assert result.invalid_artifact_handle_count == 0
    assert result.multiple_protocol_blocks_in_single_output_count == 0


def test_fake_invented_handle_failure() -> None:
    context = {
        "parsed_task": {
            "task_type": "compare_regions",
            "dataset_ref": "smoke",
            "region_ids": ["midlat_europe", "highlat_north"],
            "start": "2024-03-01",
            "end": "2024-04-01",
        },
        "data_artifacts": {"series_by_region": {}},
        "math_artifacts": {},
        "analysis_artifacts": {"findings": []},
    }
    tool_call = TypedToolCall(
        name="tec_compute_series_stats",
        arguments={"series_id": "midlat_europe_20240301_20240401"},
    )
    error = invalid_artifact_handle_error(tool_call=tool_call, context=context)
    assert error
    assert "runtime-visible" in error
    for forbidden in ["next tool", "next role", "expected_tool_sequence", "missing_goal_artifacts"]:
        assert forbidden not in error


def test_fake_multi_region_completion_state() -> None:
    assignment = RoleAssignment(
        objective="prepare_data_artifacts",
        task_summary="Prepare data.",
        scope=RoleScope(
            dataset_ref="smoke",
            regions=["equatorial_atlantic", "midlat_europe", "highlat_north"],
            start="2024-03-01",
            end="2024-04-01",
            task_intent="compare_regions",
        ),
        available_input_types=[],
        deliverables_to_produce=["series_id"],
        expected_output_type="data_artifacts",
        required_output_artifact_types=["series_id"],
        completion_criteria="Return done when data handles are visible.",
        constraints=["Use only DataAgent tools."],
    )
    context = {
        "parsed_task": {
            "task_type": "compare_regions",
            "dataset_ref": "smoke",
            "region_ids": ["equatorial_atlantic", "midlat_europe", "highlat_north"],
            "start": "2024-03-01",
            "end": "2024-04-01",
        },
        "data_artifacts": {
            "series_by_region": {
                "equatorial_atlantic": {"series_id": "series_a", "metadata": {"n_points": 744}},
                "midlat_europe": {"series_id": "series_b", "metadata": {"n_points": 744}},
                "highlat_north": {"series_id": "series_c", "metadata": {"n_points": 744}},
            }
        },
        "math_artifacts": {},
        "analysis_artifacts": {"findings": []},
    }
    packet = build_typed_state_packet(
        user_query="Compare TEC statistics.",
        current_role="data_agent",
        current_assignment=assignment,
        context=context,
    )
    assert packet["assignment_progress"]["scope_covered"] is True
    assert len(packet["available_input_artifacts"]["series_id"]) == 3


if __name__ == "__main__":
    main()
