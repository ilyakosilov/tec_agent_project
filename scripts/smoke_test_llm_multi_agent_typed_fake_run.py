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
        if "DataAgent" in system:
            return self._data_agent()
        if "MathAgent" in system:
            return self._math_agent(visible_text)
        if "AnalysisAgent" in system:
            return (
                "<role_response>"
                '{"status":"done","message":"findings ready",'
                '"findings":["A high interval artifact is available for midlat_europe."]}'
                "</role_response>"
            )
        if "ReportAgent" in system:
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
                '"available_input_types":[],"expected_output_type":"data_artifacts",'
                '"required_output_artifact_types":["series_id"],'
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
                '"available_input_types":["series_id"],"expected_output_type":"computed_artifacts",'
                '"required_output_artifact_types":["threshold_id","high_intervals"],'
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
                '"available_input_types":["high_intervals"],"expected_output_type":"findings",'
                '"required_output_artifact_types":["findings"],'
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
                '"available_input_types":["high_intervals","findings"],"expected_output_type":"final_answer",'
                '"required_output_artifact_types":["final_answer"],'
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
    build_tiny_dataset(dataset_path)
    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )
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
    assert hasattr(result, "tool_error_count")
    print("Typed fake-run smoke test finished successfully.")


if __name__ == "__main__":
    main()
