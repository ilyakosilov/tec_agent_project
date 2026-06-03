"""End-to-end fake-model smoke test for function-handoff multi-agent wiring."""

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


from tec_agents.agents.llm_multi_agent_function_handoff import (  # noqa: E402
    LLMFunctionHandoffMultiAgent,
)
from tec_agents.data.datasets import register_dataset  # noqa: E402
from tec_agents.mcp.client import LocalMCPClient  # noqa: E402
from tec_agents.mcp.server import build_local_mcp_server  # noqa: E402


class FakeFunctionHandoffChatModel:
    """Fake model that emits a complete high-TEC function-handoff flow."""

    def __init__(self) -> None:
        self.orchestrator_calls = 0
        self.role_calls = {
            "data_agent": 0,
            "math_agent": 0,
            "analysis_agent": 0,
            "report_agent": 0,
        }

    def generate(self, messages, **_: object) -> str:
        system = str(messages[0]["content"])
        visible = "\n".join(str(message.get("content") or "") for message in messages)
        if "OrchestratorAgent" in system:
            return self._orchestrator()
        if "You are LLM data_agent." in system:
            return self._data()
        if "You are LLM math_agent." in system:
            return self._math(visible)
        if "You are LLM analysis_agent." in system:
            return self._analysis()
        if "You are LLM report_agent." in system:
            return self._report()
        raise AssertionError("Unknown fake prompt")

    def _orchestrator(self) -> str:
        self.orchestrator_calls += 1
        calls = [
            ("call_data_agent", "Prepare TEC time series for midlat_europe."),
            ("call_math_agent", "Compute high TEC threshold and high intervals."),
            ("call_analysis_agent", "Interpret the computed high TEC intervals."),
            ("call_report_agent", "Write the final answer from artifacts and findings."),
        ]
        if self.orchestrator_calls <= len(calls):
            name, message = calls[self.orchestrator_calls - 1]
            return (
                f'<tool_call>{{"name":"{name}",'
                f'"arguments":{{"message":"{message}"}}}}</tool_call>'
            )
        return '<tool_call>{"name":"call_report_agent","arguments":{"message":"Write final answer."}}</tool_call>'

    def _data(self) -> str:
        self.role_calls["data_agent"] += 1
        if self.role_calls["data_agent"] == 1:
            return (
                '<tool_call>{"name":"tec_get_timeseries","arguments":'
                '{"dataset_ref":"smoke_function_handoff","region_id":"midlat_europe",'
                '"start":"2024-03-01","end":"2024-04-01"}}</tool_call>'
            )
        return (
            '<tool_call>{"name":"return_to_orchestrator","arguments":'
            '{"status":"done","message":"data ready"}}</tool_call>'
        )

    def _math(self, visible: str) -> str:
        self.role_calls["math_agent"] += 1
        series_id = _last_match(r"series_[A-Za-z0-9_]+", visible)
        if self.role_calls["math_agent"] == 1:
            return (
                '<tool_call>{"name":"tec_compute_high_threshold","arguments":'
                f'{{"series_id":"{series_id}","q":0.9}}'
                "}</tool_call>"
            )
        if self.role_calls["math_agent"] == 2:
            threshold_id = _last_match(r"thr_[A-Za-z0-9_]+", visible)
            return (
                '<tool_call>{"name":"tec_detect_high_intervals","arguments":'
                f'{{"series_id":"{series_id}","threshold_id":"{threshold_id}"}}'
                "}</tool_call>"
            )
        return (
            '<tool_call>{"name":"return_to_orchestrator","arguments":'
            '{"status":"done","message":"math artifacts ready"}}</tool_call>'
        )

    def _analysis(self) -> str:
        self.role_calls["analysis_agent"] += 1
        return (
            '<tool_call>{"name":"return_to_orchestrator","arguments":'
            '{"status":"done","message":"findings ready",'
            '"findings":["High TEC interval artifacts are available for midlat_europe."]}}'
            "</tool_call>"
        )

    def _report(self) -> str:
        self.role_calls["report_agent"] += 1
        return (
            '<tool_call>{"name":"return_to_orchestrator","arguments":'
            '{"status":"done","message":"final answer ready",'
            '"final_answer":"High TEC intervals were computed for midlat_europe."}}'
            "</tool_call>"
        )


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_function_handoff.csv"
    try:
        _build_dataset(dataset_path)
        register_dataset(
            dataset_ref="smoke_function_handoff",
            path=dataset_path,
            file_format="csv",
            time_column="time",
        )
        server = build_local_mcp_server(run_id="smoke_function_handoff")
        agent = LLMFunctionHandoffMultiAgent(
            model=FakeFunctionHandoffChatModel(),
            client=LocalMCPClient(server),
            max_orchestration_steps=8,
            max_role_steps=5,
            max_parse_retries=1,
        )
        result = agent.run(
            "Find high TEC intervals for midlat_europe from 2024-03-01 "
            "to 2024-04-01 using q=0.90 threshold."
        )
        assert result.success, result.to_dict()
        assert result.answer
        assert result.actual_tool_sequence == [
            "tec_get_timeseries",
            "tec_compute_high_threshold",
            "tec_detect_high_intervals",
        ]
        assert result.role_agent_order == [
            "data_agent",
            "math_agent",
            "analysis_agent",
            "report_agent",
        ]
        assert result.parse_error_count == 0
        assert result.forbidden_function_call_count == 0
        assert result.repeated_tool_call_count == 0
        assert result.stalled_loop_detected is False
    finally:
        if dataset_path.exists():
            dataset_path.unlink()
    print("Function-handoff fake-run smoke test finished successfully.")


def _build_dataset(path: Path) -> None:
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


def _last_match(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text)
    assert matches, f"No match for {pattern}"
    return matches[-1]


if __name__ == "__main__":
    main()
