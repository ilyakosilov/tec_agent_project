"""
Parser-only smoke tests for LLMSingleAgent.

These tests use fake models, so they do not import transformers, load Qwen, or
require a GPU.
"""

from __future__ import annotations

import json
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


from tec_agents.agents.llm_single_agent import (
    LLMSingleAgent,
    clean_model_output,
    count_tool_call_blocks,
)
from tec_agents.data.datasets import register_dataset
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


TOOL_RESULT_RE = re.compile(
    r"<tool_result>\s*(.*?)\s*</tool_result>",
    flags=re.DOTALL | re.IGNORECASE,
)


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


def latest_tool_result(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Extract the latest compact tool result from agent messages."""

    for message in reversed(messages):
        content = message.get("content", "")
        match = TOOL_RESULT_RE.search(content)
        if match:
            return json.loads(match.group(1))

    raise AssertionError("No <tool_result> block found in messages.")


def latest_artifact(messages: list[dict[str, str]], key: str) -> Any:
    """Extract one returned artifact from the latest tool observation."""

    payload = latest_tool_result(messages)
    return payload["returned_artifacts"][key]


class FakeHighTecModel:
    """Fake model that emits the expected high-TEC primitive chain."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, max_new_tokens=512, temperature=0.0, do_sample=False):
        self.calls += 1

        if self.calls == 1:
            return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
""".strip()

        if self.calls == 2:
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_compute_high_threshold", "arguments": {{"series_id": "{series_id}", "method": "quantile", "q": 0.9}}}}
</tool_call>
""".strip()

        if self.calls == 3:
            threshold_id = latest_artifact(messages, "threshold_id")
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_detect_high_intervals", "arguments": {{"series_id": "{series_id}", "threshold_id": "{threshold_id}", "min_duration_minutes": 0, "merge_gap_minutes": 60}}}}
</tool_call>
""".strip()

        return """
<final_answer>
High TEC interval detection completed from tool artifacts.
</final_answer>
""".strip()


class FakeInvalidJsonThenValidModel:
    """Fake model that first emits invalid JSON and then repairs itself."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, max_new_tokens=512, temperature=0.0, do_sample=False):
        self.calls += 1

        if self.calls == 1:
            return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01",}}
</tool_call>
""".strip()

        if self.calls == 2:
            return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
""".strip()

        if self.calls == 3:
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_compute_high_threshold", "arguments": {{"series_id": "{series_id}", "method": "quantile", "q": 0.9}}}}
</tool_call>
""".strip()

        if self.calls == 4:
            threshold_id = latest_artifact(messages, "threshold_id")
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_detect_high_intervals", "arguments": {{"series_id": "{series_id}", "threshold_id": "{threshold_id}", "min_duration_minutes": 0, "merge_gap_minutes": 60}}}}
</tool_call>
""".strip()

        return """
<final_answer>
Recovered from invalid JSON and completed high TEC detection.
</final_answer>
""".strip()


class FakeGarbageMultiToolModel:
    """Fake model that emits schema garbage and several tool_call blocks."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, max_new_tokens=512, temperature=0.0, do_sample=False):
        self.calls += 1

        if self.calls == 1:
            return """
assistant schema text that should be ignored
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
user prompt leak
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "highlat_north", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
<tool_call>
{"name": "tec_compute_high_threshold", "arguments": {"series_id": "bad", "method": "quantile", "q": 0.9}}
</tool_call>
""".strip()

        if self.calls == 2:
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_compute_high_threshold", "arguments": {{"series_id": "{series_id}", "method": "quantile", "q": 0.9}}}}
</tool_call>
""".strip()

        if self.calls == 3:
            threshold_id = latest_artifact(messages, "threshold_id")
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_detect_high_intervals", "arguments": {{"series_id": "{series_id}", "threshold_id": "{threshold_id}", "min_duration_minutes": 0, "merge_gap_minutes": 60}}}}
</tool_call>
""".strip()

        return """
<final_answer>
Completed after cleaning the first valid tool call block.
</final_answer>
""".strip()


class FakeRepeatedToolModel:
    """Fake model that keeps repeating the first tool after a valid result."""

    def generate(self, messages, max_new_tokens=512, temperature=0.0, do_sample=False):
        return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
""".strip()


class FakeRepeatThenRecoverModel:
    """Fake model that recovers after a state-aware repeated-call correction."""

    def __init__(self) -> None:
        self.calls = 0
        self.correction_message = ""

    def generate(self, messages, max_new_tokens=512, temperature=0.0, do_sample=False):
        self.calls += 1

        if self.calls == 1:
            return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
""".strip()

        if self.calls == 2:
            return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
""".strip()

        if self.calls == 3:
            self.correction_message = messages[-1]["content"]
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_compute_high_threshold", "arguments": {{"series_id": "{series_id}", "method": "quantile", "q": 0.9}}}}
</tool_call>
""".strip()

        if self.calls == 4:
            threshold_id = latest_artifact(messages, "threshold_id")
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_detect_high_intervals", "arguments": {{"series_id": "{series_id}", "threshold_id": "{threshold_id}", "min_duration_minutes": 0, "merge_gap_minutes": 60}}}}
</tool_call>
""".strip()

        return """
<final_answer>
Recovered after a repeated call correction and completed high TEC detection.
</final_answer>
""".strip()


class FakeWrongDateThenRecoverModel:
    """Fake model that first uses inclusive month-end, then corrects it."""

    def __init__(self) -> None:
        self.calls = 0
        self.date_correction_message = ""

    def generate(self, messages, max_new_tokens=512, temperature=0.0, do_sample=False):
        self.calls += 1

        if self.calls == 1:
            return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-03-31"}}
</tool_call>
""".strip()

        if self.calls == 2:
            self.date_correction_message = messages[-1]["content"]
            return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
""".strip()

        if self.calls == 3:
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_compute_high_threshold", "arguments": {{"series_id": "{series_id}", "method": "quantile", "q": 0.9}}}}
</tool_call>
""".strip()

        if self.calls == 4:
            threshold_id = latest_artifact(messages, "threshold_id")
            series_id = latest_artifact(messages, "series_id")
            return f"""
<tool_call>
{{"name": "tec_detect_high_intervals", "arguments": {{"series_id": "{series_id}", "threshold_id": "{threshold_id}", "min_duration_minutes": 0, "merge_gap_minutes": 60}}}}
</tool_call>
""".strip()

        return """
<final_answer>
Recovered from a date range correction and completed high TEC detection.
</final_answer>
""".strip()


class FakeWrongDateRepeatedModel:
    """Fake model that keeps using inclusive month-end after correction."""

    def generate(self, messages, max_new_tokens=512, temperature=0.0, do_sample=False):
        return """
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-03-31"}}
</tool_call>
""".strip()


def build_agent(
    model,
    run_id: str,
    state_feedback_mode: str = "state_aware",
) -> LLMSingleAgent:
    """Build an LLMSingleAgent with a fresh local MCP-like client."""

    server = build_local_mcp_server(run_id=run_id)
    client = LocalMCPClient(server)
    return LLMSingleAgent(
        model=model,
        client=client,
        state_feedback_mode=state_feedback_mode,
    )


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_llm_single_agent.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke_llm",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    agent = build_agent(FakeHighTecModel(), run_id="smoke_llm_single_agent_chain")
    result = agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    assert result.success is True
    assert result.tool_sequence == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert result.parse_error_count == 0
    assert result.invalid_json_count == 0
    assert result.unknown_format_count == 0
    assert result.repair_attempt_count == 0
    assert result.repeated_tool_call_count == 0
    assert result.stalled_loop_detected is False
    assert result.date_range_mismatch_detected is False
    assert result.date_range_correction_count == 0
    assert result.artifact_usage_failure is False
    assert result.state_feedback_mode == "state_aware"
    assert result.missing_goal_artifacts == []
    assert "series_id" in result.available_artifacts
    assert "threshold_id" in result.available_artifacts
    assert len(result.completed_tool_calls) == 3

    raw_multi = """
schema garbage
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "smoke_llm"}}
</tool_call>
more garbage
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "other"}}
</tool_call>
""".strip()
    cleaned = clean_model_output(raw_multi)
    assert count_tool_call_blocks(raw_multi) == 2
    assert count_tool_call_blocks(cleaned) == 1
    assert cleaned.startswith("<tool_call>")
    assert cleaned.endswith("</tool_call>")

    garbage_agent = build_agent(
        FakeGarbageMultiToolModel(),
        run_id="smoke_llm_single_agent_garbage_multi_tool",
    )
    garbage_result = garbage_agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    assert garbage_result.success is True
    assert garbage_result.tool_sequence == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert garbage_result.multi_tool_call_output_count == 1
    assert count_tool_call_blocks(garbage_result.cleaned_model_outputs[0]) == 1
    assert "schema text" not in garbage_result.cleaned_model_outputs[0]

    repair_agent = build_agent(
        FakeInvalidJsonThenValidModel(),
        run_id="smoke_llm_single_agent_invalid_json",
    )
    repair_result = repair_agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    assert repair_result.success is True
    assert repair_result.tool_sequence == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert repair_result.parse_error_count == 1
    assert repair_result.invalid_json_count == 1
    assert repair_result.repair_attempt_count > 0
    assert repair_result.artifact_usage_failure is False

    repeat_then_recover_model = FakeRepeatThenRecoverModel()
    repeat_then_recover_agent = build_agent(
        repeat_then_recover_model,
        run_id="smoke_llm_single_agent_repeat_then_recover",
    )
    repeat_then_recover_result = repeat_then_recover_agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    assert repeat_then_recover_result.success is True
    assert repeat_then_recover_result.tool_sequence == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert repeat_then_recover_result.repeated_tool_call_count == 1
    assert repeat_then_recover_result.repair_attempt_count >= 1
    assert repeat_then_recover_result.stalled_loop_detected is False
    assert repeat_then_recover_result.artifact_usage_failure is False
    assert repeat_then_recover_result.state_feedback_mode == "state_aware"
    assert "tec_compute_high_threshold" not in repeat_then_recover_model.correction_message
    assert '<tool_call>\n{"name":' not in repeat_then_recover_model.correction_message

    wrong_date_model = FakeWrongDateThenRecoverModel()
    wrong_date_agent = build_agent(
        wrong_date_model,
        run_id="smoke_llm_single_agent_wrong_date_then_recover",
    )
    wrong_date_result = wrong_date_agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    executed_timeseries_calls = [
        call
        for call in wrong_date_result.trace["calls"]
        if call["tool_name"] == "tec_get_timeseries"
    ]

    assert wrong_date_result.success is True
    assert wrong_date_result.tool_sequence == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    assert wrong_date_result.date_range_mismatch_detected is True
    assert wrong_date_result.date_range_correction_count == 1
    assert wrong_date_result.artifact_usage_failure is False
    assert len(executed_timeseries_calls) == 1
    assert executed_timeseries_calls[0]["arguments"]["end"] == "2024-04-01"
    assert "required_analysis_period: [2024-03-01, 2024-04-01)" in (
        wrong_date_model.date_correction_message
    )

    wrong_date_stalled_agent = build_agent(
        FakeWrongDateRepeatedModel(),
        run_id="smoke_llm_single_agent_wrong_date_repeated",
    )
    wrong_date_stalled_result = wrong_date_stalled_agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    assert wrong_date_stalled_result.success is False
    assert wrong_date_stalled_result.date_range_mismatch_detected is True
    assert wrong_date_stalled_result.date_range_correction_count == 2
    assert wrong_date_stalled_result.artifact_usage_failure is True
    assert wrong_date_stalled_result.trace["n_calls"] == 0

    repeated_agent = build_agent(
        FakeRepeatedToolModel(),
        run_id="smoke_llm_single_agent_repeated_tool",
    )
    repeated_result = repeated_agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    repeated_get_timeseries_calls = [
        call
        for call in repeated_result.trace["calls"]
        if call["tool_name"] == "tec_get_timeseries"
    ]

    assert repeated_result.success is False
    assert repeated_result.repeated_tool_call_count >= 2
    assert repeated_result.stalled_loop_detected is True
    assert repeated_result.artifact_usage_failure is True
    assert repeated_result.state_feedback_mode == "state_aware"
    assert len(repeated_get_timeseries_calls) == 1

    print("LLM single-agent parser-only smoke test finished successfully.")


if __name__ == "__main__":
    main()
