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


from tec_agents.agents.llm_single_agent import LLMSingleAgent
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
            payload = latest_tool_result(messages)
            series_id = payload["result"]["series_id"]
            return f"""
<tool_call>
{{"name": "tec_compute_high_threshold", "arguments": {{"series_id": "{series_id}", "method": "quantile", "q": 0.9}}}}
</tool_call>
""".strip()

        if self.calls == 3:
            payload = latest_tool_result(messages)
            result = payload["result"]
            return f"""
<tool_call>
{{"name": "tec_detect_high_intervals", "arguments": {{"series_id": "{result["series_id"]}", "threshold_id": "{result["threshold_id"]}", "min_duration_minutes": 0, "merge_gap_minutes": 60}}}}
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

        return """
<final_answer>
Recovered from invalid JSON and loaded the series.
</final_answer>
""".strip()


def build_agent(model, run_id: str) -> LLMSingleAgent:
    """Build an LLMSingleAgent with a fresh local MCP-like client."""

    server = build_local_mcp_server(run_id=run_id)
    client = LocalMCPClient(server)
    return LLMSingleAgent(model=model, client=client)


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

    repair_agent = build_agent(
        FakeInvalidJsonThenValidModel(),
        run_id="smoke_llm_single_agent_invalid_json",
    )
    repair_result = repair_agent.run(
        "Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
    )

    assert repair_result.success is True
    assert repair_result.tool_sequence == ["tec_get_timeseries"]
    assert repair_result.parse_error_count == 1
    assert repair_result.invalid_json_count == 1
    assert repair_result.repair_attempt_count > 0

    print("LLM single-agent parser-only smoke test finished successfully.")


if __name__ == "__main__":
    main()
