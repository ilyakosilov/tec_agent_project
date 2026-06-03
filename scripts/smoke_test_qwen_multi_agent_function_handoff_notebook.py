"""Smoke checks for the Qwen3.5-4B function-handoff multi-agent notebook.

This test validates notebook structure and guardrails. It does not run Qwen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "07_qwen_multi_agent_function_handoff_qwen35_4b_colab.ipynb"
)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tec_agents.eval.five_task_configs import get_five_task_configs


EXPECTED_PRESETS = {
    "hightec_midlat_europe",
    "stable_midlat_europe",
    "compare_midlat_europe_highlat_north",
    "compare_three_regions",
    "report_midlat_europe",
}


def main() -> None:
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    for marker in ["<<<<<<<", "=======", ">>>>>>>"]:
        assert marker not in text

    data = json.loads(text)
    assert data["nbformat"] >= 4
    assert all(not cell.get("outputs") for cell in data["cells"])
    assert all(
        cell.get("execution_count") is None
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    )

    sources = [_source(cell) for cell in data["cells"]]
    all_text = "\n".join(sources)
    assert "LLMFunctionHandoffMultiAgent" in all_text
    assert "Qwen/Qwen3.5-4B" in all_text
    assert "LocalQwenChatModel" in all_text
    assert "from tec_agents.eval.five_task_configs import" in all_text
    assert "GoldRunner" in all_text
    assert "compare_agent_to_gold" in all_text
    assert 'ARCHITECTURE_MODE = "qwen_multi_agent_function_handoff_full_llm"' in all_text
    assert 'FUNCTION_HANDOFF_PROTOCOL_VERSION = "function_handoff_v1"' in all_text
    assert 'PROMPT_REVISION = "function_handoff_grounded_state_v2"' in all_text
    assert 'EXPERIMENT_OUTPUT_DIR = OUTPUT_ROOT / "experiment_6_function_handoff_v2"' in all_text
    assert "qwen_multi_agent_function_handoff_v2_batch_colab.json" in all_text
    assert "qwen_multi_agent_function_handoff_v2_{preset_id}_colab.json" in all_text
    assert "outputs\" / \"metrics\" / \"real_runs\" / \"multi_agent\"" in all_text
    assert "tec_build_report" not in all_text
    assert "tec_compare_regions" not in all_text
    assert "deterministic baseline trace" not in all_text
    assert 'EXPERIMENT_OUTPUT_DIR = OUTPUT_ROOT / "experiment_5_function_handoff"' not in all_text

    first_code = next(
        _source(cell) for cell in data["cells"] if cell.get("cell_type") == "code"
    )
    assert "Pillow==12.2.0" in first_code
    assert "Runtime \u2192 Restart session" in first_code

    configs = get_five_task_configs()
    assert {item["preset_id"] for item in configs} == EXPECTED_PRESETS

    run_lines = [
        line.strip()
        for source in sources
        for line in source.splitlines()
        if "agent.run(" in line
    ]
    assert run_lines == ["result = agent.run(query)"]

    constructor_and_run_lines = [
        line.strip()
        for source in sources
        for line in source.splitlines()
        if "LLMFunctionHandoffMultiAgent(" in line
        or "agent.run(" in line
        or "model=" in line
        or "client=" in line
    ]
    forbidden_prompt_inputs = [
        "expected_tool_sequence",
        "expected_role_agent_order",
        "gold_result",
        "missing_goal_artifacts",
        "verdict_checks",
        "remaining",
        "deterministic",
    ]
    for line in constructor_and_run_lines:
        for fragment in forbidden_prompt_inputs:
            assert fragment not in line

    agent_run_index = all_text.index("result = agent.run(query)")
    gold_index = all_text.index("gold = gold_runner.run(eval_task)")
    metrics_index = all_text.index("metric_result = compare_agent_to_gold(")
    assert agent_run_index < gold_index < metrics_index

    forbidden_output_assignments = [
        'BATCH_OUTPUT_PATH = OUTPUT_DIR / "qwen_multi_agent_batch_colab.json"',
        'BATCH_OUTPUT_PATH = OUTPUT_DIR / "qwen_multi_agent_typed_batch_colab.json"',
        'BATCH_OUTPUT_PATH = OUTPUT_DIR / "qwen_multi_agent_typed_v2_batch_colab.json"',
        'BATCH_OUTPUT_PATH = EXPERIMENT_OUTPUT_DIR / "qwen_multi_agent_typed_v3_batch_colab.json"',
        'PER_TASK_OUTPUT_TEMPLATE = "qwen_multi_agent_typed_{preset_id}_colab.json"',
        'PER_TASK_OUTPUT_TEMPLATE = "qwen_multi_agent_typed_v2_{preset_id}_colab.json"',
        'PER_TASK_OUTPUT_TEMPLATE = "qwen_multi_agent_typed_v3_{preset_id}_colab.json"',
    ]
    for assignment in forbidden_output_assignments:
        assert assignment not in all_text

    for column in [
        "invalid_function_name_count",
        "forbidden_function_call_count",
        "multiple_function_blocks_in_single_output_count",
        "invalid_artifact_handle_count",
        "repeated_role_message_count",
        "successful_final_tool_without_return_count",
        "role_agent_order",
        "actual_tool_sequence",
        "final_answer_present",
    ]:
        assert column in all_text

    print("Qwen function-handoff multi-agent notebook smoke test finished successfully.")


def _source(cell: dict) -> str:
    source = cell.get("source") or ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


if __name__ == "__main__":
    main()
