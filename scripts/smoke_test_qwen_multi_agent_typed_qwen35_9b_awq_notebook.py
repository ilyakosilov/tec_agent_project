"""Smoke checks for the Qwen3.5-9B-AWQ typed model-ablation notebook.

This validates notebook structure and experiment guardrails. It does not load
the model, run Qwen, or require a GPU.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "05_qwen_multi_agent_typed_qwen35_9b_awq_colab.ipynb"
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

    assert "LLMFullTypedMultiAgent" in all_text
    assert "from tec_agents.agents.llm_multi_agent_typed import" in all_text
    assert "from tec_agents.eval.five_task_configs import" in all_text
    assert "get_five_task_configs" in all_text
    assert 'MODEL_NAME = "QuantTrio/Qwen3.5-9B-AWQ"' in all_text
    assert 'BASE_MODEL_NAME = "Qwen/Qwen3.5-9B"' in all_text
    assert 'QUANTIZATION_FORMAT = "AWQ_INT4_PREQUANTIZED"' in all_text
    assert 'MODEL_ABLATION_ID = "qwen35_9b_awq"' in all_text
    assert 'PROMPT_REVISION = "grounded_inputs_deliverables_single_block_v3"' in all_text
    assert 'ARCHITECTURE_MODE = "qwen_multi_agent_typed_full_llm"' in all_text
    assert (
        'EXPERIMENT_ID = "qwen_multi_agent_typed_v3_qwen35_9b_awq_batch_colab"'
        in all_text
    )
    assert (
        'EXPERIMENT_OUTPUT_DIR = OUTPUT_ROOT / "experiment_5_qwen35_9b_awq"'
        in all_text
    )
    assert "qwen_multi_agent_typed_v3_qwen35_9b_awq_batch_colab.json" in all_text
    assert (
        "qwen_multi_agent_typed_v3_qwen35_9b_awq_{preset_id}_colab.json"
        in all_text
    )
    assert "Model load and GPU memory verification" in all_text
    assert "Single-generation smoke test" in all_text
    assert "MIN_FREE_VRAM_GB_AFTER_LOAD" in all_text
    assert "AutoProcessor" in all_text
    assert "AutoModelForImageTextToText" in all_text
    assert "AutoModelForImageTextToText.from_pretrained(" in all_text

    forbidden_model_loading = [
        "BitsAndBytesConfig",
        "load_in_4bit=True",
        "bnb_4bit",
        "from_pretrained(BASE_MODEL_NAME",
        'from_pretrained("Qwen/Qwen3.5-9B"',
        "model_name=BASE_MODEL_NAME",
    ]
    for fragment in forbidden_model_loading:
        assert fragment not in all_text

    forbidden_output_assignments = [
        'EXPERIMENT_OUTPUT_DIR = OUTPUT_ROOT / "experiment_4"',
        'BATCH_OUTPUT_PATH = EXPERIMENT_OUTPUT_DIR / "qwen_multi_agent_typed_v3_batch_colab.json"',
        'PER_TASK_OUTPUT_TEMPLATE = "qwen_multi_agent_typed_v3_{preset_id}_colab.json"',
    ]
    for assignment in forbidden_output_assignments:
        assert assignment not in all_text

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
        if "LLMFullTypedMultiAgent(" in line
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
    ]
    for line in constructor_and_run_lines:
        for fragment in forbidden_prompt_inputs:
            assert fragment not in line

    agent_run_index = all_text.index("result = agent.run(query)")
    gold_index = all_text.index("gold = gold_runner.run(eval_task)")
    metrics_index = all_text.index("metric_result = compare_agent_to_gold(")
    assert agent_run_index < gold_index < metrics_index

    expected_sequence_lines = [
        line.strip()
        for source in sources
        for line in source.splitlines()
        if "expected_sequence" in line or "expected_tool_sequence" in line
    ]
    assert expected_sequence_lines
    for line in constructor_and_run_lines:
        assert "expected_sequence" not in line
        assert "expected_tool_sequence" not in line

    assert "typed_v3_qwen35_4b" in all_text
    assert "typed_v3_qwen35_9b_awq" in all_text
    assert 'OUTPUT_ROOT = PROJECT_DIR / "outputs" / "metrics"' in all_text
    assert "experiment_5_qwen35_9b_awq" in all_text
    assert "new prompt" not in all_text.lower()
    assert not re.search(r"def\\s+build_.*prompt", all_text)
    assert "tec_build_report" not in all_text
    assert "missing_goal_artifacts" not in all_text
    assert "deterministic baseline trace" not in all_text

    print("Qwen3.5-9B-AWQ typed model-ablation notebook smoke test finished successfully.")


def _source(cell: dict) -> str:
    source = cell.get("source") or ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


if __name__ == "__main__":
    main()
