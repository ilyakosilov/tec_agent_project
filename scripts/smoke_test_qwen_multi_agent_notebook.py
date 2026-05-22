"""Smoke checks for the Qwen full LLM multi-agent Colab notebook.

This test does not run Qwen. It validates notebook structure and guardrails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_qwen_multi_agent_colab.ipynb"
COMPARE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_five_task_architectures.py"
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


def _source(cell: dict) -> str:
    source = cell.get("source") or ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


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
    assert "from tec_agents.eval.five_task_configs import" in all_text
    assert "get_five_task_configs" in all_text
    assert "ARCHITECTURE_MODE = \"qwen_multi_agent_full_llm\"" in all_text
    assert "## Planned test questions" in all_text
    assert "for index, task_config in enumerate(SELECTED_TASK_CONFIGS" in all_text
    assert "GoldRunner" in all_text
    assert "compare_agent_to_gold" in all_text
    assert "qwen_multi_agent_batch_colab.json" in all_text
    assert "qwen_single_agent_batch_colab.json" in all_text
    assert "multi_agent_rule_based_five_task_batch.json" in all_text
    assert "tec_build_report" not in all_text
    assert "missing_goal_artifacts" not in all_text
    assert "GoldRunner result" not in all_text
    assert "deterministic baseline trace" not in all_text

    configs = get_five_task_configs()
    assert {item["preset_id"] for item in configs} == EXPECTED_PRESETS
    assert COMPARE_SCRIPT_PATH.exists()

    from tec_agents.eval.five_task_configs import build_five_task_expected_sequence

    for config in configs:
        sequence = build_five_task_expected_sequence(config)
        assert "tec_build_report" not in sequence
        assert "tec_compare_regions" not in sequence
        if config["task_type"] == "compare_regions":
            assert "tec_compare_stats" in sequence
            assert "tec_compute_series_stats" in sequence

    run_lines = [
        line.strip()
        for source in sources
        for line in source.splitlines()
        if "agent.run(" in line
    ]
    assert run_lines == ["result = agent.run(query)"]

    constructor_lines = [
        line.strip()
        for source in sources
        for line in source.splitlines()
        if "LLMFullMultiAgent(" in line
        or "model=" in line
        or "client=" in line
        or "max_orchestration_steps=" in line
    ]
    forbidden_prompt_inputs = [
        "expected_tool_sequence",
        "expected_role_agent_order",
        "GoldRunner",
        "missing_goal_artifacts",
    ]
    for line in run_lines + constructor_lines:
        for fragment in forbidden_prompt_inputs:
            assert fragment not in line

    print("Qwen multi-agent notebook smoke test finished successfully.")


if __name__ == "__main__":
    main()
