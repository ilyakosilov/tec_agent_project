"""Structure smoke test for the Qwen3-8B BNB NF4 typed Colab notebook.

This script is intentionally static: it does not import torch, download models,
load CUDA, run tools, or execute notebook cells.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "06_qwen_multi_agent_typed_qwen3_8b_bnb_nf4_colab.ipynb"
)


def _source(cell: dict) -> str:
    source = cell.get("source") or ""
    return "".join(source) if isinstance(source, list) else str(source)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]


def main() -> None:
    _assert(NOTEBOOK_PATH.exists(), f"Missing notebook: {NOTEBOOK_PATH}")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = _code_cells(notebook)
    code_text = "\n".join(_source(cell) for cell in code_cells)
    all_text = "\n".join(_source(cell) for cell in notebook.get("cells", []))

    _assert(notebook.get("nbformat") == 4, "Notebook must be nbformat 4.")
    _assert(code_cells, "Notebook has no code cells.")

    for index, cell in enumerate(code_cells):
        _assert(cell.get("execution_count") is None, f"Code cell {index} has execution_count.")
        _assert(cell.get("outputs") in ([], None), f"Code cell {index} has outputs.")
        try:
            ast.parse(_source(cell))
        except SyntaxError as exc:
            raise AssertionError(f"Code cell {index} is not valid Python: {exc}") from exc

    _assert("<<<<<<<" not in all_text and ">>>>>>>" not in all_text, "Conflict marker found.")

    required_active_strings = [
        "Qwen/Qwen3-8B",
        "Qwen3_8BBNBChatModel",
        "BNB_NF4_DYNAMIC_4BIT",
        "qwen3_8b_bnb_nf4_dynamic",
        "bnb_4bit_compute_dtype=torch.float16",
        "enable_thinking=False",
        "MAX_INPUT_TOKENS = 4096",
        "RUN_ALL_TASKS = False",
        "DIRECT_SMOKE_TESTS_PASSED",
        "hightec_midlat_europe",
        "experiment_qwen3_8b_bnb_nf4_dynamic_typed_colab",
    ]
    for needle in required_active_strings:
        _assert(needle in code_text, f"Required active code string missing: {needle}")

    forbidden_active_strings = [
        "Qwen/Qwen3.5-9B",
        "qwen35_9b",
        "QuantTrio",
        "lainlives",
        "Intel/Qwen3.5",
        "awq",
        "gptq",
        "autoround",
        "gguf",
        "vllm",
        "llama.cpp",
        "gptqmodel",
        "autoawq",
        "LocalQwenChatModel",
    ]
    lowered_code = code_text.lower()
    for needle in forbidden_active_strings:
        _assert(
            needle.lower() not in lowered_code,
            f"Forbidden active code string found: {needle}",
        )

    required_imports_or_calls = [
        "LLMFullTypedMultiAgent",
        "GoldRunner",
        "compare_agent_to_gold",
        "get_five_task_configs",
        "build_five_task_expected_sequence",
        "build_local_mcp_server",
    ]
    for needle in required_imports_or_calls:
        _assert(needle in code_text, f"Required project wiring missing: {needle}")

    _assert(
        'assert DIRECT_SMOKE_TESTS_PASSED, "Direct model smoke-tests have not passed."'
        in code_text,
        "Full benchmark is not gated by direct smoke tests.",
    )
    _assert(
        "if not RUN_ALL_TASKS:" in code_text
        and "Full batch is disabled." in code_text,
        "Full benchmark does not default to disabled state.",
    )

    _assert(
        re.search(r"result\s*=\s*agent\.run\(query\)", code_text) is not None,
        "Typed agent run should receive only query in benchmark logic.",
    )
    _assert(
        "expected_role_agent_order" not in code_text,
        "Notebook should not pass expected role order into active code.",
    )
    _assert(
        "missing_goal_artifacts" not in code_text,
        "Notebook should not pass missing goal artifacts into active code.",
    )

    old_output_assignment_patterns = [
        r"BATCH_OUTPUT_PATH\s*=.*qwen_multi_agent_batch_colab\.json",
        r"BATCH_OUTPUT_PATH\s*=.*qwen_multi_agent_typed_batch_colab\.json",
        r"BATCH_OUTPUT_PATH\s*=.*qwen_multi_agent_typed_v3_batch_colab\.json",
    ]
    for pattern in old_output_assignment_patterns:
        _assert(
            re.search(pattern, code_text) is None,
            f"Notebook appears to overwrite an older output path: {pattern}",
        )

    _assert(
        "qwen_multi_agent_typed_qwen3_8b_bnb_nf4_dynamic_batch_colab.json"
        in code_text,
        "New batch output filename is missing.",
    )
    _assert(
        "qwen3_8b_bnb_nf4_dynamic_direct_smoke_tests.json" in code_text,
        "Direct smoke-test output filename is missing.",
    )
    _assert(
        "qwen_multi_agent_typed_qwen3_8b_bnb_nf4_dynamic_pilot_hightec_midlat_europe_colab.json"
        in code_text,
        "Pilot output filename is missing.",
    )

    print("OK: Qwen3-8B BNB NF4 typed notebook structure is valid.")


if __name__ == "__main__":
    main()
