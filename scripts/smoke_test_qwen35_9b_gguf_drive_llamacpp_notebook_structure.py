"""Static structure smoke test for the Google Drive GGUF llama.cpp notebook.

This test does not download the model, mount Drive, build llama.cpp, start a
server, or run GPU inference. It validates notebook structure and guardrails.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "06_qwen_multi_agent_typed_qwen35_9b_gguf_drive_llamacpp_colab.ipynb"
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
    code_sources = [
        _source(cell) for cell in data["cells"] if cell.get("cell_type") == "code"
    ]
    all_text = "\n".join(sources)
    code_text = "\n".join(code_sources)
    lower_code = code_text.lower()

    for idx, source in enumerate(code_sources):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"Code cell {idx} is not syntactically valid: {exc}") from exc

    configs = get_five_task_configs()
    assert {item["preset_id"] for item in configs} == EXPECTED_PRESETS

    required_strings = [
        "Qwen3.5-9B-Q4_K_M.gguf",
        "google_drive_persistent_copy",
        "llama.cpp_cuda_server",
        "RUN_FULL_BATCH = False",
        "SEMANTIC_SMOKE_PASSED",
        "TYPED_XML_SMOKE_PASSED",
        "ORCHESTRATOR_SMOKE_PASSED",
        "PILOT_WORKFLOW_PASSED",
        "/content/drive/MyDrive/tec_agent_models/qwen35_9b_gguf",
        "/content/models",
        "drive.mount(\"/content/drive\")",
        "shutil.copy2(DRIVE_GGUF_PATH, GGUF_PATH)",
        "MODEL_SOURCE_SNAPSHOT_HASH",
        "1379f25c6b505a3fc737bd7818cb09389cf807c1",
        "experiment_5c_qwen35_9b_gguf_q4_k_m_drive_llamacpp",
        "qwen_multi_agent_typed_v3_qwen35_9b_gguf_q4_k_m_drive_llamacpp_batch_colab.json",
        "qwen_multi_agent_typed_v3_qwen35_9b_gguf_q4_k_m_drive_llamacpp_{preset_id}_colab.json",
    ]
    for fragment in required_strings:
        assert fragment in all_text, fragment

    assert "LLAMA_CONTEXT_SIZE = 4096" in code_text
    assert "MAX_INPUT_TOKENS = 4096" in code_text
    assert "MAX_NEW_TOKENS = 512" in code_text
    assert "-DGGML_CUDA=ON" in code_text
    assert "-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}" in code_text
    assert "ccache" in code_text
    assert "libcurl4-openssl-dev" in code_text
    assert "LLAMA_N_GPU_LAYERS = 99" in code_text
    assert "Qwen35GGUFLlamaCppChatModel" in code_text
    assert "model = Qwen35GGUFLlamaCppChatModel(" in code_text
    assert "/v1/chat/completions" in code_text
    assert "requests.post(" in code_text
    assert "result = agent.run(query)" in code_text
    assert "gold = gold_runner.run(eval_task)" in code_text
    assert "metric_result = compare_agent_to_gold(" in code_text

    forbidden_anywhere = [
        "QuantTrio/Qwen3.5-9B-AWQ",
        "lainlives/Qwen3.5-9B-bnb-4bit",
    ]
    for fragment in forbidden_anywhere:
        assert fragment not in all_text

    forbidden_code_fragments = [
        "bitsandbytes",
        "autoawq",
        "gptqmodel",
        "automodelforimagetexttotext",
        "autoprocessor.from_pretrained",
        "hf_hub_download",
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit",
        "quanttrio",
        "lainlives",
    ]
    for fragment in forbidden_code_fragments:
        assert fragment not in lower_code, fragment

    run_id_lines = [line for line in code_text.splitlines() if "run_id" in line]
    assert run_id_lines
    for line in run_id_lines:
        assert "qwen35_9b_gguf_q4_k_m_drive_llamacpp" in line
        assert "awq" not in line.lower()
        assert "bnb" not in line.lower()

    assert "assert SEMANTIC_SMOKE_PASSED" in code_text
    assert "assert TYPED_XML_SMOKE_PASSED" in code_text
    assert "assert ORCHESTRATOR_SMOKE_PASSED" in code_text
    assert "assert PILOT_WORKFLOW_PASSED" in code_text
    assert "if not RUN_FULL_BATCH:" in code_text
    assert "Set RUN_FULL_BATCH = True in CONFIG" in code_text
    assert "tool_call_count" in code_text
    assert "len(pilot_result.actual_tool_sequence) >= 1" in code_text

    constructor_and_run_lines = [
        line.strip()
        for source in code_sources
        for line in source.splitlines()
        if "LLMFullTypedMultiAgent(" in line
        or ".run(query)" in line
        or "agent.run(" in line
        or "model=" in line
        or "client=" in line
    ]
    for line in constructor_and_run_lines:
        assert "expected_tool_sequence" not in line
        assert "expected_role_agent_order" not in line
        assert "gold_result" not in line
        assert "missing_goal_artifacts" not in line
        assert "verdict_checks" not in line

    agent_run_index = code_text.index("result = agent.run(query)")
    gold_index = code_text.index("gold = gold_runner.run(eval_task)")
    metrics_index = code_text.index("metric_result = compare_agent_to_gold(")
    assert agent_run_index < gold_index < metrics_index

    assert "build_five_task_expected_sequence(config)" in code_text
    assert "build_five_task_expected_sequence(task_config)" in code_text
    assert "tec_build_report" not in code_text
    assert "def build_typed_" not in code_text
    assert "def tec_" not in code_text

    print("Qwen3.5-9B GGUF Drive llama.cpp notebook structure smoke test finished successfully.")


def _source(cell: dict) -> str:
    source = cell.get("source") or ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


if __name__ == "__main__":
    main()
