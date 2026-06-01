"""Structure smoke test for the Qwen3.5-9B GGUF llama.cpp Colab notebook.

This test does not download GGUF weights, build llama.cpp, start a server, or
run any LLM inference. It validates that the notebook is a clean model-ablation
branch over the existing typed v3 architecture.
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
    / "06_qwen_multi_agent_typed_qwen35_9b_gguf_llamacpp_colab.ipynb"
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

    for idx, source in enumerate(code_sources):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"Code cell {idx} is not syntactically valid: {exc}") from exc

    configs = get_five_task_configs()
    assert {item["preset_id"] for item in configs} == EXPECTED_PRESETS

    assert "Experiment 5c" in all_text
    assert "LLMFullTypedMultiAgent" in code_text
    assert "from tec_agents.agents.llm_multi_agent_typed import" in code_text
    assert "from tec_agents.eval.five_task_configs import" in code_text
    assert "get_five_task_configs" in code_text
    assert 'BASE_MODEL_NAME = "Qwen/Qwen3.5-9B"' in code_text
    assert 'QUANTIZED_MODEL_REPO = "lmstudio-community/Qwen3.5-9B-GGUF"' in code_text
    assert 'GGUF_FILENAME = "Qwen3.5-9B-Q4_K_M.gguf"' in code_text
    assert 'QUANTIZATION_FORMAT = "GGUF_Q4_K_M"' in code_text
    assert 'INFERENCE_BACKEND = "llama.cpp_cuda_server"' in code_text
    assert 'MODEL_ABLATION_ID = "qwen35_9b_gguf_q4_k_m_llamacpp"' in code_text
    assert 'EXPERIMENT_ID = "experiment_5c_qwen35_9b_gguf_q4_k_m_llamacpp"' in code_text
    assert "experiment_5c_qwen35_9b_gguf_q4_k_m_llamacpp" in code_text
    assert "qwen_multi_agent_typed_v3_qwen35_9b_gguf_q4_k_m_llamacpp_batch_colab.json" in code_text
    assert "qwen_multi_agent_typed_v3_qwen35_9b_gguf_q4_k_m_llamacpp_{preset_id}_colab.json" in code_text
    assert "qwen35_9b_gguf_q4_k_m_llamacpp_infrastructure_smoke_failure.json" in code_text

    assert "llama.cpp" in all_text
    assert "llama-server" in code_text
    assert "-DGGML_CUDA=ON" in code_text
    assert "--jinja" in code_text
    assert "/v1/chat/completions" in code_text
    assert "Qwen35GGUFLlamaCppChatModel" in code_text
    assert "model = Qwen35GGUFLlamaCppChatModel(" in code_text
    assert "requests.post(" in code_text
    assert "hf_hub_download(" in code_text
    assert "repo_id=QUANTIZED_MODEL_REPO" in code_text
    assert "filename=GGUF_FILENAME" in code_text

    forbidden_anywhere = [
        "QuantTrio/Qwen3.5-9B-AWQ",
        "lainlives/Qwen3.5-9B-bnb-4bit",
    ]
    for fragment in forbidden_anywhere:
        assert fragment not in all_text

    forbidden_code = [
        "bitsandbytes",
        "autoawq",
        "gptqmodel",
        "AutoModel",
        "AutoProcessor",
        "transformers",
        "from llama_cpp",
        "llama-cpp-python",
        "vllm",
        "sglang",
        "ollama",
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit",
        "AWQ_INT4",
        "BNB",
    ]
    lower_code = code_text.lower()
    for fragment in forbidden_code:
        assert fragment.lower() not in lower_code

    run_id_lines = [
        line for line in code_text.splitlines() if "run_id=" in line or "run_id" in line
    ]
    assert run_id_lines
    for line in run_id_lines:
        assert "qwen35_9b_gguf_q4_k_m_llamacpp" in line
        assert "awq" not in line.lower()
        assert "bnb" not in line.lower()

    for required_gate in [
        "semantic_smoke",
        "typed_xml_smoke",
        "orchestrator_smoke",
        "pilot_workflow",
        "SEMANTIC_SMOKE_PASSED",
        "TYPED_XML_SMOKE_PASSED",
        "ORCHESTRATOR_SMOKE_PASSED",
        "PILOT_WORKFLOW_PASSED",
    ]:
        assert required_gate in code_text

    assert "Infrastructure validation failed" in code_text
    assert "This run must not be included in model-quality or architecture comparison" in code_text

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

    assert "result = agent.run(query)" in code_text
    agent_run_index = code_text.index("result = agent.run(query)")
    gold_index = code_text.index("gold = gold_runner.run(eval_task)")
    metrics_index = code_text.index("metric_result = compare_agent_to_gold(")
    assert agent_run_index < gold_index < metrics_index

    assert "expected_tool_sequence" in code_text
    assert "build_five_task_expected_sequence(config)" in code_text
    assert "build_five_task_expected_sequence(task_config)" in code_text
    assert "tec_build_report" not in code_text
    assert "def build_typed_" not in code_text
    assert "def tec_" not in code_text

    print("Qwen3.5-9B GGUF llama.cpp notebook structure smoke test finished successfully.")


def _source(cell: dict) -> str:
    source = cell.get("source") or ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


if __name__ == "__main__":
    main()
