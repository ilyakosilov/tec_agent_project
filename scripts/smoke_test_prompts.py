"""
Smoke test for future LLM system prompt definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.llm.prompts import (
    BASE_DOMAIN_CONTEXT,
    build_all_multi_agent_prompts,
    build_analysis_agent_system_prompt,
    build_data_agent_system_prompt,
    build_math_agent_system_prompt,
    build_orchestrator_system_prompt,
    build_report_agent_system_prompt,
    build_single_agent_system_prompt,
)


def assert_nonempty(name: str, prompt: str) -> None:
    """Assert that a prompt contains useful text."""

    assert isinstance(prompt, str), name
    assert prompt.strip(), name


def assert_contains_all(name: str, prompt: str, substrings: list[str]) -> None:
    """Assert that all substrings are present, case-insensitively."""

    lower_prompt = prompt.lower()
    missing = [text for text in substrings if text.lower() not in lower_prompt]
    assert not missing, f"{name} is missing: {missing}"


def assert_legacy_report_tool_only_marked_removed(prompt: str) -> None:
    """Ensure tec_build_report is not described as an allowed tool."""

    marker = "tec_build_report was removed and must not be used"
    text = prompt.lower()

    if "tec_build_report" not in text:
        return

    remaining = text.replace(marker, "")
    assert "tec_build_report" not in remaining


def main() -> None:
    prompts = {
        "single_agent": build_single_agent_system_prompt(),
        "orchestrator": build_orchestrator_system_prompt(),
        "data_agent": build_data_agent_system_prompt(),
        "math_agent": build_math_agent_system_prompt(),
        "analysis_agent": build_analysis_agent_system_prompt(),
        "report_agent": build_report_agent_system_prompt(),
    }

    multi_prompts = build_all_multi_agent_prompts()
    assert set(multi_prompts) == {
        "orchestrator",
        "data_agent",
        "math_agent",
        "analysis_agent",
        "report_agent",
    }

    for name, prompt in prompts.items():
        assert_nonempty(name, prompt)
        assert BASE_DOMAIN_CONTEXT in prompt, name
        assert_legacy_report_tool_only_marked_removed(prompt)

    for name, prompt in multi_prompts.items():
        assert prompts[name] == prompt

    assert_contains_all(
        "report_agent",
        prompts["report_agent"],
        [
            "Allowed tools: none",
            "do not fabricate",
            "do not invent",
            "missing_artifacts",
        ],
    )

    assert_contains_all(
        "orchestrator",
        prompts["orchestrator"],
        [
            "data_agent",
            "math_agent",
            "analysis_agent",
            "report_agent",
            "retry",
            "missing_artifacts",
        ],
    )

    assert_contains_all(
        "single_agent",
        prompts["single_agent"],
        [
            "Tool protocol context",
            "Preferred primitive chains",
            "tec_get_timeseries",
            "tec_compute_high_threshold",
            "tec_detect_stable_intervals",
            "tec_compare_stats",
        ],
    )

    print("Prompt smoke test finished successfully.")


if __name__ == "__main__":
    main()
