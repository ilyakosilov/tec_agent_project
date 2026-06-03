"""Parser/protocol smoke tests for function-handoff multi-agent experiment.

No Qwen, GPU, dataset, or tool execution is used here.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.llm_multi_agent_function_handoff import (
    build_function_handoff_state_packet,
)
from tec_agents.agents.llm_multi_agent_function_handoff_prompts import (
    build_function_handoff_orchestrator_prompt,
    build_function_handoff_role_prompt,
    build_function_handoff_state_message,
)
from tec_agents.agents.llm_multi_agent_function_handoff_protocol import (
    FORBIDDEN_PROMPT_KEYS,
    RETURN_FUNCTION,
    parse_function_handoff_output,
    role_for_handoff_function,
    validate_function_call_for_role,
    validate_role_return,
)


def main() -> None:
    test_orchestrator_handoffs()
    test_worker_allowlists()
    test_return_to_orchestrator()
    test_prompt_and_state_no_forbidden_hints()
    print("Function-handoff parser-only smoke test finished successfully.")


def test_orchestrator_handoffs() -> None:
    for name, role in {
        "call_data_agent": "data_agent",
        "call_math_agent": "math_agent",
        "call_analysis_agent": "analysis_agent",
        "call_report_agent": "report_agent",
    }.items():
        parsed = parse_function_handoff_output(
            f'<tool_call>{{"name":"{name}","arguments":{{"message":"work"}}}}</tool_call>'
        )
        assert parsed.ok, parsed.to_dict()
        assert parsed.call is not None
        assert role_for_handoff_function(parsed.call.name) == role
        assert validate_function_call_for_role("orchestrator", parsed.call.name) is None
    assert validate_function_call_for_role("orchestrator", "tec_get_timeseries")
    assert validate_function_call_for_role("orchestrator", "return_to_orchestrator")


def test_worker_allowlists() -> None:
    assert validate_function_call_for_role("data_agent", "tec_get_timeseries") is None
    assert validate_function_call_for_role("data_agent", RETURN_FUNCTION) is None
    assert validate_function_call_for_role("data_agent", "call_math_agent")
    assert validate_function_call_for_role("math_agent", "tec_compute_high_threshold") is None
    assert validate_function_call_for_role("math_agent", "tec_detect_high_intervals") is None
    assert validate_function_call_for_role("math_agent", "data_agent")
    assert validate_function_call_for_role("analysis_agent", "tec_compare_stats")
    assert validate_function_call_for_role("report_agent", "tec_get_timeseries")


def test_return_to_orchestrator() -> None:
    assert (
        validate_role_return(
            {"status": "done", "message": "done", "final_answer": "answer"},
            role="report_agent",
        )
        is None
    )
    assert validate_role_return(
        {"status": "done", "message": "done", "final_answer": "answer"},
        role="analysis_agent",
    )
    parsed = parse_function_handoff_output(
        '<tool_call>{"name":"return_to_orchestrator","arguments":{"status":"done","message":"done"}}</tool_call>'
    )
    assert parsed.ok, parsed.to_dict()
    assert parsed.call is not None
    assert parsed.call.name == RETURN_FUNCTION


def test_prompt_and_state_no_forbidden_hints() -> None:
    context = {
        "parsed_task": {
            "task_type": "high_tec",
            "dataset_ref": "default",
            "region_ids": ["midlat_europe"],
            "start": "2024-03-01",
            "end": "2024-04-01",
            "q": 0.9,
        },
        "data_artifacts": {},
        "math_artifacts": {},
        "analysis_artifacts": {"findings": []},
        # This key must be stripped from prompt-visible state.
        "missing_goal_artifacts": ["forbidden"],
    }
    packet = build_function_handoff_state_packet(
        user_query="Find high TEC intervals.",
        current_role="data_agent",
        current_message="prepare data",
        context=context,
    )
    text = "\n".join(
        [
            build_function_handoff_orchestrator_prompt(),
            build_function_handoff_role_prompt("data_agent"),
            build_function_handoff_role_prompt("math_agent"),
            build_function_handoff_role_prompt("analysis_agent"),
            build_function_handoff_role_prompt("report_agent"),
            build_function_handoff_state_message(
                role="data_agent",
                user_query="Find high TEC intervals.",
                state_packet=packet,
            ),
        ]
    )
    forbidden = [
        "expected_tool_sequence",
        "expected_role_agent_order",
        "GoldRunner",
        "gold_result",
        "verdict_checks",
        "missing_goal_artifacts",
        "remaining_goals",
        "deterministic_baseline_trace",
        "next tool",
        "next role",
    ]
    for item in forbidden:
        assert item.lower() not in text.lower(), item


if __name__ == "__main__":
    main()
