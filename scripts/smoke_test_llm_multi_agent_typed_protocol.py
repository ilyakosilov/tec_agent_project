"""Smoke tests for the typed full LLM multi-agent protocol.

This script does not load Qwen and does not require GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.agents.llm_multi_agent_typed import (
    build_assignment_progress,
    build_typed_state_packet,
    count_tool_errors_from_trace,
    count_tool_schema_validation_errors_from_trace,
    enrich_typed_role_response,
    equivalent_role_assignment_key,
    missing_required_output_artifact_types,
)
from tec_agents.agents.llm_multi_agent_typed_prompts import (
    build_typed_tool_contract_catalogue,
    build_typed_orchestrator_prompt,
    build_typed_orchestrator_state_message,
    build_typed_role_response_as_tool_repair_message,
    build_typed_role_prompt,
    build_typed_role_state_message,
)
from tec_agents.agents.llm_multi_agent_typed_protocol import (
    RoleAssignment,
    RoleResponse,
    RoleScope,
    clean_typed_output,
    make_tool_observation,
    parse_typed_role_action,
    parse_typed_role_output,
    parse_typed_role_response,
    tool_call_key,
    tool_argument_contract,
)


FORBIDDEN_PROMPT_FRAGMENTS = [
    "expected_tool_sequence",
    "expected_role_agent_order",
    "gold_result",
    "GoldRunner",
    "missing_goal_artifacts",
    "missing goals",
    "remaining goals",
    "next tool",
    "next role",
    "deterministic baseline trace",
]


def main() -> None:
    test_role_action_schema()
    test_role_output_protocols()
    test_tool_observation_and_duplicates()
    test_minimal_role_response_enrichment_and_repair()
    test_output_cleaner_removes_chat_template_noise()
    test_prompts_and_state_packets_hide_forbidden_hints()
    print("Typed LLM multi-agent protocol smoke test finished successfully.")


def _assignment_json() -> str:
    return """
{
  "objective": "prepare_data_artifacts",
  "task_summary": "Prepare TEC handles for a requested scope.",
  "scope": {
    "dataset_ref": "default",
    "regions": ["midlat_europe"],
    "start": "2024-03-01",
    "end": "2024-04-01",
    "task_intent": "high_tec"
  },
  "available_input_types": [],
  "expected_output_type": "data_artifacts",
  "required_output_artifact_types": ["series_id"],
  "completion_criteria": "Return role_response when handles are available.",
  "constraints": ["Use only DataAgent tools."]
}
""".strip()


def test_role_action_schema() -> None:
    valid = (
        "<role_action>"
        '{"action":"call_role","role":"data_agent","assignment":'
        + _assignment_json()
        + ',"reason":"prepare data"}'
        "</role_action>"
    )
    parsed = parse_typed_role_action(valid)
    assert parsed.ok, parsed.to_dict()
    assert parsed.value.role == "data_agent"
    assert parsed.value.assignment.expected_output_type == "data_artifacts"
    assert parsed.value.assignment.required_output_artifact_types == ["series_id"]

    missing_assignment = (
        '<role_action>{"action":"call_role","role":"data_agent","assignment":null,"reason":"bad"}</role_action>'
    )
    parsed = parse_typed_role_action(missing_assignment)
    assert not parsed.ok
    assert parsed.error_code == "schema_error"

    for forbidden_key in [
        "expected_tool_sequence",
        "expected_role_agent_order",
        "gold_result",
        "missing_goal_artifacts",
    ]:
        forbidden_field = (
            "<role_action>"
            '{"action":"call_role","role":"data_agent","assignment":'
            '{"objective":"x","task_summary":"x","scope":{},"available_input_types":[],'
            '"expected_output_type":"data_artifacts","required_output_artifact_types":[],"completion_criteria":"x",'
            f'"{forbidden_key}":["bad"],"constraints":[]}},'
            '"reason":"bad"}'
            "</role_action>"
        )
        parsed = parse_typed_role_action(forbidden_field)
        assert not parsed.ok
        assert parsed.error_code in {"schema_error", "forbidden_field"}

    orchestrator_tool = '<tool_call>{"name":"tec_get_timeseries","arguments":{}}</tool_call>'
    parsed = parse_typed_role_output(orchestrator_tool, "orchestrator")
    assert not parsed.ok
    assert parsed.error_code == "protocol_violation"


def test_role_output_protocols() -> None:
    data_tool = (
        '<tool_call>{"name":"tec_get_timeseries","arguments":'
        '{"dataset_ref":"default","region_id":"midlat_europe","start":"2024-03-01","end":"2024-04-01"}}'
        "</tool_call>"
    )
    parsed = parse_typed_role_output(data_tool, "data_agent")
    assert parsed.ok
    assert parsed.value.name == "tec_get_timeseries"

    data_response = '<role_response>{"status":"done","message":"done"}</role_response>'
    parsed = parse_typed_role_output(data_response, "data_agent")
    assert parsed.ok
    assert parsed.value.role == ""
    assert parsed.value.message == "done"

    minimal_cannot_complete = (
        '<role_response>{"status":"cannot_complete","message":"cannot complete"}</role_response>'
    )
    parsed = parse_typed_role_output(minimal_cannot_complete, "math_agent")
    assert parsed.ok
    assert parsed.value.status == "cannot_complete"

    role_response_as_tool = '<tool_call>{"name":"role_response","arguments":{}}</tool_call>'
    parsed = parse_typed_role_output(role_response_as_tool, "data_agent")
    assert not parsed.ok
    assert parsed.error_code == "protocol_violation"
    assert "not a tool" in (parsed.error_message or "")
    assert parsed.error_code != "invalid_xml"

    agent_as_tool = '<tool_call>{"name":"report_agent","arguments":{}}</tool_call>'
    parsed = parse_typed_role_output(agent_as_tool, "data_agent")
    assert not parsed.ok
    assert parsed.error_code == "protocol_violation"

    math_tool = '<tool_call>{"name":"tec_compute_series_stats","arguments":{"series_id":"series_1"}}</tool_call>'
    parsed = parse_typed_role_output(math_tool, "math_agent")
    assert parsed.ok

    aggregate_compare = '<tool_call>{"name":"compare_regions","arguments":{}}</tool_call>'
    parsed = parse_typed_role_output(aggregate_compare, "math_agent")
    assert not parsed.ok
    assert parsed.error_code == "protocol_violation"

    math_cannot_complete = (
        '<role_response>{"status":"cannot_complete","role":"math_agent","summary":"need series",'
        '"produced_artifact_types":[],"artifact_refs":[],"findings":[],'
        '"needs":[{"type":"series_id","reason":"No visible series handles."}]}</role_response>'
    )
    parsed = parse_typed_role_output(math_cannot_complete, "math_agent")
    assert parsed.ok
    assert parsed.value.needs[0].type == "series_id"

    analysis_tool = '<tool_call>{"name":"tec_compare_stats","arguments":{}}</tool_call>'
    parsed = parse_typed_role_output(analysis_tool, "analysis_agent")
    assert not parsed.ok
    assert parsed.error_code == "protocol_violation"

    analysis_response = (
        '<role_response>{"status":"done","role":"analysis_agent","summary":"findings",'
        '"produced_artifact_types":["findings"],"artifact_refs":[],"findings":["x"],"needs":[]}</role_response>'
    )
    parsed = parse_typed_role_response(analysis_response)
    assert parsed.ok

    final = '<final_answer>{"answer":"Done."}</final_answer>'
    parsed = parse_typed_role_output(final, "report_agent")
    assert parsed.ok
    assert parsed.value.answer == "Done."

    report_tool = '<tool_call>{"name":"tec_compute_series_stats","arguments":{}}</tool_call>'
    parsed = parse_typed_role_output(report_tool, "report_agent")
    assert not parsed.ok
    assert parsed.error_code == "protocol_violation"


def test_tool_observation_and_duplicates() -> None:
    obs = make_tool_observation(
        tool_name="tec_get_timeseries",
        status="ok",
        result={
            "series_id": "series_abc",
            "metadata": {"region_id": "midlat_europe", "n_points": 744},
        },
    )
    assert obs.produced_artifact_type == "series_id"
    assert obs.artifact_id == "series_abc"
    assert "744" in obs.summary

    first = tool_call_key("tec_get_timeseries", {"region_id": "midlat_europe"})
    second = tool_call_key("tec_get_timeseries", {"region_id": "highlat_north"})
    duplicate = tool_call_key("tec_get_timeseries", {"region_id": "midlat_europe"})
    assert first != second
    assert first == duplicate

    validation_obs = make_tool_observation(
        tool_name="tec_compare_stats",
        status="error",
        error={
            "error": {
                "error_type": "validation_error",
                "message": "stats_ids field required",
                "tool_name": "tec_compare_stats",
            }
        },
    )
    assert validation_obs.error_type == "validation_error"
    assert validation_obs.expected_argument_contract == tool_argument_contract(
        "tec_compare_stats"
    )
    assert "stats_ids" in (validation_obs.error_summary or "")
    rendered = str(validation_obs.to_dict())
    for fragment in FORBIDDEN_PROMPT_FRAGMENTS:
        assert fragment not in rendered, fragment


def test_minimal_role_response_enrichment_and_repair() -> None:
    assignment = RoleAssignment(
        objective="prepare_data_artifacts",
        task_summary="Prepare data.",
        scope=RoleScope(
            dataset_ref="default",
            regions=["midlat_europe"],
            start="2024-03-01",
            end="2024-04-01",
            task_intent="high_tec",
        ),
        available_input_types=[],
        expected_output_type="data_artifacts",
        required_output_artifact_types=["series_id"],
        completion_criteria="Return role_response when data handles are visible.",
        constraints=["Use only data tools."],
    )
    observation = make_tool_observation(
        tool_name="tec_get_timeseries",
        status="ok",
        result={
            "series_id": "series_abc",
            "metadata": {"region_id": "midlat_europe", "n_points": 744},
        },
    )
    context = {
        "tool_observations": [observation.to_dict()],
        "data_artifacts": {
            "series_by_region": {
                "midlat_europe": {
                    "series_id": "series_abc",
                    "metadata": {"n_points": 744},
                }
            }
        },
    }
    response = RoleResponse(status="done", message="done")
    enriched = enrich_typed_role_response(
        response=response,
        role="data_agent",
        assignment=assignment,
        context=context,
        observation_start=0,
    )
    assert enriched.role == "data_agent"
    assert enriched.status == "done"
    assert enriched.message == "done"
    assert "series_id" in enriched.produced_artifact_types
    assert "series_abc" in enriched.artifact_refs

    premature_assignment = RoleAssignment(
        objective="compute_high_intervals",
        task_summary="Compute requested high TEC interval artifacts.",
        scope=RoleScope(regions=["midlat_europe"]),
        available_input_types=["series_id"],
        expected_output_type="computed_artifacts",
        required_output_artifact_types=["threshold_id", "high_intervals"],
        completion_criteria="Return done only when required artifact types exist.",
        constraints=["Use only math tools."],
    )
    premature_response = RoleResponse(
        status="done",
        message="done",
        role="math_agent",
        produced_artifact_types=["threshold_id"],
    )
    missing = missing_required_output_artifact_types(
        response=premature_response,
        assignment=premature_assignment,
        context={},
    )
    assert missing == ["high_intervals"]

    key_one = equivalent_role_assignment_key(
        role="analysis_agent",
        assignment=premature_assignment,
        context=context,
    )
    key_two = equivalent_role_assignment_key(
        role="analysis_agent",
        assignment=premature_assignment,
        context=context,
    )
    assert key_one == key_two

    trace = {
        "calls": [
            {"tool_name": "tec_get_timeseries", "status": "ok"},
            {
                "tool_name": "tec_compare_stats",
                "status": "error",
                "error_type": "validation_error",
            },
        ]
    }
    assert count_tool_errors_from_trace(trace) == 1
    assert count_tool_schema_validation_errors_from_trace(trace) == 1

    repair = build_typed_role_response_as_tool_repair_message()
    assert '<role_response>\n{"status":"done","message":"done"}\n</role_response>' in repair
    assert "Do not use `<tool_call>`." in repair
    for fragment in FORBIDDEN_PROMPT_FRAGMENTS:
        assert fragment not in repair, fragment


def test_output_cleaner_removes_chat_template_noise() -> None:
    raw = """
<|im_start|>assistant
<think><role_response>{"status":"failed","message":"hidden"}</role_response></think>
user
The previous message was invalid.
assistant
<role_response>
{"status":"done","message":"done"}
</role_response>
extra text
<|im_end|>
"""
    cleaned = clean_typed_output(raw)
    assert cleaned == '<role_response>\n{"status":"done","message":"done"}\n</role_response>'


def test_prompts_and_state_packets_hide_forbidden_hints() -> None:
    assignment = RoleAssignment(
        objective="prepare_data_artifacts",
        task_summary="Prepare data.",
        scope=RoleScope(
            dataset_ref="default",
            regions=["midlat_europe"],
            start="2024-03-01",
            end="2024-04-01",
            task_intent="high_tec",
        ),
        available_input_types=[],
        expected_output_type="data_artifacts",
        required_output_artifact_types=["series_id"],
        completion_criteria="Return role_response when data handles are visible.",
        constraints=["Use only data tools."],
    )
    context = {
        "available_artifacts": {},
        "role_outputs": [],
        "tool_observations": [],
        "completed_tool_calls": [],
    }
    packet = build_typed_state_packet(
        user_query="Find high TEC intervals.",
        current_role="data_agent",
        current_assignment=assignment,
        context=context,
    )
    assert "assignment_progress" in packet
    assert packet["assignment_progress"]["requested_regions"] == ["midlat_europe"]
    assert packet["assignment_progress"]["scope_covered"] is False
    rendered = "\n".join(
        [
            build_typed_orchestrator_prompt(),
            build_typed_orchestrator_state_message(
                user_query="Find high TEC intervals.",
                state_packet=packet,
            ),
            build_typed_role_prompt("data_agent"),
            build_typed_role_prompt("math_agent"),
            build_typed_role_prompt("analysis_agent"),
            build_typed_role_prompt("report_agent"),
            build_typed_role_state_message(role="data_agent", state_packet=packet),
        ]
    )
    for fragment in FORBIDDEN_PROMPT_FRAGMENTS:
        assert fragment not in rendered, fragment
    assert '<tool_call>\n{"name":"role_response","arguments":{}}\n</tool_call>' in rendered
    assert '<role_response>\n{"status":"done","message":"done"}\n</role_response>' in rendered
    assert "role_response is not a tool" in rendered
    orchestrator_prompt = build_typed_orchestrator_prompt()
    assert "MathAgent creates:" in orchestrator_prompt
    assert "AnalysisAgent only interprets numerical artifacts" in orchestrator_prompt
    assert "ReportAgent is the only role that produces final_answer" in orchestrator_prompt
    assert "Calling AnalysisAgent to detect high TEC intervals" in orchestrator_prompt
    assert "Repeatedly calling AnalysisAgent" in orchestrator_prompt

    math_prompt = build_typed_role_prompt("math_agent")
    assert "tec_compute_series_stats" in math_prompt
    assert "one visible series_id" in math_prompt
    assert "tec_compare_stats" in math_prompt
    assert "stats_ids" in math_prompt
    assert "never pass series_ids" in math_prompt
    assert "A high TEC threshold is intermediate" in math_prompt
    assert "required_output_artifact_types" in math_prompt

    data_prompt = build_typed_role_prompt("data_agent")
    assert "MULTI-REGION COMPLETION" in data_prompt
    assert "every requested region has a visible series_id handle" in data_prompt

    analysis_prompt = build_typed_role_prompt("analysis_agent")
    assert "at least one non-empty finding" in analysis_prompt

    report_prompt = build_typed_role_prompt("report_agent")
    assert "only worker role that returns <final_answer>" in report_prompt

    catalogue = build_typed_tool_contract_catalogue("math_agent")
    assert '"series_id": "<one visible series_id>"' in catalogue
    assert '"stats_ids": ["<visible stats_id>", "<visible stats_id>"]' in catalogue


if __name__ == "__main__":
    main()
