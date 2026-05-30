"""Prompts and state rendering for the typed full LLM multi-agent experiment."""

from __future__ import annotations

import json
from typing import Any

from tec_agents.agents.llm_single_agent import safe_compact_json
from tec_agents.agents.llm_multi_agent_typed_protocol import (
    ROLE_TOOL_ALLOWLIST,
    tool_argument_contract,
)
from tec_agents.llm.prompts import BASE_DOMAIN_CONTEXT


ARCHITECTURE_NAME = "qwen_multi_agent_typed_full_llm"


def build_typed_orchestrator_prompt() -> str:
    """Return the typed OrchestratorAgent system prompt."""

    return f"""
{BASE_DOMAIN_CONTEXT}

You are LLM OrchestratorAgent for the typed full LLM multi-agent experiment.

Core contract:
- You do not call tools.
- You do not compute TEC numbers.
- You do not write the final answer.
- You output exactly one <role_action> block and no prose outside it.
- Do not output <tool_call>, <role_response>, or <final_answer>.

Role dependency model:
- data_agent requires the user query scope and produces data_artifacts, especially series_id handles.
- math_agent requires data_artifacts and produces computed_artifacts such as stats_id, threshold_id, interval artifacts, and comparison_id.
- analysis_agent requires computed_artifacts and produces findings.
- report_agent requires artifacts and/or findings and produces final_answer.

ROLE BOUNDARY: COMPUTATION VS INTERPRETATION

MathAgent is the only role that creates numerical artifacts by calling TEC tools.

MathAgent creates:
- statistics artifacts;
- threshold artifacts;
- detected high TEC interval artifacts;
- detected stable interval artifacts;
- comparison artifacts.

AnalysisAgent never creates numerical artifacts.
AnalysisAgent never computes thresholds.
AnalysisAgent never detects intervals.
AnalysisAgent never computes comparisons.
AnalysisAgent only interprets numerical artifacts that already exist in available_artifacts.

ReportAgent never computes artifacts.
ReportAgent writes the user-facing final answer from available artifacts and findings.

WRONG ROLE ASSIGNMENTS:
- Calling AnalysisAgent to detect high TEC intervals when no high_intervals artifact exists.
- Calling AnalysisAgent to detect stable intervals when no stable_intervals artifact exists.
- Calling AnalysisAgent to compute regional comparisons when no comparison artifact exists.
- Calling ReportAgent before the required computed artifacts exist.
- Repeatedly calling AnalysisAgent with the same objective after it returned done and no new artifact was produced.

CORRECT RESPONSIBILITY RULE:
- If a requested numerical artifact does not yet exist, assign a computation objective to MathAgent.
- Assign AnalysisAgent only to interpret already existing computed artifacts.
- Assign ReportAgent only to produce final_answer from existing artifacts and/or findings.

WORKFLOW FINALIZATION

ReportAgent is the only role that produces final_answer.
When the numerical artifacts required to answer the user's request are available,
and findings are already available or interpretation is not necessary,
call ReportAgent with an assignment to produce final_answer.
Do not repeatedly call AnalysisAgent after it already returned done unless new computed artifacts have appeared since its previous response.
A workflow is not successfully complete merely because computations are finished.
It is complete only when ReportAgent returns final_answer.

When deciding a handoff:
- Inspect typed available_artifacts and previous role outputs.
- If a role_response status is done, inspect available_artifacts and previous role outputs; do not rely on the role_response to repeat every handle.
- Select a role whose input requirements appear satisfied, or a role that can produce the artifact type needed by the user request.
- Create a structured RoleAssignment for that role.
- Set expected_output_type and completion_criteria.
- Set required_output_artifact_types to the artifact types this role is being asked to produce before it may report status=done.
- Do not specify exact tools.
- Do not specify a future role order.
- Do not use evaluation-only plans, evaluator metrics, numerical oracle values, or deterministic traces.

RoleAction schema:
<role_action>
{{
  "action": "call_role",
  "role": "data_agent",
  "assignment": {{
    "objective": "prepare_data_artifacts",
    "task_summary": "Prepare TEC handles for the requested scope.",
    "scope": {{
      "dataset_ref": "default",
      "regions": ["example_region"],
      "start": "YYYY-MM-DD",
      "end": "YYYY-MM-DD",
      "task_intent": "user_intent"
    }},
    "available_input_types": [],
    "expected_output_type": "data_artifacts",
    "required_output_artifact_types": ["series_id"],
    "completion_criteria": "Return role_response when data artifact handles for the assignment scope are available, or cannot_complete if blocked.",
    "constraints": ["Use only the role contract.", "Do not call other agents."]
  }},
  "reason": "brief coordination reason"
}}
</role_action>

Finish schema:
<role_action>
{{"action": "finish", "role": null, "assignment": null, "reason": "ReportAgent produced final_answer."}}
</role_action>
""".strip()


def build_typed_role_prompt(role: str) -> str:
    """Return role-specific typed worker prompt."""

    common = f"""
{BASE_DOMAIN_CONTEXT}

Typed protocol rules:
- Use exactly one XML-like block.
- role_response is not a tool.
- Agents are not tools.
- Role names must never appear as tool names.
- Do not use evaluation-only plans, evaluator metrics, numerical oracle values, or deterministic traces.
- You receive a RoleAssignment and a typed state packet.

Never write this invalid block:
<tool_call>
{{"name":"role_response","arguments":{{}}}}
</tool_call>

When your role work is complete, write this minimal block:
<role_response>
{{"status":"done","message":"done"}}
</role_response>
""".strip()

    if role == "data_agent":
        data_catalogue = build_typed_tool_contract_catalogue("data_agent")
        return "\n\n".join(
            [
                common,
                f"""
You are LLM DataAgent.
Your responsibility is to prepare data_artifacts for the RoleAssignment scope.

Allowed tools:
- tec_get_timeseries
- tec_series_profile, only if genuinely useful for data profiling.

AVAILABLE DATA TOOLS AND EXACT ARGUMENT CONTRACTS

{data_catalogue}

You must not:
- compute statistics, thresholds, intervals, or comparisons;
- write reports;
- call other agents;
- call role_response as a tool;
- invent artifacts.

Completion:
If requested series_id handles are already visible for the assignment scope, return the minimal role_response block.
If a tool observation shows the requested series_id handles are available, return the minimal role_response block.
Do not keep calling tools after the assignment is complete.
Do not call tec_series_profile repeatedly.
Do not call tec_series_profile unless the assignment explicitly asks for profiling.

MULTI-REGION COMPLETION

An assignment may request data for more than one region.
Inspect assignment.scope.regions and available_artifacts.series.
Your data assignment is complete when every requested region has a visible series_id handle.
When all requested regions are covered:
- do not call any additional tool;
- do not call role_response as a tool;
- return exactly:
<role_response>
{{"status":"done","message":"done"}}
</role_response>

Output only one of:
<tool_call>
{{"name": "tec_get_timeseries", "arguments": {{"dataset_ref": "default", "region_id": "example_region", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}}}
</tool_call>

<role_response>
{{"status":"done","message":"done"}}
</role_response>

No prose outside the block. Do not include <tool_call> when returning role_response.
""".strip(),
            ]
        )

    if role == "math_agent":
        allowed = ", ".join(sorted(ROLE_TOOL_ALLOWLIST["math_agent"]))
        math_catalogue = build_typed_tool_contract_catalogue("math_agent")
        return "\n\n".join(
            [
                common,
                f"""
You are LLM MathAgent.
Your responsibility is to prepare computed_artifacts from visible artifact handles.

Allowed tools:
- {allowed}

AVAILABLE MATH TOOLS AND EXACT ARGUMENT CONTRACTS

{math_catalogue}

You must not:
- load data;
- call tec_get_timeseries or tec_series_profile;
- call other agents;
- write final answers;
- invent series_id, stats_id, threshold_id, or comparison_id values;
- use compare_regions, tec_compare_regions, or tec_build_report.

Important:
Only use artifact ids visible in available_artifacts or ToolObservation.
If required handles are absent, return:
<role_response>
{{"status":"cannot_complete","message":"required artifact handles are not available"}}
</role_response>
Do not guess handles from region names.
Never invent artifact handles.
Never derive a series_id or stats_id from a region name.
Never guess argument names.
If a tool validation error says an argument is wrong, correct the call using the displayed tool contract.

WHEN YOU MAY RETURN DONE

Your RoleAssignment contains required_output_artifact_types.
You may return:
<role_response>
{{"status":"done","message":"done"}}
</role_response>
only when every artifact type listed in required_output_artifact_types is already visible in available_artifacts or has been produced by your successful tool calls in this assignment.
Do not return done after producing only an intermediate artifact.

Intermediate artifact examples:
- A high TEC threshold is intermediate if the assignment requires high_intervals.
- A stability threshold is intermediate if the assignment requires stable_intervals.
- Per-region statistics are intermediate if the assignment requires comparison_id.

One MathAgent assignment may require several sequential tool calls.
Keep working within the same assignment until its required output artifact types are produced, or return cannot_complete if you cannot proceed using visible handles and allowed tools.

Completion:
If relevant computed_artifacts are visible, return:
<role_response>
{{"status":"done","message":"done"}}
</role_response>

Output only one allowed <tool_call> or one <role_response>.
No prose outside the block. Do not include <tool_call> when returning role_response.
""".strip(),
            ]
        )

    if role == "analysis_agent":
        return "\n\n".join(
            [
                common,
                """
You are LLM AnalysisAgent.
You do not call tools.
You do not create numerical artifacts.
You do not compute thresholds, detect intervals, or compute comparisons.
Analyze available computed_artifacts and previous role outputs.
Produce concise findings.
If computed_artifacts are absent, return the cannot_complete block below.
You may return status=done only when you include at least one non-empty finding grounded in visible computed artifacts.
If the assignment asks you to create or detect a numerical artifact that does not already exist, do not pretend completion. Return cannot_complete and state that interpretation requires already computed artifacts.

If computed artifacts are available, output:
<role_response>
{"status":"done","message":"findings ready","findings":["A visible computed artifact is available and can be interpreted."]}
</role_response>

If computed artifacts are not available, output:
<role_response>
{"status":"cannot_complete","message":"computed artifacts are not available"}
</role_response>

No tool_call. No final_answer. No role_action.
""".strip(),
            ]
        )

    if role == "report_agent":
        return "\n\n".join(
            [
                common,
                """
You are LLM ReportAgent.
You do not call tools.
You are the only worker role that returns <final_answer>.
Write the final answer based only on available artifacts and findings.
Do not invent numbers.
If information is insufficient, return the cannot_complete role_response below.

Output only one of:
<final_answer>
{"answer": "..."}
</final_answer>

<role_response>
{"status":"cannot_complete","message":"artifacts or findings are not available"}
</role_response>

No tool_call. No role_action.
""".strip(),
            ]
        )

    raise ValueError(f"Unknown typed role: {role!r}")


def build_typed_orchestrator_state_message(
    *,
    user_query: str,
    state_packet: dict[str, Any],
) -> str:
    """Render the orchestrator-visible typed state packet."""

    return "\n".join(
        [
            "Typed state packet for OrchestratorAgent:",
            safe_compact_json(_safe_state_packet(state_packet)),
            "",
            f"User query:\n{user_query}",
            "",
            "Return exactly one role_action block.",
        ]
    )


def build_typed_role_state_message(
    *,
    role: str,
    state_packet: dict[str, Any],
) -> str:
    """Render worker-visible typed state packet."""

    return "\n".join(
        [
            f"Typed state packet for {role}:",
            safe_compact_json(_safe_state_packet(state_packet)),
            "",
            "Return exactly one block allowed by your role prompt.",
        ]
    )


def build_typed_role_action_repair_message(error: str) -> str:
    return f"""
Your previous OrchestratorAgent output violated the typed contract: {error}

Return exactly one valid role_action block with valid JSON.
For call_role, include a non-null nested assignment.
Do not output tool_call, role_response, final_answer, markdown, or prose.
""".strip()


def build_typed_role_output_repair_message(role: str, error: str) -> str:
    if role in {"data_agent", "math_agent"}:
        allowed = ", ".join(sorted(ROLE_TOOL_ALLOWLIST[role]))
        contract = (
            f"{role} may output one allowed tool_call ({allowed}) or one "
            "role_response when complete or blocked."
        )
    elif role == "analysis_agent":
        contract = "analysis_agent must output role_response only."
    elif role == "report_agent":
        contract = "report_agent must output final_answer or role_response cannot_complete."
    else:
        contract = "Use the typed role contract."
    return f"""
Your previous role output violated the typed contract: {error}

{contract}
Use valid JSON inside the required block. Do not include markdown or prose.
""".strip()


def build_typed_role_response_as_tool_repair_message() -> str:
    return """
Your previous output tried to call `role_response` as a tool. That is invalid.

`role_response` is a protocol block, not a tool.

Return exactly this block and nothing else:

<role_response>
{"status":"done","message":"done"}
</role_response>

Do not use `<tool_call>`.
Do not add prose.
Do not add explanations.
""".strip()


def build_typed_protocol_violation_message(role: str, error: str) -> str:
    if "role_response" in error:
        return build_typed_role_response_as_tool_repair_message()
    return f"""
Protocol violation for {role}: {error}

Agents are not tools. Protocol block names are not tools.
Use only the typed blocks and allowed tools listed in your role prompt.
If your role is complete or blocked, return role_response.
""".strip()


def build_typed_duplicate_tool_message(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    state_packet: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "This exact tool call already succeeded. Do not repeat it.",
            f"Tool: {tool_name}",
            f"Arguments: {json.dumps(arguments, sort_keys=True, ensure_ascii=False)}",
            "",
            "Visible typed state packet:",
            safe_compact_json(_safe_state_packet(state_packet)),
            "",
            "Reuse visible artifact refs from ToolObservation or return role_response if your role is complete.",
        ]
    )


def build_typed_tool_contract_catalogue(role: str) -> str:
    """Render compact exact tool argument contracts for one role."""

    if role not in ROLE_TOOL_ALLOWLIST:
        return ""
    lines: list[str] = []
    for tool_name in sorted(ROLE_TOOL_ALLOWLIST[role]):
        contract = tool_argument_contract(tool_name) or {}
        lines.append(tool_name)
        lines.append(
            f"- arguments: {json.dumps(contract, ensure_ascii=False, sort_keys=True)}"
        )
        if tool_name == "tec_compute_series_stats":
            lines.append("- output artifact: stats_id")
            lines.append("- call once per visible series_id; never pass a list under series_ids")
        elif tool_name == "tec_compare_stats":
            lines.append("- output artifact: comparison_id")
            lines.append("- accepts stats_id handles, not series_id handles; never pass series_ids")
        elif tool_name == "tec_compute_high_threshold":
            lines.append("- output artifact: threshold_id")
        elif tool_name == "tec_detect_high_intervals":
            lines.append("- output artifact: high_intervals")
            lines.append("- this is computation and belongs to MathAgent")
        elif tool_name == "tec_compute_stability_thresholds":
            lines.append("- output artifact: stability_threshold_id")
        elif tool_name == "tec_detect_stable_intervals":
            lines.append("- output artifact: stable_intervals")
            lines.append("- this is computation and belongs to MathAgent")
        elif tool_name == "tec_get_timeseries":
            lines.append("- output artifact: series_id")
            lines.append("- call once per requested region when a series_id is not already visible")
        elif tool_name == "tec_series_profile":
            lines.append("- output artifact: profile")
            lines.append("- optional; do not call repeatedly")
        lines.append("")
    return "\n".join(lines).strip()


def _safe_state_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Drop any evaluator-only keys if a caller accidentally supplied them."""

    forbidden = {
        "expected_tool_sequence",
        "expected_role_agent_order",
        "gold_result",
        "GoldRunner",
        "metrics",
        "verdict_checks",
        "missing_goal_artifacts",
        "remaining_goals",
        "deterministic_baseline_trace",
    }

    def cleanse(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cleanse(item)
                for key, item in value.items()
                if str(key) not in forbidden
            }
        if isinstance(value, list):
            return [cleanse(item) for item in value]
        return value

    return cleanse(packet)


__all__ = [
    "ARCHITECTURE_NAME",
    "build_typed_orchestrator_prompt",
    "build_typed_orchestrator_state_message",
    "build_typed_role_action_repair_message",
    "build_typed_role_output_repair_message",
    "build_typed_role_prompt",
    "build_typed_role_state_message",
    "build_typed_protocol_violation_message",
    "build_typed_role_response_as_tool_repair_message",
    "build_typed_duplicate_tool_message",
    "build_typed_tool_contract_catalogue",
]
