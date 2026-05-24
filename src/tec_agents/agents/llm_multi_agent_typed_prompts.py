"""Prompts and state rendering for the typed full LLM multi-agent experiment."""

from __future__ import annotations

import json
from typing import Any

from tec_agents.agents.llm_single_agent import safe_compact_json
from tec_agents.agents.llm_multi_agent_typed_protocol import ROLE_TOOL_ALLOWLIST
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

When deciding a handoff:
- Inspect typed available_artifacts and previous role outputs.
- If a role_response status is done, inspect available_artifacts and previous role outputs; do not rely on the role_response to repeat every handle.
- Select a role whose input requirements appear satisfied, or a role that can produce the artifact type needed by the user request.
- Create a structured RoleAssignment for that role.
- Set expected_output_type and completion_criteria.
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
        return "\n\n".join(
            [
                common,
                """
You are LLM DataAgent.
Your responsibility is to prepare data_artifacts for the RoleAssignment scope.

Allowed tools:
- tec_get_timeseries
- tec_series_profile, only if genuinely useful for data profiling.

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

Output only one of:
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "default", "region_id": "example_region", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
</tool_call>

<role_response>
{"status":"done","message":"done"}
</role_response>

No prose outside the block. Do not include <tool_call> when returning role_response.
""".strip(),
            ]
        )

    if role == "math_agent":
        allowed = ", ".join(sorted(ROLE_TOOL_ALLOWLIST["math_agent"]))
        return "\n\n".join(
            [
                common,
                f"""
You are LLM MathAgent.
Your responsibility is to prepare computed_artifacts from visible artifact handles.

Allowed tools:
- {allowed}

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
Analyze available computed_artifacts and previous role outputs.
Produce concise findings.
If computed_artifacts are absent, return the cannot_complete block below.

If computed artifacts are available, output:
<role_response>
{"status":"done","message":"findings ready"}
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
]
