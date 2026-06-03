"""Prompts for the simplified function-handoff multi-agent experiment."""

from __future__ import annotations

import json
from typing import Any

from tec_agents.agents.llm_multi_agent_function_handoff_protocol import (
    ARCHITECTURE_NAME,
    DATA_TOOLS,
    FORBIDDEN_PROMPT_KEYS,
    MATH_TOOLS,
    ORCHESTRATOR_FUNCTIONS,
    RETURN_FUNCTION,
    ROLE_FUNCTION_ALLOWLIST,
    safe_state_payload,
    tool_argument_contract,
)
from tec_agents.agents.llm_single_agent import safe_compact_json
from tec_agents.llm.prompts import BASE_DOMAIN_CONTEXT


PROMPT_REVISION = "function_handoff_minimal_v1"


def build_function_handoff_orchestrator_prompt() -> str:
    functions = "\n".join(
        f"- {name}: call {role}" for name, role in sorted(ORCHESTRATOR_FUNCTIONS.items())
    )
    return f"""
{BASE_DOMAIN_CONTEXT}

You are LLM OrchestratorAgent in the function-handoff multi-agent experiment.

You coordinate roles. You do not call TEC tools. You do not compute numbers.
You do not write the final answer yourself.

You may call exactly one internal function per turn:
{functions}

Function schema:
<tool_call>
{{"name":"call_data_agent","arguments":{{"message":"short instruction"}}}}
</tool_call>

Role responsibilities:
- data_agent loads TEC time series handles.
- math_agent creates numerical artifacts using TEC tools: stats, thresholds, intervals, comparisons.
- analysis_agent interprets existing computed artifacts and returns findings.
- report_agent writes the user-facing final answer from visible artifacts/findings.

Decision rules:
- Inspect the user query, actual available artifacts, previous role returns, and handoff history.
- Choose the role that should work next from the actual state.
- Do not specify a future role sequence.
- Do not specify exact TEC tool calls.
- Do not call the same role with the same message repeatedly if state has not changed.
- The run is complete only when report_agent returns final_answer to the runtime.

Forbidden:
- Do not output role_action, role_response, final_answer, markdown, or prose.
- Do not call TEC tools directly.
- Do not use evaluator-only hidden data.

Return exactly one <tool_call> block and nothing else.
""".strip()


def build_function_handoff_role_prompt(role: str) -> str:
    if role not in ROLE_FUNCTION_ALLOWLIST:
        raise ValueError(f"Unknown function-handoff role: {role!r}")
    allowed = ", ".join(sorted(ROLE_FUNCTION_ALLOWLIST[role]))
    common = f"""
{BASE_DOMAIN_CONTEXT}

You are LLM {role}.

ONE TURN = ONE FUNCTION CALL.
Return exactly one <tool_call> block and no prose.
Allowed functions/tools for your role:
{allowed}

Use this format:
<tool_call>
{{"name":"function_or_tool_name","arguments":{{...}}}}
</tool_call>

return_to_orchestrator is an internal function, not a TEC tool.
When your role is done or blocked, call:
<tool_call>
{{"name":"return_to_orchestrator","arguments":{{"status":"done","message":"done"}}}}
</tool_call>

Never call role names as functions. Never call role_response, role_action, or final_answer.
Do not use evaluator-only hidden data.
""".strip()

    if role == "data_agent":
        return "\n\n".join(
            [
                common,
                f"""
Your responsibility:
- Load missing TEC time series for the requested regions/period.
- Use only data tools or {RETURN_FUNCTION}.
- Do not compute stats, thresholds, intervals, comparisons, or reports.

Allowed data tool contracts:
{_tool_catalogue(DATA_TOOLS)}

Completion:
- If every requested region already has a visible series_id, call return_to_orchestrator.
- If you successfully loaded the needed series handles, call return_to_orchestrator.
- Do not call tec_series_profile unless profiling is explicitly requested.
""".strip(),
            ]
        )

    if role == "math_agent":
        return "\n\n".join(
            [
                common,
                f"""
Your responsibility:
- Create numerical artifacts using visible handles and allowed math tools.
- Use only exact artifact handles shown in state/observations.
- Do not load data. Do not call other agents. Do not write final answers.
- If required input handles are absent, call {RETURN_FUNCTION} with status cannot_complete.

Allowed math tool contracts:
{_tool_catalogue(MATH_TOOLS)}

Important:
- For comparison, compute stats_id handles first, then compare stats_id handles.
- tec_compare_stats accepts stats_ids, not series_id handles.
- Do not invent artifact handles from region names or dates.
- After useful computation is done or you are blocked, call return_to_orchestrator.
""".strip(),
            ]
        )

    if role == "analysis_agent":
        return "\n\n".join(
            [
                common,
                """
Your responsibility:
- Interpret existing computed artifacts only.
- You have no TEC tools.
- Return findings through return_to_orchestrator.
- If computed artifacts are absent, return cannot_complete.

Example:
<tool_call>
{"name":"return_to_orchestrator","arguments":{"status":"done","message":"findings ready","findings":["A computed artifact is available and can answer the request."]}}
</tool_call>
""".strip(),
            ]
        )

    if role == "report_agent":
        return "\n\n".join(
            [
                common,
                """
Your responsibility:
- Write the final user-facing answer using only visible artifacts/findings.
- You have no TEC tools.
- Do not invent numbers.
- Put the final answer inside return_to_orchestrator.final_answer.

Example:
<tool_call>
{"name":"return_to_orchestrator","arguments":{"status":"done","message":"final answer ready","final_answer":"..."}}
</tool_call>
""".strip(),
            ]
        )

    raise ValueError(f"Unknown function-handoff role: {role!r}")


def build_function_handoff_state_message(
    *,
    role: str,
    user_query: str,
    state_packet: dict[str, Any],
) -> str:
    safe_packet = safe_state_payload(state_packet)
    if role == "orchestrator":
        header = "State for OrchestratorAgent."
    else:
        header = f"State for {role}."
    return "\n".join(
        [
            header,
            f"User query: {user_query}",
            "",
            safe_compact_json(safe_packet),
            "",
            "Return exactly one <tool_call> block.",
        ]
    )


def build_function_handoff_repair_message(role: str, error: str) -> str:
    allowed = ", ".join(sorted(ROLE_FUNCTION_ALLOWLIST.get(role, set())))
    return f"""
Your previous output was invalid for {role}: {error}

Return exactly one <tool_call> block with valid JSON.
Allowed names for your role: {allowed}
Do not add prose, markdown, role_response, role_action, or final_answer blocks.
""".strip()


def assert_no_forbidden_hints(text: str) -> None:
    lower = text.lower()
    for item in FORBIDDEN_PROMPT_KEYS:
        if str(item).lower() in lower:
            raise AssertionError(f"Forbidden prompt/state hint found: {item}")


def _tool_catalogue(tool_names: set[str]) -> str:
    lines: list[str] = []
    for tool_name in sorted(tool_names):
        contract = tool_argument_contract(tool_name) or {}
        lines.append(tool_name)
        lines.append(f"- arguments: {json.dumps(contract, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)


__all__ = [
    "ARCHITECTURE_NAME",
    "PROMPT_REVISION",
    "assert_no_forbidden_hints",
    "build_function_handoff_orchestrator_prompt",
    "build_function_handoff_repair_message",
    "build_function_handoff_role_prompt",
    "build_function_handoff_state_message",
]
