"""
Full LLM-driven multi-agent TEC workflow.

This module is intentionally separate from the deterministic role-based
baseline. Every role is driven by model.generate(); the runtime only validates
role actions, parses textual protocols, enforces role tool permissions, executes
tools through LocalMCPClient, and records diagnostics.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from tec_agents.agents.llm_single_agent import (
    canonical_arguments_json,
    clean_model_output,
    count_final_answer_blocks,
    count_tool_call_blocks,
    detect_output_wrapper_anomalies,
    infer_task_state,
    safe_compact_json,
)
from tec_agents.llm.prompts import BASE_DOMAIN_CONTEXT
from tec_agents.llm.tool_call_parser import (
    ParseErrorCode,
    parse_final_answer,
    parse_tool_call,
)
from tec_agents.mcp.client import LocalMCPClient


ROLE_NAMES = {
    "data_agent",
    "math_agent",
    "analysis_agent",
    "report_agent",
}

ROLE_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "orchestrator": set(),
    "data_agent": {"tec_get_timeseries", "tec_series_profile"},
    "math_agent": {
        "tec_compute_series_stats",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
        "tec_compute_stability_thresholds",
        "tec_detect_stable_intervals",
        "tec_compare_stats",
    },
    "analysis_agent": set(),
    "report_agent": set(),
}

AGENT_RESPONSE_STATUS = {
    "ok",
    "missing_artifacts",
    "invalid_input",
    "tool_error",
    "partial",
    "final",
}

ROLE_ACTION_RE = re.compile(
    r"<role_action\b[^>]*>\s*(.*?)\s*</role_action>",
    flags=re.DOTALL | re.IGNORECASE,
)
ROLE_RESPONSE_RE = re.compile(
    r"<role_response\b[^>]*>\s*(.*?)\s*</role_response>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class LLMRoleOutput:
    """Result of one LLM role invocation."""

    role: str
    status: str
    message: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    raw_model_outputs: list[str] = field(default_factory=list)
    cleaned_model_outputs: list[str] = field(default_factory=list)
    parse_error_count: int = 0
    invalid_json_count: int = 0
    unknown_format_count: int = 0
    invalid_tool_name_count: int = 0
    forbidden_tool_call_count: int = 0
    repeated_tool_call_count: int = 0
    repair_attempt_count: int = 0
    stalled_loop_detected: bool = False
    tool_sequence: list[str] = field(default_factory=list)
    role_steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMMultiAgentResult:
    """Structured result returned by LLMFullMultiAgent."""

    answer: str
    parsed_task: dict[str, Any] | None
    tool_results: dict[str, Any]
    trace: dict[str, Any]
    orchestration_steps: list[dict[str, Any]]
    role_outputs: list[dict[str, Any]]
    orchestrator_decisions: list[dict[str, Any]]
    raw_model_outputs: list[str]
    cleaned_model_outputs: list[str]
    role_agent_order: list[str]
    actual_tool_sequence: list[str]
    parse_error_count: int
    invalid_json_count: int
    unknown_format_count: int
    invalid_role_action_count: int
    invalid_tool_name_count: int
    forbidden_tool_call_count: int
    repeated_tool_call_count: int
    stalled_loop_detected: bool
    repair_attempt_count: int
    retry_count: int
    recovery_attempt_count: int
    recovery_success_count: int
    recovery_failure_count: int
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable result."""

        return asdict(self)


class LLMOrchestratorAgent:
    """LLM-driven orchestrator. It selects role handoffs through role_action."""

    def __init__(self, model, *, temperature: float = 0.0, max_new_tokens: int = 256):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def decide(
        self,
        *,
        user_query: str,
        context: dict[str, Any],
        max_parse_retries: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Ask the LLM orchestrator for a role action."""

        diagnostics = {
            "raw_model_outputs": [],
            "cleaned_model_outputs": [],
            "parse_error_count": 0,
            "invalid_json_count": 0,
            "unknown_format_count": 0,
            "invalid_role_action_count": 0,
            "repair_attempt_count": 0,
        }
        messages = [
            {
                "role": "system",
                "content": build_orchestrator_prompt(),
            },
            {
                "role": "user",
                "content": build_orchestrator_state_message(user_query, context),
            },
        ]

        for _ in range(max_parse_retries + 1):
            raw = self.model.generate(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            cleaned = clean_multi_agent_output(raw)
            diagnostics["raw_model_outputs"].append(raw)
            diagnostics["cleaned_model_outputs"].append(cleaned)
            action, error_code, error_message = parse_role_action(cleaned)
            if action is not None:
                validation_error = validate_role_action(action, context=context)
                if validation_error is None:
                    return action, diagnostics
                diagnostics["invalid_role_action_count"] += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_role_action_repair_message(validation_error),
                    }
                )
                continue

            diagnostics["parse_error_count"] += 1
            if error_code == ParseErrorCode.INVALID_JSON:
                diagnostics["invalid_json_count"] += 1
            elif error_code == ParseErrorCode.UNKNOWN_FORMAT:
                diagnostics["unknown_format_count"] += 1
            diagnostics["repair_attempt_count"] += 1
            messages.append(
                {
                    "role": "user",
                    "content": build_role_action_repair_message(
                        error_message or "Could not parse role_action."
                    ),
                }
            )

        return None, diagnostics


class LLMRoleAgent:
    """LLM-driven worker role with hard runtime tool restrictions."""

    def __init__(
        self,
        role: str,
        model,
        client: LocalMCPClient,
        *,
        max_role_steps: int = 8,
        max_tool_calls: int = 8,
        max_parse_retries: int = 2,
        temperature: float = 0.0,
        tool_max_new_tokens: int = 256,
        response_max_new_tokens: int = 512,
    ) -> None:
        if role not in ROLE_TOOL_ALLOWLIST:
            raise ValueError(f"Unknown LLM role: {role!r}")
        self.role = role
        self.model = model
        self.client = client
        self.max_role_steps = max_role_steps
        self.max_tool_calls = max_tool_calls
        self.max_parse_retries = max_parse_retries
        self.temperature = temperature
        self.tool_max_new_tokens = tool_max_new_tokens
        self.response_max_new_tokens = response_max_new_tokens

    def run(
        self,
        *,
        user_query: str,
        role_message: str,
        context: dict[str, Any],
    ) -> LLMRoleOutput:
        """Run one LLM role until it emits role_response or final_answer."""

        messages = [
            {"role": "system", "content": build_role_prompt(self.role)},
            {
                "role": "user",
                "content": build_role_input_message(
                    role=self.role,
                    user_query=user_query,
                    role_message=role_message,
                    context=context,
                ),
            },
        ]

        output = LLMRoleOutput(role=self.role, status="tool_error")
        consecutive_parse_errors = 0
        tool_call_count = 0

        for step in range(1, self.max_role_steps + 1):
            raw = self.model.generate(
                messages,
                max_new_tokens=self.response_max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            cleaned = clean_multi_agent_output(raw)
            output.raw_model_outputs.append(raw)
            output.cleaned_model_outputs.append(cleaned)
            parsed = parse_role_output(cleaned)
            role_step = {
                "step": step,
                "raw_model_output": raw,
                "cleaned_model_output": cleaned,
                "parsed_type": parsed["type"],
            }

            if parsed["type"] == "parse_error":
                output.parse_error_count += 1
                consecutive_parse_errors += 1
                error_code = parsed.get("error_code")
                if error_code == ParseErrorCode.INVALID_JSON.value:
                    output.invalid_json_count += 1
                elif error_code == ParseErrorCode.UNKNOWN_FORMAT.value:
                    output.unknown_format_count += 1
                role_step["parse_error_message"] = parsed.get("error_message")
                output.role_steps.append(role_step)
                if consecutive_parse_errors > self.max_parse_retries:
                    output.status = "invalid_input"
                    output.message = parsed.get("error_message") or "Parse failed."
                    return output
                output.repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_role_output_repair_message(
                            parsed.get("error_message") or "Unknown format."
                        ),
                    }
                )
                continue

            consecutive_parse_errors = 0

            if parsed["type"] == "tool_call":
                tool_name = parsed["tool_name"]
                arguments = parsed["arguments"]
                role_step["tool_name"] = tool_name
                role_step["arguments"] = arguments

                if tool_name not in self.client.list_tool_names():
                    output.invalid_tool_name_count += 1
                    role_step["tool_status"] = "invalid_tool_name"
                    output.role_steps.append(role_step)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Tool {tool_name!r} is not available. Return a "
                                "valid role output using only the allowed tools."
                            ),
                        }
                    )
                    continue

                if tool_name not in ROLE_TOOL_ALLOWLIST[self.role]:
                    output.forbidden_tool_call_count += 1
                    role_step["tool_status"] = "forbidden_tool"
                    output.role_steps.append(role_step)
                    messages.append(
                        {
                            "role": "user",
                            "content": build_forbidden_tool_message(self.role, tool_name),
                        }
                    )
                    continue

                key = tool_call_key(tool_name, arguments)
                successful_keys = context.setdefault("successful_tool_call_keys", set())
                if key in successful_keys:
                    output.repeated_tool_call_count += 1
                    role_step["tool_status"] = "skipped_repeated"
                    output.role_steps.append(role_step)
                    if output.repeated_tool_call_count >= 2:
                        output.stalled_loop_detected = True
                        output.status = "tool_error"
                        output.message = (
                            "Repeated identical successful tool call caused a stall."
                        )
                        return output
                    messages.append(
                        {
                            "role": "user",
                            "content": build_repeated_tool_message(
                                tool_name,
                                arguments,
                                context,
                            ),
                        }
                    )
                    continue

                if tool_call_count >= self.max_tool_calls:
                    output.status = "tool_error"
                    output.message = f"Exceeded max_tool_calls={self.max_tool_calls}."
                    output.role_steps.append(role_step)
                    return output

                tool_call_count += 1
                response = self.client.call_tool(
                    tool_name,
                    arguments,
                    agent_name=self.role,
                    step=tool_call_count,
                )
                response_dict = response.to_dict()
                role_step["tool_status"] = response.status
                role_step["tool_result"] = response_dict
                output.role_steps.append(role_step)
                if response.status == "ok" and response.result is not None:
                    successful_keys.add(key)
                    output.tool_sequence.append(tool_name)
                    record_successful_tool_call(
                        context=context,
                        role=self.role,
                        tool_name=tool_name,
                        arguments=arguments,
                        tool_response=response_dict,
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": build_tool_observation_for_role(
                            role=self.role,
                            tool_name=tool_name,
                            tool_response=response_dict,
                            context=context,
                        ),
                    }
                )
                continue

            if parsed["type"] == "role_response":
                output.status = parsed["status"]
                output.message = parsed["message"]
                output.artifacts = parsed["artifacts"]
                output.findings = parsed["findings"]
                output.role_steps.append(role_step)
                merge_role_response(context, self.role, parsed)
                return output

            if parsed["type"] == "final_answer":
                role_step["final_answer"] = parsed["final_answer"]
                output.role_steps.append(role_step)
                if self.role != "report_agent":
                    output.status = "invalid_input"
                    output.message = "Only report_agent may emit final_answer."
                    return output
                output.status = "final"
                output.final_answer = parsed["final_answer"]
                output.message = "Final answer produced."
                context["final_answer"] = parsed["final_answer"]
                return output

        output.status = "tool_error"
        output.message = f"Exceeded max_role_steps={self.max_role_steps}."
        output.stalled_loop_detected = True
        return output


class LLMDataAgent(LLMRoleAgent):
    """LLM-driven DataAgent."""

    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("data_agent", model, client, **kwargs)


class LLMMathAgent(LLMRoleAgent):
    """LLM-driven MathAgent."""

    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("math_agent", model, client, **kwargs)


class LLMAnalysisAgent(LLMRoleAgent):
    """LLM-driven AnalysisAgent."""

    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("analysis_agent", model, client, **kwargs)


class LLMReportAgent(LLMRoleAgent):
    """LLM-driven ReportAgent."""

    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("report_agent", model, client, **kwargs)


class LLMFullMultiAgent:
    """Full LLM multi-agent runner with LLM-driven orchestrator and roles."""

    def __init__(
        self,
        model,
        client: LocalMCPClient,
        *,
        max_orchestration_steps: int = 12,
        max_role_steps: int = 8,
        max_tool_calls_per_role: int = 8,
        max_parse_retries: int = 2,
        max_tool_retries: int = 2,
        temperature: float = 0.0,
        state_feedback_mode: str = "state_aware",
    ) -> None:
        self.model = model
        self.client = client
        self.max_orchestration_steps = max_orchestration_steps
        self.max_role_steps = max_role_steps
        self.max_tool_calls_per_role = max_tool_calls_per_role
        self.max_parse_retries = max_parse_retries
        self.max_tool_retries = max_tool_retries
        self.temperature = temperature
        self.state_feedback_mode = state_feedback_mode
        self.orchestrator = LLMOrchestratorAgent(
            model,
            temperature=temperature,
        )
        role_kwargs = {
            "max_role_steps": max_role_steps,
            "max_tool_calls": max_tool_calls_per_role,
            "max_parse_retries": max_parse_retries,
            "temperature": temperature,
        }
        self.roles = {
            "data_agent": LLMDataAgent(model, client, **role_kwargs),
            "math_agent": LLMMathAgent(model, client, **role_kwargs),
            "analysis_agent": LLMAnalysisAgent(model, client, **role_kwargs),
            "report_agent": LLMReportAgent(model, client, **role_kwargs),
        }

    def reset(self) -> None:
        """Reset underlying tool state."""

        reset_fn = getattr(self.client, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def run(self, user_query: str) -> LLMMultiAgentResult:
        """Run the full LLM-driven multi-agent network."""

        parsed_task = public_task_state(infer_task_state(user_query))
        context = initial_multi_agent_context(user_query, parsed_task)
        orchestration_steps: list[dict[str, Any]] = []
        role_outputs: list[dict[str, Any]] = []
        orchestrator_decisions: list[dict[str, Any]] = []
        raw_model_outputs: list[str] = []
        cleaned_model_outputs: list[str] = []
        counters = {
            "parse_error_count": 0,
            "invalid_json_count": 0,
            "unknown_format_count": 0,
            "invalid_role_action_count": 0,
            "invalid_tool_name_count": 0,
            "forbidden_tool_call_count": 0,
            "repeated_tool_call_count": 0,
            "repair_attempt_count": 0,
            "retry_count": 0,
            "recovery_attempt_count": 0,
            "recovery_success_count": 0,
            "recovery_failure_count": 0,
        }
        stalled_loop_detected = False
        first_orchestrator_step: dict[str, Any] | None = None

        for orchestration_index in range(1, self.max_orchestration_steps + 1):
            action, diagnostics = self.orchestrator.decide(
                user_query=user_query,
                context=context,
                max_parse_retries=self.max_parse_retries,
            )
            accumulate_orchestrator_diagnostics(counters, diagnostics)
            raw_model_outputs.extend(diagnostics["raw_model_outputs"])
            cleaned_model_outputs.extend(diagnostics["cleaned_model_outputs"])

            if action is None:
                counters["invalid_role_action_count"] += 1
                counters["recovery_failure_count"] += 1
                return self._result(
                    user_query=user_query,
                    parsed_task=parsed_task,
                    context=context,
                    orchestration_steps=orchestration_steps,
                    role_outputs=role_outputs,
                    orchestrator_decisions=orchestrator_decisions,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    counters=counters,
                    stalled_loop_detected=True,
                    success=False,
                    error_message="Orchestrator could not produce a valid role_action.",
                )

            orchestrator_decisions.append(action)
            if first_orchestrator_step is None:
                first_orchestrator_step = make_orchestration_step(
                    node="orchestrator",
                    action="llm_decide_role",
                    status="ok",
                    details={
                        "worker": "role_based_workflow",
                        "selected_worker": "role_based_workflow",
                        "llm_driven": True,
                        "decisions": orchestrator_decisions,
                        "agent_response": {
                            "status": "ok",
                            "agent": "orchestrator",
                            "artifacts": {},
                            "message": "LLM orchestrator produced role action.",
                            "can_continue": True,
                            "requires_retry": False,
                            "missing_artifacts": [],
                            "attempt": 1,
                            "max_attempts": self.max_parse_retries + 1,
                        },
                    },
                    decision="continue",
                )
                orchestration_steps.append(first_orchestrator_step)
            else:
                first_orchestrator_step["details"]["decisions"] = orchestrator_decisions

            if action["action"] == "finish":
                answer = str(context.get("final_answer") or "")
                success = bool(answer)
                return self._result(
                    user_query=user_query,
                    parsed_task=parsed_task,
                    context=context,
                    orchestration_steps=orchestration_steps,
                    role_outputs=role_outputs,
                    orchestrator_decisions=orchestrator_decisions,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    counters=counters,
                    stalled_loop_detected=stalled_loop_detected,
                    success=success,
                    error_message=None if success else "Orchestrator finished without final answer.",
                )

            role = str(action["role"])
            role_message = str(action.get("message") or "")
            context.setdefault("role_handoffs", []).append(
                {
                    "step": orchestration_index,
                    "role": role,
                    "message": role_message,
                }
            )
            role_output = self.roles[role].run(
                user_query=user_query,
                role_message=role_message,
                context=context,
            )
            role_output_dict = asdict(role_output)
            role_outputs.append(role_output_dict)
            raw_model_outputs.extend(role_output.raw_model_outputs)
            cleaned_model_outputs.extend(role_output.cleaned_model_outputs)
            accumulate_role_diagnostics(counters, role_output)
            stalled_loop_detected = (
                stalled_loop_detected or role_output.stalled_loop_detected
            )
            context.setdefault("role_agent_order", []).append(role)
            context.setdefault("role_outputs", []).append(
                {
                    "role": role,
                    "status": role_output.status,
                    "message": role_output.message,
                    "artifacts": role_output.artifacts,
                    "findings": role_output.findings,
                }
            )
            orchestration_steps.append(
                make_orchestration_step(
                    node=role,
                    action="llm_role_run",
                    status=role_output.status,
                    details={
                        "llm_driven": True,
                        "orchestrator_decision": action,
                        "agent_response": {
                            "status": role_output.status,
                            "agent": role,
                            "artifacts": role_output.artifacts,
                            "message": role_output.message,
                            "can_continue": role_output.status
                            in {"ok", "partial", "final"},
                            "requires_retry": role_output.status
                            in {"tool_error", "invalid_input"},
                            "missing_artifacts": [],
                            "attempt": 1,
                            "max_attempts": self.max_parse_retries + 1,
                        },
                    },
                    decision=(
                        "continue"
                        if role_output.status in {"ok", "partial", "final"}
                        else "fail"
                    ),
                )
            )

            if role_output.status not in {"ok", "partial", "final"}:
                counters["recovery_failure_count"] += 1
                return self._result(
                    user_query=user_query,
                    parsed_task=parsed_task,
                    context=context,
                    orchestration_steps=orchestration_steps,
                    role_outputs=role_outputs,
                    orchestrator_decisions=orchestrator_decisions,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    counters=counters,
                    stalled_loop_detected=stalled_loop_detected,
                    success=False,
                    error_message=f"{role} failed: {role_output.message}",
                )

        return self._result(
            user_query=user_query,
            parsed_task=parsed_task,
            context=context,
            orchestration_steps=orchestration_steps,
            role_outputs=role_outputs,
            orchestrator_decisions=orchestrator_decisions,
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            counters=counters,
            stalled_loop_detected=True,
            success=False,
            error_message=(
                f"Exceeded max_orchestration_steps={self.max_orchestration_steps}."
            ),
        )

    def _result(
        self,
        *,
        user_query: str,
        parsed_task: dict[str, Any],
        context: dict[str, Any],
        orchestration_steps: list[dict[str, Any]],
        role_outputs: list[dict[str, Any]],
        orchestrator_decisions: list[dict[str, Any]],
        raw_model_outputs: list[str],
        cleaned_model_outputs: list[str],
        counters: dict[str, int],
        stalled_loop_detected: bool,
        success: bool,
        error_message: str | None,
    ) -> LLMMultiAgentResult:
        tool_results = build_multi_agent_tool_results(context)
        answer = str(context.get("final_answer") or "")
        trace = self.client.get_trace()
        return LLMMultiAgentResult(
            answer=answer,
            parsed_task=parsed_task,
            tool_results=tool_results,
            trace=trace,
            orchestration_steps=orchestration_steps,
            role_outputs=role_outputs,
            orchestrator_decisions=orchestrator_decisions,
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            role_agent_order=list(context.get("role_agent_order") or []),
            actual_tool_sequence=[
                str(call.get("tool_name")) for call in trace.get("calls", [])
            ],
            parse_error_count=counters["parse_error_count"],
            invalid_json_count=counters["invalid_json_count"],
            unknown_format_count=counters["unknown_format_count"],
            invalid_role_action_count=counters["invalid_role_action_count"],
            invalid_tool_name_count=counters["invalid_tool_name_count"],
            forbidden_tool_call_count=counters["forbidden_tool_call_count"],
            repeated_tool_call_count=counters["repeated_tool_call_count"],
            stalled_loop_detected=stalled_loop_detected,
            repair_attempt_count=counters["repair_attempt_count"],
            retry_count=counters["retry_count"],
            recovery_attempt_count=counters["recovery_attempt_count"],
            recovery_success_count=counters["recovery_success_count"],
            recovery_failure_count=counters["recovery_failure_count"],
            success=success,
            error_message=error_message,
        )


def build_orchestrator_prompt() -> str:
    """Return the full-LLM orchestrator prompt without evaluator hints."""

    return f"""
{BASE_DOMAIN_CONTEXT}

LLM OrchestratorAgent:
- You coordinate LLM role agents and do not call tools.
- You do not compute TEC values and do not write the final answer yourself.
- Available roles:
  data_agent: retrieves time-series data artifacts.
  math_agent: computes numerical artifacts from available handles.
  analysis_agent: writes structured findings from artifacts.
  report_agent: writes the final answer from artifacts and findings.
- Use only the user query, role outputs, handoff history, and available
  artifacts shown in the current state.
- Avoid repeated handoffs that do not add information.

Return exactly one block:
<role_action>
{{"action": "call_role", "role": "data_agent|math_agent|analysis_agent|report_agent", "message": "brief instruction for that role"}}
</role_action>

or:
<role_action>
{{"action": "finish", "reason": "why the workflow is complete"}}
</role_action>

No markdown or explanation outside the tags.
""".strip()


def build_role_prompt(role: str) -> str:
    """Return role-specific prompt for the full LLM multi-agent network."""

    common = f"""
{BASE_DOMAIN_CONTEXT}

Shared role protocol:
- Use only the user query, role message, available artifacts, completed calls,
  and role outputs shown to you.
- Do not invent numerical values.
- You may use the same tool name multiple times when arguments differ and the
  task requires it, such as separate regions or separate series handles.
- Do not repeat an identical successful tool call with the same tool name and
  the same arguments. Reuse the returned artifact id when relevant.
- Do not use tec_build_report.
- Do not use aggregate tec_compare_regions for primitive comparisons.
""".strip()

    if role == "data_agent":
        details = """
LLM DataAgent:
- Retrieve data artifacts only.
- Allowed tools:
  tec_get_timeseries
  tec_series_profile
- Do not compute statistics, thresholds, intervals, comparisons, analysis, or
  report text.
- Use dataset_ref="default" unless the user or state specifies another value.
- Dates use [start, end), with end exclusive.
""".strip()
    elif role == "math_agent":
        details = """
LLM MathAgent:
- Compute numerical artifacts from available handles.
- Allowed tools:
  tec_compute_series_stats
  tec_compare_stats
  tec_compute_high_threshold
  tec_detect_high_intervals
  tec_compute_stability_thresholds
  tec_detect_stable_intervals
- Do not load data.
- Do not write the final user-facing answer.
""".strip()
    elif role == "analysis_agent":
        details = """
LLM AnalysisAgent:
- No tools are allowed.
- Use available data and math artifacts to produce concise structured findings.
- Do not write the final user-facing answer.
""".strip()
    elif role == "report_agent":
        details = """
LLM ReportAgent:
- No tools are allowed.
- Produce the final user-facing answer using only available artifacts and
  findings.
- If limitations remain, state them plainly without fabricating numbers.
""".strip()
    else:
        raise ValueError(f"Unknown role: {role!r}")

    protocol = """
Return exactly one of:
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>

<role_response>
{"status": "ok", "message": "brief summary", "artifacts": {}, "findings": []}
</role_response>

<final_answer>
Final user-facing answer here. Only report_agent may use this.
</final_answer>

No markdown or explanation outside the tags.
""".strip()
    return "\n\n".join([common, details, protocol])


def build_orchestrator_state_message(user_query: str, context: dict[str, Any]) -> str:
    """Build orchestrator-visible state without evaluator expectations."""

    return "\n".join(
        [
            f"User query:\n{user_query}",
            "",
            "Available artifacts:",
            safe_compact_json(context.get("available_artifacts") or {}),
            "",
            "Role handoff history:",
            safe_compact_json(context.get("role_handoffs") or []),
            "",
            "Previous role outputs:",
            safe_compact_json(context.get("role_outputs") or []),
            "",
            "Final answer present:",
            "yes" if context.get("final_answer") else "no",
        ]
    )


def build_role_input_message(
    *,
    role: str,
    user_query: str,
    role_message: str,
    context: dict[str, Any],
) -> str:
    """Build role-visible state without evaluator expectations."""

    return "\n".join(
        [
            f"Role: {role}",
            f"User query:\n{user_query}",
            "",
            f"Message from orchestrator:\n{role_message}",
            "",
            "Available artifacts:",
            safe_compact_json(context.get("available_artifacts") or {}),
            "",
            "Completed successful tool calls:",
            safe_compact_json(context.get("completed_tool_calls") or []),
            "",
            "Previous role outputs:",
            safe_compact_json(context.get("role_outputs") or []),
        ]
    )


def clean_multi_agent_output(raw_text: str) -> str:
    """Return the first complete protocol block from model output."""

    candidates: list[tuple[int, int]] = []
    for tag in ["role_action", "role_response", "tool_call", "final_answer"]:
        block = _first_closed_tag_block(raw_text, tag)
        if block is not None:
            candidates.append(block)
    if not candidates:
        return ""
    start, end = min(candidates, key=lambda item: item[0])
    return raw_text[start:end].strip()


def parse_role_action(text: str) -> tuple[dict[str, Any] | None, ParseErrorCode | None, str | None]:
    """Parse an orchestrator role_action block."""

    match = ROLE_ACTION_RE.search(text)
    if not match:
        return None, ParseErrorCode.UNKNOWN_FORMAT, "No role_action block found."
    try:
        data = json.loads(match.group(1).strip())
    except Exception as exc:
        return None, ParseErrorCode.INVALID_JSON, f"Invalid role_action JSON: {exc}"
    if not isinstance(data, dict):
        return None, ParseErrorCode.INVALID_JSON, "role_action JSON must be an object."
    return data, None, None


def validate_role_action(action: dict[str, Any], *, context: dict[str, Any]) -> str | None:
    """Return validation error for a role action, or None."""

    action_name = action.get("action")
    if action_name == "call_role":
        role = action.get("role")
        if role not in ROLE_NAMES:
            return f"Unknown role {role!r}."
        return None
    if action_name == "finish":
        if not context.get("final_answer"):
            return "Cannot finish before a report_agent final_answer exists."
        return None
    return "role_action.action must be 'call_role' or 'finish'."


def parse_role_output(text: str) -> dict[str, Any]:
    """Parse worker role output."""

    final_answer = parse_final_answer(text)
    if final_answer is not None:
        return {"type": "final_answer", "final_answer": final_answer}

    role_response_match = ROLE_RESPONSE_RE.search(text)
    if role_response_match:
        block = role_response_match.group(1).strip()
        try:
            data = json.loads(block)
        except Exception as exc:
            return {
                "type": "parse_error",
                "error_code": ParseErrorCode.INVALID_JSON.value,
                "error_message": f"Invalid role_response JSON: {exc}",
            }
        if not isinstance(data, dict):
            return {
                "type": "parse_error",
                "error_code": ParseErrorCode.INVALID_JSON.value,
                "error_message": "role_response JSON must be an object.",
            }
        status = str(data.get("status") or "ok")
        if status not in AGENT_RESPONSE_STATUS:
            status = "ok"
        findings = data.get("findings") or []
        if not isinstance(findings, list):
            findings = []
        artifacts = data.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        return {
            "type": "role_response",
            "status": status,
            "message": str(data.get("message") or data.get("summary") or ""),
            "artifacts": artifacts,
            "findings": findings,
        }

    if "<tool_call" in text.lower():
        parsed = parse_tool_call(text)
        if parsed.tool_call is not None:
            return {
                "type": "tool_call",
                "tool_name": parsed.tool_call.name,
                "arguments": parsed.tool_call.arguments,
            }
        return {
            "type": "parse_error",
            "error_code": (
                parsed.error_code.value
                if parsed.error_code is not None
                else ParseErrorCode.UNKNOWN_FORMAT.value
            ),
            "error_message": parsed.error_message,
        }

    return {
        "type": "parse_error",
        "error_code": ParseErrorCode.UNKNOWN_FORMAT.value,
        "error_message": "No supported protocol block found.",
    }


def build_role_action_repair_message(error: str) -> str:
    """Return an orchestrator repair message."""

    return f"""
Your previous orchestrator output was invalid: {error}

Return exactly one valid block:
<role_action>
{{"action": "call_role", "role": "data_agent", "message": "brief message"}}
</role_action>

or:
<role_action>
{{"action": "finish", "reason": "final answer is available"}}
</role_action>
No markdown or explanation outside the tags.
""".strip()


def build_role_output_repair_message(error: str) -> str:
    """Return a worker role repair message."""

    return f"""
Your previous role output was invalid: {error}

Return exactly one valid <tool_call>, <role_response>, or <final_answer> block.
Use valid JSON where JSON is required. No markdown outside the tags.
""".strip()


def build_forbidden_tool_message(role: str, tool_name: str) -> str:
    """Return a role tool-permission correction."""

    allowed = sorted(ROLE_TOOL_ALLOWLIST[role])
    return "\n".join(
        [
            f"Tool {tool_name!r} is forbidden for {role}.",
            f"Allowed tools for this role: {allowed or '<none>'}.",
            "Return a valid role output using only permitted actions.",
        ]
    )


def build_repeated_tool_message(
    tool_name: str,
    arguments: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Return a duplicate-call correction without task-plan hints."""

    return "\n".join(
        [
            "The previous tool call was rejected because the identical call already succeeded.",
            f"Tool: {tool_name}",
            f"Arguments: {safe_compact_json(arguments)}",
            "",
            "Available artifacts:",
            safe_compact_json(context.get("available_artifacts") or {}),
            "",
            "Use an artifact id from the successful call when it is relevant, or choose another permitted action.",
        ]
    )


def build_tool_observation_for_role(
    *,
    role: str,
    tool_name: str,
    tool_response: dict[str, Any],
    context: dict[str, Any],
) -> str:
    """Build a role-visible tool observation without missing-goal hints."""

    observation = {
        "tool_name": tool_name,
        "status": tool_response.get("status"),
        "returned_artifacts": returned_artifacts(tool_response.get("result") or {}),
    }
    return "\n".join(
        [
            "<tool_result>",
            safe_compact_json(observation),
            "</tool_result>",
            "",
            "Available artifacts:",
            safe_compact_json(context.get("available_artifacts") or {}),
            "",
            "Completed successful tool calls:",
            safe_compact_json(context.get("completed_tool_calls") or []),
            "",
            "Continue using only the protocol block for your role.",
        ]
    )


def initial_multi_agent_context(
    user_query: str,
    parsed_task: dict[str, Any],
) -> dict[str, Any]:
    """Build internal execution context."""

    return {
        "user_query": user_query,
        "parsed_task": parsed_task,
        "data_artifacts": {
            "series_by_region": {},
            "regions": [],
            "dataset_ref": parsed_task.get("dataset_ref") or "default",
            "start": parsed_task.get("start"),
            "end": parsed_task.get("end"),
        },
        "math_artifacts": {},
        "analysis_artifacts": {"findings": []},
        "available_artifacts": {},
        "completed_tool_calls": [],
        "successful_tool_call_keys": set(),
        "role_handoffs": [],
        "role_outputs": [],
        "role_agent_order": [],
        "final_answer": None,
    }


def public_task_state(task_state: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable parsed-task view without internal missing-goal data."""

    return {
        "task_type": task_state.get("task_type"),
        "dataset_ref": task_state.get("dataset_ref"),
        "region_id": task_state.get("region_id"),
        "region_ids": list(task_state.get("region_ids") or []),
        "regions": list(task_state.get("region_ids") or []),
        "start": task_state.get("start"),
        "end": task_state.get("end"),
        "q": task_state.get("q"),
        "raw_query": task_state.get("raw_query"),
    }


def record_successful_tool_call(
    *,
    context: dict[str, Any],
    role: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_response: dict[str, Any],
) -> None:
    """Record a successful tool call and update artifacts."""

    result = tool_response.get("result") or {}
    record = {
        "role": role,
        "tool_name": tool_name,
        "arguments": arguments,
        "returned_artifacts": returned_artifacts(result),
    }
    context.setdefault("completed_tool_calls", []).append(record)
    update_artifacts_from_tool(
        context=context,
        role=role,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
    )


def update_artifacts_from_tool(
    *,
    context: dict[str, Any],
    role: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Update internal artifact stores from a successful primitive tool result."""

    available = context.setdefault("available_artifacts", {})
    data = context.setdefault("data_artifacts", {})
    math = context.setdefault("math_artifacts", {})
    parsed = context.get("parsed_task") or {}
    task_type = parsed.get("task_type")

    if tool_name == "tec_get_timeseries":
        metadata = result.get("metadata") or {}
        region_id = (
            result.get("region_id")
            or metadata.get("region_id")
            or arguments.get("region_id")
            or "unknown"
        )
        data.setdefault("series_by_region", {})[region_id] = {
            "series_id": result.get("series_id"),
            "tool_result": result,
            "metadata": metadata,
        }
        regions = data.setdefault("regions", [])
        if region_id not in regions:
            regions.append(region_id)
        data["dataset_ref"] = metadata.get("dataset_ref") or arguments.get("dataset_ref")
        data["start"] = metadata.get("requested_start") or arguments.get("start")
        data["end"] = metadata.get("requested_end") or arguments.get("end")
        available.setdefault("series_by_region", {})[region_id] = result.get("series_id")
        available["series_id"] = result.get("series_id")
        available["n_points"] = metadata.get("n_points")
        return

    if tool_name == "tec_compute_series_stats":
        region_id = result.get("region_id") or region_for_series(
            context,
            str(result.get("series_id") or arguments.get("series_id")),
        )
        item = {
            "stats_id": result.get("stats_id"),
            "series_id": result.get("series_id") or arguments.get("series_id"),
            "stats": result,
            "metrics": result.get("metrics", {}),
        }
        math.setdefault("stats_by_region", {})[region_id] = item
        available.setdefault("stats_by_region", {})[region_id] = result.get("stats_id")
        if task_type == "report":
            report_inputs = math.setdefault("report_inputs", {})
            basic = report_inputs.setdefault("basic_stats", {"by_region": {}})
            basic.setdefault("by_region", {})[region_id] = item
        return

    if tool_name == "tec_compare_stats":
        math["comparison"] = result
        available["comparison_id"] = result.get("comparison_id")
        report_inputs = math.get("report_inputs")
        if isinstance(report_inputs, dict):
            basic = report_inputs.setdefault("basic_stats", {"by_region": {}})
            basic["comparison"] = result
        return

    if tool_name == "tec_compute_high_threshold":
        region_id = region_for_series(context, str(arguments.get("series_id")))
        item = math.setdefault("high_tec", {}).setdefault(region_id, {})
        item["threshold"] = result
        available.setdefault("high_threshold_by_region", {})[region_id] = result.get(
            "threshold_id"
        )
        if task_type == "report":
            report_inputs = math.setdefault("report_inputs", {})
            high = report_inputs.setdefault("high_tec", {"by_region": {}})
            high.setdefault("by_region", {}).setdefault(region_id, {})[
                "threshold"
            ] = result
        return

    if tool_name == "tec_detect_high_intervals":
        region_id = region_for_series(context, str(arguments.get("series_id")))
        item = math.setdefault("high_tec", {}).setdefault(region_id, {})
        item["intervals"] = result
        available.setdefault("high_intervals_by_region", {})[region_id] = result.get(
            "n_intervals"
        )
        if task_type == "report":
            report_inputs = math.setdefault("report_inputs", {})
            high = report_inputs.setdefault("high_tec", {"by_region": {}})
            high.setdefault("by_region", {}).setdefault(region_id, {})[
                "intervals"
            ] = result
        return

    if tool_name == "tec_compute_stability_thresholds":
        region_id = region_for_series(context, str(arguments.get("series_id")))
        item = math.setdefault("stable_intervals", {}).setdefault(region_id, {})
        item["thresholds"] = result
        available.setdefault("stable_threshold_by_region", {})[region_id] = result.get(
            "threshold_id"
        )
        if task_type == "report":
            report_inputs = math.setdefault("report_inputs", {})
            stable = report_inputs.setdefault("stable_intervals", {"by_region": {}})
            stable.setdefault("by_region", {}).setdefault(region_id, {})[
                "thresholds"
            ] = result
        return

    if tool_name == "tec_detect_stable_intervals":
        region_id = region_for_series(context, str(arguments.get("series_id")))
        item = math.setdefault("stable_intervals", {}).setdefault(region_id, {})
        item["intervals"] = result
        available.setdefault("stable_intervals_by_region", {})[region_id] = result.get(
            "n_intervals"
        )
        if task_type == "report":
            report_inputs = math.setdefault("report_inputs", {})
            stable = report_inputs.setdefault("stable_intervals", {"by_region": {}})
            stable.setdefault("by_region", {}).setdefault(region_id, {})[
                "intervals"
            ] = result


def merge_role_response(
    context: dict[str, Any],
    role: str,
    parsed: dict[str, Any],
) -> None:
    """Merge role_response artifacts/findings into context."""

    artifacts = parsed.get("artifacts") or {}
    if role == "data_agent" and artifacts:
        context.setdefault("data_artifacts", {}).update(artifacts)
    elif role == "math_agent" and artifacts:
        context.setdefault("math_artifacts", {}).update(artifacts)
    elif role == "analysis_agent":
        findings = parsed.get("findings") or []
        context.setdefault("analysis_artifacts", {}).setdefault("findings", []).extend(
            findings
        )
        if artifacts:
            context.setdefault("analysis_artifacts", {}).update(artifacts)
    elif role == "report_agent" and artifacts:
        context.setdefault("report_artifacts", {}).update(artifacts)


def build_multi_agent_tool_results(context: dict[str, Any]) -> dict[str, Any]:
    """Return role-shaped tool_results compatible with existing metrics."""

    result = {
        "data": context.get("data_artifacts") or {},
        "math": context.get("math_artifacts") or {},
        "analysis": context.get("analysis_artifacts") or {},
        "answer": context.get("final_answer") or "",
        "final_answer": context.get("final_answer") or "",
    }
    parsed = context.get("parsed_task") or {}
    if parsed.get("task_type") == "report":
        data = result["data"]
        regions = data.get("regions") or list((data.get("series_by_region") or {}))
        result["regions"] = list(regions)
        result["region_ids"] = list(regions)
    return result


def returned_artifacts(result: dict[str, Any]) -> dict[str, Any]:
    """Extract compact artifact identifiers from a tool result."""

    artifacts: dict[str, Any] = {}
    for key in [
        "series_id",
        "stats_id",
        "threshold_id",
        "comparison_id",
        "region_id",
        "stats_ids",
        "n_intervals",
        "value",
    ]:
        if key in result:
            artifacts[key] = result[key]
    metadata = result.get("metadata") or {}
    if metadata.get("region_id") is not None:
        artifacts["region_id"] = metadata["region_id"]
    if metadata.get("n_points") is not None:
        artifacts["n_points"] = metadata["n_points"]
    return artifacts


def region_for_series(context: dict[str, Any], series_id: str) -> str:
    """Find region_id for a series_id."""

    series_by_region = (
        context.get("data_artifacts", {}).get("series_by_region") or {}
    )
    for region_id, item in series_by_region.items():
        if str(item.get("series_id")) == str(series_id):
            return str(region_id)
    return "unknown"


def make_orchestration_step(
    *,
    node: str,
    action: str,
    status: str,
    details: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    """Build a metrics-compatible orchestration step."""

    return {
        "node": node,
        "action": action,
        "details": details,
        "status": status,
        "missing_artifacts": [],
        "requested_next_action": None,
        "can_continue": status in {"ok", "partial", "final"},
        "requires_retry": status in {"tool_error", "invalid_input"},
        "attempt": 1,
        "max_attempts": 3,
        "decision": decision,
    }


def accumulate_orchestrator_diagnostics(
    counters: dict[str, int],
    diagnostics: dict[str, Any],
) -> None:
    """Add orchestrator diagnostics into global counters."""

    for key in [
        "parse_error_count",
        "invalid_json_count",
        "unknown_format_count",
        "invalid_role_action_count",
        "repair_attempt_count",
    ]:
        counters[key] += int(diagnostics.get(key) or 0)


def accumulate_role_diagnostics(
    counters: dict[str, int],
    role_output: LLMRoleOutput,
) -> None:
    """Add role diagnostics into global counters."""

    counters["parse_error_count"] += role_output.parse_error_count
    counters["invalid_json_count"] += role_output.invalid_json_count
    counters["unknown_format_count"] += role_output.unknown_format_count
    counters["invalid_tool_name_count"] += role_output.invalid_tool_name_count
    counters["forbidden_tool_call_count"] += role_output.forbidden_tool_call_count
    counters["repeated_tool_call_count"] += role_output.repeated_tool_call_count
    counters["repair_attempt_count"] += role_output.repair_attempt_count


def tool_call_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return canonical key for duplicate successful tool call detection."""

    return f"{tool_name}:{canonical_arguments_json(arguments)}"


def _first_closed_tag_block(text: str, tag: str) -> tuple[int, int] | None:
    pattern = re.compile(
        rf"<{tag}\b[^>]*>.*?</{tag}>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return match.start(), match.end()


__all__ = [
    "LLMAnalysisAgent",
    "LLMDataAgent",
    "LLMFullMultiAgent",
    "LLMMathAgent",
    "LLMMultiAgentResult",
    "LLMOrchestratorAgent",
    "LLMReportAgent",
    "ROLE_NAMES",
    "ROLE_TOOL_ALLOWLIST",
    "build_orchestrator_prompt",
    "build_role_prompt",
    "clean_multi_agent_output",
    "parse_role_action",
    "parse_role_output",
]
