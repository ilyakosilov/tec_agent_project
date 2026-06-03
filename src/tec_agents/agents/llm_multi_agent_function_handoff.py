"""Simplified function-handoff full LLM multi-agent TEC workflow.

This experiment is separate from the historical untyped and typed runners. It
keeps all role choices LLM-driven, but makes role handoffs look like ordinary
function calls in the same <tool_call> protocol used for TEC tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tec_agents.agents.llm_multi_agent import (
    build_multi_agent_tool_results,
    initial_multi_agent_context,
    record_successful_tool_call,
    tool_call_key,
)
from tec_agents.agents.llm_multi_agent_function_handoff_prompts import (
    ARCHITECTURE_NAME,
    PROMPT_REVISION,
    build_function_handoff_orchestrator_prompt,
    build_function_handoff_repair_message,
    build_function_handoff_role_prompt,
    build_function_handoff_state_message,
)
from tec_agents.agents.llm_multi_agent_function_handoff_protocol import (
    FUNCTION_HANDOFF_PROTOCOL_VERSION,
    FunctionHandoffCall,
    RETURN_FUNCTION,
    clean_function_handoff_output,
    is_orchestrator_handoff,
    is_return_to_orchestrator,
    is_tec_tool,
    normalize_role_return,
    parse_function_handoff_output,
    role_for_handoff_function,
    tool_argument_contract,
    validate_function_call_for_role,
    validate_handoff_arguments,
    validate_role_return,
)
from tec_agents.agents.llm_single_agent import infer_task_state
from tec_agents.mcp.client import LocalMCPClient


@dataclass
class FunctionHandoffRoleOutput:
    role: str
    status: str = "failed"
    message: str = ""
    findings: list[str] = field(default_factory=list)
    final_answer: str = ""
    raw_model_outputs: list[str] = field(default_factory=list)
    cleaned_model_outputs: list[str] = field(default_factory=list)
    role_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    parse_error_count: int = 0
    invalid_json_count: int = 0
    invalid_function_name_count: int = 0
    forbidden_function_call_count: int = 0
    repeated_tool_call_count: int = 0
    multiple_function_blocks_in_single_output_count: int = 0
    tool_error_count: int = 0
    repair_attempt_count: int = 0
    stalled_loop_detected: bool = False


@dataclass
class LLMFunctionHandoffMultiAgentResult:
    answer: str
    parsed_task: dict[str, Any] | None
    tool_results: dict[str, Any]
    trace: dict[str, Any]
    orchestration_steps: list[dict[str, Any]]
    role_outputs: list[dict[str, Any]]
    orchestrator_decisions: list[dict[str, Any]]
    function_calls: list[dict[str, Any]]
    tool_observations: list[dict[str, Any]]
    raw_model_outputs: list[str]
    cleaned_model_outputs: list[str]
    role_agent_order: list[str]
    actual_tool_sequence: list[str]
    function_handoff_protocol_version: str
    architecture: str
    parse_error_count: int
    invalid_json_count: int
    invalid_function_name_count: int
    forbidden_function_call_count: int
    repeated_tool_call_count: int
    multiple_function_blocks_in_single_output_count: int
    tool_error_count: int
    stalled_loop_detected: bool
    repair_attempt_count: int
    retry_count: int
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMFunctionHandoffOrchestratorAgent:
    """LLM-driven orchestrator that can only call role handoff functions."""

    def __init__(self, model, *, temperature: float = 0.0, max_new_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def decide(
        self,
        *,
        user_query: str,
        context: dict[str, Any],
        max_parse_retries: int,
    ) -> tuple[FunctionHandoffCall | None, dict[str, Any]]:
        diagnostics = _empty_diagnostics()
        messages = [
            {"role": "system", "content": build_function_handoff_orchestrator_prompt()},
            {
                "role": "user",
                "content": build_function_handoff_state_message(
                    role="orchestrator",
                    user_query=user_query,
                    state_packet=build_function_handoff_state_packet(
                        user_query=user_query,
                        current_role="orchestrator",
                        current_message="",
                        context=context,
                    ),
                ),
            },
        ]
        for _ in range(max_parse_retries + 1):
            raw = self.model.generate(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            parsed = parse_function_handoff_output(raw)
            diagnostics["raw_model_outputs"].append(raw)
            diagnostics["cleaned_model_outputs"].append(parsed.cleaned_text)
            if parsed.block_count > 1:
                diagnostics["multiple_function_blocks_in_single_output_count"] += 1
            if parsed.ok and parsed.call is not None:
                error = validate_function_call_for_role("orchestrator", parsed.call.name)
                if not error:
                    error = validate_handoff_arguments(parsed.call.arguments)
                if error is None:
                    return parsed.call, diagnostics
                diagnostics["invalid_function_name_count"] += 1
                diagnostics["forbidden_function_call_count"] += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_function_handoff_repair_message(
                            "orchestrator",
                            error,
                        ),
                    }
                )
                diagnostics["repair_attempt_count"] += 1
                continue
            _accumulate_parse_error(diagnostics, parsed.error_code)
            messages.append(
                {
                    "role": "user",
                    "content": build_function_handoff_repair_message(
                        "orchestrator",
                        parsed.error_message or "Invalid function call.",
                    ),
                }
            )
            diagnostics["repair_attempt_count"] += 1
        return None, diagnostics


class LLMFunctionHandoffRoleAgent:
    """LLM-driven worker role using TEC tools or return_to_orchestrator."""

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
        max_new_tokens: int = 512,
    ) -> None:
        self.role = role
        self.model = model
        self.client = client
        self.max_role_steps = max_role_steps
        self.max_tool_calls = max_tool_calls
        self.max_parse_retries = max_parse_retries
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def run(
        self,
        *,
        user_query: str,
        message: str,
        context: dict[str, Any],
    ) -> FunctionHandoffRoleOutput:
        output = FunctionHandoffRoleOutput(role=self.role)
        messages = [
            {"role": "system", "content": build_function_handoff_role_prompt(self.role)},
            {
                "role": "user",
                "content": build_function_handoff_state_message(
                    role=self.role,
                    user_query=user_query,
                    state_packet=build_function_handoff_state_packet(
                        user_query=user_query,
                        current_role=self.role,
                        current_message=message,
                        context=context,
                    ),
                ),
            },
        ]
        consecutive_parse_errors = 0
        tool_call_count = 0

        for step in range(1, self.max_role_steps + 1):
            raw = self.model.generate(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            parsed = parse_function_handoff_output(raw)
            output.raw_model_outputs.append(raw)
            output.cleaned_model_outputs.append(parsed.cleaned_text)
            if parsed.block_count > 1:
                output.multiple_function_blocks_in_single_output_count += 1
            role_step: dict[str, Any] = {
                "step": step,
                "raw_model_output": raw,
                "cleaned_model_output": parsed.cleaned_text,
                "parse_ok": parsed.ok,
            }
            if not parsed.ok or parsed.call is None:
                consecutive_parse_errors += 1
                _record_role_parse_error(output, parsed.error_code)
                role_step["error_code"] = parsed.error_code
                role_step["error_message"] = parsed.error_message
                output.role_steps.append(role_step)
                if consecutive_parse_errors > self.max_parse_retries:
                    output.status = "failed"
                    output.message = parsed.error_message or "Parse failed."
                    return output
                output.repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_function_handoff_repair_message(
                            self.role,
                            parsed.error_message or "Invalid function call.",
                        ),
                    }
                )
                continue

            consecutive_parse_errors = 0
            call = parsed.call
            role_step["function_name"] = call.name
            role_step["arguments"] = call.arguments
            validation_error = validate_function_call_for_role(self.role, call.name)
            if validation_error:
                output.invalid_function_name_count += 1
                output.forbidden_function_call_count += 1
                role_step["function_status"] = "forbidden"
                role_step["error_message"] = validation_error
                output.role_steps.append(role_step)
                output.repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_function_handoff_repair_message(
                            self.role,
                            validation_error,
                        ),
                    }
                )
                continue

            if is_return_to_orchestrator(call.name):
                return_error = validate_role_return(call.arguments, role=self.role)
                if return_error:
                    output.invalid_function_name_count += 1
                    role_step["function_status"] = "invalid_return"
                    role_step["error_message"] = return_error
                    output.role_steps.append(role_step)
                    output.repair_attempt_count += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": build_function_handoff_repair_message(
                                self.role,
                                return_error,
                            ),
                        }
                    )
                    continue
                normalized = normalize_role_return(call.arguments, role=self.role)
                output.status = normalized["status"]
                output.message = normalized["message"]
                output.findings = list(normalized["findings"])
                output.final_answer = str(normalized["final_answer"] or "")
                role_step["function_status"] = "returned"
                role_step["role_return"] = normalized
                output.role_steps.append(role_step)
                merge_function_role_return(context, normalized)
                return output

            if not is_tec_tool(call.name):
                output.invalid_function_name_count += 1
                role_step["function_status"] = "unknown_function"
                output.role_steps.append(role_step)
                continue

            if call.name not in self.client.list_tool_names():
                output.invalid_function_name_count += 1
                role_step["function_status"] = "unregistered_tool"
                output.role_steps.append(role_step)
                messages.append(
                    {
                        "role": "user",
                        "content": build_function_handoff_repair_message(
                            self.role,
                            f"Tool {call.name!r} is not registered.",
                        ),
                    }
                )
                output.repair_attempt_count += 1
                continue

            key = tool_call_key(call.name, call.arguments)
            successful_keys = context.setdefault("successful_tool_call_keys", set())
            if key in successful_keys:
                output.repeated_tool_call_count += 1
                role_step["function_status"] = "skipped_repeated"
                output.role_steps.append(role_step)
                if output.repeated_tool_call_count >= 2:
                    output.stalled_loop_detected = True
                    output.status = "failed"
                    output.message = "Repeated identical successful tool call caused a stall."
                    return output
                output.repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_function_handoff_repair_message(
                            self.role,
                            "This exact tool call already succeeded. Use visible artifacts or return_to_orchestrator.",
                        ),
                    }
                )
                continue

            if tool_call_count >= self.max_tool_calls:
                output.status = "failed"
                output.message = f"Exceeded max_tool_calls={self.max_tool_calls}."
                output.role_steps.append(role_step)
                return output

            tool_call_count += 1
            response = self.client.call_tool(
                call.name,
                call.arguments,
                agent_name=self.role,
                step=tool_call_count,
            )
            response_dict = response.to_dict()
            role_step["function_status"] = response.status
            role_step["tool_result"] = response_dict
            output.role_steps.append(role_step)

            observation = make_function_tool_observation(
                tool_name=call.name,
                status=response.status,
                result=response.result if isinstance(response.result, dict) else {},
                error=response.error,
            )
            context.setdefault("function_tool_observations", []).append(observation)
            output.tool_observations.append(observation)
            if response.status != "ok":
                output.tool_error_count += 1
            if response.status == "ok" and response.result is not None:
                successful_keys.add(key)
                output.tool_sequence.append(call.name)
                record_successful_tool_call(
                    context=context,
                    role=self.role,
                    tool_name=call.name,
                    arguments=call.arguments,
                    tool_response=response_dict,
                )
            messages.append(
                {
                    "role": "user",
                    "content": build_function_handoff_state_message(
                        role=self.role,
                        user_query=user_query,
                        state_packet=build_function_handoff_state_packet(
                            user_query=user_query,
                            current_role=self.role,
                            current_message=message,
                            context=context,
                        ),
                    ),
                }
            )

        output.status = "failed"
        output.message = f"Exceeded max_role_steps={self.max_role_steps}."
        return output


class LLMFunctionHandoffMultiAgent:
    """Full LLM multi-agent runner using synthetic function handoffs."""

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
        state_feedback_mode: str = "function_handoff_state",
        max_new_tokens: int = 512,
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
        self.max_new_tokens = max_new_tokens
        self.orchestrator = LLMFunctionHandoffOrchestratorAgent(
            model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def run(self, query: str) -> LLMFunctionHandoffMultiAgentResult:
        parsed_task = infer_task_state(query)
        context = initial_multi_agent_context(query, parsed_task)
        context["function_role_returns"] = []
        context["function_handoffs"] = []
        context["function_tool_observations"] = []

        counters = _empty_global_counters()
        orchestration_steps: list[dict[str, Any]] = []
        role_outputs: list[dict[str, Any]] = []
        orchestrator_decisions: list[dict[str, Any]] = []
        function_calls: list[dict[str, Any]] = []
        raw_model_outputs: list[str] = []
        cleaned_model_outputs: list[str] = []
        role_agent_order: list[str] = []
        error_message: str | None = None

        for step in range(1, self.max_orchestration_steps + 1):
            call, diagnostics = self.orchestrator.decide(
                user_query=query,
                context=context,
                max_parse_retries=self.max_parse_retries,
            )
            _accumulate_diagnostics(counters, diagnostics)
            raw_model_outputs.extend(diagnostics.get("raw_model_outputs") or [])
            cleaned_model_outputs.extend(diagnostics.get("cleaned_model_outputs") or [])
            if call is None:
                error_message = "Orchestrator could not produce a valid function handoff."
                break
            role = role_for_handoff_function(call.name)
            if role is None:
                error_message = f"Orchestrator selected invalid function {call.name!r}."
                break
            message = str(call.arguments.get("message") or "")
            decision = {
                "step": step,
                "function_name": call.name,
                "role": role,
                "message": message,
                "arguments": dict(call.arguments),
            }
            orchestrator_decisions.append(decision)
            function_calls.append({"caller": "orchestrator", **decision})
            context.setdefault("function_handoffs", []).append(decision)
            orchestration_steps.append(
                {
                    "node": "orchestrator",
                    "action": call.name,
                    "details": {
                        "worker": "function_handoff_full_llm_multi_agent",
                        "selected_worker": "function_handoff_full_llm_multi_agent",
                        "selected_role": role,
                    },
                }
            )

            role_agent = LLMFunctionHandoffRoleAgent(
                role,
                self.model,
                self.client,
                max_role_steps=self.max_role_steps,
                max_tool_calls=self.max_tool_calls_per_role,
                max_parse_retries=self.max_parse_retries,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            role_output = role_agent.run(
                user_query=query,
                message=message,
                context=context,
            )
            _accumulate_role_output(counters, role_output)
            raw_model_outputs.extend(role_output.raw_model_outputs)
            cleaned_model_outputs.extend(role_output.cleaned_model_outputs)
            role_agent_order.append(role)
            role_outputs.append(asdict(role_output))
            function_calls.append(
                {
                    "caller": role,
                    "function_name": RETURN_FUNCTION,
                    "status": role_output.status,
                    "message": role_output.message,
                    "final_answer_present": bool(role_output.final_answer),
                }
            )
            orchestration_steps.append(
                {
                    "node": role,
                    "action": "function_role_run",
                    "status": role_output.status,
                    "details": {
                        "worker": "function_handoff_full_llm_multi_agent",
                        "role": role,
                        "message": role_output.message,
                    },
                }
            )
            if role_output.stalled_loop_detected:
                counters["stalled_loop_detected"] = 1
            if role_output.status == "failed" and not role_output.final_answer:
                error_message = role_output.message or f"{role} failed."
                break
            if role == "report_agent" and role_output.final_answer:
                context["final_answer"] = role_output.final_answer
                break
        else:
            error_message = f"Exceeded max_orchestration_steps={self.max_orchestration_steps}."
            counters["stalled_loop_detected"] = 1

        trace = self.client.get_trace()
        actual_tool_sequence = [
            str(call.get("tool_name"))
            for call in trace.get("calls", [])
            if call.get("status") == "ok" and call.get("tool_name")
        ]
        tool_results = build_multi_agent_tool_results(context)
        answer = context.get("final_answer") or ""
        success = bool(answer)

        return LLMFunctionHandoffMultiAgentResult(
            answer=answer,
            parsed_task=parsed_task,
            tool_results=tool_results,
            trace=trace,
            orchestration_steps=orchestration_steps,
            role_outputs=role_outputs,
            orchestrator_decisions=orchestrator_decisions,
            function_calls=function_calls,
            tool_observations=list(context.get("function_tool_observations") or []),
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            role_agent_order=role_agent_order,
            actual_tool_sequence=actual_tool_sequence,
            function_handoff_protocol_version=FUNCTION_HANDOFF_PROTOCOL_VERSION,
            architecture=ARCHITECTURE_NAME,
            parse_error_count=counters["parse_error_count"],
            invalid_json_count=counters["invalid_json_count"],
            invalid_function_name_count=counters["invalid_function_name_count"],
            forbidden_function_call_count=counters["forbidden_function_call_count"],
            repeated_tool_call_count=counters["repeated_tool_call_count"],
            multiple_function_blocks_in_single_output_count=counters[
                "multiple_function_blocks_in_single_output_count"
            ],
            tool_error_count=counters["tool_error_count"],
            stalled_loop_detected=bool(counters["stalled_loop_detected"]),
            repair_attempt_count=counters["repair_attempt_count"],
            retry_count=counters["repair_attempt_count"],
            success=success,
            error_message=None if success else error_message,
        )


def build_function_handoff_state_packet(
    *,
    user_query: str,
    current_role: str,
    current_message: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "user_query": user_query,
        "current_role": current_role,
        "current_message": current_message,
        "parsed_task": _public_parsed_task(context.get("parsed_task") or {}),
        "available_artifacts": summarize_available_artifacts(context),
        "previous_role_returns": list(context.get("function_role_returns") or [])[-8:],
        "handoff_history": list(context.get("function_handoffs") or [])[-8:],
        "recent_tool_observations": list(context.get("function_tool_observations") or [])[-8:],
        "completed_successful_tool_calls": list(context.get("completed_tool_calls") or [])[-12:],
    }


def summarize_available_artifacts(context: dict[str, Any]) -> dict[str, Any]:
    data = context.get("data_artifacts") or {}
    math = context.get("math_artifacts") or {}
    series = []
    for region_id, item in (data.get("series_by_region") or {}).items():
        metadata = item.get("metadata") or {}
        series.append(
            {
                "region_id": region_id,
                "series_id": item.get("series_id"),
                "n_points": metadata.get("n_points"),
            }
        )
    stats = [
        {
            "region_id": region_id,
            "stats_id": item.get("stats_id"),
            "series_id": item.get("series_id"),
        }
        for region_id, item in (math.get("stats_by_region") or {}).items()
    ]
    high = []
    for region_id, item in (math.get("high_tec") or {}).items():
        high.append(
            {
                "region_id": region_id,
                "threshold_id": (item.get("threshold") or {}).get("threshold_id"),
                "high_intervals": bool(item.get("intervals")),
                "n_intervals": (item.get("intervals") or {}).get("n_intervals"),
            }
        )
    stable = []
    for region_id, item in (math.get("stable_intervals") or {}).items():
        stable.append(
            {
                "region_id": region_id,
                "threshold_id": (item.get("thresholds") or {}).get("threshold_id"),
                "stable_intervals": bool(item.get("intervals")),
                "n_intervals": (item.get("intervals") or {}).get("n_intervals"),
            }
        )
    comparison = math.get("comparison") or {}
    return {
        "series": series,
        "stats": stats,
        "high_tec": high,
        "stable_intervals": stable,
        "comparison": {
            "comparison_id": comparison.get("comparison_id"),
            "regions": comparison.get("regions"),
        }
        if comparison
        else None,
        "findings": list((context.get("analysis_artifacts") or {}).get("findings") or []),
        "final_answer_present": bool(context.get("final_answer")),
    }


def merge_function_role_return(context: dict[str, Any], role_return: dict[str, Any]) -> None:
    context.setdefault("function_role_returns", []).append(dict(role_return))
    role = role_return.get("role")
    if role == "analysis_agent":
        context.setdefault("analysis_artifacts", {}).setdefault("findings", []).extend(
            role_return.get("findings") or []
        )
    if role == "report_agent" and role_return.get("final_answer"):
        context["final_answer"] = role_return["final_answer"]


def make_function_tool_observation(
    *,
    tool_name: str,
    status: str,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    result = result or {}
    artifact_id = _artifact_id_for_tool(tool_name, result)
    return {
        "tool_name": tool_name,
        "status": status,
        "artifact_id": artifact_id,
        "summary": _tool_summary(tool_name, status, result, error),
        "result_compact": _compact_tool_result(result),
        "error": error,
        "expected_argument_contract": None if status == "ok" else tool_argument_contract(tool_name),
    }


def _artifact_id_for_tool(tool_name: str, result: dict[str, Any]) -> str | None:
    if tool_name == "tec_get_timeseries":
        return result.get("series_id")
    if tool_name == "tec_compute_series_stats":
        return result.get("stats_id")
    if tool_name in {"tec_compute_high_threshold", "tec_compute_stability_thresholds"}:
        return result.get("threshold_id")
    if tool_name == "tec_compare_stats":
        return result.get("comparison_id")
    if tool_name in {"tec_detect_high_intervals", "tec_detect_stable_intervals"}:
        return result.get("intervals_id") or result.get("threshold_id")
    return None


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "series_id",
        "stats_id",
        "threshold_id",
        "comparison_id",
        "region_id",
        "n_intervals",
        "n_points",
        "metrics",
        "regions",
    ]
    compact = {key: result[key] for key in keys if key in result}
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        compact["metadata"] = {
            key: metadata[key]
            for key in ["dataset_ref", "region_id", "requested_start", "requested_end", "n_points"]
            if key in metadata
        }
    return compact


def _tool_summary(
    tool_name: str,
    status: str,
    result: dict[str, Any],
    error: dict[str, Any] | None,
) -> str:
    if status != "ok":
        message = (error or {}).get("message") or (error or {}).get("error_type")
        return f"{tool_name} failed: {message or 'error'}"
    if tool_name == "tec_get_timeseries":
        metadata = result.get("metadata") or {}
        return f"Loaded series {result.get('series_id')} for {metadata.get('region_id') or result.get('region_id')}."
    if tool_name in {"tec_detect_high_intervals", "tec_detect_stable_intervals"}:
        return f"Detected {result.get('n_intervals')} intervals."
    return f"{tool_name} completed."


def _public_parsed_task(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": parsed.get("task_type"),
        "dataset_ref": parsed.get("dataset_ref"),
        "region_id": parsed.get("region_id"),
        "region_ids": list(parsed.get("region_ids") or []),
        "start": parsed.get("start"),
        "end": parsed.get("end"),
        "q": parsed.get("q"),
    }


def _empty_diagnostics() -> dict[str, Any]:
    return {
        "raw_model_outputs": [],
        "cleaned_model_outputs": [],
        "parse_error_count": 0,
        "invalid_json_count": 0,
        "invalid_function_name_count": 0,
        "forbidden_function_call_count": 0,
        "multiple_function_blocks_in_single_output_count": 0,
        "repair_attempt_count": 0,
    }


def _empty_global_counters() -> dict[str, int]:
    return {
        "parse_error_count": 0,
        "invalid_json_count": 0,
        "invalid_function_name_count": 0,
        "forbidden_function_call_count": 0,
        "repeated_tool_call_count": 0,
        "multiple_function_blocks_in_single_output_count": 0,
        "tool_error_count": 0,
        "stalled_loop_detected": 0,
        "repair_attempt_count": 0,
    }


def _accumulate_parse_error(diagnostics: dict[str, Any], error_code: str | None) -> None:
    diagnostics["parse_error_count"] += 1
    if error_code == "invalid_json":
        diagnostics["invalid_json_count"] += 1


def _record_role_parse_error(
    output: FunctionHandoffRoleOutput,
    error_code: str | None,
) -> None:
    output.parse_error_count += 1
    if error_code == "invalid_json":
        output.invalid_json_count += 1


def _accumulate_diagnostics(counters: dict[str, int], diagnostics: dict[str, Any]) -> None:
    for key in [
        "parse_error_count",
        "invalid_json_count",
        "invalid_function_name_count",
        "forbidden_function_call_count",
        "multiple_function_blocks_in_single_output_count",
        "repair_attempt_count",
    ]:
        counters[key] += int(diagnostics.get(key) or 0)


def _accumulate_role_output(
    counters: dict[str, int],
    role_output: FunctionHandoffRoleOutput,
) -> None:
    for key in [
        "parse_error_count",
        "invalid_json_count",
        "invalid_function_name_count",
        "forbidden_function_call_count",
        "repeated_tool_call_count",
        "multiple_function_blocks_in_single_output_count",
        "tool_error_count",
        "repair_attempt_count",
    ]:
        counters[key] += int(getattr(role_output, key))
    if role_output.stalled_loop_detected:
        counters["stalled_loop_detected"] += 1


__all__ = [
    "ARCHITECTURE_NAME",
    "FUNCTION_HANDOFF_PROTOCOL_VERSION",
    "LLMFunctionHandoffMultiAgent",
    "LLMFunctionHandoffMultiAgentResult",
    "LLMFunctionHandoffOrchestratorAgent",
    "LLMFunctionHandoffRoleAgent",
    "PROMPT_REVISION",
    "build_function_handoff_state_packet",
    "make_function_tool_observation",
    "merge_function_role_return",
    "summarize_available_artifacts",
]
