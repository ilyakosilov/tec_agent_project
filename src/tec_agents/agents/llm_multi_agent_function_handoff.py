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
    llm_call_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    role_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    attempted_tec_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    skipped_repeated_tec_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    failed_tec_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    parse_error_count: int = 0
    invalid_json_count: int = 0
    invalid_function_name_count: int = 0
    forbidden_function_call_count: int = 0
    repeated_tool_call_count: int = 0
    multiple_function_blocks_in_single_output_count: int = 0
    invalid_artifact_handle_count: int = 0
    successful_final_tool_without_return_count: int = 0
    data_agent_repeated_retrieval_after_all_assignment_series_present_count: int = 0
    math_repeated_tool_after_terminal_artifact_count: int = 0
    math_repeated_tool_after_new_intermediate_artifact_count: int = 0
    math_failed_to_return_after_assignment_artifact_present_count: int = 0
    math_recomputed_existing_region_stats_count: int = 0
    math_assignment_completed_but_not_returned_count: int = 0
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
    llm_call_diagnostics: list[dict[str, Any]]
    raw_model_outputs: list[str]
    cleaned_model_outputs: list[str]
    role_agent_order: list[str]
    actual_tool_sequence: list[str]
    attempted_tec_tool_calls: list[dict[str, Any]]
    attempted_tec_tool_sequence: list[str]
    successful_tec_tool_sequence: list[str]
    skipped_repeated_tec_tool_calls: list[dict[str, Any]]
    failed_tec_tool_calls: list[dict[str, Any]]
    handoff_function_sequence: list[str]
    role_call_sequence: list[str]
    function_handoff_protocol_version: str
    architecture: str
    parse_error_count: int
    invalid_json_count: int
    invalid_function_name_count: int
    forbidden_function_call_count: int
    repeated_tool_call_count: int
    multiple_function_blocks_in_single_output_count: int
    multiple_protocol_blocks_in_single_output_count: int
    invalid_artifact_handle_count: int
    repeated_role_message_count: int
    orchestrator_equivalent_assignment_without_state_change_count: int
    successful_final_tool_without_return_count: int
    data_agent_repeated_retrieval_after_all_assignment_series_present_count: int
    math_repeated_tool_after_terminal_artifact_count: int
    math_repeated_tool_after_new_intermediate_artifact_count: int
    math_failed_to_return_after_assignment_artifact_present_count: int
    math_recomputed_existing_region_stats_count: int
    math_assignment_completed_but_not_returned_count: int
    tool_error_count: int
    stalled_loop_detected: bool
    repair_attempt_count: int
    retry_count: int
    success: bool
    workflow_completed: bool
    final_answer_present: bool
    terminal_numeric_artifact_present: bool
    numeric_match_with_gold: bool | None
    findings_present: bool
    analysis_agent_called: bool
    report_agent_called: bool
    failure_reason: str | None
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
        orchestration_step: int,
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
        for attempt in range(max_parse_retries + 1):
            raw = self.model.generate(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            parsed = parse_function_handoff_output(raw)
            call_diagnostics = build_llm_call_diagnostics(
                model=self.model,
                messages=messages,
                raw_output=raw,
                cleaned_output=parsed.cleaned_text,
                caller_role="orchestrator",
                orchestration_step=orchestration_step,
                role_step=None,
                attempt=attempt + 1,
            )
            diagnostics["raw_model_outputs"].append(raw)
            diagnostics["cleaned_model_outputs"].append(parsed.cleaned_text)
            diagnostics["llm_call_diagnostics"].append(call_diagnostics)
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
        orchestration_step: int,
    ) -> FunctionHandoffRoleOutput:
        output = FunctionHandoffRoleOutput(role=self.role)
        current_role_successful_tool_calls: list[dict[str, Any]] = []
        current_role_tool_observations: list[dict[str, Any]] = []
        current_role_attempted_tec_tool_calls: list[dict[str, Any]] = []

        def state_message() -> str:
            return build_function_handoff_state_message(
                role=self.role,
                user_query=user_query,
                state_packet=build_function_handoff_state_packet(
                    user_query=user_query,
                    current_role=self.role,
                    current_message=message,
                    context=context,
                    current_role_successful_tool_calls=current_role_successful_tool_calls,
                    current_role_tool_observations=current_role_tool_observations,
                    current_role_attempted_tec_tool_calls=current_role_attempted_tec_tool_calls,
                ),
            )

        messages = [
            {"role": "system", "content": build_function_handoff_role_prompt(self.role)},
            {"role": "user", "content": state_message()},
        ]
        consecutive_parse_errors = 0
        tool_call_count = 0
        previous_successful_final_tool = False

        for step in range(1, self.max_role_steps + 1):
            raw = self.model.generate(
                messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            parsed = parse_function_handoff_output(raw)
            call_diagnostics = build_llm_call_diagnostics(
                model=self.model,
                messages=messages,
                raw_output=raw,
                cleaned_output=parsed.cleaned_text,
                caller_role=self.role,
                orchestration_step=orchestration_step,
                role_step=step,
                attempt=1,
            )
            output.raw_model_outputs.append(raw)
            output.cleaned_model_outputs.append(parsed.cleaned_text)
            output.llm_call_diagnostics.append(call_diagnostics)
            if parsed.block_count > 1:
                output.multiple_function_blocks_in_single_output_count += 1
            role_step: dict[str, Any] = {
                "step": step,
                "raw_model_output": raw,
                "cleaned_model_output": parsed.cleaned_text,
                "llm_call_diagnostics": call_diagnostics,
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

            attempted_call = {
                "role": self.role,
                "role_step": step,
                "tool_name": call.name,
                "arguments": dict(call.arguments),
                "status": "parsed",
            }
            context.setdefault("function_attempted_tec_tool_calls", []).append(attempted_call)
            output.attempted_tec_tool_calls.append(attempted_call)
            current_role_attempted_tec_tool_calls.append(attempted_call)
            role_step["attempted_tec_tool_call"] = attempted_call

            if previous_successful_final_tool:
                output.successful_final_tool_without_return_count += 1
                role_step["final_tool_return_expected"] = True
                if self.role == "math_agent":
                    output.math_failed_to_return_after_assignment_artifact_present_count += 1
                    output.math_assignment_completed_but_not_returned_count += 1

            if call.name not in self.client.list_tool_names():
                output.invalid_function_name_count += 1
                role_step["function_status"] = "unregistered_tool"
                attempted_call["status"] = "unregistered_tool"
                output.failed_tec_tool_calls.append(attempted_call)
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

            invalid_handle_error = validate_visible_artifact_handles(call.name, call.arguments, context)
            if invalid_handle_error:
                output.invalid_artifact_handle_count += 1
                role_step["function_status"] = "invalid_artifact_handle"
                role_step["error_message"] = invalid_handle_error
                attempted_call["status"] = "invalid_artifact_handle"
                output.failed_tec_tool_calls.append(attempted_call)
                context.setdefault("function_failed_tec_tool_calls", []).append(attempted_call)
                observation = make_function_tool_observation(
                    tool_name=call.name,
                    status="error",
                    result={},
                    error={
                        "error_type": "invalid_artifact_handle",
                        "message": invalid_handle_error,
                        "tool_name": call.name,
                    },
                )
                role_step["observation"] = observation
                context.setdefault("function_tool_observations", []).append(observation)
                current_role_tool_observations.append(observation)
                context.setdefault("function_runtime_feedback", []).append(
                    {
                        "kind": "invalid_artifact_handle",
                        "role": self.role,
                        "tool_name": call.name,
                        "message": (
                            "The supplied artifact handle is not runtime-visible. "
                            "Use only exact handles listed in CURRENT RUNTIME FACTS."
                        ),
                    }
                )
                output.tool_observations.append(observation)
                output.role_steps.append(role_step)
                output.repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": state_message(),
                    }
                )
                continue

            key = tool_call_key(call.name, call.arguments)
            successful_keys = context.setdefault("successful_tool_call_keys", set())
            if key in successful_keys:
                output.repeated_tool_call_count += 1
                role_step["function_status"] = "skipped_repeated"
                attempted_call["status"] = "skipped_repeated"
                output.skipped_repeated_tec_tool_calls.append(attempted_call)
                context.setdefault("function_skipped_repeated_tec_tool_calls", []).append(
                    attempted_call
                )
                if self.role == "data_agent" and call.name == "tec_get_timeseries":
                    progress = assignment_progress_for_role(self.role, context)
                    if progress.get("scope_covered"):
                        output.data_agent_repeated_retrieval_after_all_assignment_series_present_count += 1
                        role_step[
                            "data_agent_repeated_retrieval_after_all_assignment_series_present"
                        ] = True
                if self.role == "math_agent":
                    if _terminal_numeric_artifact_present(context):
                        output.math_repeated_tool_after_terminal_artifact_count += 1
                        role_step["math_repeated_tool_after_terminal_artifact"] = True
                    elif call.name in {
                        "tec_compute_series_stats",
                        "tec_compute_high_threshold",
                        "tec_compute_stability_thresholds",
                    }:
                        output.math_repeated_tool_after_new_intermediate_artifact_count += 1
                        role_step["math_repeated_tool_after_new_intermediate_artifact"] = True
                output.role_steps.append(role_step)
                context.setdefault("function_runtime_feedback", []).append(
                    {
                        "kind": "repeated_successful_tool_call",
                        "role": self.role,
                        "tool_name": call.name,
                        "message": (
                            "This exact tool call already succeeded and was not executed again. "
                            "Use the artifact shown in CURRENT RUNTIME FACTS or return_to_orchestrator."
                        ),
                    }
                )
                if output.repeated_tool_call_count >= 2:
                    output.stalled_loop_detected = True
                    output.status = "failed"
                    output.message = "Repeated identical successful tool call caused a stall."
                    return output
                output.repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": state_message(),
                    }
                )
                continue

            if tool_call_count >= self.max_tool_calls:
                output.status = "failed"
                output.message = f"Exceeded max_tool_calls={self.max_tool_calls}."
                attempted_call["status"] = "max_tool_calls_exceeded"
                output.failed_tec_tool_calls.append(attempted_call)
                output.role_steps.append(role_step)
                return output

            tool_call_count += 1
            if self.role == "math_agent" and call.name == "tec_compute_series_stats":
                series_id = call.arguments.get("series_id")
                if isinstance(series_id, str) and _stats_exists_for_series(context, series_id):
                    output.math_recomputed_existing_region_stats_count += 1
                    role_step["math_recomputed_existing_region_stats"] = True
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
            current_role_tool_observations.append(observation)
            output.tool_observations.append(observation)
            if response.status != "ok":
                output.tool_error_count += 1
                attempted_call["status"] = response.status
                output.failed_tec_tool_calls.append(attempted_call)
                context.setdefault("function_failed_tec_tool_calls", []).append(attempted_call)
            if response.status == "ok" and response.result is not None:
                attempted_call["status"] = "ok"
                successful_keys.add(key)
                output.tool_sequence.append(call.name)
                previous_successful_final_tool = is_final_like_tool(call.name)
                record_successful_tool_call(
                    context=context,
                    role=self.role,
                    tool_name=call.name,
                    arguments=call.arguments,
                    tool_response=response_dict,
                )
                success_record = _last_item(context.get("completed_tool_calls") or [])
                if isinstance(success_record, dict):
                    current_role_successful_tool_calls.append(success_record)
            messages.append(
                {
                    "role": "user",
                    "content": state_message(),
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
        context["function_runtime_feedback"] = []
        context["function_attempted_tec_tool_calls"] = []
        context["function_skipped_repeated_tec_tool_calls"] = []
        context["function_failed_tec_tool_calls"] = []

        counters = _empty_global_counters()
        orchestration_steps: list[dict[str, Any]] = []
        role_outputs: list[dict[str, Any]] = []
        orchestrator_decisions: list[dict[str, Any]] = []
        function_calls: list[dict[str, Any]] = []
        llm_call_diagnostics: list[dict[str, Any]] = []
        raw_model_outputs: list[str] = []
        cleaned_model_outputs: list[str] = []
        role_agent_order: list[str] = []
        error_message: str | None = None

        for step in range(1, self.max_orchestration_steps + 1):
            call, diagnostics = self.orchestrator.decide(
                user_query=query,
                context=context,
                max_parse_retries=self.max_parse_retries,
                orchestration_step=step,
            )
            _accumulate_diagnostics(counters, diagnostics)
            llm_call_diagnostics.extend(diagnostics.get("llm_call_diagnostics") or [])
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
            role_message_key = (
                role,
                " ".join(message.lower().split()),
                _artifact_state_signature(context),
            )
            if role_message_key in context.setdefault("function_role_message_keys", set()):
                counters["repeated_role_message_count"] += 1
                counters["orchestrator_equivalent_assignment_without_state_change_count"] += 1
                context.setdefault("function_runtime_feedback", []).append(
                    {
                        "kind": "repeated_role_message",
                        "role": role,
                        "message": (
                            "The same role was called with the same message again "
                            "while runtime-visible state should be inspected."
                        ),
                    }
                )
            else:
                context["function_role_message_keys"].add(role_message_key)
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
                orchestration_step=step,
            )
            _accumulate_role_output(counters, role_output)
            llm_call_diagnostics.extend(role_output.llm_call_diagnostics)
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
        terminal_numeric_artifact_present = _terminal_numeric_artifact_present(context)
        findings_present = bool((context.get("analysis_artifacts") or {}).get("findings"))
        analysis_agent_called = "analysis_agent" in role_agent_order
        report_agent_called = "report_agent" in role_agent_order
        attempted_tec_tool_calls = list(context.get("function_attempted_tec_tool_calls") or [])
        skipped_repeated = list(context.get("function_skipped_repeated_tec_tool_calls") or [])
        failed_tec = list(context.get("function_failed_tec_tool_calls") or [])
        handoff_function_sequence = [
            str(item.get("function_name"))
            for item in function_calls
            if item.get("caller") == "orchestrator" and item.get("function_name")
        ]

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
            llm_call_diagnostics=llm_call_diagnostics,
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            role_agent_order=role_agent_order,
            actual_tool_sequence=actual_tool_sequence,
            attempted_tec_tool_calls=attempted_tec_tool_calls,
            attempted_tec_tool_sequence=[
                str(item.get("tool_name")) for item in attempted_tec_tool_calls
            ],
            successful_tec_tool_sequence=actual_tool_sequence,
            skipped_repeated_tec_tool_calls=skipped_repeated,
            failed_tec_tool_calls=failed_tec,
            handoff_function_sequence=handoff_function_sequence,
            role_call_sequence=role_agent_order,
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
            multiple_protocol_blocks_in_single_output_count=counters[
                "multiple_function_blocks_in_single_output_count"
            ],
            invalid_artifact_handle_count=counters["invalid_artifact_handle_count"],
            repeated_role_message_count=counters["repeated_role_message_count"],
            orchestrator_equivalent_assignment_without_state_change_count=counters[
                "orchestrator_equivalent_assignment_without_state_change_count"
            ],
            successful_final_tool_without_return_count=counters[
                "successful_final_tool_without_return_count"
            ],
            data_agent_repeated_retrieval_after_all_assignment_series_present_count=counters[
                "data_agent_repeated_retrieval_after_all_assignment_series_present_count"
            ],
            math_repeated_tool_after_terminal_artifact_count=counters[
                "math_repeated_tool_after_terminal_artifact_count"
            ],
            math_repeated_tool_after_new_intermediate_artifact_count=counters[
                "math_repeated_tool_after_new_intermediate_artifact_count"
            ],
            math_failed_to_return_after_assignment_artifact_present_count=counters[
                "math_failed_to_return_after_assignment_artifact_present_count"
            ],
            math_recomputed_existing_region_stats_count=counters[
                "math_recomputed_existing_region_stats_count"
            ],
            math_assignment_completed_but_not_returned_count=counters[
                "math_assignment_completed_but_not_returned_count"
            ],
            tool_error_count=counters["tool_error_count"],
            stalled_loop_detected=bool(counters["stalled_loop_detected"]),
            repair_attempt_count=counters["repair_attempt_count"],
            retry_count=counters["repair_attempt_count"],
            success=success,
            workflow_completed=success,
            final_answer_present=bool(answer),
            terminal_numeric_artifact_present=terminal_numeric_artifact_present,
            numeric_match_with_gold=None,
            findings_present=findings_present,
            analysis_agent_called=analysis_agent_called,
            report_agent_called=report_agent_called,
            failure_reason=None if success else error_message,
            error_message=None if success else error_message,
        )


def build_function_handoff_state_packet(
    *,
    user_query: str,
    current_role: str,
    current_message: str,
    context: dict[str, Any],
    current_role_successful_tool_calls: list[dict[str, Any]] | None = None,
    current_role_tool_observations: list[dict[str, Any]] | None = None,
    current_role_attempted_tec_tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    available_artifacts = summarize_available_artifacts(context)
    return {
        "user_query": user_query,
        "current_role": current_role,
        "current_message": current_message,
        "parsed_task": _public_parsed_task(context.get("parsed_task") or {}),
        "available_artifacts": available_artifacts,
        "runtime_visible_handles": runtime_visible_handles(context),
        "assignment_progress": assignment_progress_for_role(current_role, context),
        "current_role_successful_tool_calls": list(current_role_successful_tool_calls or []),
        "current_role_tool_observations": list(current_role_tool_observations or []),
        "current_role_attempted_tec_tool_calls": list(
            current_role_attempted_tec_tool_calls or []
        ),
        "last_role_return": _last_item(context.get("function_role_returns") or []),
        "runtime_feedback": list(context.get("function_runtime_feedback") or [])[-4:],
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


def runtime_visible_handles(context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = context.get("data_artifacts") or {}
    math = context.get("math_artifacts") or {}
    handles: dict[str, list[dict[str, Any]]] = {
        "series_id": [],
        "stats_id": [],
        "threshold_id": [],
        "high_intervals": [],
        "stable_intervals": [],
        "comparison_id": [],
    }
    for region_id, item in (data.get("series_by_region") or {}).items():
        if item.get("series_id"):
            handles["series_id"].append(
                {"region_id": region_id, "id": item.get("series_id")}
            )
    for region_id, item in (math.get("stats_by_region") or {}).items():
        if item.get("stats_id"):
            handles["stats_id"].append(
                {
                    "region_id": region_id,
                    "id": item.get("stats_id"),
                    "series_id": item.get("series_id"),
                }
            )
    for region_id, item in (math.get("high_tec") or {}).items():
        threshold = item.get("threshold") or {}
        intervals = item.get("intervals") or {}
        if threshold.get("threshold_id"):
            handles["threshold_id"].append(
                {
                    "region_id": region_id,
                    "id": threshold.get("threshold_id"),
                    "kind": "high_threshold",
                }
            )
        if intervals:
            handles["high_intervals"].append(
                {
                    "region_id": region_id,
                    "threshold_id": intervals.get("threshold_id")
                    or threshold.get("threshold_id"),
                    "n_intervals": intervals.get("n_intervals"),
                }
            )
    for region_id, item in (math.get("stable_intervals") or {}).items():
        threshold = item.get("thresholds") or {}
        intervals = item.get("intervals") or {}
        if threshold.get("threshold_id"):
            handles["threshold_id"].append(
                {
                    "region_id": region_id,
                    "id": threshold.get("threshold_id"),
                    "kind": "stability_threshold",
                }
            )
        if intervals:
            handles["stable_intervals"].append(
                {
                    "region_id": region_id,
                    "threshold_id": intervals.get("threshold_id")
                    or threshold.get("threshold_id"),
                    "n_intervals": intervals.get("n_intervals"),
                }
            )
    comparison = math.get("comparison") or {}
    if comparison.get("comparison_id"):
        handles["comparison_id"].append(
            {
                "id": comparison.get("comparison_id"),
                "regions": comparison.get("regions"),
            }
        )
    return handles


def assignment_progress_for_role(role: str, context: dict[str, Any]) -> dict[str, Any]:
    parsed = context.get("parsed_task") or {}
    requested = requested_regions_from_task(parsed)
    covered = {
        region_id
        for region_id, item in (
            (context.get("data_artifacts") or {}).get("series_by_region") or {}
        ).items()
        if item.get("series_id")
    }
    progress: dict[str, Any] = {
        "requested_regions": requested,
        "covered_regions": [region for region in requested if region in covered],
        "scope_covered": bool(requested)
        and all(region in covered for region in requested),
    }
    if role == "math_agent":
        progress["visible_input_handles"] = runtime_visible_handles(context)
    return progress


def requested_regions_from_task(parsed: dict[str, Any]) -> list[str]:
    regions = list(parsed.get("region_ids") or [])
    region = parsed.get("region_id")
    if region and region not in regions:
        regions.append(str(region))
    return [str(item) for item in regions if str(item).strip()]


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
        "produced_artifact_type": _artifact_type_for_tool(tool_name),
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


def _artifact_type_for_tool(tool_name: str) -> str | None:
    if tool_name == "tec_get_timeseries":
        return "series_id"
    if tool_name == "tec_compute_series_stats":
        return "stats_id"
    if tool_name == "tec_compute_high_threshold":
        return "threshold_id"
    if tool_name == "tec_compute_stability_thresholds":
        return "threshold_id"
    if tool_name == "tec_detect_high_intervals":
        return "high_intervals"
    if tool_name == "tec_detect_stable_intervals":
        return "stable_intervals"
    if tool_name == "tec_compare_stats":
        return "comparison_id"
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
        "window_minutes": parsed.get("window_minutes"),
        "q_delta": parsed.get("q_delta"),
        "q_std": parsed.get("q_std"),
    }


def validate_visible_artifact_handles(
    tool_name: str,
    arguments: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    visible = _visible_handle_sets(context)
    checks: list[tuple[str, str]] = []
    for key in ["series_id", "threshold_id", "stats_id", "comparison_id"]:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            checks.append((key, value))
    stats_ids = arguments.get("stats_ids")
    if isinstance(stats_ids, list):
        for value in stats_ids:
            if isinstance(value, str) and value.strip():
                checks.append(("stats_id", value))
    reference_stats_id = arguments.get("reference_stats_id")
    if isinstance(reference_stats_id, str) and reference_stats_id.strip():
        checks.append(("stats_id", reference_stats_id))

    for key, value in checks:
        if value not in visible.get(key, set()):
            known = sorted(visible.get(key, set()))
            return (
                f"{tool_name} argument {key}={value!r} is not runtime-visible. "
                f"Known {key} handles: {known or '<none>'}."
            )
    return None


def _visible_handle_sets(context: dict[str, Any]) -> dict[str, set[str]]:
    handles = runtime_visible_handles(context)
    visible: dict[str, set[str]] = {
        "series_id": set(),
        "stats_id": set(),
        "threshold_id": set(),
        "comparison_id": set(),
    }
    for item in handles.get("series_id") or []:
        if item.get("id"):
            visible["series_id"].add(str(item["id"]))
    for item in handles.get("stats_id") or []:
        if item.get("id"):
            visible["stats_id"].add(str(item["id"]))
    for item in handles.get("threshold_id") or []:
        if item.get("id"):
            visible["threshold_id"].add(str(item["id"]))
    for item in handles.get("comparison_id") or []:
        if item.get("id"):
            visible["comparison_id"].add(str(item["id"]))
    return visible


def is_final_like_tool(tool_name: str) -> bool:
    return tool_name in {
        "tec_detect_high_intervals",
        "tec_detect_stable_intervals",
        "tec_compare_stats",
    }


def _stats_exists_for_series(context: dict[str, Any], series_id: str) -> bool:
    for item in ((context.get("math_artifacts") or {}).get("stats_by_region") or {}).values():
        if str(item.get("series_id") or "") == series_id and item.get("stats_id"):
            return True
    return False


def _terminal_numeric_artifact_present(context: dict[str, Any]) -> bool:
    parsed = context.get("parsed_task") or {}
    task_type = parsed.get("task_type")
    math = context.get("math_artifacts") or {}
    if task_type == "high_tec":
        return any(
            bool((item or {}).get("intervals"))
            for item in (math.get("high_tec") or {}).values()
        )
    if task_type == "stable_intervals":
        return any(
            bool((item or {}).get("intervals"))
            for item in (math.get("stable_intervals") or {}).values()
        )
    if task_type == "compare_regions":
        return bool((math.get("comparison") or {}).get("comparison_id"))
    if task_type == "report":
        region_ids = requested_regions_from_task(parsed)
        stats = math.get("stats_by_region") or {}
        high = math.get("high_tec") or {}
        stable = math.get("stable_intervals") or {}
        return bool(region_ids) and all(
            region in stats
            and bool((high.get(region) or {}).get("intervals"))
            and bool((stable.get(region) or {}).get("intervals"))
            for region in region_ids
        )
    return False


def _artifact_state_signature(context: dict[str, Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    handles = runtime_visible_handles(context)
    signature: list[tuple[str, tuple[str, ...]]] = []
    for key in sorted(handles):
        ids: list[str] = []
        for item in handles.get(key) or []:
            artifact_id = item.get("id") or item.get("threshold_id") or item.get("region_id")
            if artifact_id is not None:
                ids.append(str(artifact_id))
        signature.append((key, tuple(sorted(ids))))
    return tuple(signature)


def build_llm_call_diagnostics(
    *,
    model,
    messages: list[dict[str, str]],
    raw_output: str,
    cleaned_output: str,
    caller_role: str,
    orchestration_step: int | None,
    role_step: int | None,
    attempt: int,
) -> dict[str, Any]:
    """Return compact prompt/generation diagnostics without storing full prompts."""

    prompt = _render_prompt_for_diagnostics(model, messages)
    before_tokens = _count_tokens(model, prompt)
    max_input_tokens = getattr(model, "max_input_tokens", None)
    after_tokens = (
        min(before_tokens, int(max_input_tokens))
        if isinstance(before_tokens, int) and isinstance(max_input_tokens, int)
        else before_tokens
    )
    raw = str(raw_output or "")
    first_tool_index = raw.lower().find("<tool_call")
    return {
        "caller_role": caller_role,
        "orchestration_step": orchestration_step,
        "role_step": role_step,
        "attempt": attempt,
        "prompt_character_count": len(prompt),
        "prompt_token_count_before_truncation": before_tokens,
        "prompt_token_count_after_truncation": after_tokens,
        "prompt_was_truncated": (
            bool(isinstance(before_tokens, int) and isinstance(after_tokens, int) and before_tokens > after_tokens)
        ),
        "generated_token_count": _count_tokens(model, raw, add_special_tokens=False),
        "raw_output_had_prefix_before_tool_call": bool(first_tool_index > 0 and raw[:first_tool_index].strip()),
        "output_cleaning_applied": str(cleaned_output or "").strip() != raw.strip(),
        "prompt_tail_preview": prompt[-500:],
    }


def _render_prompt_for_diagnostics(model, messages: list[dict[str, str]]) -> str:
    renderer = getattr(model, "_render_prompt", None)
    if callable(renderer):
        try:
            return str(renderer(messages))
        except Exception:
            pass
    chunks = []
    for message in messages:
        chunks.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
    chunks.append("assistant:")
    return "\n\n".join(chunks)


def _count_tokens(model, text: str, *, add_special_tokens: bool = True) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer(
            str(text or ""),
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
    except Exception:
        return None
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if input_ids is None:
        return None
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _last_item(items: list[Any]) -> Any:
    return items[-1] if items else None


def _empty_diagnostics() -> dict[str, Any]:
    return {
        "raw_model_outputs": [],
        "cleaned_model_outputs": [],
        "llm_call_diagnostics": [],
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
        "invalid_artifact_handle_count": 0,
        "repeated_role_message_count": 0,
        "orchestrator_equivalent_assignment_without_state_change_count": 0,
        "successful_final_tool_without_return_count": 0,
        "data_agent_repeated_retrieval_after_all_assignment_series_present_count": 0,
        "math_repeated_tool_after_terminal_artifact_count": 0,
        "math_repeated_tool_after_new_intermediate_artifact_count": 0,
        "math_failed_to_return_after_assignment_artifact_present_count": 0,
        "math_recomputed_existing_region_stats_count": 0,
        "math_assignment_completed_but_not_returned_count": 0,
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
        "invalid_artifact_handle_count",
        "successful_final_tool_without_return_count",
        "data_agent_repeated_retrieval_after_all_assignment_series_present_count",
        "math_repeated_tool_after_terminal_artifact_count",
        "math_repeated_tool_after_new_intermediate_artifact_count",
        "math_failed_to_return_after_assignment_artifact_present_count",
        "math_recomputed_existing_region_stats_count",
        "math_assignment_completed_but_not_returned_count",
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
    "assignment_progress_for_role",
    "is_final_like_tool",
    "make_function_tool_observation",
    "merge_function_role_return",
    "requested_regions_from_task",
    "runtime_visible_handles",
    "summarize_available_artifacts",
    "validate_visible_artifact_handles",
]
