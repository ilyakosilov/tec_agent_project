"""Typed-contract full LLM multi-agent TEC workflow.

This runner is a new experiment alongside the historical ``llm_multi_agent``
module. All roles remain LLM-driven; the runtime only validates typed blocks,
role tool permissions, duplicate calls, and logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tec_agents.agents.llm_multi_agent import (
    build_multi_agent_tool_results,
    initial_multi_agent_context,
    make_orchestration_step,
    public_task_state,
    record_successful_tool_call,
)
from tec_agents.agents.llm_single_agent import infer_task_state
from tec_agents.agents.llm_multi_agent_typed_prompts import (
    ARCHITECTURE_NAME,
    build_typed_duplicate_tool_message,
    build_typed_orchestrator_prompt,
    build_typed_orchestrator_state_message,
    build_typed_protocol_violation_message,
    build_typed_role_action_repair_message,
    build_typed_role_output_repair_message,
    build_typed_role_prompt,
    build_typed_role_state_message,
)
from tec_agents.agents.llm_multi_agent_typed_protocol import (
    ROLE_TOOL_ALLOWLIST,
    TYPED_PROTOCOL_VERSION,
    RoleAction,
    RoleAssignment,
    RoleResponse,
    ToolObservation,
    TypedParseResult,
    TypedToolCall,
    clean_typed_output,
    compact_tool_result,
    make_tool_observation,
    parse_typed_role_output,
    tool_call_key,
    validate_tool_call_for_role,
)
from tec_agents.mcp.client import LocalMCPClient


@dataclass
class LLMTypedRoleOutput:
    role: str
    status: str
    summary: str = ""
    assignment: dict[str, Any] | None = None
    produced_artifact_types: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    needs: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    raw_model_outputs: list[str] = field(default_factory=list)
    cleaned_model_outputs: list[str] = field(default_factory=list)
    role_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_sequence: list[str] = field(default_factory=list)
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    parse_error_count: int = 0
    invalid_json_count: int = 0
    unknown_format_count: int = 0
    invalid_role_protocol_count: int = 0
    invalid_tool_name_count: int = 0
    forbidden_tool_call_count: int = 0
    invalid_role_response_count: int = 0
    invalid_final_answer_count: int = 0
    repeated_tool_call_count: int = 0
    repair_attempt_count: int = 0
    stalled_loop_detected: bool = False


@dataclass
class LLMTypedMultiAgentResult:
    answer: str
    parsed_task: dict[str, Any] | None
    tool_results: dict[str, Any]
    trace: dict[str, Any]
    orchestration_steps: list[dict[str, Any]]
    role_outputs: list[dict[str, Any]]
    orchestrator_decisions: list[dict[str, Any]]
    role_actions: list[dict[str, Any]]
    role_assignments: list[dict[str, Any]]
    tool_observations: list[dict[str, Any]]
    available_artifacts_snapshots: list[dict[str, Any]]
    raw_model_outputs: list[str]
    cleaned_model_outputs: list[str]
    role_agent_order: list[str]
    actual_tool_sequence: list[str]
    typed_protocol_version: str
    architecture: str
    parse_error_count: int
    invalid_json_count: int
    unknown_format_count: int
    invalid_assignment_count: int
    invalid_role_action_count: int
    invalid_role_response_count: int
    invalid_final_answer_count: int
    invalid_tool_name_count: int
    invalid_role_protocol_count: int
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
        return asdict(self)


class LLMTypedOrchestratorAgent:
    """LLM-driven orchestrator using RoleAction + nested RoleAssignment."""

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
    ) -> tuple[RoleAction | None, dict[str, Any]]:
        diagnostics = _empty_orchestrator_diagnostics()
        messages = [
            {"role": "system", "content": build_typed_orchestrator_prompt()},
            {
                "role": "user",
                "content": build_typed_orchestrator_state_message(
                    user_query=user_query,
                    state_packet=build_typed_state_packet(
                        user_query=user_query,
                        current_role="orchestrator",
                        current_assignment=None,
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
            cleaned = clean_typed_output(raw)
            diagnostics["raw_model_outputs"].append(raw)
            diagnostics["cleaned_model_outputs"].append(cleaned)
            parsed = parse_typed_role_output(cleaned or raw, "orchestrator")
            if parsed.ok:
                action = parsed.value
                if action.action == "finish" and not context.get("final_answer"):
                    diagnostics["invalid_role_action_count"] += 1
                    diagnostics["repair_attempt_count"] += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": build_typed_role_action_repair_message(
                                "finish is valid only after ReportAgent final_answer is present."
                            ),
                        }
                    )
                    continue
                return action, diagnostics

            _accumulate_parse_error(diagnostics, parsed)
            if parsed.error_code == "schema_error" and "assignment" in (
                parsed.error_message or ""
            ):
                diagnostics["invalid_assignment_count"] += 1
            diagnostics["invalid_role_action_count"] += 1
            diagnostics["repair_attempt_count"] += 1
            messages.append(
                {
                    "role": "user",
                    "content": build_typed_role_action_repair_message(
                        parsed.error_message or "Invalid role_action."
                    ),
                }
            )

        return None, diagnostics


class LLMTypedRoleAgent:
    """LLM-driven typed worker role."""

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
        if role not in ROLE_TOOL_ALLOWLIST:
            raise ValueError(f"Unknown typed role: {role!r}")
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
        assignment: RoleAssignment,
        context: dict[str, Any],
    ) -> LLMTypedRoleOutput:
        output = LLMTypedRoleOutput(
            role=self.role,
            status="failed",
            assignment=assignment.to_dict(),
        )
        messages = [
            {"role": "system", "content": build_typed_role_prompt(self.role)},
            {
                "role": "user",
                "content": build_typed_role_state_message(
                    role=self.role,
                    state_packet=build_typed_state_packet(
                        user_query=user_query,
                        current_role=self.role,
                        current_assignment=assignment,
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
            cleaned = clean_typed_output(raw)
            output.raw_model_outputs.append(raw)
            output.cleaned_model_outputs.append(cleaned)
            parsed = parse_typed_role_output(cleaned or raw, self.role)
            role_step: dict[str, Any] = {
                "step": step,
                "raw_model_output": raw,
                "cleaned_model_output": cleaned,
                "parsed_kind": parsed.kind,
                "parse_ok": parsed.ok,
            }

            if not parsed.ok:
                self._record_parse_error(output, parsed)
                consecutive_parse_errors += 1
                role_step["error_code"] = parsed.error_code
                role_step["error_message"] = parsed.error_message
                output.role_steps.append(role_step)
                if consecutive_parse_errors > self.max_parse_retries:
                    output.status = "failed"
                    output.summary = parsed.error_message or "Typed parse failed."
                    return output
                output.repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_typed_role_output_repair_message(
                            self.role,
                            parsed.error_message or "Invalid typed output.",
                        ),
                    }
                )
                continue

            consecutive_parse_errors = 0
            if parsed.kind == "tool_call":
                tool_call: TypedToolCall = parsed.value
                role_step["tool_name"] = tool_call.name
                role_step["arguments"] = tool_call.arguments
                protocol_error = validate_tool_call_for_role(self.role, tool_call.name)
                if protocol_error:
                    output.invalid_role_protocol_count += 1
                    output.forbidden_tool_call_count += 1
                    role_step["tool_status"] = "protocol_violation"
                    role_step["protocol_error"] = protocol_error
                    output.role_steps.append(role_step)
                    messages.append(
                        {
                            "role": "user",
                            "content": build_typed_protocol_violation_message(
                                self.role,
                                protocol_error,
                            ),
                        }
                    )
                    continue

                if tool_call.name not in self.client.list_tool_names():
                    output.invalid_tool_name_count += 1
                    role_step["tool_status"] = "invalid_tool_name"
                    output.role_steps.append(role_step)
                    messages.append(
                        {
                            "role": "user",
                            "content": build_typed_protocol_violation_message(
                                self.role,
                                f"Tool {tool_call.name!r} is not registered.",
                            ),
                        }
                    )
                    continue

                key = tool_call_key(tool_call.name, tool_call.arguments)
                successful_keys = context.setdefault("successful_tool_call_keys", set())
                if key in successful_keys:
                    output.repeated_tool_call_count += 1
                    role_step["tool_status"] = "skipped_repeated"
                    output.role_steps.append(role_step)
                    if output.repeated_tool_call_count >= 2:
                        output.stalled_loop_detected = True
                        output.status = "failed"
                        output.summary = "Repeated identical successful tool call caused a stall."
                        return output
                    output.repair_attempt_count += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": build_typed_duplicate_tool_message(
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments,
                                state_packet=build_typed_state_packet(
                                    user_query=user_query,
                                    current_role=self.role,
                                    current_assignment=assignment,
                                    context=context,
                                ),
                            ),
                        }
                    )
                    continue

                if tool_call_count >= self.max_tool_calls:
                    output.status = "failed"
                    output.summary = f"Exceeded max_tool_calls={self.max_tool_calls}."
                    output.role_steps.append(role_step)
                    return output

                tool_call_count += 1
                response = self.client.call_tool(
                    tool_call.name,
                    tool_call.arguments,
                    agent_name=self.role,
                    step=tool_call_count,
                )
                response_dict = response.to_dict()
                role_step["tool_status"] = response.status
                role_step["tool_result"] = response_dict
                output.role_steps.append(role_step)

                result = response.result if response.result is not None else {}
                observation = make_tool_observation(
                    tool_name=tool_call.name,
                    status=response.status,
                    result=result if isinstance(result, dict) else {},
                    error=None if response.status == "ok" else response_dict,
                )
                context.setdefault("tool_observations", []).append(
                    observation.to_dict()
                )
                output.tool_observations.append(observation.to_dict())

                if response.status == "ok" and response.result is not None:
                    successful_keys.add(key)
                    output.tool_sequence.append(tool_call.name)
                    record_successful_tool_call(
                        context=context,
                        role=self.role,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        tool_response=response_dict,
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": build_typed_role_state_message(
                            role=self.role,
                            state_packet=build_typed_state_packet(
                                user_query=user_query,
                                current_role=self.role,
                                current_assignment=assignment,
                                context=context,
                            ),
                        ),
                    }
                )
                continue

            if parsed.kind == "role_response":
                response: RoleResponse = parsed.value
                output.status = response.status
                output.summary = response.summary
                output.produced_artifact_types = response.produced_artifact_types
                output.artifact_refs = response.artifact_refs
                output.findings = response.findings
                output.needs = [need.to_dict() for need in response.needs]
                output.role_steps.append(role_step)
                _merge_typed_role_response(context, response)
                return output

            if parsed.kind == "final_answer":
                answer = parsed.value.answer
                if self.role != "report_agent":
                    output.invalid_final_answer_count += 1
                    output.invalid_role_protocol_count += 1
                    role_step["protocol_error"] = "Only report_agent may emit final_answer."
                    output.role_steps.append(role_step)
                    messages.append(
                        {
                            "role": "user",
                            "content": build_typed_role_output_repair_message(
                                self.role,
                                "Only report_agent may emit final_answer.",
                            ),
                        }
                    )
                    continue
                output.status = "done"
                output.summary = "Final answer produced."
                output.final_answer = answer
                context["final_answer"] = answer
                output.role_steps.append(role_step)
                return output

        output.status = "failed"
        output.summary = f"Exceeded max_role_steps={self.max_role_steps}."
        output.stalled_loop_detected = True
        return output

    def _record_parse_error(
        self,
        output: LLMTypedRoleOutput,
        parsed: TypedParseResult,
    ) -> None:
        output.parse_error_count += 1
        if parsed.error_code == "invalid_json":
            output.invalid_json_count += 1
        elif parsed.error_code == "unknown_format":
            output.unknown_format_count += 1
        elif parsed.error_code == "protocol_violation":
            output.invalid_role_protocol_count += 1
            output.forbidden_tool_call_count += 1
        elif parsed.error_code == "schema_error" and parsed.kind == "role_response":
            output.invalid_role_response_count += 1
        elif parsed.error_code == "schema_error" and parsed.kind == "final_answer":
            output.invalid_final_answer_count += 1


class LLMTypedDataAgent(LLMTypedRoleAgent):
    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("data_agent", model, client, **kwargs)


class LLMTypedMathAgent(LLMTypedRoleAgent):
    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("math_agent", model, client, **kwargs)


class LLMTypedAnalysisAgent(LLMTypedRoleAgent):
    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("analysis_agent", model, client, **kwargs)


class LLMTypedReportAgent(LLMTypedRoleAgent):
    def __init__(self, model, client: LocalMCPClient, **kwargs: Any) -> None:
        super().__init__("report_agent", model, client, **kwargs)


class LLMFullTypedMultiAgent:
    """Full LLM multi-agent network with typed role contracts."""

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
        state_feedback_mode: str = "typed_state",
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
        self.orchestrator = LLMTypedOrchestratorAgent(
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
            "data_agent": LLMTypedDataAgent(model, client, **role_kwargs),
            "math_agent": LLMTypedMathAgent(model, client, **role_kwargs),
            "analysis_agent": LLMTypedAnalysisAgent(model, client, **role_kwargs),
            "report_agent": LLMTypedReportAgent(model, client, **role_kwargs),
        }

    def reset(self) -> None:
        reset_fn = getattr(self.client, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def run(self, user_query: str) -> LLMTypedMultiAgentResult:
        parsed_task = public_task_state(infer_task_state(user_query))
        context = initial_multi_agent_context(user_query, parsed_task)
        context["typed_protocol_version"] = TYPED_PROTOCOL_VERSION
        context["tool_observations"] = []

        orchestration_steps: list[dict[str, Any]] = []
        role_outputs: list[dict[str, Any]] = []
        orchestrator_decisions: list[dict[str, Any]] = []
        role_actions: list[dict[str, Any]] = []
        role_assignments: list[dict[str, Any]] = []
        available_artifacts_snapshots: list[dict[str, Any]] = []
        raw_model_outputs: list[str] = []
        cleaned_model_outputs: list[str] = []
        counters = _empty_global_counters()
        stalled_loop_detected = False

        for orchestration_index in range(1, self.max_orchestration_steps + 1):
            action, diagnostics = self.orchestrator.decide(
                user_query=user_query,
                context=context,
                max_parse_retries=self.max_parse_retries,
            )
            _accumulate_orchestrator(counters, diagnostics)
            raw_model_outputs.extend(diagnostics["raw_model_outputs"])
            cleaned_model_outputs.extend(diagnostics["cleaned_model_outputs"])

            if action is None:
                counters["recovery_failure_count"] += 1
                return self._result(
                    parsed_task=parsed_task,
                    context=context,
                    orchestration_steps=orchestration_steps,
                    role_outputs=role_outputs,
                    orchestrator_decisions=orchestrator_decisions,
                    role_actions=role_actions,
                    role_assignments=role_assignments,
                    available_artifacts_snapshots=available_artifacts_snapshots,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    counters=counters,
                    stalled_loop_detected=True,
                    success=False,
                    error_message="Orchestrator could not produce valid typed RoleAction.",
                )

            action_dict = action.to_dict()
            orchestrator_decisions.append(action_dict)
            role_actions.append(action_dict)
            if action.assignment is not None:
                role_assignments.append(action.assignment.to_dict())

            orchestration_steps.append(
                make_orchestration_step(
                    node="orchestrator",
                    action="typed_llm_decide_role",
                    status="ok",
                    details={
                        "worker": "typed_full_llm_multi_agent",
                        "selected_worker": "typed_full_llm_multi_agent",
                        "workflow": "typed_full_llm_role_workflow",
                        "architecture": ARCHITECTURE_NAME,
                        "typed_protocol_version": TYPED_PROTOCOL_VERSION,
                        "llm_driven": True,
                        "role_action": action_dict,
                        "agent_response": {
                            "status": "ok",
                            "agent": "orchestrator",
                            "artifacts": {},
                            "message": "Typed LLM orchestrator produced RoleAction.",
                            "can_continue": True,
                            "requires_retry": False,
                            "missing_artifacts": [],
                            "attempt": 1,
                            "max_attempts": self.max_parse_retries + 1,
                        },
                    },
                    decision="continue",
                )
            )

            if action.action == "finish":
                success = bool(context.get("final_answer"))
                return self._result(
                    parsed_task=parsed_task,
                    context=context,
                    orchestration_steps=orchestration_steps,
                    role_outputs=role_outputs,
                    orchestrator_decisions=orchestrator_decisions,
                    role_actions=role_actions,
                    role_assignments=role_assignments,
                    available_artifacts_snapshots=available_artifacts_snapshots,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    counters=counters,
                    stalled_loop_detected=stalled_loop_detected,
                    success=success,
                    error_message=None if success else "Finished without final_answer.",
                )

            role = str(action.role)
            assignment = action.assignment
            if role not in self.roles or assignment is None:
                counters["invalid_role_action_count"] += 1
                counters["recovery_failure_count"] += 1
                return self._result(
                    parsed_task=parsed_task,
                    context=context,
                    orchestration_steps=orchestration_steps,
                    role_outputs=role_outputs,
                    orchestrator_decisions=orchestrator_decisions,
                    role_actions=role_actions,
                    role_assignments=role_assignments,
                    available_artifacts_snapshots=available_artifacts_snapshots,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    counters=counters,
                    stalled_loop_detected=True,
                    success=False,
                    error_message="Invalid typed role handoff.",
                )

            context.setdefault("role_handoffs", []).append(
                {
                    "step": orchestration_index,
                    "role": role,
                    "assignment": assignment.to_dict(),
                }
            )
            role_output = self.roles[role].run(
                user_query=context["user_query"],
                assignment=assignment,
                context=context,
            )
            output_dict = asdict(role_output)
            role_outputs.append(output_dict)
            raw_model_outputs.extend(role_output.raw_model_outputs)
            cleaned_model_outputs.extend(role_output.cleaned_model_outputs)
            _accumulate_role(counters, role_output)
            stalled_loop_detected = (
                stalled_loop_detected or role_output.stalled_loop_detected
            )
            context.setdefault("role_agent_order", []).append(role)
            context.setdefault("role_outputs", []).append(
                {
                    "role": role,
                    "status": role_output.status,
                    "summary": role_output.summary,
                    "produced_artifact_types": role_output.produced_artifact_types,
                    "artifact_refs": role_output.artifact_refs,
                    "findings": role_output.findings,
                    "needs": role_output.needs,
                }
            )
            available_artifacts_snapshots.append(
                {
                    "step": orchestration_index,
                    "role": role,
                    "available_artifacts": build_typed_available_artifacts(context),
                }
            )
            orchestration_steps.append(
                make_orchestration_step(
                    node=role,
                    action="typed_llm_role_run",
                    status=role_output.status,
                    details={
                        "llm_driven": True,
                        "typed_protocol_version": TYPED_PROTOCOL_VERSION,
                        "assignment": assignment.to_dict(),
                        "agent_response": {
                            "status": role_output.status,
                            "agent": role,
                            "artifacts": {
                                "produced_artifact_types": role_output.produced_artifact_types,
                                "artifact_refs": role_output.artifact_refs,
                            },
                            "message": role_output.summary,
                            "can_continue": role_output.status
                            in {"done", "partial", "cannot_complete"},
                            "requires_retry": role_output.status in {"failed"},
                            "missing_artifacts": [],
                            "attempt": 1,
                            "max_attempts": self.max_parse_retries + 1,
                        },
                    },
                    decision=(
                        "continue"
                        if role_output.status in {"done", "partial", "cannot_complete"}
                        else "fail"
                    ),
                )
            )

            if role_output.status == "failed":
                counters["recovery_failure_count"] += 1
                return self._result(
                    parsed_task=parsed_task,
                    context=context,
                    orchestration_steps=orchestration_steps,
                    role_outputs=role_outputs,
                    orchestrator_decisions=orchestrator_decisions,
                    role_actions=role_actions,
                    role_assignments=role_assignments,
                    available_artifacts_snapshots=available_artifacts_snapshots,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    counters=counters,
                    stalled_loop_detected=stalled_loop_detected,
                    success=False,
                    error_message=f"{role} failed: {role_output.summary}",
                )

        return self._result(
            parsed_task=parsed_task,
            context=context,
            orchestration_steps=orchestration_steps,
            role_outputs=role_outputs,
            orchestrator_decisions=orchestrator_decisions,
            role_actions=role_actions,
            role_assignments=role_assignments,
            available_artifacts_snapshots=available_artifacts_snapshots,
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            counters=counters,
            stalled_loop_detected=True,
            success=False,
            error_message=f"Exceeded max_orchestration_steps={self.max_orchestration_steps}.",
        )

    def _result(
        self,
        *,
        parsed_task: dict[str, Any],
        context: dict[str, Any],
        orchestration_steps: list[dict[str, Any]],
        role_outputs: list[dict[str, Any]],
        orchestrator_decisions: list[dict[str, Any]],
        role_actions: list[dict[str, Any]],
        role_assignments: list[dict[str, Any]],
        available_artifacts_snapshots: list[dict[str, Any]],
        raw_model_outputs: list[str],
        cleaned_model_outputs: list[str],
        counters: dict[str, int],
        stalled_loop_detected: bool,
        success: bool,
        error_message: str | None,
    ) -> LLMTypedMultiAgentResult:
        trace = self.client.get_trace()
        return LLMTypedMultiAgentResult(
            answer=str(context.get("final_answer") or ""),
            parsed_task=parsed_task,
            tool_results=build_multi_agent_tool_results(context),
            trace=trace,
            orchestration_steps=orchestration_steps,
            role_outputs=role_outputs,
            orchestrator_decisions=orchestrator_decisions,
            role_actions=role_actions,
            role_assignments=role_assignments,
            tool_observations=list(context.get("tool_observations") or []),
            available_artifacts_snapshots=available_artifacts_snapshots,
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            role_agent_order=list(context.get("role_agent_order") or []),
            actual_tool_sequence=[
                str(call.get("tool_name")) for call in trace.get("calls", [])
            ],
            typed_protocol_version=TYPED_PROTOCOL_VERSION,
            architecture=ARCHITECTURE_NAME,
            parse_error_count=counters["parse_error_count"],
            invalid_json_count=counters["invalid_json_count"],
            unknown_format_count=counters["unknown_format_count"],
            invalid_assignment_count=counters["invalid_assignment_count"],
            invalid_role_action_count=counters["invalid_role_action_count"],
            invalid_role_response_count=counters["invalid_role_response_count"],
            invalid_final_answer_count=counters["invalid_final_answer_count"],
            invalid_tool_name_count=counters["invalid_tool_name_count"],
            invalid_role_protocol_count=counters["invalid_role_protocol_count"],
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


def build_typed_state_packet(
    *,
    user_query: str,
    current_role: str,
    current_assignment: RoleAssignment | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build compact role-visible typed state without evaluator hints."""

    return {
        "user_query": user_query,
        "current_role": current_role,
        "current_assignment": (
            current_assignment.to_dict() if current_assignment is not None else None
        ),
        "available_artifacts": build_typed_available_artifacts(context),
        "previous_role_outputs": list(context.get("role_outputs") or [])[-8:],
        "recent_tool_observations": list(context.get("tool_observations") or [])[-6:],
        "completed_successful_tool_calls": list(
            context.get("completed_tool_calls") or []
        )[-10:],
    }


def build_typed_available_artifacts(context: dict[str, Any]) -> dict[str, Any]:
    """Return grouped available artifacts for typed state packets."""

    data = context.get("data_artifacts") or {}
    math = context.get("math_artifacts") or {}
    series = []
    for region_id, item in (data.get("series_by_region") or {}).items():
        series.append(
            {
                "region_id": region_id,
                "series_id": item.get("series_id"),
                "n_points": (item.get("metadata") or {}).get("n_points"),
            }
        )
    stats = []
    for region_id, item in (math.get("stats_by_region") or {}).items():
        stats.append({"region_id": region_id, "stats_id": item.get("stats_id")})
    thresholds = []
    for collection_name in ["high_tec", "stable_intervals"]:
        for region_id, item in (math.get(collection_name) or {}).items():
            threshold = item.get("threshold") or item.get("thresholds") or {}
            if threshold.get("threshold_id"):
                thresholds.append(
                    {
                        "region_id": region_id,
                        "threshold_id": threshold.get("threshold_id"),
                        "kind": collection_name,
                    }
                )
    high_intervals = []
    for region_id, item in (math.get("high_tec") or {}).items():
        intervals = item.get("intervals") or {}
        if intervals:
            high_intervals.append(
                {
                    "region_id": region_id,
                    "n_intervals": intervals.get("n_intervals"),
                    "threshold_id": intervals.get("threshold_id"),
                }
            )
    stable_intervals = []
    for region_id, item in (math.get("stable_intervals") or {}).items():
        intervals = item.get("intervals") or {}
        if intervals:
            stable_intervals.append(
                {
                    "region_id": region_id,
                    "n_intervals": intervals.get("n_intervals"),
                    "threshold_id": intervals.get("threshold_id"),
                }
            )
    comparisons = []
    comparison = math.get("comparison")
    if isinstance(comparison, dict) and comparison:
        comparisons.append(
            {
                "comparison_id": comparison.get("comparison_id"),
                "regions": comparison.get("regions"),
            }
        )
    return {
        "series": series,
        "stats": stats,
        "thresholds": thresholds,
        "high_intervals": high_intervals,
        "stable_intervals": stable_intervals,
        "comparisons": comparisons,
    }


def _merge_typed_role_response(context: dict[str, Any], response: RoleResponse) -> None:
    context.setdefault("typed_role_responses", []).append(response.to_dict())
    if response.role == "analysis_agent":
        context.setdefault("analysis_artifacts", {}).setdefault("findings", []).extend(
            response.findings
        )


def _empty_orchestrator_diagnostics() -> dict[str, Any]:
    return {
        "raw_model_outputs": [],
        "cleaned_model_outputs": [],
        "parse_error_count": 0,
        "invalid_json_count": 0,
        "unknown_format_count": 0,
        "invalid_assignment_count": 0,
        "invalid_role_action_count": 0,
        "repair_attempt_count": 0,
    }


def _empty_global_counters() -> dict[str, int]:
    return {
        "parse_error_count": 0,
        "invalid_json_count": 0,
        "unknown_format_count": 0,
        "invalid_assignment_count": 0,
        "invalid_role_action_count": 0,
        "invalid_role_response_count": 0,
        "invalid_final_answer_count": 0,
        "invalid_tool_name_count": 0,
        "invalid_role_protocol_count": 0,
        "forbidden_tool_call_count": 0,
        "repeated_tool_call_count": 0,
        "repair_attempt_count": 0,
        "retry_count": 0,
        "recovery_attempt_count": 0,
        "recovery_success_count": 0,
        "recovery_failure_count": 0,
    }


def _accumulate_parse_error(
    diagnostics: dict[str, Any],
    parsed: TypedParseResult,
) -> None:
    diagnostics["parse_error_count"] += 1
    if parsed.error_code == "invalid_json":
        diagnostics["invalid_json_count"] += 1
    elif parsed.error_code == "unknown_format":
        diagnostics["unknown_format_count"] += 1


def _accumulate_orchestrator(
    counters: dict[str, int],
    diagnostics: dict[str, Any],
) -> None:
    for key in [
        "parse_error_count",
        "invalid_json_count",
        "unknown_format_count",
        "invalid_assignment_count",
        "invalid_role_action_count",
        "repair_attempt_count",
    ]:
        counters[key] += int(diagnostics.get(key) or 0)


def _accumulate_role(counters: dict[str, int], role_output: LLMTypedRoleOutput) -> None:
    for key in [
        "parse_error_count",
        "invalid_json_count",
        "unknown_format_count",
        "invalid_role_protocol_count",
        "invalid_tool_name_count",
        "forbidden_tool_call_count",
        "invalid_role_response_count",
        "invalid_final_answer_count",
        "repeated_tool_call_count",
        "repair_attempt_count",
    ]:
        counters[key] += int(getattr(role_output, key))


__all__ = [
    "ARCHITECTURE_NAME",
    "LLMFullTypedMultiAgent",
    "LLMTypedAnalysisAgent",
    "LLMTypedDataAgent",
    "LLMTypedMathAgent",
    "LLMTypedMultiAgentResult",
    "LLMTypedOrchestratorAgent",
    "LLMTypedReportAgent",
    "LLMTypedRoleAgent",
    "LLMTypedRoleOutput",
    "TYPED_PROTOCOL_VERSION",
    "ToolObservation",
    "build_typed_available_artifacts",
    "build_typed_state_packet",
    "compact_tool_result",
]
