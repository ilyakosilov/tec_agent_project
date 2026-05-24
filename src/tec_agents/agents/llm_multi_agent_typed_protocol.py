"""Typed protocol models and parsers for the typed full LLM multi-agent run.

This module is intentionally separate from the historical free-form
``llm_multi_agent`` experiment. It defines typed XML-like blocks for role
handoffs, role assignments, tool calls, tool observations, role responses, and
final answers without exposing evaluator expectations to the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from tec_agents.agents.llm_single_agent import canonical_arguments_json


TYPED_PROTOCOL_VERSION = "typed_role_contract_v1"

ROLE_NAMES = {"data_agent", "math_agent", "analysis_agent", "report_agent"}

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

PROTOCOL_TOOL_NAMES = {"role_action", "role_response", "final_answer", "tool_call"}
FORBIDDEN_TOOL_NAMES = {
    "orchestrator",
    "data_agent",
    "math_agent",
    "analysis_agent",
    "report_agent",
    "role_action",
    "role_response",
    "final_answer",
    "compare_regions",
    "tec_compare_regions",
    "tec_build_report",
}
FORBIDDEN_PROMPT_KEYS = {
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

BLOCK_TAGS = ("role_action", "tool_call", "role_response", "final_answer")


@dataclass
class RoleScope:
    dataset_ref: str | None = None
    regions: list[str] = field(default_factory=list)
    start: str | None = None
    end: str | None = None
    task_intent: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RoleScope":
        data = data or {}
        regions = data.get("regions") or []
        if isinstance(regions, str):
            regions = [regions]
        return cls(
            dataset_ref=_nullable_str(data.get("dataset_ref")),
            regions=[str(region) for region in regions if region is not None],
            start=_nullable_str(data.get("start")),
            end=_nullable_str(data.get("end")),
            task_intent=_nullable_str(data.get("task_intent")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoleAssignment:
    objective: str
    task_summary: str
    scope: RoleScope
    available_input_types: list[str]
    expected_output_type: str
    completion_criteria: str
    constraints: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoleAssignment":
        return cls(
            objective=str(data.get("objective") or ""),
            task_summary=str(data.get("task_summary") or ""),
            scope=RoleScope.from_dict(data.get("scope") or {}),
            available_input_types=[
                str(item) for item in (data.get("available_input_types") or [])
            ],
            expected_output_type=str(data.get("expected_output_type") or "other"),
            completion_criteria=str(data.get("completion_criteria") or ""),
            constraints=[str(item) for item in (data.get("constraints") or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoleAction:
    action: str
    role: str | None = None
    assignment: RoleAssignment | None = None
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoleAction":
        assignment = data.get("assignment")
        return cls(
            action=str(data.get("action") or ""),
            role=_nullable_str(data.get("role")),
            assignment=(
                RoleAssignment.from_dict(assignment)
                if isinstance(assignment, dict)
                else None
            ),
            reason=str(data.get("reason") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TypedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedToolCall":
        arguments = data.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        return cls(name=str(data.get("name") or ""), arguments=arguments)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    tool_name: str
    status: str
    produced_artifact_type: str | None = None
    artifact_id: str | None = None
    artifact_refs: list[str] = field(default_factory=list)
    summary: str = ""
    raw_result_compact: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoleNeed:
    type: str
    reason: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoleNeed":
        return cls(type=str(data.get("type") or ""), reason=str(data.get("reason") or ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoleResponse:
    status: str
    role: str = ""
    summary: str = ""
    message: str = ""
    produced_artifact_types: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    needs: list[RoleNeed] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoleResponse":
        needs = data.get("needs") or []
        message = str(data.get("message") or data.get("summary") or "")
        return cls(
            status=str(data.get("status") or "done"),
            role=str(data.get("role") or ""),
            summary=str(data.get("summary") or data.get("message") or ""),
            message=message,
            produced_artifact_types=[
                str(item) for item in (data.get("produced_artifact_types") or [])
            ],
            artifact_refs=[str(item) for item in (data.get("artifact_refs") or [])],
            findings=[str(item) for item in (data.get("findings") or [])],
            needs=[
                RoleNeed.from_dict(item)
                for item in needs
                if isinstance(item, dict)
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalAnswer:
    answer: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalAnswer":
        return cls(answer=str(data.get("answer") or ""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TypedParseResult:
    ok: bool
    kind: str
    value: Any = None
    error_code: str | None = None
    error_message: str | None = None
    raw_text: str = ""
    cleaned_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        return {
            "ok": self.ok,
            "kind": self.kind,
            "value": value,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
        }


def parse_typed_role_action(text: str) -> TypedParseResult:
    """Parse and validate one typed role_action block."""

    result = _parse_json_block(text, "role_action")
    if not result.ok:
        return result
    data = result.value
    if not isinstance(data, dict):
        return _error("role_action", "invalid_json", "role_action JSON must be an object.", text)
    if _contains_forbidden_key(data):
        return _error("role_action", "forbidden_field", "role_action contains evaluator-only fields.", text)
    action = RoleAction.from_dict(data)
    validation_error = validate_role_action(action)
    if validation_error:
        return _error("role_action", "schema_error", validation_error, text)
    result.value = action
    return result


def parse_typed_tool_call(text: str) -> TypedParseResult:
    """Parse one typed tool_call block."""

    result = _parse_json_block(text, "tool_call")
    if not result.ok:
        return result
    data = result.value
    if not isinstance(data, dict):
        return _error("tool_call", "invalid_json", "tool_call JSON must be an object.", text)
    call = TypedToolCall.from_dict(data)
    if not call.name:
        return _error("tool_call", "schema_error", "tool_call.name is required.", text)
    result.value = call
    return result


def parse_typed_role_response(text: str) -> TypedParseResult:
    """Parse and validate one typed role_response block."""

    result = _parse_json_block(text, "role_response")
    if not result.ok:
        return result
    data = result.value
    if not isinstance(data, dict):
        return _error("role_response", "invalid_json", "role_response JSON must be an object.", text)
    response = RoleResponse.from_dict(data)
    validation_error = validate_role_response(response)
    if validation_error:
        return _error("role_response", "schema_error", validation_error, text)
    result.value = response
    return result


def parse_typed_final_answer(text: str) -> TypedParseResult:
    """Parse and validate one final_answer block."""

    result = _parse_json_block(text, "final_answer")
    if not result.ok:
        return result
    data = result.value
    if isinstance(data, str):
        data = {"answer": data}
    if not isinstance(data, dict):
        return _error("final_answer", "invalid_json", "final_answer JSON must be an object.", text)
    answer = FinalAnswer.from_dict(data)
    if not answer.answer.strip():
        return _error("final_answer", "schema_error", "final_answer.answer is required.", text)
    result.value = answer
    return result


def parse_typed_role_output(text: str, role: str) -> TypedParseResult:
    """Parse role output and enforce the block type allowed for that role."""

    blocks = find_protocol_blocks(text)
    if not blocks:
        return _error("unknown", "unknown_format", "No typed protocol block found.", text)
    if len(blocks) > 1:
        return _error("unknown", "multiple_blocks", "Return exactly one protocol block.", text)

    tag = blocks[0]["tag"]
    if role == "orchestrator":
        if tag != "role_action":
            return _error(tag, "protocol_violation", "Orchestrator may output only role_action.", text)
        return parse_typed_role_action(text)

    if role in {"data_agent", "math_agent"}:
        if tag == "tool_call":
            parsed = parse_typed_tool_call(text)
            if not parsed.ok:
                return parsed
            call = parsed.value
            protocol_error = validate_tool_call_for_role(role, call.name)
            if protocol_error:
                return _error("tool_call", "protocol_violation", protocol_error, text)
            return parsed
        if tag == "role_response":
            parsed = parse_typed_role_response(text)
            if parsed.ok and parsed.value.role and parsed.value.role != role:
                return _error("role_response", "schema_error", f"role_response.role must be {role}.", text)
            return parsed
        return _error(tag, "protocol_violation", f"{role} may output only tool_call or role_response.", text)

    if role == "analysis_agent":
        if tag != "role_response":
            return _error(tag, "protocol_violation", "AnalysisAgent may output only role_response.", text)
        parsed = parse_typed_role_response(text)
        if parsed.ok and parsed.value.role and parsed.value.role != "analysis_agent":
            return _error("role_response", "schema_error", "role_response.role must be analysis_agent.", text)
        return parsed

    if role == "report_agent":
        if tag == "final_answer":
            return parse_typed_final_answer(text)
        if tag == "role_response":
            parsed = parse_typed_role_response(text)
            if parsed.ok and parsed.value.role and parsed.value.role != "report_agent":
                return _error("role_response", "schema_error", "role_response.role must be report_agent.", text)
            return parsed
        return _error(tag, "protocol_violation", "ReportAgent may output only final_answer or role_response.", text)

    return _error("unknown", "protocol_violation", f"Unknown role: {role}", text)


def validate_role_action(action: RoleAction) -> str | None:
    """Return validation error for a typed RoleAction."""

    if action.action == "call_role":
        if action.role not in ROLE_NAMES:
            return f"Unknown or missing role for call_role: {action.role!r}."
        if action.assignment is None:
            return "call_role requires a non-null assignment."
        return validate_role_assignment(action.assignment)
    if action.action == "finish":
        if action.assignment is not None:
            return "finish must not include an assignment."
        return None
    return "RoleAction.action must be 'call_role' or 'finish'."


def validate_role_assignment(assignment: RoleAssignment) -> str | None:
    """Return validation error for RoleAssignment."""

    data = assignment.to_dict()
    if _contains_forbidden_key(data):
        return "RoleAssignment contains evaluator-only forbidden fields."
    if not assignment.objective:
        return "RoleAssignment.objective is required."
    if not assignment.task_summary:
        return "RoleAssignment.task_summary is required."
    if assignment.expected_output_type not in {
        "data_artifacts",
        "computed_artifacts",
        "findings",
        "final_answer",
        "other",
    }:
        return "RoleAssignment.expected_output_type is invalid."
    if not assignment.completion_criteria:
        return "RoleAssignment.completion_criteria is required."
    return None


def validate_role_response(response: RoleResponse) -> str | None:
    """Return validation error for RoleResponse."""

    if response.status not in {"done", "partial", "cannot_complete", "failed"}:
        return "RoleResponse.status must be done, partial, cannot_complete, or failed."
    if response.role and response.role not in ROLE_NAMES:
        return "RoleResponse.role must be a known worker role."
    if not (response.message.strip() or response.summary.strip()):
        return "RoleResponse.message is required."
    return None


def validate_tool_call_for_role(role: str, tool_name: str) -> str | None:
    """Return protocol/tool validation error for a tool call in a role."""

    if tool_name in ROLE_NAMES or tool_name in PROTOCOL_TOOL_NAMES or tool_name == "orchestrator":
        return f"{tool_name!r} is not a tool. Agents and protocol blocks cannot be tool names."
    if tool_name in {"compare_regions", "tec_compare_regions", "tec_build_report"}:
        return f"{tool_name!r} is an aggregate/legacy tool and is not allowed in the typed primitive workflow."
    if role in {"analysis_agent", "report_agent", "orchestrator"}:
        return f"{role} must not call tools."
    if tool_name not in ROLE_TOOL_ALLOWLIST.get(role, set()):
        return f"Tool {tool_name!r} is not allowed for {role}."
    return None


def make_tool_observation(
    *,
    tool_name: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> ToolObservation:
    """Build a compact typed ToolObservation from an existing tool result."""

    result = result or {}
    artifact_type = artifact_type_for_tool(tool_name)
    artifact_id = artifact_id_from_result(tool_name, result)
    refs = [artifact_id] if artifact_id else []
    summary = _summary_for_tool(tool_name, result, status)
    return ToolObservation(
        tool_name=tool_name,
        status=status,
        produced_artifact_type=artifact_type,
        artifact_id=artifact_id,
        artifact_refs=refs,
        summary=summary,
        raw_result_compact=compact_tool_result(tool_name, result) if result else None,
        error=error,
    )


def artifact_type_for_tool(tool_name: str) -> str | None:
    """Return the artifact type normally produced by a tool."""

    return {
        "tec_get_timeseries": "series_id",
        "tec_series_profile": "profile",
        "tec_compute_series_stats": "stats_id",
        "tec_compute_high_threshold": "threshold_id",
        "tec_detect_high_intervals": "high_intervals",
        "tec_compute_stability_thresholds": "stability_threshold_id",
        "tec_detect_stable_intervals": "stable_intervals",
        "tec_compare_stats": "comparison_id",
    }.get(tool_name)


def artifact_id_from_result(tool_name: str, result: dict[str, Any]) -> str | None:
    """Extract primary artifact id from a tool result."""

    if tool_name == "tec_get_timeseries":
        return _nullable_str(result.get("series_id"))
    if tool_name == "tec_compute_series_stats":
        return _nullable_str(result.get("stats_id"))
    if tool_name in {"tec_compute_high_threshold", "tec_compute_stability_thresholds"}:
        return _nullable_str(result.get("threshold_id"))
    if tool_name == "tec_compare_stats":
        return _nullable_str(result.get("comparison_id"))
    if tool_name in {"tec_detect_high_intervals", "tec_detect_stable_intervals"}:
        return _nullable_str(result.get("intervals_id") or result.get("threshold_id"))
    if tool_name == "tec_series_profile":
        return _nullable_str(result.get("profile_id") or result.get("series_id"))
    return None


def compact_tool_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return prompt-safe compact tool result metadata."""

    keys = [
        "series_id",
        "stats_id",
        "threshold_id",
        "comparison_id",
        "region_id",
        "n_intervals",
        "threshold_value",
        "value",
        "n_points",
        "metrics",
        "regions",
        "stats_ids",
    ]
    compact = {key: result[key] for key in keys if key in result}
    metadata = result.get("metadata") or {}
    if isinstance(metadata, dict):
        compact["metadata"] = {
            key: metadata[key]
            for key in [
                "dataset_ref",
                "region_id",
                "requested_start",
                "requested_end",
                "actual_start",
                "actual_end",
                "interval_convention",
                "n_points",
            ]
            if key in metadata
        }
    return compact


def tool_call_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return canonical duplicate-call key."""

    return f"{tool_name}:{canonical_arguments_json(arguments)}"


def find_protocol_blocks(text: str) -> list[dict[str, Any]]:
    """Return all supported protocol blocks in text."""

    blocks: list[dict[str, Any]] = []
    for tag in BLOCK_TAGS:
        pattern = re.compile(
            rf"<{tag}\b[^>]*>\s*(.*?)\s*</{tag}>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            blocks.append(
                {
                    "tag": tag,
                    "start": match.start(),
                    "end": match.end(),
                    "body": match.group(1).strip(),
                    "block": match.group(0).strip(),
                }
            )
    return sorted(blocks, key=lambda item: item["start"])


def clean_typed_output(raw_text: str) -> str:
    """Return the first complete typed protocol block after chat-template cleanup."""

    text = _strip_generation_noise(raw_text)
    blocks = find_protocol_blocks(text)
    if not blocks:
        return ""
    return blocks[0]["block"]


def _strip_generation_noise(raw_text: str) -> str:
    """Remove common chat-template and reasoning fragments from model output."""

    text = str(raw_text or "")
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<\|im_start\|>\s*(?:system|user|assistant)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("<|im_end|>", "")
    text = re.sub(r"(?im)^\s*(?:system|user|assistant)\s*$", "", text)
    return text.strip()


def _parse_json_block(text: str, tag: str) -> TypedParseResult:
    blocks = [block for block in find_protocol_blocks(text) if block["tag"] == tag]
    if not blocks:
        return _error(tag, "unknown_format", f"No {tag} block found.", text)
    if len(find_protocol_blocks(text)) > 1:
        return _error(tag, "multiple_blocks", "Return exactly one protocol block.", text)
    body = blocks[0]["body"]
    try:
        data = json.loads(body)
    except Exception as exc:
        return _error(tag, "invalid_json", f"Invalid {tag} JSON: {exc}", text, cleaned=blocks[0]["block"])
    return TypedParseResult(
        ok=True,
        kind=tag,
        value=data,
        raw_text=text,
        cleaned_text=blocks[0]["block"],
    )


def _error(
    kind: str,
    code: str,
    message: str,
    text: str,
    *,
    cleaned: str = "",
) -> TypedParseResult:
    return TypedParseResult(
        ok=False,
        kind=kind,
        error_code=code,
        error_message=message,
        raw_text=text,
        cleaned_text=cleaned,
    )


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in FORBIDDEN_PROMPT_KEYS:
                return True
            if _contains_forbidden_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _summary_for_tool(tool_name: str, result: dict[str, Any], status: str) -> str:
    if status != "ok":
        return f"{tool_name} returned status={status}."
    metadata = result.get("metadata") or {}
    region = result.get("region_id") or metadata.get("region_id")
    n_points = result.get("n_points") or metadata.get("n_points")
    if tool_name == "tec_get_timeseries":
        return f"Loaded TEC time series for {region or 'unknown region'} with {n_points or 'unknown'} points."
    if tool_name == "tec_compute_series_stats":
        return f"Computed statistics for {region or 'a series'}."
    if tool_name == "tec_compare_stats":
        return "Computed pairwise statistics comparison."
    if tool_name in {"tec_detect_high_intervals", "tec_detect_stable_intervals"}:
        return f"Detected {result.get('n_intervals', 'unknown')} intervals."
    return f"{tool_name} completed successfully."


__all__ = [
    "FORBIDDEN_PROMPT_KEYS",
    "FORBIDDEN_TOOL_NAMES",
    "FinalAnswer",
    "ROLE_NAMES",
    "ROLE_TOOL_ALLOWLIST",
    "RoleAction",
    "RoleAssignment",
    "RoleNeed",
    "RoleResponse",
    "RoleScope",
    "TYPED_PROTOCOL_VERSION",
    "ToolObservation",
    "TypedParseResult",
    "TypedToolCall",
    "artifact_type_for_tool",
    "clean_typed_output",
    "compact_tool_result",
    "find_protocol_blocks",
    "make_tool_observation",
    "parse_typed_final_answer",
    "parse_typed_role_action",
    "parse_typed_role_output",
    "parse_typed_role_response",
    "parse_typed_tool_call",
    "tool_call_key",
    "validate_role_action",
    "validate_role_assignment",
    "validate_role_response",
    "validate_tool_call_for_role",
]
