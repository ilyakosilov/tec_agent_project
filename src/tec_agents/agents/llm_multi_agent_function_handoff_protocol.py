"""Function-handoff protocol for a simplified full LLM multi-agent run.

This experiment intentionally uses one output shape for both internal role
handoffs and TEC tool calls:

<tool_call>
{"name": "...", "arguments": {...}}
</tool_call>

The runtime distinguishes internal functions from TEC tools by name and role
allowlists. This keeps all roles LLM-driven while avoiding a separate
role_action / role_response / final_answer block family.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from tec_agents.llm.tool_call_parser import ToolCall, parse_tool_call


FUNCTION_HANDOFF_PROTOCOL_VERSION = "function_handoff_v2"
ARCHITECTURE_NAME = "qwen_multi_agent_function_handoff_full_llm"

ROLE_NAMES = {"data_agent", "math_agent", "analysis_agent", "report_agent"}

ORCHESTRATOR_FUNCTIONS: dict[str, str] = {
    "call_data_agent": "data_agent",
    "call_math_agent": "math_agent",
    "call_analysis_agent": "analysis_agent",
    "call_report_agent": "report_agent",
}

RETURN_FUNCTION = "return_to_orchestrator"

DATA_TOOLS = {"tec_get_timeseries", "tec_series_profile"}
MATH_TOOLS = {
    "tec_compute_series_stats",
    "tec_compute_high_threshold",
    "tec_detect_high_intervals",
    "tec_compute_stability_thresholds",
    "tec_detect_stable_intervals",
    "tec_compare_stats",
}

ROLE_FUNCTION_ALLOWLIST: dict[str, set[str]] = {
    "orchestrator": set(ORCHESTRATOR_FUNCTIONS),
    "data_agent": DATA_TOOLS | {RETURN_FUNCTION},
    "math_agent": MATH_TOOLS | {RETURN_FUNCTION},
    "analysis_agent": {RETURN_FUNCTION},
    "report_agent": {RETURN_FUNCTION},
}

FORBIDDEN_FUNCTION_NAMES = {
    "role_response",
    "role_action",
    "final_answer",
    "orchestrator",
    "data_agent",
    "math_agent",
    "analysis_agent",
    "report_agent",
    "tec_build_report",
    "tec_compare_regions",
    "compare_regions",
}

FORBIDDEN_PROMPT_KEYS = {
    "expected_tool_sequence",
    "expected_role_agent_order",
    "expected_role_order",
    "GoldRunner",
    "gold_result",
    "gold_outputs",
    "metrics",
    "verdict_checks",
    "missing_goal_artifacts",
    "remaining_goal_artifacts",
    "remaining_goals",
    "remaining evaluator goals",
    "remaining_goal",
    "missing_goal",
    "deterministic_baseline_trace",
    "deterministic baseline trace",
    "tool_sequence_match",
    "role_agent_order_match",
    "artifact_flow_valid",
    "next_required_tool",
    "next_required_role",
    "next tool",
    "next role",
}


@dataclass
class FunctionHandoffCall:
    """Parsed function/tool call emitted by a role."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> "FunctionHandoffCall":
        return cls(name=call.name, arguments=dict(call.arguments or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionParseResult:
    ok: bool
    call: FunctionHandoffCall | None = None
    error_code: str | None = None
    error_message: str | None = None
    raw_text: str = ""
    cleaned_text: str = ""
    block_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "call": self.call.to_dict() if self.call is not None else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "block_count": self.block_count,
        }


def parse_function_handoff_output(text: str) -> FunctionParseResult:
    """Parse the first complete tool_call block from a model response."""

    raw = str(text or "")
    block_count = count_tool_call_blocks(raw)
    cleaned = clean_function_handoff_output(raw)
    parsed = parse_tool_call(cleaned or raw)
    if parsed.tool_call is None:
        return FunctionParseResult(
            ok=False,
            error_code=str(parsed.error_code.value if parsed.error_code else "unknown_format"),
            error_message=parsed.error_message or "No valid tool_call block found.",
            raw_text=raw,
            cleaned_text=cleaned,
            block_count=block_count,
        )
    return FunctionParseResult(
        ok=True,
        call=FunctionHandoffCall.from_tool_call(parsed.tool_call),
        raw_text=raw,
        cleaned_text=cleaned,
        block_count=block_count,
    )


def clean_function_handoff_output(raw_text: str) -> str:
    """Remove common chat-template noise and return the first tool_call block."""

    text = _strip_generation_noise(raw_text)
    match = _tool_call_pattern().search(text)
    return match.group(0).strip() if match else ""


def count_tool_call_blocks(raw_text: str) -> int:
    """Return the number of complete tool_call blocks after cleanup."""

    return len(_tool_call_pattern().findall(_strip_generation_noise(raw_text)))


def validate_function_call_for_role(role: str, function_name: str) -> str | None:
    """Return a role/protocol error, or None if the function call is allowed."""

    if function_name in FORBIDDEN_FUNCTION_NAMES:
        return f"{function_name!r} is not an allowed function in this protocol."
    if role not in ROLE_FUNCTION_ALLOWLIST:
        return f"Unknown role: {role!r}."
    if function_name not in ROLE_FUNCTION_ALLOWLIST[role]:
        allowed = ", ".join(sorted(ROLE_FUNCTION_ALLOWLIST[role]))
        return f"{function_name!r} is not allowed for {role}. Allowed: {allowed}."
    return None


def is_orchestrator_handoff(function_name: str) -> bool:
    return function_name in ORCHESTRATOR_FUNCTIONS


def role_for_handoff_function(function_name: str) -> str | None:
    return ORCHESTRATOR_FUNCTIONS.get(function_name)


def is_return_to_orchestrator(function_name: str) -> bool:
    return function_name == RETURN_FUNCTION


def is_tec_tool(function_name: str) -> bool:
    return function_name in DATA_TOOLS or function_name in MATH_TOOLS


def validate_handoff_arguments(arguments: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "Handoff function arguments must be an object."
    if "message" in arguments and not isinstance(arguments.get("message"), str):
        return "Optional handoff field 'message' must be a string when present."
    return None


def normalize_role_return(arguments: dict[str, Any], *, role: str) -> dict[str, Any]:
    """Normalize return_to_orchestrator arguments into a trace-safe dict."""

    status = str(arguments.get("status") or "done")
    if status not in {"done", "cannot_complete", "failed"}:
        status = "failed"
    message = str(arguments.get("message") or status)
    findings_raw = arguments.get("findings") or []
    if isinstance(findings_raw, str):
        findings = [findings_raw] if findings_raw.strip() else []
    elif isinstance(findings_raw, list):
        findings = [str(item) for item in findings_raw if str(item).strip()]
    else:
        findings = []
    final_answer = str(arguments.get("final_answer") or "")
    return {
        "role": role,
        "status": status,
        "message": message,
        "findings": findings,
        "final_answer": final_answer,
    }


def validate_role_return(arguments: dict[str, Any], *, role: str) -> str | None:
    normalized = normalize_role_return(arguments, role=role)
    if not normalized["message"].strip():
        return "return_to_orchestrator requires a non-empty message."
    if role != "report_agent" and normalized["final_answer"]:
        return "Only report_agent may include final_answer."
    return None


def safe_state_payload(value: Any) -> Any:
    """Drop forbidden evaluator-only keys from prompt-visible state."""

    def forbidden(key: str) -> bool:
        return key in FORBIDDEN_PROMPT_KEYS

    if isinstance(value, dict):
        return {
            str(key): safe_state_payload(item)
            for key, item in value.items()
            if not forbidden(str(key))
        }
    if isinstance(value, list):
        return [safe_state_payload(item) for item in value]
    return value


def tool_argument_contract(tool_name: str) -> dict[str, Any] | None:
    """Compact exact argument contract for prompt-visible TEC tools."""

    contracts: dict[str, dict[str, Any]] = {
        "tec_get_timeseries": {
            "dataset_ref": "default",
            "region_id": "<one region_id>",
            "start": "<inclusive start date>",
            "end": "<exclusive end date>",
            "freq": None,
        },
        "tec_series_profile": {"series_id": "<one visible series_id>"},
        "tec_compute_series_stats": {
            "series_id": "<one visible series_id>",
            "metrics": ["<optional metric names>"],
        },
        "tec_compute_high_threshold": {
            "series_id": "<one visible series_id>",
            "method": "quantile",
            "q": 0.9,
            "value": None,
        },
        "tec_detect_high_intervals": {
            "series_id": "<one visible series_id>",
            "threshold_id": "<visible threshold_id>",
            "min_duration_minutes": 0.0,
            "merge_gap_minutes": 0.0,
        },
        "tec_compute_stability_thresholds": {
            "series_id": "<one visible series_id>",
            "window_minutes": 180,
            "method": "quantile",
            "q_delta": 0.6,
            "q_std": 0.6,
        },
        "tec_detect_stable_intervals": {
            "series_id": "<one visible series_id>",
            "threshold_id": "<visible stability threshold_id>",
            "min_duration_minutes": 180.0,
            "merge_gap_minutes": 60.0,
        },
        "tec_compare_stats": {
            "stats_ids": ["<visible stats_id>", "<visible stats_id>"],
            "reference_stats_id": None,
            "metrics": ["<optional metric names>"],
        },
    }
    contract = contracts.get(tool_name)
    return dict(contract) if contract is not None else None


def _tool_call_pattern() -> re.Pattern[str]:
    return re.compile(
        r"<tool_call\b[^>]*>\s*(.*?)\s*</tool_call>",
        flags=re.DOTALL | re.IGNORECASE,
    )


def _strip_generation_noise(raw_text: str) -> str:
    text = str(raw_text or "")
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<\|im_start\|>\s*(?:system|user|assistant)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("<|im_end|>", "")
    text = re.sub(r"(?im)^\s*(?:system|user|assistant|function)\s*$", "", text)
    return text.strip()


__all__ = [
    "ARCHITECTURE_NAME",
    "DATA_TOOLS",
    "FORBIDDEN_FUNCTION_NAMES",
    "FORBIDDEN_PROMPT_KEYS",
    "FUNCTION_HANDOFF_PROTOCOL_VERSION",
    "FunctionHandoffCall",
    "FunctionParseResult",
    "MATH_TOOLS",
    "ORCHESTRATOR_FUNCTIONS",
    "RETURN_FUNCTION",
    "ROLE_FUNCTION_ALLOWLIST",
    "ROLE_NAMES",
    "clean_function_handoff_output",
    "count_tool_call_blocks",
    "is_orchestrator_handoff",
    "is_return_to_orchestrator",
    "is_tec_tool",
    "normalize_role_return",
    "parse_function_handoff_output",
    "role_for_handoff_function",
    "safe_state_payload",
    "tool_argument_contract",
    "validate_function_call_for_role",
    "validate_handoff_arguments",
    "validate_role_return",
]
