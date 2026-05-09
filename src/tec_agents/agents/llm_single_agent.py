"""
LLM single-agent TEC workflow using textual tool calls.

The agent does not implement a new tool layer. It parses model text with the
project parser and sends validated tool calls through LocalMCPClient.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from tec_agents.llm.prompts import build_single_agent_system_prompt
from tec_agents.llm.tool_call_parser import (
    ParseErrorCode,
    parse_final_answer,
    parse_model_output,
    parse_tool_call,
)
from tec_agents.mcp.client import LocalMCPClient, MCPToolView
from tec_agents.mcp.server import ToolCallResponse


AGENT_NAME = "llm_single_agent"


@dataclass
class LLMSingleAgentStep:
    """One LLM step, parse result, and optional tool execution result."""

    step: int
    raw_model_output: str
    parsed_type: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    parse_error_code: str | None = None
    parse_error_message: str | None = None
    tool_status: str | None = None
    tool_result: dict[str, Any] | None = None


@dataclass
class LLMSingleAgentResult:
    """Structured result returned by LLMSingleAgent."""

    answer: str
    parsed_task: dict[str, Any] | None
    tool_results: dict[str, Any] | list[dict[str, Any]]
    trace: dict[str, Any]
    orchestration_steps: list[dict[str, Any]]
    raw_model_outputs: list[str]
    parse_error_count: int
    invalid_json_count: int
    unknown_format_count: int
    repair_attempt_count: int
    tool_sequence: list[str]
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


class LLMSingleAgent:
    """Single LLM agent with textual tool-call parsing and MCP-like execution."""

    def __init__(
        self,
        model,
        client: LocalMCPClient,
        max_steps: int = 20,
        max_tool_calls: int = 20,
        max_parse_retries: int = 2,
        max_tool_retries: int = 2,
        temperature: float = 0.0,
    ):
        self.model = model
        self.client = client
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_parse_retries = max_parse_retries
        self.max_tool_retries = max_tool_retries
        self.temperature = temperature

    def reset(self) -> None:
        """Reset the underlying MCP-like client, if supported."""

        reset_fn = getattr(self.client, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def run(self, user_query: str) -> LLMSingleAgentResult:
        """Run the LLM tool-calling loop for one user query."""

        messages = self._initial_messages(user_query)
        raw_model_outputs: list[str] = []
        step_records: list[LLMSingleAgentStep] = []
        tool_records: list[dict[str, Any]] = []
        tool_sequence: list[str] = []
        parse_error_count = 0
        invalid_json_count = 0
        unknown_format_count = 0
        repair_attempt_count = 0
        consecutive_parse_errors = 0
        tool_call_count = 0
        tool_error_counts: dict[str, int] = {}
        available_tool_names = set(self.client.list_tool_names())

        for step in range(1, self.max_steps + 1):
            raw_output = self.model.generate(
                messages,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            raw_model_outputs.append(raw_output)

            parsed = _parse_agent_output(raw_output)

            if parsed.final_answer is not None:
                record = LLMSingleAgentStep(
                    step=step,
                    raw_model_output=raw_output,
                    parsed_type="final_answer",
                )
                step_records.append(record)
                return self._result(
                    answer=parsed.final_answer,
                    user_query=user_query,
                    tool_records=tool_records,
                    step_records=step_records,
                    raw_model_outputs=raw_model_outputs,
                    parse_error_count=parse_error_count,
                    invalid_json_count=invalid_json_count,
                    unknown_format_count=unknown_format_count,
                    repair_attempt_count=repair_attempt_count,
                    tool_sequence=tool_sequence,
                    success=True,
                    error_message=None,
                )

            if parsed.tool_call is None:
                parse_error_count += 1
                consecutive_parse_errors += 1

                error_code = parsed.error_code
                if error_code == ParseErrorCode.INVALID_JSON:
                    invalid_json_count += 1
                elif error_code == ParseErrorCode.UNKNOWN_FORMAT:
                    unknown_format_count += 1

                record = LLMSingleAgentStep(
                    step=step,
                    raw_model_output=raw_output,
                    parsed_type="parse_error",
                    parse_error_code=(
                        error_code.value if error_code is not None else None
                    ),
                    parse_error_message=parsed.error_message,
                )
                step_records.append(record)

                if consecutive_parse_errors <= self.max_parse_retries:
                    repair_attempt_count += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": _repair_message_for_parse_error(parsed),
                        }
                    )
                    continue

                return self._result(
                    answer="",
                    user_query=user_query,
                    tool_records=tool_records,
                    step_records=step_records,
                    raw_model_outputs=raw_model_outputs,
                    parse_error_count=parse_error_count,
                    invalid_json_count=invalid_json_count,
                    unknown_format_count=unknown_format_count,
                    repair_attempt_count=repair_attempt_count,
                    tool_sequence=tool_sequence,
                    success=False,
                    error_message=parsed.error_message or "Could not parse model output.",
                )

            consecutive_parse_errors = 0
            tool_call = parsed.tool_call

            if tool_call_count >= self.max_tool_calls:
                record = LLMSingleAgentStep(
                    step=step,
                    raw_model_output=raw_output,
                    parsed_type="tool_call",
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    tool_status="error",
                    tool_result={
                        "error": {
                            "error_type": "max_tool_calls_exceeded",
                            "message": (
                                f"Exceeded max_tool_calls={self.max_tool_calls}."
                            ),
                        }
                    },
                )
                step_records.append(record)
                return self._result(
                    answer="",
                    user_query=user_query,
                    tool_records=tool_records,
                    step_records=step_records,
                    raw_model_outputs=raw_model_outputs,
                    parse_error_count=parse_error_count,
                    invalid_json_count=invalid_json_count,
                    unknown_format_count=unknown_format_count,
                    repair_attempt_count=repair_attempt_count,
                    tool_sequence=tool_sequence,
                    success=False,
                    error_message=f"Exceeded max_tool_calls={self.max_tool_calls}.",
                )

            tool_call_count += 1

            if tool_call.name not in available_tool_names:
                response = ToolCallResponse(
                    tool_name=tool_call.name,
                    status="error",
                    result=None,
                    error={
                        "error_type": "unknown_tool",
                        "message": (
                            f"Unknown tool {tool_call.name!r}. Available tools: "
                            f"{sorted(available_tool_names)}"
                        ),
                    },
                )
            else:
                response = self.client.call_tool(
                    tool_call.name,
                    tool_call.arguments,
                    agent_name=AGENT_NAME,
                    step=step,
                )

            response_dict = response.to_dict()
            tool_records.append(
                {
                    "step": step,
                    "tool_name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "response": response_dict,
                }
            )

            if response.status == "ok":
                tool_sequence.append(tool_call.name)

            record = LLMSingleAgentStep(
                step=step,
                raw_model_output=raw_output,
                parsed_type="tool_call",
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                tool_status=response.status,
                tool_result=response_dict,
            )
            step_records.append(record)

            if response.status != "ok":
                retry_key = _tool_retry_key(tool_call.name, tool_call.arguments)
                tool_error_counts[retry_key] = tool_error_counts.get(retry_key, 0) + 1

                if tool_error_counts[retry_key] > self.max_tool_retries:
                    message = _tool_error_message(response)
                    return self._result(
                        answer="",
                        user_query=user_query,
                        tool_records=tool_records,
                        step_records=step_records,
                        raw_model_outputs=raw_model_outputs,
                        parse_error_count=parse_error_count,
                        invalid_json_count=invalid_json_count,
                        unknown_format_count=unknown_format_count,
                        repair_attempt_count=repair_attempt_count,
                        tool_sequence=tool_sequence,
                        success=False,
                        error_message=message,
                    )

            messages.append(_build_tool_observation_message(response))

        return self._result(
            answer="",
            user_query=user_query,
            tool_records=tool_records,
            step_records=step_records,
            raw_model_outputs=raw_model_outputs,
            parse_error_count=parse_error_count,
            invalid_json_count=invalid_json_count,
            unknown_format_count=unknown_format_count,
            repair_attempt_count=repair_attempt_count,
            tool_sequence=tool_sequence,
            success=False,
            error_message=f"Exceeded max_steps={self.max_steps}.",
        )

    def _initial_messages(self, user_query: str) -> list[dict[str, str]]:
        """Build system and user messages for the first model call."""

        system_prompt = "\n\n".join(
            [
                build_single_agent_system_prompt(),
                build_tool_calling_instruction(),
                _build_available_tools_context(self.client.list_tool_views()),
            ]
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

    def _result(
        self,
        *,
        answer: str,
        user_query: str,
        tool_records: list[dict[str, Any]],
        step_records: list[LLMSingleAgentStep],
        raw_model_outputs: list[str],
        parse_error_count: int,
        invalid_json_count: int,
        unknown_format_count: int,
        repair_attempt_count: int,
        tool_sequence: list[str],
        success: bool,
        error_message: str | None,
    ) -> LLMSingleAgentResult:
        """Build an LLMSingleAgentResult from accumulated state."""

        return LLMSingleAgentResult(
            answer=answer,
            parsed_task={
                "raw_query": user_query,
                "agent_type": AGENT_NAME,
                "inferred_tool_sequence": list(tool_sequence),
            },
            tool_results=_derive_tool_results(tool_records),
            trace=self.client.get_trace(),
            orchestration_steps=[asdict(record) for record in step_records],
            raw_model_outputs=raw_model_outputs,
            parse_error_count=parse_error_count,
            invalid_json_count=invalid_json_count,
            unknown_format_count=unknown_format_count,
            repair_attempt_count=repair_attempt_count,
            tool_sequence=tool_sequence,
            success=success,
            error_message=error_message,
        )


def build_tool_calling_instruction() -> str:
    """Return textual tool-call instructions appended to the system prompt."""

    return """
Textual tool-call protocol:
- You must return exactly one of:
  <tool_call>...</tool_call>
  or
  <final_answer>...</final_answer>
- During tool use, output exactly one tool call.
- Tool call JSON must have this shape:
  {"name": "...", "arguments": {...}}
- Use double quotes in JSON.
- Do not write markdown around tool calls.
- Do not write explanation outside the tags.
- Use tools for numeric TEC results.
- Do not fabricate thresholds, intervals, means, p90 values, or peaks.
- If the task is complete, return <final_answer>...</final_answer>.
- If a tool result gives an artifact id, use that id in the next tool call.
""".strip()


def _parse_agent_output(text: str):
    """Parse LLM text through the shared project parser helpers."""

    if parse_final_answer(text) is not None:
        return parse_model_output(text)

    if "<tool_call" in text.lower():
        return parse_tool_call(text)

    return parse_model_output(text)


def build_invalid_json_repair_message(error: str) -> str:
    """Return a repair prompt for invalid JSON inside a tool call."""

    return f"""
Your previous response contained invalid JSON inside <tool_call>.
Parser error: {error}

Return exactly one valid tool call:
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>
Use double quotes only. No trailing commas. No explanation.
""".strip()


def build_unknown_format_repair_message() -> str:
    """Return a repair prompt for output without recognized tags."""

    return """
Your previous response did not contain <tool_call> or <final_answer>.
Return exactly one of these formats:
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>

<final_answer>
Your final answer here.
</final_answer>
No markdown and no explanation outside the tags.
""".strip()


def build_missing_field_repair_message(error: str) -> str:
    """Return a repair prompt for missing tool-call fields."""

    return f"""
Your previous tool call was incomplete.
Parser error: {error}

Your tool call must include both "name" and "arguments":
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>
No markdown and no explanation outside the tags.
""".strip()


def safe_compact_json(value: Any, *, max_list_items: int = 8) -> str:
    """Return compact JSON for model observations without huge arrays."""

    compact = _compact_value(value, max_list_items=max_list_items)
    return json.dumps(compact, ensure_ascii=False, indent=2, default=str)


def _build_available_tools_context(tools: list[MCPToolView]) -> str:
    """Build a compact tool schema block for the model prompt."""

    tool_payload = []
    for tool in tools:
        schema = tool.input_schema or {}
        tool_payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "arguments_schema": {
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            }
        )

    return (
        "Available deterministic tools. Use these exact names and argument "
        "schemas:\n"
        f"{json.dumps(tool_payload, ensure_ascii=False, indent=2)}"
    )


def _build_tool_observation_message(response: ToolCallResponse) -> dict[str, str]:
    """Build the next user message containing a compact tool observation."""

    payload = response.to_dict()
    compact_payload = _compact_value(payload, max_list_items=8)
    content = (
        "Tool result:\n"
        "<tool_result>\n"
        f"{json.dumps(compact_payload, ensure_ascii=False, indent=2, default=str)}\n"
        "</tool_result>\n\n"
        "Continue the task. If more computation is needed, return exactly one "
        "<tool_call>. If the task is complete, return <final_answer>."
    )
    return {"role": "user", "content": content}


def _repair_message_for_parse_error(parsed) -> str:
    """Choose the appropriate repair prompt for a parser error."""

    message = parsed.error_message or "Unknown parser error."

    if parsed.error_code == ParseErrorCode.INVALID_JSON:
        return build_invalid_json_repair_message(message)

    if parsed.error_code in {
        ParseErrorCode.MISSING_TOOL_NAME,
        ParseErrorCode.MISSING_ARGUMENTS,
    }:
        return build_missing_field_repair_message(message)

    return build_unknown_format_repair_message()


def _compact_value(
    value: Any,
    *,
    max_list_items: int,
    max_string_chars: int = 1000,
    max_depth: int = 8,
) -> Any:
    """Recursively compact values before putting them back into the prompt."""

    if max_depth < 0:
        return "<max depth reached>"

    if isinstance(value, dict):
        return {
            str(key): _compact_value(
                item,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
                max_depth=max_depth - 1,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        items = [
            _compact_value(
                item,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
                max_depth=max_depth - 1,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            items.append(
                {
                    "_truncated": True,
                    "original_length": len(value),
                    "shown_items": max_list_items,
                }
            )
        return items

    if isinstance(value, tuple):
        return _compact_value(
            list(value),
            max_list_items=max_list_items,
            max_string_chars=max_string_chars,
            max_depth=max_depth,
        )

    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars] + "...<truncated>"

    return value


def _derive_tool_results(tool_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a result payload that keeps full calls and common task views."""

    payload: dict[str, Any] = {
        "tool_calls": tool_records,
        "by_tool": {},
    }

    ok_results: list[tuple[str, dict[str, Any]]] = []

    for record in tool_records:
        tool_name = record["tool_name"]
        response = record["response"]
        payload["by_tool"].setdefault(tool_name, []).append(response)

        if response.get("status") == "ok" and response.get("result") is not None:
            ok_results.append((tool_name, response["result"]))

    names = [name for name, _ in ok_results]

    if names == [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]:
        payload.update(
            {
                "timeseries": ok_results[0][1],
                "threshold": ok_results[1][1],
                "intervals": ok_results[2][1],
            }
        )

    elif names == [
        "tec_get_timeseries",
        "tec_compute_stability_thresholds",
        "tec_detect_stable_intervals",
    ]:
        payload.update(
            {
                "timeseries": ok_results[0][1],
                "thresholds": ok_results[1][1],
                "intervals": ok_results[2][1],
            }
        )

    elif "tec_compare_stats" in names:
        payload.update(
            {
                "timeseries": [
                    result
                    for name, result in ok_results
                    if name == "tec_get_timeseries"
                ],
                "stats": [
                    result
                    for name, result in ok_results
                    if name == "tec_compute_series_stats"
                ],
                "comparison": next(
                    result
                    for name, result in reversed(ok_results)
                    if name == "tec_compare_stats"
                ),
            }
        )

    return payload


def _tool_retry_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a stable retry key for a failed tool call."""

    try:
        encoded = json.dumps(arguments, sort_keys=True, default=str)
    except TypeError:
        encoded = repr(arguments)
    return f"{tool_name}:{encoded}"


def _tool_error_message(response: ToolCallResponse) -> str:
    """Return a concise tool error message."""

    error = response.error or {}
    error_type = error.get("error_type", "tool_error")
    message = error.get("message", "Tool call failed.")
    return f"{response.tool_name} failed with {error_type}: {message}"


__all__ = [
    "LLMSingleAgent",
    "LLMSingleAgentResult",
    "LLMSingleAgentStep",
    "build_invalid_json_repair_message",
    "build_missing_field_repair_message",
    "build_tool_calling_instruction",
    "build_unknown_format_repair_message",
    "safe_compact_json",
]
