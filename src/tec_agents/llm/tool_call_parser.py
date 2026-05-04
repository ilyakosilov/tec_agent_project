from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ParseErrorCode(str, Enum):
    INVALID_JSON = "invalid_json"
    MISSING_TOOL_NAME = "missing_tool_name"
    MISSING_ARGUMENTS = "missing_arguments"
    UNKNOWN_FORMAT = "unknown_format"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ParseResult:
    tool_call: ToolCall | None = None
    final_answer: str | None = None
    error_code: ParseErrorCode | None = None
    error_message: str | None = None
    raw_text: str | None = None


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)

_FINAL_ANSWER_RE = re.compile(
    r"<final_answer>\s*(.*?)\s*</final_answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_json_block(text: str) -> dict[str, Any]:
    """
    Extract and parse JSON object from text.

    Expected formats:
    1. Pure JSON:
       {"name": "...", "arguments": {...}}

    2. JSON with surrounding text:
       some text {"name": "...", "arguments": {...}} some text
    """
    text = text.strip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not an object.")
        return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    json_text = text[start : end + 1]
    parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")

    return parsed


def parse_final_answer(text: str) -> str | None:
    match = _FINAL_ANSWER_RE.search(text)

    if not match:
        return None

    answer = match.group(1).strip()
    return answer or None


def parse_tool_call(text: str) -> ParseResult:
    """
    Parse one tool call from model output.

    Supported format:

    <tool_call>
    {"name": "tec_get_timeseries", "arguments": {"dataset_ref": "default"}}
    </tool_call>
    """
    result = ParseResult(raw_text=text)

    match = _TOOL_CALL_RE.search(text)

    if not match:
        result.error_code = ParseErrorCode.UNKNOWN_FORMAT
        result.error_message = "No <tool_call>...</tool_call> block found."
        return result

    block = match.group(1).strip()

    try:
        data = extract_json_block(block)
    except Exception as exc:
        result.error_code = ParseErrorCode.INVALID_JSON
        result.error_message = f"Could not parse tool call JSON: {exc}"
        return result

    name = data.get("name")
    arguments = data.get("arguments")

    if not isinstance(name, str) or not name.strip():
        result.error_code = ParseErrorCode.MISSING_TOOL_NAME
        result.error_message = "Tool call JSON must contain non-empty string field 'name'."
        return result

    if arguments is None:
        result.error_code = ParseErrorCode.MISSING_ARGUMENTS
        result.error_message = "Tool call JSON must contain field 'arguments'."
        return result

    if not isinstance(arguments, dict):
        result.error_code = ParseErrorCode.MISSING_ARGUMENTS
        result.error_message = "Tool call field 'arguments' must be an object."
        return result

    result.tool_call = ToolCall(name=name.strip(), arguments=arguments)
    return result


def parse_model_output(text: str) -> ParseResult:
    """
    Parse model output as either final answer or tool call.

    Priority:
    1. final_answer
    2. tool_call
    3. unknown format
    """
    final_answer = parse_final_answer(text)

    if final_answer is not None:
        return ParseResult(
            final_answer=final_answer,
            raw_text=text,
        )

    return parse_tool_call(text)