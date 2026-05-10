"""
LLM single-agent TEC workflow using textual tool calls.

The agent does not implement a new tool layer. It parses model text with the
project parser and sends validated tool calls through LocalMCPClient.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from tec_agents.data.regions import list_region_ids
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
REPEATED_TOOL_CALL_STALL_LIMIT = 2

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call\b[^>]*>.*?</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_FINAL_ANSWER_BLOCK_RE = re.compile(
    r"<final_answer\b[^>]*>.*?</final_answer>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class LLMSingleAgentStep:
    """One LLM step, parse result, and optional tool execution result."""

    step: int
    raw_model_output: str
    cleaned_model_output: str
    parsed_type: str
    warnings: list[str] = field(default_factory=list)
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
    cleaned_model_outputs: list[str]
    parse_error_count: int
    invalid_json_count: int
    unknown_format_count: int
    multi_tool_call_output_count: int
    multi_final_answer_output_count: int
    repair_attempt_count: int
    repeated_tool_call_count: int
    stalled_loop_detected: bool
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
        tool_max_new_tokens: int = 256,
        final_max_new_tokens: int = 512,
    ):
        self.model = model
        self.client = client
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_parse_retries = max_parse_retries
        self.max_tool_retries = max_tool_retries
        self.temperature = temperature
        self.tool_max_new_tokens = tool_max_new_tokens
        self.final_max_new_tokens = final_max_new_tokens

    def reset(self) -> None:
        """Reset the underlying MCP-like client, if supported."""

        reset_fn = getattr(self.client, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def run(self, user_query: str) -> LLMSingleAgentResult:
        """Run the LLM tool-calling loop for one user query."""

        messages = self._initial_messages(user_query)
        task_state = infer_task_state(user_query)
        raw_model_outputs: list[str] = []
        cleaned_model_outputs: list[str] = []
        step_records: list[LLMSingleAgentStep] = []
        tool_records: list[dict[str, Any]] = []
        tool_sequence: list[str] = []
        parse_error_count = 0
        invalid_json_count = 0
        unknown_format_count = 0
        multi_tool_call_output_count = 0
        multi_final_answer_output_count = 0
        repair_attempt_count = 0
        repeated_tool_call_count = 0
        stalled_loop_detected = False
        consecutive_parse_errors = 0
        tool_call_count = 0
        tool_error_counts: dict[str, int] = {}
        previous_tool_calls: list[tuple[str, str]] = []
        successful_tool_call_keys: set[str] = set()
        repeated_attempts_by_call: dict[str, int] = {}
        available_tool_names = set(self.client.list_tool_names())

        for step in range(1, self.max_steps + 1):
            max_new_tokens = (
                self.final_max_new_tokens
                if _should_request_final_answer(task_state)
                else self.tool_max_new_tokens
            )
            raw_output = self.model.generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
            raw_model_outputs.append(raw_output)

            warnings: list[str] = []
            tool_block_count = count_tool_call_blocks(raw_output)
            final_block_count = count_final_answer_blocks(raw_output)

            if tool_block_count > 1:
                multi_tool_call_output_count += 1
                warnings.append("multiple_tool_calls_in_output")

            if final_block_count > 1:
                multi_final_answer_output_count += 1
                warnings.append("multiple_final_answers_in_output")

            cleaned_output = clean_model_output(raw_output)
            if not cleaned_output.strip():
                warnings.append("cleaning_failed")
                cleaned_output = raw_output

            cleaned_model_outputs.append(cleaned_output)
            parsed = _parse_agent_output(cleaned_output)

            if parsed.final_answer is not None:
                success = _final_answer_is_allowed(task_state, tool_sequence)
                error_message = None
                if not success:
                    error_message = (
                        "Final answer produced before required high_tec tools "
                        "completed."
                    )
                record = LLMSingleAgentStep(
                    step=step,
                    raw_model_output=raw_output,
                    cleaned_model_output=cleaned_output,
                    parsed_type="final_answer",
                    warnings=warnings,
                )
                step_records.append(record)
                return self._result(
                    answer=parsed.final_answer,
                    user_query=user_query,
                    task_state=task_state,
                    tool_records=tool_records,
                    step_records=step_records,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    parse_error_count=parse_error_count,
                    invalid_json_count=invalid_json_count,
                    unknown_format_count=unknown_format_count,
                    multi_tool_call_output_count=multi_tool_call_output_count,
                    multi_final_answer_output_count=multi_final_answer_output_count,
                    repair_attempt_count=repair_attempt_count,
                    repeated_tool_call_count=repeated_tool_call_count,
                    stalled_loop_detected=stalled_loop_detected,
                    tool_sequence=tool_sequence,
                    success=success,
                    error_message=error_message,
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
                    cleaned_model_output=cleaned_output,
                    parsed_type="parse_error",
                    warnings=warnings,
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
                    task_state=task_state,
                    tool_records=tool_records,
                    step_records=step_records,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    parse_error_count=parse_error_count,
                    invalid_json_count=invalid_json_count,
                    unknown_format_count=unknown_format_count,
                    multi_tool_call_output_count=multi_tool_call_output_count,
                    multi_final_answer_output_count=multi_final_answer_output_count,
                    repair_attempt_count=repair_attempt_count,
                    repeated_tool_call_count=repeated_tool_call_count,
                    stalled_loop_detected=stalled_loop_detected,
                    tool_sequence=tool_sequence,
                    success=False,
                    error_message=parsed.error_message or "Could not parse model output.",
                )

            consecutive_parse_errors = 0
            tool_call = parsed.tool_call
            canonical_arguments = canonical_arguments_json(tool_call.arguments)
            tool_call_key = (tool_call.name, canonical_arguments)
            tool_call_key_text = _tool_call_key_text(tool_call_key)

            if (
                previous_tool_calls
                and previous_tool_calls[-1] == tool_call_key
                and tool_call_key_text in successful_tool_call_keys
            ):
                repeated_tool_call_count += 1
                repeated_attempts_by_call[tool_call_key_text] = (
                    repeated_attempts_by_call.get(tool_call_key_text, 0) + 1
                )
                previous_tool_calls.append(tool_call_key)
                warnings.append("repeated_identical_tool_call_after_success")

                record = LLMSingleAgentStep(
                    step=step,
                    raw_model_output=raw_output,
                    cleaned_model_output=cleaned_output,
                    parsed_type="tool_call",
                    warnings=warnings,
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    tool_status="skipped_repeated",
                    tool_result={
                        "error": {
                            "error_type": "repeated_identical_tool_call",
                            "message": (
                                "Tool call skipped because the same call already "
                                "returned a valid result."
                            ),
                        }
                    },
                )
                step_records.append(record)

                if (
                    repeated_attempts_by_call[tool_call_key_text]
                    >= REPEATED_TOOL_CALL_STALL_LIMIT
                    or repeated_tool_call_count >= REPEATED_TOOL_CALL_STALL_LIMIT
                    or repair_attempt_count >= self.max_parse_retries
                ):
                    stalled_loop_detected = True
                    return self._result(
                        answer="",
                        user_query=user_query,
                        task_state=task_state,
                        tool_records=tool_records,
                        step_records=step_records,
                        raw_model_outputs=raw_model_outputs,
                        cleaned_model_outputs=cleaned_model_outputs,
                        parse_error_count=parse_error_count,
                        invalid_json_count=invalid_json_count,
                        unknown_format_count=unknown_format_count,
                        multi_tool_call_output_count=multi_tool_call_output_count,
                        multi_final_answer_output_count=multi_final_answer_output_count,
                        repair_attempt_count=repair_attempt_count,
                        repeated_tool_call_count=repeated_tool_call_count,
                        stalled_loop_detected=stalled_loop_detected,
                        tool_sequence=tool_sequence,
                        success=False,
                        error_message=(
                            "Stalled tool loop: repeated identical tool call "
                            "after successful result."
                        ),
                    )

                repair_attempt_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": build_repeated_tool_call_repair_message(
                            tool_call.name,
                            tool_call.arguments,
                            task_state,
                        ),
                    }
                )
                continue

            if tool_call_count >= self.max_tool_calls:
                record = LLMSingleAgentStep(
                    step=step,
                    raw_model_output=raw_output,
                    cleaned_model_output=cleaned_output,
                    parsed_type="tool_call",
                    warnings=warnings,
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
                    task_state=task_state,
                    tool_records=tool_records,
                    step_records=step_records,
                    raw_model_outputs=raw_model_outputs,
                    cleaned_model_outputs=cleaned_model_outputs,
                    parse_error_count=parse_error_count,
                    invalid_json_count=invalid_json_count,
                    unknown_format_count=unknown_format_count,
                    multi_tool_call_output_count=multi_tool_call_output_count,
                    multi_final_answer_output_count=multi_final_answer_output_count,
                    repair_attempt_count=repair_attempt_count,
                    repeated_tool_call_count=repeated_tool_call_count,
                    stalled_loop_detected=stalled_loop_detected,
                    tool_sequence=tool_sequence,
                    success=False,
                    error_message=f"Exceeded max_tool_calls={self.max_tool_calls}.",
                )

            tool_call_count += 1
            previous_tool_calls.append(tool_call_key)

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
            update_task_state_from_tool_result(
                task_state=task_state,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                response=response_dict,
            )
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
                successful_tool_call_keys.add(tool_call_key_text)

            record = LLMSingleAgentStep(
                step=step,
                raw_model_output=raw_output,
                cleaned_model_output=cleaned_output,
                parsed_type="tool_call",
                warnings=warnings,
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
                        task_state=task_state,
                        tool_records=tool_records,
                        step_records=step_records,
                        raw_model_outputs=raw_model_outputs,
                        cleaned_model_outputs=cleaned_model_outputs,
                        parse_error_count=parse_error_count,
                        invalid_json_count=invalid_json_count,
                        unknown_format_count=unknown_format_count,
                        multi_tool_call_output_count=multi_tool_call_output_count,
                        multi_final_answer_output_count=multi_final_answer_output_count,
                        repair_attempt_count=repair_attempt_count,
                        repeated_tool_call_count=repeated_tool_call_count,
                        stalled_loop_detected=stalled_loop_detected,
                        tool_sequence=tool_sequence,
                        success=False,
                        error_message=message,
                    )

            messages.append(
                {
                    "role": "user",
                    "content": build_tool_observation_message(
                        tool_call.name,
                        response_dict,
                        task_state,
                    ),
                }
            )

        return self._result(
            answer="",
            user_query=user_query,
            task_state=task_state,
            tool_records=tool_records,
            step_records=step_records,
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            parse_error_count=parse_error_count,
            invalid_json_count=invalid_json_count,
            unknown_format_count=unknown_format_count,
            multi_tool_call_output_count=multi_tool_call_output_count,
            multi_final_answer_output_count=multi_final_answer_output_count,
            repair_attempt_count=repair_attempt_count,
            repeated_tool_call_count=repeated_tool_call_count,
            stalled_loop_detected=stalled_loop_detected,
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
        task_state: dict[str, Any],
        tool_records: list[dict[str, Any]],
        step_records: list[LLMSingleAgentStep],
        raw_model_outputs: list[str],
        cleaned_model_outputs: list[str],
        parse_error_count: int,
        invalid_json_count: int,
        unknown_format_count: int,
        multi_tool_call_output_count: int,
        multi_final_answer_output_count: int,
        repair_attempt_count: int,
        repeated_tool_call_count: int,
        stalled_loop_detected: bool,
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
                "task_state": task_state,
                "inferred_tool_sequence": list(tool_sequence),
            },
            tool_results=_derive_tool_results(tool_records),
            trace=self.client.get_trace(),
            orchestration_steps=[asdict(record) for record in step_records],
            raw_model_outputs=raw_model_outputs,
            cleaned_model_outputs=cleaned_model_outputs,
            parse_error_count=parse_error_count,
            invalid_json_count=invalid_json_count,
            unknown_format_count=unknown_format_count,
            multi_tool_call_output_count=multi_tool_call_output_count,
            multi_final_answer_output_count=multi_final_answer_output_count,
            repair_attempt_count=repair_attempt_count,
            repeated_tool_call_count=repeated_tool_call_count,
            stalled_loop_detected=stalled_loop_detected,
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

Current experiment constraints:
- For high_tec tasks, use this chain:
  1. tec_get_timeseries
  2. tec_compute_high_threshold
  3. tec_detect_high_intervals
  4. final_answer
- After tec_get_timeseries returns a series_id, do not call
  tec_get_timeseries again for the same region/date.
- Use series_id in tec_compute_high_threshold.
- After threshold_id is returned, use it in tec_detect_high_intervals.
- Never output more than one tool call.
- Do not include role labels, schema text, examples, or markdown.
""".strip()


def clean_model_output(raw_text: str) -> str:
    """
    Return only the first complete tool_call or final_answer block.

    If both block types are present, the block that starts earlier wins.
    Garbage before the first relevant tag and after its closing tag is removed.
    """

    candidates: list[tuple[int, int, str]] = []

    for tag in ["tool_call", "final_answer"]:
        block = _first_closed_tag_block(raw_text, tag)
        if block is not None:
            start, end = block
            candidates.append((start, end, tag))

    if not candidates:
        return ""

    start, end, _ = min(candidates, key=lambda item: item[0])
    return raw_text[start:end].strip()


def count_tool_call_blocks(text: str) -> int:
    """Return the number of complete <tool_call> blocks in text."""

    return len(_TOOL_CALL_BLOCK_RE.findall(text))


def count_final_answer_blocks(text: str) -> int:
    """Return the number of complete <final_answer> blocks in text."""

    return len(_FINAL_ANSWER_BLOCK_RE.findall(text))


def canonical_arguments_json(arguments: dict[str, Any]) -> str:
    """Return stable JSON for comparing repeated tool calls."""

    return json.dumps(
        arguments,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def infer_task_state(user_query: str) -> dict[str, Any]:
    """Infer lightweight task state used only for guidance and diagnostics."""

    lower = user_query.lower()
    task_type = "unknown"

    if "stable" in lower or "low variability" in lower or "quiet" in lower:
        task_type = "stable_intervals"
    elif "compare" in lower or "comparison" in lower:
        task_type = "compare_regions"
    elif "report" in lower or "summary" in lower or "summarize" in lower:
        task_type = "report"
    elif "high_tec" in lower or ("high" in lower and "tec" in lower):
        task_type = "high_tec"

    region_id = None
    for candidate in list_region_ids():
        if candidate.lower() in lower:
            region_id = candidate
            break

    q = 0.9
    q_match = re.search(r"\bq\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", lower)
    if q_match:
        q = float(q_match.group(1))

    start = None
    end = None
    if "march" in lower and "2024" in lower:
        start = "2024-03-01"
        end = "2024-04-01"

    return {
        "raw_query": user_query,
        "task_type": task_type,
        "dataset_ref": "default",
        "region_id": region_id,
        "q": q,
        "start": start,
        "end": end,
        "series_id": None,
        "threshold_id": None,
        "intervals_ready": False,
        "n_intervals": None,
        "series_by_region": {},
        "available_artifacts": {},
    }


def update_task_state_from_tool_result(
    *,
    task_state: dict[str, Any],
    tool_name: str,
    arguments: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """Update lightweight task state from a compact tool response."""

    if response.get("status") != "ok":
        return

    result = response.get("result") or {}
    artifacts = task_state.setdefault("available_artifacts", {})

    if tool_name == "tec_get_timeseries":
        series_id = result.get("series_id")
        metadata = result.get("metadata") or {}
        region_id = (
            metadata.get("region_id")
            or arguments.get("region_id")
            or task_state.get("region_id")
        )

        if series_id is not None:
            artifacts["series_id"] = series_id
            if region_id is not None:
                task_state.setdefault("series_by_region", {})[str(region_id)] = series_id
            if (
                task_state.get("region_id") in {None, region_id}
                or task_state.get("task_type") == "high_tec"
            ):
                task_state["series_id"] = series_id
                task_state["region_id"] = region_id

    elif tool_name == "tec_compute_high_threshold":
        threshold_id = result.get("threshold_id")
        if threshold_id is not None:
            task_state["threshold_id"] = threshold_id
            artifacts["threshold_id"] = threshold_id
        if result.get("series_id") is not None:
            task_state["series_id"] = result["series_id"]
            artifacts["series_id"] = result["series_id"]

    elif tool_name == "tec_detect_high_intervals":
        task_state["intervals_ready"] = True
        task_state["n_intervals"] = result.get("n_intervals")
        artifacts["intervals"] = "ready"
        if result.get("series_id") is not None:
            task_state["series_id"] = result["series_id"]
            artifacts["series_id"] = result["series_id"]
        if result.get("threshold_id") is not None:
            task_state["threshold_id"] = result["threshold_id"]
            artifacts["threshold_id"] = result["threshold_id"]

    elif tool_name == "tec_compute_series_stats":
        if result.get("stats_id") is not None:
            artifacts.setdefault("stats_ids", []).append(result["stats_id"])

    elif tool_name == "tec_compare_stats":
        if result.get("comparison_id") is not None:
            artifacts["comparison_id"] = result["comparison_id"]


def build_tool_observation_message(
    tool_name: str,
    tool_response: dict[str, Any],
    task_state: dict[str, Any],
) -> str:
    """Build a stateful observation message after a tool call."""

    returned_artifacts = _returned_artifacts(tool_response)
    observation = {
        "tool_name": tool_name,
        "status": tool_response.get("status"),
        "returned_artifacts": returned_artifacts,
    }

    lines = [
        "Tool result:",
        "<tool_result>",
        json.dumps(observation, ensure_ascii=False, indent=2, default=str),
        "</tool_result>",
        "",
        f"Current task: {task_state.get('task_type')}.",
        "Available artifacts:",
    ]

    artifact_lines = _available_artifact_lines(task_state)
    lines.extend(artifact_lines if artifact_lines else ["- <none>"])

    repeated_warning = _do_not_repeat_line(tool_name, task_state)
    if repeated_warning is not None:
        lines.extend(["", repeated_warning])

    lines.extend(
        [
            "",
            "Valid next tools:",
            _valid_next_tools_line(task_state),
            "",
            "Next required step:",
            _next_required_step_text(task_state),
            "",
            "Return exactly one <tool_call> or <final_answer>. No explanations.",
        ]
    )

    return "\n".join(lines)


def build_repeated_tool_call_repair_message(
    tool_name: str,
    arguments: dict[str, Any],
    task_state: dict[str, Any],
) -> str:
    """Return a repair instruction for a repeated identical tool call."""

    return "\n".join(
        [
            (
                "You already called this tool with the same arguments and "
                "received a valid result."
            ),
            f"Repeated tool: {tool_name}",
            f"Repeated arguments: {safe_compact_json(arguments)}",
            "Use the returned artifact id instead of calling it again.",
            "",
            "Available artifacts:",
            *(_available_artifact_lines(task_state) or ["- <none>"]),
            "",
            "For this task, the next expected step is:",
            _next_required_step_text(task_state),
            "",
            "Return exactly one <tool_call> or <final_answer>. No explanations.",
        ]
    )


def _first_closed_tag_block(text: str, tag: str) -> tuple[int, int] | None:
    """Return start/end offsets for the first complete tag block."""

    lower = text.lower()
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = lower.find(open_tag)
    if start < 0:
        return None

    end = lower.find(close_tag, start + len(open_tag))
    if end < 0:
        return None

    return start, end + len(close_tag)


def _tool_call_key_text(tool_call_key: tuple[str, str]) -> str:
    """Return a compact string key for a canonicalized tool call."""

    return f"{tool_call_key[0]}:{tool_call_key[1]}"


def _should_request_final_answer(task_state: dict[str, Any]) -> bool:
    """Return True when the next model turn is expected to be a final answer."""

    return (
        task_state.get("task_type") == "high_tec"
        and bool(task_state.get("intervals_ready"))
    )


def _final_answer_is_allowed(
    task_state: dict[str, Any],
    tool_sequence: list[str],
) -> bool:
    """Validate that a final answer is not premature for known task types."""

    if task_state.get("task_type") != "high_tec":
        return True

    expected = [
        "tec_get_timeseries",
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
    ]
    return _is_ordered_subsequence(expected, tool_sequence) and bool(
        task_state.get("intervals_ready")
    )


def _is_ordered_subsequence(expected: list[str], actual: list[str]) -> bool:
    """Return True when expected appears in order inside actual."""

    cursor = 0
    for item in actual:
        if cursor < len(expected) and item == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def _returned_artifacts(tool_response: dict[str, Any]) -> dict[str, Any]:
    """Extract artifact handles and compact result facts from a tool response."""

    result = tool_response.get("result") or {}
    artifacts: dict[str, Any] = {}

    for key in [
        "series_id",
        "threshold_id",
        "stats_id",
        "comparison_id",
        "n_intervals",
        "threshold_value",
        "value",
    ]:
        if key in result:
            artifacts[key] = result[key]

    metadata = result.get("metadata") or {}
    if metadata.get("region_id") is not None:
        artifacts["region_id"] = metadata["region_id"]

    return artifacts


def _available_artifact_lines(task_state: dict[str, Any]) -> list[str]:
    """Return human-readable artifact inventory lines."""

    lines: list[str] = []
    region_id = task_state.get("region_id")
    series_id = task_state.get("series_id")
    threshold_id = task_state.get("threshold_id")

    if series_id is not None:
        if region_id is not None:
            lines.append(f"- series_id for {region_id}: {series_id}")
        else:
            lines.append(f"- series_id: {series_id}")

    if threshold_id is not None:
        lines.append(f"- threshold_id: {threshold_id}")

    if task_state.get("intervals_ready"):
        n_intervals = task_state.get("n_intervals")
        lines.append(f"- high TEC intervals: ready; n_intervals={n_intervals}")

    stats_ids = (task_state.get("available_artifacts") or {}).get("stats_ids") or []
    for stats_id in stats_ids:
        lines.append(f"- stats_id: {stats_id}")

    comparison_id = (task_state.get("available_artifacts") or {}).get("comparison_id")
    if comparison_id is not None:
        lines.append(f"- comparison_id: {comparison_id}")

    return lines


def _do_not_repeat_line(
    tool_name: str,
    task_state: dict[str, Any],
) -> str | None:
    """Return a targeted don't-repeat instruction for the last successful tool."""

    if tool_name == "tec_get_timeseries" and task_state.get("series_id") is not None:
        region_id = task_state.get("region_id") or "this region"
        return f"Do not call tec_get_timeseries again for {region_id}."

    if (
        tool_name == "tec_compute_high_threshold"
        and task_state.get("threshold_id") is not None
    ):
        return "Do not call tec_compute_high_threshold again for this series."

    if tool_name == "tec_detect_high_intervals" and task_state.get("intervals_ready"):
        return "Do not call high TEC tools again unless the user asks for more."

    return None


def _valid_next_tools_line(task_state: dict[str, Any]) -> str:
    """Return a concise list of valid next tools for known task state."""

    if task_state.get("task_type") == "high_tec":
        if task_state.get("series_id") is None:
            return "- tec_get_timeseries"
        if task_state.get("threshold_id") is None:
            return "- tec_compute_high_threshold"
        if not task_state.get("intervals_ready"):
            return "- tec_detect_high_intervals"
        return "- none; return final_answer"

    return "- choose the next primitive tool required by the task"


def _next_required_step_text(task_state: dict[str, Any]) -> str:
    """Return a deterministic next-step hint for the model."""

    if task_state.get("task_type") == "high_tec":
        series_id = task_state.get("series_id")
        threshold_id = task_state.get("threshold_id")
        region_id = task_state.get("region_id")
        start = task_state.get("start")
        end = task_state.get("end")
        dataset_ref = task_state.get("dataset_ref") or "default"
        q = task_state.get("q", 0.9)

        if series_id is None:
            args = {
                "dataset_ref": dataset_ref,
                "region_id": region_id,
                "start": start,
                "end": end,
            }
            args = {key: value for key, value in args.items() if value is not None}
            return _format_tool_call_hint("tec_get_timeseries", args)

        if threshold_id is None:
            return _format_tool_call_hint(
                "tec_compute_high_threshold",
                {
                    "series_id": series_id,
                    "method": "quantile",
                    "q": q,
                },
            )

        if not task_state.get("intervals_ready"):
            return _format_tool_call_hint(
                "tec_detect_high_intervals",
                {
                    "series_id": series_id,
                    "threshold_id": threshold_id,
                    "min_duration_minutes": 0,
                    "merge_gap_minutes": 60,
                },
            )

        return (
            "<final_answer>\n"
            "Summarize the high TEC threshold and detected intervals using only "
            "tool results.\n"
            "</final_answer>"
        )

    return "Use the primitive tool chain for the current task, or return final_answer if complete."


def _format_tool_call_hint(tool_name: str, arguments: dict[str, Any]) -> str:
    """Format an exact textual tool-call hint."""

    payload = {
        "name": tool_name,
        "arguments": arguments,
    }
    return (
        "<tool_call>\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "</tool_call>"
    )


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
    "build_repeated_tool_call_repair_message",
    "build_tool_observation_message",
    "build_tool_calling_instruction",
    "build_unknown_format_repair_message",
    "canonical_arguments_json",
    "clean_model_output",
    "count_final_answer_blocks",
    "count_tool_call_blocks",
    "infer_task_state",
    "safe_compact_json",
]
