"""
Tool executor for TEC analysis tools.

The executor is the single entry point for calling deterministic tools.
It validates inputs, calls the tool implementation, validates outputs, and
records a structured trace for later evaluation.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from tec_agents.tools.registry import ToolRegistry, build_tool_registry
from tec_agents.tools.tec_tools import ToolStore


@dataclass
class ToolCallTrace:
    """One structured tool call trace record."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str
    latency_sec: float
    output: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None
    agent_name: str | None = None
    step: int | None = None


@dataclass
class ExecutorTrace:
    """Trace for one executor run."""

    run_id: str
    calls: list[ToolCallTrace] = field(default_factory=list)

    def add_call(self, trace: ToolCallTrace) -> None:
        """Append one tool call trace."""

        self.calls.append(trace)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable trace."""

        return {
            "run_id": self.run_id,
            "n_calls": len(self.calls),
            "calls": [
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "status": call.status,
                    "latency_sec": call.latency_sec,
                    "output": call.output,
                    "error_type": call.error_type,
                    "error_message": call.error_message,
                    "traceback_text": call.traceback_text,
                    "agent_name": call.agent_name,
                    "step": call.step,
                }
                for call in self.calls
            ],
        }


class ToolExecutionError(RuntimeError):
    """Raised when a tool call fails."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        error_type: str,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.error_type = error_type
        self.original_error = original_error


class ToolExecutor:
    """
    Validating executor for registered TEC tools.

    Parameters
    ----------
    registry:
        Tool registry. If omitted, the default TEC registry is used.
    store:
        In-memory artifact store. If omitted, a new clean ToolStore is created.
    run_id:
        Optional run identifier for trace logs.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        store: ToolStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self.registry = registry or build_tool_registry()
        self.store = store or ToolStore()
        self.trace = ExecutorTrace(run_id=run_id or f"run_{uuid4().hex[:10]}")

    def call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | BaseModel,
        *,
        agent_name: str | None = None,
        step: int | None = None,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        """
        Call a registered tool.

        Parameters
        ----------
        tool_name:
            Public tool name from ToolRegistry.
        arguments:
            Raw dictionary or already-created Pydantic model.
        agent_name:
            Optional name of the agent that requested the call.
        step:
            Optional step index in the agent loop.
        raise_on_error:
            If True, raise ToolExecutionError on failure.
            If False, return a structured error dictionary.

        Returns
        -------
        dict
            JSON-serializable validated tool output.
        """

        call_id = f"call_{uuid4().hex[:10]}"
        started = time.perf_counter()

        raw_arguments = (
            arguments.model_dump()
            if isinstance(arguments, BaseModel)
            else dict(arguments)
        )

        try:
            tool = self.registry.get(tool_name)

            if isinstance(arguments, tool.input_model):
                validated_input = arguments
            else:
                validated_input = tool.input_model.model_validate(raw_arguments)

            raw_output = tool.func(validated_input, self.store)

            if isinstance(raw_output, tool.output_model):
                validated_output = raw_output
            else:
                validated_output = tool.output_model.model_validate(raw_output)

            output_dict = validated_output.model_dump()

            latency = time.perf_counter() - started
            self.trace.add_call(
                ToolCallTrace(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=raw_arguments,
                    status="ok",
                    latency_sec=latency,
                    output=output_dict,
                    agent_name=agent_name,
                    step=step,
                )
            )

            return output_dict

        except ValidationError as exc:
            return self._handle_error(
                call_id=call_id,
                tool_name=tool_name,
                arguments=raw_arguments,
                started=started,
                error_type="validation_error",
                error=exc,
                agent_name=agent_name,
                step=step,
                raise_on_error=raise_on_error,
            )

        except Exception as exc:
            return self._handle_error(
                call_id=call_id,
                tool_name=tool_name,
                arguments=raw_arguments,
                started=started,
                error_type=type(exc).__name__,
                error=exc,
                agent_name=agent_name,
                step=step,
                raise_on_error=raise_on_error,
            )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return registered tools as MCP-like schemas."""

        return self.registry.list_mcp_like_schemas()

    def list_openai_tools(self) -> list[dict[str, Any]]:
        """Return registered tools as OpenAI-compatible schemas."""

        return self.registry.list_openai_tool_schemas()

    def get_trace(self) -> dict[str, Any]:
        """Return executor trace as JSON-serializable dictionary."""

        return self.trace.to_dict()

    def reset_store(self) -> None:
        """Reset intermediate artifacts while keeping the same registry and trace."""

        self.store = ToolStore()

    def reset_trace(self, run_id: str | None = None) -> None:
        """Reset trace log."""

        self.trace = ExecutorTrace(run_id=run_id or f"run_{uuid4().hex[:10]}")

    def _handle_error(
        self,
        *,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        started: float,
        error_type: str,
        error: Exception,
        agent_name: str | None,
        step: int | None,
        raise_on_error: bool,
    ) -> dict[str, Any]:
        """Record and optionally raise a structured tool error."""

        latency = time.perf_counter() - started
        tb = traceback.format_exc()

        error_output = {
            "error": {
                "error_type": error_type,
                "message": str(error),
                "tool_name": tool_name,
            }
        }

        self.trace.add_call(
            ToolCallTrace(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                status="error",
                latency_sec=latency,
                output=None,
                error_type=error_type,
                error_message=str(error),
                traceback_text=tb,
                agent_name=agent_name,
                step=step,
            )
        )

        if raise_on_error:
            raise ToolExecutionError(
                f"Tool {tool_name!r} failed with {error_type}: {error}",
                tool_name=tool_name,
                error_type=error_type,
                original_error=error,
            ) from error

        return error_output


def build_default_executor(run_id: str | None = None) -> ToolExecutor:
    """Convenience factory for the default TEC tool executor."""

    return ToolExecutor(
        registry=build_tool_registry(),
        store=ToolStore(),
        run_id=run_id,
    )