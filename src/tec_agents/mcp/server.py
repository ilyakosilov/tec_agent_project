"""
Local MCP-like tool server.

This is not a full official MCP implementation yet. It provides the same core
idea for the project:

- tools are listed through a normalized interface;
- tools are called by name with JSON-like arguments;
- agents do not call Python tool functions directly.

Later this module can be replaced or wrapped by a real MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tec_agents.tools.executor import ToolExecutor, build_default_executor


@dataclass
class ToolCallRequest:
    """Simple MCP-like tool call request."""

    tool_name: str
    arguments: dict[str, Any]
    agent_name: str | None = None
    step: int | None = None


@dataclass
class ToolCallResponse:
    """Simple MCP-like tool call response."""

    tool_name: str
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable response."""

        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


class LocalMCPServer:
    """
    Local in-process MCP-like server.

    It exposes a minimal server interface:

    - list_tools()
    - call_tool(...)
    - get_trace()

    The server owns one ToolExecutor and one ToolStore. For independent
    experiments, create a new server per run or call reset().
    """

    def __init__(self, executor: ToolExecutor | None = None) -> None:
        self.executor = executor or build_default_executor()

    def list_tools(self) -> list[dict[str, Any]]:
        """Return available tools as JSON-serializable schemas."""

        return self.executor.list_tools()

    def list_openai_tools(self) -> list[dict[str, Any]]:
        """Return available tools as OpenAI-compatible function schemas."""

        return self.executor.list_openai_tools()

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        agent_name: str | None = None,
        step: int | None = None,
        raise_on_error: bool = False,
    ) -> ToolCallResponse:
        """
        Call a tool by name.

        By default this method does not raise tool errors. Instead it returns a
        structured response with status='error'. This behavior is convenient for
        agents because tool failures can be returned to the LLM as observations.
        """

        result = self.executor.call(
            tool_name=tool_name,
            arguments=arguments,
            agent_name=agent_name,
            step=step,
            raise_on_error=raise_on_error,
        )

        if "error" in result:
            return ToolCallResponse(
                tool_name=tool_name,
                status="error",
                result=None,
                error=result["error"],
            )

        return ToolCallResponse(
            tool_name=tool_name,
            status="ok",
            result=result,
            error=None,
        )

    def call(self, request: ToolCallRequest) -> ToolCallResponse:
        """Call a tool using a ToolCallRequest object."""

        return self.call_tool(
            tool_name=request.tool_name,
            arguments=request.arguments,
            agent_name=request.agent_name,
            step=request.step,
        )

    def get_trace(self) -> dict[str, Any]:
        """Return executor trace."""

        return self.executor.get_trace()

    def reset(self) -> None:
        """Reset intermediate store and trace."""

        self.executor.reset_store()
        self.executor.reset_trace()


def build_local_mcp_server(run_id: str | None = None) -> LocalMCPServer:
    """Create a local MCP-like server with a fresh executor."""

    return LocalMCPServer(executor=build_default_executor(run_id=run_id))