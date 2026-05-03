"""
Local MCP-like client.

The client is the interface used by agents. It hides the concrete tool execution
backend. Today the backend is an in-process LocalMCPServer; later it can become
a real MCP server, HTTP service, or stdio process without changing agent code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tec_agents.mcp.server import LocalMCPServer, ToolCallResponse


@dataclass(frozen=True)
class MCPToolView:
    """Compact tool description visible to agents."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LocalMCPClient:
    """
    Client for a local in-process MCP-like server.

    Agents should depend on this class instead of ToolExecutor.
    """

    def __init__(self, server: LocalMCPServer) -> None:
        self.server = server

    def list_tools(self) -> list[dict[str, Any]]:
        """Return full tool schemas."""

        return self.server.list_tools()

    def list_tool_views(self) -> list[MCPToolView]:
        """
        Return compact tool descriptions.

        This is convenient for prompts and agent diagnostics.
        """

        views: list[MCPToolView] = []

        for tool in self.server.list_tools():
            views.append(
                MCPToolView(
                    name=tool["name"],
                    description=tool["description"],
                    input_schema=tool["input_schema"],
                )
            )

        return views

    def list_tool_names(self) -> list[str]:
        """Return available tool names."""

        return sorted(tool["name"] for tool in self.server.list_tools())

    def list_openai_tools(self) -> list[dict[str, Any]]:
        """
        Return OpenAI-compatible tool schemas.

        This will be useful when we connect vLLM/OpenAI-style tool calling.
        """

        return self.server.list_openai_tools()

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        agent_name: str | None = None,
        step: int | None = None,
    ) -> ToolCallResponse:
        """
        Call a tool through the MCP-like server.

        Errors are returned as ToolCallResponse(status='error'), not raised.
        """

        return self.server.call_tool(
            tool_name=tool_name,
            arguments=arguments,
            agent_name=agent_name,
            step=step,
            raise_on_error=False,
        )

    def call_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        agent_name: str | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """
        Call a tool and return only the result dictionary.

        Raises RuntimeError if the tool fails. This helper is useful for tests
        and deterministic code, while agents may prefer call_tool().
        """

        response = self.call_tool(
            tool_name=tool_name,
            arguments=arguments,
            agent_name=agent_name,
            step=step,
        )

        if response.status != "ok":
            raise RuntimeError(
                f"Tool {tool_name!r} failed: {response.error}"
            )

        assert response.result is not None
        return response.result

    def get_trace(self) -> dict[str, Any]:
        """Return server trace."""

        return self.server.get_trace()

    def reset(self) -> None:
        """Reset server state."""

        self.server.reset()