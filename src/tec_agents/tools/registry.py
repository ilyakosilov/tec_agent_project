"""
Tool registry for deterministic TEC tools.

The registry is the central catalog of available tools. It connects:

- public tool name;
- human-readable description;
- input schema;
- output schema;
- Python implementation.

Agents should not call tool functions directly. They should use ToolExecutor,
which relies on this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from tec_agents.tools import tec_tools
from tec_agents.tools.schemas import (
    BuildReportInput,
    BuildReportOutput,
    CompareRegionsInput,
    CompareRegionsOutput,
    ComputeHighThresholdInput,
    ComputeHighThresholdOutput,
    ComputeStabilityThresholdsInput,
    ComputeStabilityThresholdsOutput,
    DetectHighIntervalsInput,
    DetectHighIntervalsOutput,
    DetectStableIntervalsInput,
    DetectStableIntervalsOutput,
    FindStableIntervalsDirectInput,
    GetTimeseriesInput,
    GetTimeseriesOutput,
    SeriesProfileInput,
    SeriesProfileOutput,
)


ToolFunction = Callable[[BaseModel, tec_tools.ToolStore], BaseModel]


@dataclass(frozen=True)
class ToolSpec:
    """Complete specification of a tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    func: ToolFunction

    def input_json_schema(self) -> dict:
        """Return JSON schema for tool input."""

        return self.input_model.model_json_schema()

    def output_json_schema(self) -> dict:
        """Return JSON schema for tool output."""

        return self.output_model.model_json_schema()

    def to_openai_tool_schema(self) -> dict:
        """
        Convert the tool spec to OpenAI-compatible function tool schema.

        This is useful for vLLM/OpenAI-style tool calling.
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }

    def to_mcp_like_schema(self) -> dict:
        """
        Convert the tool spec to a simple MCP-like schema.

        This is not full official MCP yet. It is a local normalized description
        that can later be exposed through a real MCP server.
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }


class ToolRegistry:
    """Registry of available deterministic tools."""

    def __init__(self, tools: list[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}

        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        """Register one tool."""

        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        """Return tool specification by name."""

        try:
            return self._tools[name]
        except KeyError as exc:
            allowed = ", ".join(sorted(self._tools)) or "<none>"
            raise ValueError(
                f"Unknown tool: {name!r}. Available tools: {allowed}"
            ) from exc

    def names(self) -> list[str]:
        """Return registered tool names."""

        return sorted(self._tools)

    def list_specs(self) -> list[ToolSpec]:
        """Return all tool specs sorted by name."""

        return [self._tools[name] for name in self.names()]

    def list_mcp_like_schemas(self) -> list[dict]:
        """Return all tools as MCP-like JSON-serializable schemas."""

        return [tool.to_mcp_like_schema() for tool in self.list_specs()]

    def list_openai_tool_schemas(self) -> list[dict]:
        """Return all tools as OpenAI-compatible tool schemas."""

        return [tool.to_openai_tool_schema() for tool in self.list_specs()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def build_tool_registry() -> ToolRegistry:
    """Build the default TEC tool registry."""

    return ToolRegistry(
        tools=[
            ToolSpec(
                name="tec_get_timeseries",
                description=(
                    "Load a TEC time series for one predefined geographic region "
                    "and a half-open time interval [start, end)."
                ),
                input_model=GetTimeseriesInput,
                output_model=GetTimeseriesOutput,
                func=tec_tools.tec_get_timeseries,
            ),
            ToolSpec(
                name="tec_series_profile",
                description=(
                    "Return descriptive statistics for a previously loaded TEC "
                    "time series."
                ),
                input_model=SeriesProfileInput,
                output_model=SeriesProfileOutput,
                func=tec_tools.tec_series_profile,
            ),
            ToolSpec(
                name="tec_compute_high_threshold",
                description=(
                    "Compute a high-TEC threshold for a stored time series. "
                    "Usually use method='quantile' and q=0.9 unless the user "
                    "explicitly asks for another threshold."
                ),
                input_model=ComputeHighThresholdInput,
                output_model=ComputeHighThresholdOutput,
                func=tec_tools.tec_compute_high_threshold,
            ),
            ToolSpec(
                name="tec_detect_high_intervals",
                description=(
                    "Detect intervals where TEC is greater than or equal to a "
                    "previously computed high-TEC threshold."
                ),
                input_model=DetectHighIntervalsInput,
                output_model=DetectHighIntervalsOutput,
                func=tec_tools.tec_detect_high_intervals,
            ),
            ToolSpec(
                name="tec_compute_stability_thresholds",
                description=(
                    "Compute rolling-window thresholds for stable interval detection "
                    "using quantiles of rolling TEC variability."
                ),
                input_model=ComputeStabilityThresholdsInput,
                output_model=ComputeStabilityThresholdsOutput,
                func=tec_tools.tec_compute_stability_thresholds,
            ),
            ToolSpec(
                name="tec_detect_stable_intervals",
                description=(
                    "Detect stable TEC intervals using previously computed stability "
                    "thresholds."
                ),
                input_model=DetectStableIntervalsInput,
                output_model=DetectStableIntervalsOutput,
                func=tec_tools.tec_detect_stable_intervals,
            ),
            ToolSpec(
                name="tec_find_stable_intervals_direct",
                description=(
                    "Detect stable TEC intervals using explicit variability limits "
                    "without a separate threshold computation step."
                ),
                input_model=FindStableIntervalsDirectInput,
                output_model=DetectStableIntervalsOutput,
                func=tec_tools.tec_find_stable_intervals_direct,
            ),
            ToolSpec(
                name="tec_compare_regions",
                description=(
                    "Compare aggregated TEC statistics across two or more predefined "
                    "regions for the same time interval."
                ),
                input_model=CompareRegionsInput,
                output_model=CompareRegionsOutput,
                func=tec_tools.tec_compare_regions,
            ),
            ToolSpec(
                name="tec_build_report",
                description=(
                    "Build a compact deterministic TEC report for one or more regions, "
                    "including basic statistics and optionally high-TEC intervals."
                ),
                input_model=BuildReportInput,
                output_model=BuildReportOutput,
                func=tec_tools.tec_build_report,
            ),
        ]
    )