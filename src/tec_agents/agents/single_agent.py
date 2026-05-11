"""
Simple single-agent baseline.

This agent does not use an LLM yet. It is a deterministic rule-based baseline
that uses the same MCP-like client and the same TEC tools as future LLM agents.

Supported scenarios:
- high TEC interval detection for one region and one month;
- TEC statistics comparison for two or more regions and one month.
- stable/low-variability TEC interval detection;
- structured TEC reports for one or more regions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tec_agents.agents.protocol import (
    AgentResponse,
    AgentStatus,
    RequestedNextAction,
    agent_missing_artifacts,
    agent_ok,
    agent_tool_error,
)
from tec_agents.data.dates import parse_date_range_from_text
from tec_agents.data.regions import list_region_ids
from tec_agents.mcp.client import LocalMCPClient


MAX_TOOL_RETRIES = 2
DEFAULT_REPORT_INCLUDE = ["basic_stats", "high_tec", "stable_intervals"]


@dataclass
class ParsedSingleAgentTask:
    """Parsed task for the deterministic single-agent baseline."""

    task_type: str
    dataset_ref: str = "default"
    region_id: str | None = "midlat_europe"
    region_ids: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    start: str = ""
    end: str = ""
    q: float = 0.9
    window_minutes: int = 180
    q_delta: float = 0.60
    q_std: float = 0.60
    include: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    raw_query: str = ""


@dataclass
class SingleAgentStep:
    """One high-level single-agent orchestration step."""

    node: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    missing_artifacts: list[str] = field(default_factory=list)
    requested_next_action: dict[str, Any] | None = None
    can_continue: bool = True
    requires_retry: bool = False
    attempt: int = 1
    max_attempts: int = MAX_TOOL_RETRIES
    decision: str | None = None


@dataclass
class SingleAgentResult:
    """Result of a single-agent run."""

    answer: str
    parsed_task: ParsedSingleAgentTask
    tool_results: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    orchestration_steps: list[SingleAgentStep] = field(default_factory=list)


class _SingleAgentFailure(RuntimeError):
    """Internal structured single-agent failure."""

    def __init__(self, response: AgentResponse) -> None:
        super().__init__(response.message)
        self.response = response


class RuleBasedSingleAgent:
    """
    Deterministic single-agent baseline.

    The class deliberately uses MCP-like client instead of direct Python tool
    calls. This keeps the interface aligned with future LLM-based agents.
    """

    def __init__(
        self,
        client: LocalMCPClient,
        dataset_ref: str = "default",
        agent_name: str = "rule_based_single_agent",
    ) -> None:
        self.client = client
        self.dataset_ref = dataset_ref
        self.agent_name = agent_name
        self._last_step_responses: list[AgentResponse] = []

    def reset(self) -> None:
        """
        Reset underlying MCP client state if supported.

        This is important for evaluation: each task should have an isolated
        tool store and trace, otherwise tool-call metrics accumulate across
        tasks and become inflated.
        """

        if hasattr(self.client, "reset"):
            self.client.reset()

    def run(self, query: str) -> SingleAgentResult:
        """Run the agent on one user query."""

        self._last_step_responses = []
        parsed = self.parse_query(query)
        failure_response: AgentResponse | None = None

        try:
            if parsed.task_type == "high_tec":
                tool_results = self._run_high_tec(parsed)
                answer = self._format_high_tec_answer(parsed, tool_results)

            elif parsed.task_type == "compare_regions":
                tool_results = self._run_compare_regions(parsed)
                answer = self._format_compare_regions_answer(parsed, tool_results)

            elif parsed.task_type == "stable_intervals":
                tool_results = self._run_stable_intervals(parsed)
                answer = self._format_stable_intervals_answer(parsed, tool_results)

            elif parsed.task_type == "report":
                tool_results = self._run_report(parsed)
                answer = self._format_report_answer(parsed, tool_results)

            else:
                raise ValueError(
                    f"Unsupported task_type={parsed.task_type!r}. "
                    "This baseline currently supports high_tec, compare_regions, "
                    "stable_intervals and report."
                )

        except _SingleAgentFailure as exc:
            failure_response = exc.response
            tool_results = {"_failure": failure_response.to_dict()}
            missing = ", ".join(failure_response.missing_artifacts) or "<none>"
            answer = (
                f"[ERROR] Cannot complete {parsed.task_type} task: "
                f"{failure_response.message}; missing artifacts: {missing}."
            )

        orchestration_steps = [
            SingleAgentStep(
                node=self.agent_name,
                action="parse_route_execute_and_report",
                details={
                    "task_type": parsed.task_type,
                    "region_id": parsed.region_id,
                    "region_ids": parsed.region_ids,
                    "regions": parsed.regions,
                    "start": parsed.start,
                    "end": parsed.end,
                    "q": parsed.q,
                    "window_minutes": parsed.window_minutes,
                    "q_delta": parsed.q_delta,
                    "q_std": parsed.q_std,
                    "include": parsed.include,
                    "metrics": parsed.metrics,
                    "selected_worker": "single_agent",
                    "step_responses": [
                        response.to_dict()
                        for response in self._last_step_responses
                    ],
                },
                status=(
                    failure_response.status.value
                    if failure_response is not None
                    else "ok"
                ),
                missing_artifacts=(
                    failure_response.missing_artifacts
                    if failure_response is not None
                    else []
                ),
                requested_next_action=(
                    failure_response.requested_next_action.to_dict()
                    if failure_response is not None
                    and failure_response.requested_next_action is not None
                    else None
                ),
                can_continue=(
                    failure_response.can_continue
                    if failure_response is not None
                    else True
                ),
                requires_retry=(
                    failure_response.requires_retry
                    if failure_response is not None
                    else False
                ),
                attempt=(
                    failure_response.attempt
                    if failure_response is not None
                    else 1
                ),
                max_attempts=MAX_TOOL_RETRIES,
                decision="fail" if failure_response is not None else "continue",
            )
        ]

        return SingleAgentResult(
            answer=answer,
            parsed_task=parsed,
            tool_results=tool_results,
            trace=self.client.get_trace(),
            orchestration_steps=orchestration_steps,
        )

    def parse_query(self, query: str) -> ParsedSingleAgentTask:
        """
        Parse a simple query.

        This is intentionally simple. Later, Qwen will replace this parser.
        """

        lower = query.lower()

        task_type = self._extract_task_type(lower)
        start, end = self._extract_month_range(lower)
        q = self._extract_quantile(lower)

        if task_type == "compare_regions":
            region_ids = self._extract_region_ids(lower)
            if len(region_ids) < 2:
                raise ValueError(
                    "Compare task requires at least two recognized regions"
                )
            region_id = None

        elif task_type == "report":
            region_ids = self._extract_region_ids(lower)
            if not region_ids:
                region_ids = ["midlat_europe"]
            region_id = None

        else:
            region_id = self._extract_region_id(lower)
            region_ids = [region_id]

        return ParsedSingleAgentTask(
            task_type=task_type,
            dataset_ref=self.dataset_ref,
            region_id=region_id,
            region_ids=region_ids,
            regions=region_ids,
            start=start,
            end=end,
            q=q,
            window_minutes=180,
            q_delta=0.60,
            q_std=0.60,
            include=self._extract_report_include(lower),
            metrics=self._default_compare_metrics(),
            raw_query=query,
        )

    def _call_tool_with_retry(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        step: int,
        required: bool = True,
        missing_artifacts: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Call a tool with bounded deterministic retry handling."""

        last_response: AgentResponse | None = None

        for attempt in range(1, MAX_TOOL_RETRIES + 1):
            tool_response = self.client.call_tool(
                tool_name,
                arguments,
                agent_name=self.agent_name,
                step=step,
            )
            if tool_response.status == "ok" and tool_response.result is not None:
                response = agent_ok(
                    agent=self.agent_name,
                    artifacts={
                        "tool_name": tool_name,
                        "result": tool_response.result,
                    },
                    message=f"{tool_name} succeeded.",
                    attempt=attempt,
                    max_attempts=MAX_TOOL_RETRIES,
                )
                self._last_step_responses.append(response)
                return tool_response.result

            missing = missing_artifacts or [f"tool.{tool_name}.result"]
            last_response = agent_tool_error(
                agent=self.agent_name,
                message=f"{tool_name} failed.",
                missing_artifacts=missing,
                requested_next_action=RequestedNextAction(
                    target_agent=self.agent_name,
                    task="retry_tool_call",
                    reason="Tool call failed",
                    required_artifacts=missing if required else [],
                    optional_artifacts=[] if required else missing,
                    params={"tool_name": tool_name, "arguments": arguments},
                ),
                can_continue=not required,
                requires_retry=attempt < MAX_TOOL_RETRIES,
                attempt=attempt,
                max_attempts=MAX_TOOL_RETRIES,
            )
            self._last_step_responses.append(last_response)

            if attempt < MAX_TOOL_RETRIES:
                continue

        if required:
            assert last_response is not None
            raise _SingleAgentFailure(last_response)

        return None

    def _require_artifact(
        self,
        source: dict[str, Any],
        key: str,
        artifact_path: str,
    ) -> Any:
        """Return a required artifact field or raise a structured failure."""

        if key in source and source[key] is not None:
            return source[key]

        response = agent_missing_artifacts(
            agent=self.agent_name,
            missing_artifacts=[artifact_path],
            requested_next_action=RequestedNextAction(
                target_agent=self.agent_name,
                task="recover_missing_artifact",
                reason=f"Missing required artifact {artifact_path}",
                required_artifacts=[artifact_path],
            ),
            message=f"Cannot continue without {artifact_path}.",
            attempt=1,
            max_attempts=MAX_TOOL_RETRIES,
        )
        self._last_step_responses.append(response)
        raise _SingleAgentFailure(response)

    def _run_high_tec(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute high-TEC detection through MCP-like tools."""

        if parsed.region_id is None:
            raise ValueError("High-TEC task requires region_id")

        ts_result = self._call_tool_with_retry(
            "tec_get_timeseries",
            {
                "dataset_ref": parsed.dataset_ref,
                "region_id": parsed.region_id,
                "start": parsed.start,
                "end": parsed.end,
            },
            step=1,
            missing_artifacts=[f"data.series_by_region.{parsed.region_id}"],
        )
        assert ts_result is not None

        series_id = self._require_artifact(
            ts_result,
            "series_id",
            f"data.series_by_region.{parsed.region_id}.series_id",
        )

        threshold_result = self._call_tool_with_retry(
            "tec_compute_high_threshold",
            {
                "series_id": series_id,
                "method": "quantile",
                "q": parsed.q,
            },
            step=2,
            missing_artifacts=["math.high_tec.threshold"],
        )
        assert threshold_result is not None

        threshold_id = self._require_artifact(
            threshold_result,
            "threshold_id",
            "math.high_tec.threshold.threshold_id",
        )

        intervals_result = self._call_tool_with_retry(
            "tec_detect_high_intervals",
            {
                "series_id": series_id,
                "threshold_id": threshold_id,
                "min_duration_minutes": 0,
                "merge_gap_minutes": 60,
            },
            step=3,
            missing_artifacts=["math.high_tec.intervals"],
        )
        assert intervals_result is not None

        return {
            "timeseries": ts_result,
            "threshold": threshold_result,
            "intervals": intervals_result,
        }

    def _run_compare_regions(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute region comparison through primitive MCP-like tools."""

        if len(parsed.region_ids) < 2:
            raise ValueError("Comparison task requires at least two region_ids")

        timeseries_results: list[dict[str, Any]] = []
        stats_results: list[dict[str, Any]] = []
        stats_ids: list[str] = []

        step = 1
        for region_id in parsed.region_ids:
            ts_result = self._call_tool_with_retry(
                "tec_get_timeseries",
                {
                    "dataset_ref": parsed.dataset_ref,
                    "region_id": region_id,
                    "start": parsed.start,
                    "end": parsed.end,
                },
                step=step,
                missing_artifacts=[f"data.series_by_region.{region_id}"],
            )
            assert ts_result is not None
            timeseries_results.append(ts_result)
            step += 1

        for ts_result in timeseries_results:
            series_id = self._require_artifact(
                ts_result,
                "series_id",
                "data.series_by_region.series_id",
            )
            stats_result = self._call_tool_with_retry(
                "tec_compute_series_stats",
                {
                    "series_id": series_id,
                    "metrics": parsed.metrics,
                },
                step=step,
                missing_artifacts=["math.stats_by_region"],
            )
            assert stats_result is not None
            stats_results.append(stats_result)
            stats_ids.append(
                self._require_artifact(
                    stats_result,
                    "stats_id",
                    "math.stats_by_region.stats_id",
                )
            )
            step += 1

        comparison_result = self._call_tool_with_retry(
            "tec_compare_stats",
            {
                "stats_ids": stats_ids,
                "metrics": parsed.metrics,
            },
            step=step,
            missing_artifacts=["math.comparison"],
        )
        assert comparison_result is not None

        return {
            "timeseries": timeseries_results,
            "stats": stats_results,
            "comparison": comparison_result,
        }

    def _run_stable_intervals(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute stable-interval detection through MCP-like tools."""

        if parsed.region_id is None:
            raise ValueError("Stable-interval task requires region_id")

        ts_result = self._call_tool_with_retry(
            "tec_get_timeseries",
            {
                "dataset_ref": parsed.dataset_ref,
                "region_id": parsed.region_id,
                "start": parsed.start,
                "end": parsed.end,
            },
            step=1,
            missing_artifacts=[f"data.series_by_region.{parsed.region_id}"],
        )
        assert ts_result is not None

        series_id = self._require_artifact(
            ts_result,
            "series_id",
            f"data.series_by_region.{parsed.region_id}.series_id",
        )

        thresholds_result = self._call_tool_with_retry(
            "tec_compute_stability_thresholds",
            {
                "series_id": series_id,
                "window_minutes": parsed.window_minutes,
                "method": "quantile",
                "q_delta": parsed.q_delta,
                "q_std": parsed.q_std,
            },
            step=2,
            missing_artifacts=["math.stable_intervals.thresholds"],
        )
        assert thresholds_result is not None

        threshold_id = self._require_artifact(
            thresholds_result,
            "threshold_id",
            "math.stable_intervals.thresholds.threshold_id",
        )

        intervals_result = self._call_tool_with_retry(
            "tec_detect_stable_intervals",
            {
                "series_id": series_id,
                "threshold_id": threshold_id,
                "min_duration_minutes": parsed.window_minutes,
                "merge_gap_minutes": 60,
            },
            step=3,
            missing_artifacts=["math.stable_intervals.intervals"],
        )
        assert intervals_result is not None

        return {
            "timeseries": ts_result,
            "thresholds": thresholds_result,
            "intervals": intervals_result,
        }

    def _run_report(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute structured report generation through primitive tools."""

        if not parsed.region_ids:
            raise ValueError("Report task requires at least one region")

        include = _report_include(parsed.include)
        series_by_region: dict[str, dict[str, Any]] = {}
        report_inputs: dict[str, Any] = {}
        step = 1

        for region_id in parsed.region_ids:
            ts_result = self._call_tool_with_retry(
                "tec_get_timeseries",
                {
                    "dataset_ref": parsed.dataset_ref,
                    "region_id": region_id,
                    "start": parsed.start,
                    "end": parsed.end,
                },
                step=step,
                missing_artifacts=[f"data.series_by_region.{region_id}"],
            )
            assert ts_result is not None
            step += 1
            series_id = self._require_artifact(
                ts_result,
                "series_id",
                f"data.series_by_region.{region_id}.series_id",
            )
            series_by_region[region_id] = {
                "series_id": series_id,
                "tool_result": ts_result,
                "metadata": ts_result.get("metadata"),
            }

        if "basic_stats" in include:
            by_region: dict[str, dict[str, Any]] = {}
            stats_ids: list[str] = []

            for region_id in parsed.region_ids:
                series_id = series_by_region[region_id]["series_id"]
                stats_result = self._call_tool_with_retry(
                    "tec_compute_series_stats",
                    {
                        "series_id": series_id,
                        "metrics": parsed.metrics,
                    },
                    step=step,
                    missing_artifacts=[f"math.report_inputs.basic_stats.{region_id}"],
                )
                assert stats_result is not None
                step += 1
                stats_id = self._require_artifact(
                    stats_result,
                    "stats_id",
                    f"math.report_inputs.basic_stats.{region_id}.stats_id",
                )

                by_region[region_id] = {
                    "stats_id": stats_id,
                    "series_id": series_id,
                    "stats": stats_result,
                    "metrics": stats_result.get("metrics", {}),
                }
                stats_ids.append(stats_id)

            comparison = None
            if len(stats_ids) >= 2:
                comparison = self._call_tool_with_retry(
                    "tec_compare_stats",
                    {
                        "stats_ids": stats_ids,
                        "metrics": parsed.metrics,
                    },
                    step=step,
                    missing_artifacts=["math.report_inputs.basic_stats.comparison"],
                )
                assert comparison is not None
                step += 1

            report_inputs["basic_stats"] = {
                "by_region": by_region,
                "comparison": comparison,
            }

        if "high_tec" in include:
            by_region = {}

            for region_id in parsed.region_ids:
                series_id = series_by_region[region_id]["series_id"]
                threshold_result = self._call_tool_with_retry(
                    "tec_compute_high_threshold",
                    {
                        "series_id": series_id,
                        "method": "quantile",
                        "q": parsed.q,
                    },
                    step=step,
                    missing_artifacts=[f"math.report_inputs.high_tec.{region_id}.threshold"],
                )
                assert threshold_result is not None
                step += 1
                threshold_id = self._require_artifact(
                    threshold_result,
                    "threshold_id",
                    f"math.report_inputs.high_tec.{region_id}.threshold_id",
                )

                intervals_result = self._call_tool_with_retry(
                    "tec_detect_high_intervals",
                    {
                        "series_id": series_id,
                        "threshold_id": threshold_id,
                        "min_duration_minutes": 0,
                        "merge_gap_minutes": 60,
                    },
                    step=step,
                    missing_artifacts=[f"math.report_inputs.high_tec.{region_id}.intervals"],
                )
                assert intervals_result is not None
                step += 1

                by_region[region_id] = {
                    "threshold": threshold_result,
                    "intervals": intervals_result,
                }

            report_inputs["high_tec"] = {"by_region": by_region}

        if "stable_intervals" in include:
            by_region = {}

            for region_id in parsed.region_ids:
                series_id = series_by_region[region_id]["series_id"]
                thresholds_result = self._call_tool_with_retry(
                    "tec_compute_stability_thresholds",
                    {
                        "series_id": series_id,
                        "window_minutes": parsed.window_minutes,
                        "method": "quantile",
                        "q_delta": parsed.q_delta,
                        "q_std": parsed.q_std,
                    },
                    step=step,
                    missing_artifacts=[
                        f"math.report_inputs.stable_intervals.{region_id}.thresholds"
                    ],
                )
                assert thresholds_result is not None
                step += 1
                threshold_id = self._require_artifact(
                    thresholds_result,
                    "threshold_id",
                    f"math.report_inputs.stable_intervals.{region_id}.threshold_id",
                )

                intervals_result = self._call_tool_with_retry(
                    "tec_detect_stable_intervals",
                    {
                        "series_id": series_id,
                        "threshold_id": threshold_id,
                        "min_duration_minutes": parsed.window_minutes,
                        "merge_gap_minutes": 60,
                    },
                    step=step,
                    missing_artifacts=[
                        f"math.report_inputs.stable_intervals.{region_id}.intervals"
                    ],
                )
                assert intervals_result is not None
                step += 1

                by_region[region_id] = {
                    "thresholds": thresholds_result,
                    "intervals": intervals_result,
                }

            report_inputs["stable_intervals"] = {"by_region": by_region}

        return {
            "data": {
                "series_by_region": series_by_region,
                "regions": list(parsed.region_ids),
                "dataset_ref": parsed.dataset_ref,
                "start": parsed.start,
                "end": parsed.end,
            },
            "math": {
                "report_inputs": report_inputs,
            },
            "analysis": {
                "findings": [],
                "summary": {
                    "primary_region": parsed.region_ids[0],
                    "n_regions": len(parsed.region_ids),
                },
            },
        }

    def _format_high_tec_answer(
        self,
        parsed: ParsedSingleAgentTask,
        tool_results: dict[str, Any],
    ) -> str:
        """Create a compact human-readable answer for high-TEC task."""

        threshold = tool_results["threshold"]
        intervals = tool_results["intervals"]

        lines = [
            (
                f"High TEC intervals for {parsed.region_id} from "
                f"{parsed.start} to {parsed.end} using q={parsed.q}:"
            ),
            (
                f"Threshold: {threshold['value']:.3f} TECU "
                f"({threshold['method']}, n={threshold['n_points_used']})."
            ),
            f"Detected intervals: {intervals['n_intervals']}.",
        ]

        if intervals["n_intervals"] == 0:
            lines.append("No high-TEC intervals were detected.")
            return "\n".join(lines)

        lines.append("")
        lines.append("Intervals:")

        for i, item in enumerate(intervals["intervals"], start=1):
            peak_value = item["peak_value"]
            mean_value = item["mean_value"]

            peak_text = (
                f"{peak_value:.3f} TECU at {item['peak_time']}"
                if peak_value is not None
                else "n/a"
            )
            mean_text = f"{mean_value:.3f} TECU" if mean_value is not None else "n/a"

            lines.append(
                f"{i}. {item['start']} -> {item['end']}; "
                f"duration={item['duration_minutes']:.1f} min; "
                f"peak={peak_text}; "
                f"mean={mean_text}."
            )

        return "\n".join(lines)

    def _format_compare_regions_answer(
        self,
        parsed: ParsedSingleAgentTask,
        tool_results: dict[str, Any],
    ) -> str:
        """Create a compact human-readable answer for region comparison."""

        lines = [
            (
                f"TEC statistics comparison for {', '.join(parsed.region_ids)} "
                f"from {parsed.start} to {parsed.end}:"
            ),
            "",
        ]

        comparison = _extract_comparison_payload(tool_results)
        stats = comparison["items"]

        for item in stats:
            metrics = item.get("metrics", {})
            lines.append(
                f"- {item['region_id']}: "
                f"mean={_fmt_float(metrics.get('mean'))} TECU, "
                f"median={_fmt_float(metrics.get('median'))} TECU, "
                f"max={_fmt_float(metrics.get('max'))} TECU, "
                f"std={_fmt_float(metrics.get('std'))} TECU, "
                f"p90={_fmt_float(metrics.get('p90'))} TECU."
            )

        return "\n".join(lines)

    def _format_stable_intervals_answer(
        self,
        parsed: ParsedSingleAgentTask,
        tool_results: dict[str, Any],
    ) -> str:
        """Create a compact human-readable answer for stable intervals."""

        thresholds = tool_results["thresholds"]
        intervals = tool_results["intervals"]

        return (
            f"Stable TEC intervals for {parsed.region_id} from {parsed.start} "
            f"to {parsed.end}: detected {intervals['n_intervals']} intervals "
            f"using window={thresholds['window_minutes']} min, "
            f"max_delta={thresholds['max_delta_threshold']:.3f}, "
            f"rolling_std={thresholds['rolling_std_threshold']:.3f}."
        )

    def _format_report_answer(
        self,
        parsed: ParsedSingleAgentTask,
        tool_results: dict[str, Any],
    ) -> str:
        """Create a compact human-readable answer for structured report tasks."""

        if "sections" in tool_results:
            sections = sorted(tool_results.get("sections", {}))
        else:
            report_inputs = (tool_results.get("math") or {}).get("report_inputs") or {}
            sections = sorted(report_inputs)

        return (
            "Built TEC report from primitive artifacts for "
            f"{', '.join(parsed.region_ids)} from {parsed.start} to {parsed.end}. "
            f"Sections: {', '.join(sections)}."
        )

    def _extract_task_type(self, lower_query: str) -> str:
        """Extract supported task type."""

        report_markers = [
            "report",
            "summary",
            "summarize",
            "summarise",
            "build report",
            "create summary",
        ]

        if any(marker in lower_query for marker in report_markers):
            return "report"

        stable_markers = [
            "stable",
            "low variability",
            "low-variability",
            "quiet tec",
            "quiet interval",
            "quiet period",
        ]

        if any(marker in lower_query for marker in stable_markers):
            return "stable_intervals"

        compare_markers = [
            "compare",
            "comparison",
            "\u0441\u0440\u0430\u0432\u043d\u0438",
            "\u0441\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435",
        ]

        if any(marker in lower_query for marker in compare_markers):
            return "compare_regions"

        if "high" in lower_query and "tec" in lower_query:
            return "high_tec"

        return "unknown"

    def _extract_region_id(self, lower_query: str) -> str:
        """Extract one region_id from query or use default."""

        region_ids = self._extract_region_ids(lower_query)
        if region_ids:
            return region_ids[0]

        return "midlat_europe"

    def _extract_region_ids(self, lower_query: str) -> list[str]:
        """Extract all region IDs mentioned in the query."""

        found: list[str] = []

        for region_id in list_region_ids():
            if region_id.lower() in lower_query:
                found.append(region_id)

        aliases = {
            "europe": "midlat_europe",
            "north high latitudes": "highlat_north",
            "northern high latitudes": "highlat_north",
            "high latitude north": "highlat_north",
            "atlantic": "equatorial_atlantic",
            "africa": "equatorial_africa",
            "pacific": "equatorial_pacific",
            "usa": "midlat_usa",
            "asia": "midlat_asia",
            "australia": "midlat_australia",
            "south america": "midlat_south_america",
        }

        for alias, region_id in aliases.items():
            if alias in lower_query and region_id not in found:
                found.append(region_id)

        return found

    def _extract_month_range(self, lower_query: str) -> tuple[str, str]:
        """Extract month and year as half-open date interval [start, end)."""

        return parse_date_range_from_text(lower_query)

    def _extract_quantile(self, lower_query: str) -> float:
        """Extract q from query or use default q=0.9."""

        patterns = [
            r"\bq\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)",
            r"\bquantile\s*=?\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, lower_query)
            if match:
                return float(match.group(1))

        return 0.9

    def _extract_report_include(self, lower_query: str) -> list[str]:
        """Extract report sections or return the deterministic default."""

        include = ["basic_stats", "high_tec", "stable_intervals"]

        if "basic_stats" in lower_query or "basic stats" in lower_query:
            return ["basic_stats"]

        return include

    def _default_compare_metrics(self) -> list[str]:
        """Return deterministic metrics used in compare-region primitive chains."""

        return ["mean", "median", "min", "max", "std", "p90", "p95"]


def _fmt_float(value: Any) -> str:
    """Format optional numeric value."""

    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _extract_comparison_payload(tool_results: dict[str, Any]) -> dict[str, Any]:
    """Extract comparison payload from current or legacy result shapes."""

    if "comparison" in tool_results:
        return tool_results["comparison"]

    if "items" in tool_results:
        return tool_results

    if "stats" in tool_results:
        return {
            "items": [
                {
                    "stats_id": item.get("stats_id", ""),
                    "series_id": item.get("series_id", ""),
                    "region_id": item.get("region_id"),
                    "metrics": item,
                }
                for item in tool_results["stats"]
            ]
        }

    raise KeyError("Could not find comparison payload")


def _report_include(include: list[str] | None) -> list[str]:
    """Return normalized report include sections."""

    selected = include or DEFAULT_REPORT_INCLUDE
    allowed = set(DEFAULT_REPORT_INCLUDE)
    return [section for section in dict.fromkeys(selected) if section in allowed]
