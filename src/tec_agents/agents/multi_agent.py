"""
Rule-based role-oriented multi-agent baseline.

This module implements a deterministic multi-agent workflow without an LLM.
It uses the same MCP-like client and TEC tools as the single-agent baseline,
but separates responsibilities by role instead of by task type:

- orchestrator: parse the request and build the workflow plan;
- data_agent: load and validate data artifacts;
- math_agent: compute deterministic numerical artifacts;
- analysis_agent: convert artifacts into structured findings;
- report_agent: format the final answer from artifacts and findings.

Agents do not call each other directly. RuleBasedMultiAgent coordinates all
stage transitions through the orchestrator-owned workflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tec_agents.agents.protocol import (
    AgentResponse,
    AgentStatus,
    RequestedNextAction,
    StepRecoveryDecision,
    agent_final,
    agent_invalid_input,
    agent_missing_artifacts,
    agent_ok,
    agent_partial,
    agent_tool_error,
    required_artifacts_for_task,
)
from tec_agents.data.dates import parse_date_range_from_text
from tec_agents.data.regions import list_region_ids
from tec_agents.mcp.client import LocalMCPClient


MAX_AGENT_RETRIES = 2
DEFAULT_REPORT_INCLUDE = ["basic_stats", "high_tec", "stable_intervals"]
EXPECTED_ROLE_AGENT_ORDER = [
    "orchestrator",
    "data_agent",
    "math_agent",
    "analysis_agent",
    "report_agent",
]


@dataclass
class ParsedMultiAgentTask:
    """Parsed task for the rule-based multi-agent baseline."""

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
class MultiAgentStep:
    """One high-level multi-agent orchestration step."""

    node: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    missing_artifacts: list[str] = field(default_factory=list)
    requested_next_action: dict[str, Any] | None = None
    can_continue: bool = True
    requires_retry: bool = False
    attempt: int = 1
    max_attempts: int = MAX_AGENT_RETRIES
    decision: str | None = None


@dataclass
class MultiAgentResult:
    """Result of a multi-agent run."""

    answer: str
    parsed_task: ParsedMultiAgentTask
    tool_results: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    orchestration_steps: list[MultiAgentStep] = field(default_factory=list)


class RuleBasedOrchestrator:
    """Rule-based orchestrator that parses queries and builds role plans."""

    def __init__(self, dataset_ref: str = "default") -> None:
        self.dataset_ref = dataset_ref

    def parse(self, query: str) -> ParsedMultiAgentTask:
        """Parse a supported natural-language query into structured fields."""

        lower = query.lower()

        task_type = self._extract_task_type(lower)
        start, end = self._extract_month_range(lower)
        q = self._extract_quantile(lower)

        if task_type == "compare_regions":
            region_ids = self._extract_region_ids(lower)
            if len(region_ids) < 2:
                raise ValueError("Compare task requires at least two recognized regions")

            return ParsedMultiAgentTask(
                task_type=task_type,
                dataset_ref=self.dataset_ref,
                region_id=None,
                region_ids=region_ids,
                regions=region_ids,
                start=start,
                end=end,
                q=q,
                include=self._extract_report_include(lower),
                metrics=self._default_compare_metrics(),
                raw_query=query,
            )

        if task_type == "high_tec":
            region_id = self._extract_region_id(lower)
            return ParsedMultiAgentTask(
                task_type=task_type,
                dataset_ref=self.dataset_ref,
                region_id=region_id,
                region_ids=[region_id],
                regions=[region_id],
                start=start,
                end=end,
                q=q,
                include=self._extract_report_include(lower),
                metrics=self._default_compare_metrics(),
                raw_query=query,
            )

        if task_type == "stable_intervals":
            region_id = self._extract_region_id(lower)
            return ParsedMultiAgentTask(
                task_type=task_type,
                dataset_ref=self.dataset_ref,
                region_id=region_id,
                region_ids=[region_id],
                regions=[region_id],
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

        if task_type == "report":
            region_ids = self._extract_region_ids(lower)
            if not region_ids:
                region_ids = ["midlat_europe"]

            return ParsedMultiAgentTask(
                task_type=task_type,
                dataset_ref=self.dataset_ref,
                region_id=None,
                region_ids=region_ids,
                regions=region_ids,
                start=start,
                end=end,
                q=q,
                include=self._extract_report_include(lower),
                metrics=self._default_compare_metrics(),
                raw_query=query,
            )

        raise ValueError(
            f"Unsupported task_type={task_type!r}. "
            "Multi-agent baseline supports high_tec, compare_regions, "
            "stable_intervals and report."
        )

    def build_plan(self, parsed: ParsedMultiAgentTask) -> MultiAgentStep:
        """Build a deterministic role workflow plan for a parsed task."""

        details = {
            "task_type": parsed.task_type,
            "workflow": "role_based",
            "worker": "role_based_workflow",
            "selected_worker": "role_based_workflow",
            "stages": ["data", "math", "analysis", "report"],
            "expected_role_agent_order": EXPECTED_ROLE_AGENT_ORDER,
            "region_id": parsed.region_id,
            "region_ids": parsed.region_ids,
            "regions": parsed.regions,
            "start": parsed.start,
            "end": parsed.end,
            "q": parsed.q,
            "window_minutes": parsed.window_minutes,
            "q_delta": parsed.q_delta,
            "q_std": parsed.q_std,
            "include": _report_include(parsed.include),
            "metrics": parsed.metrics,
        }

        return MultiAgentStep(
            node="orchestrator",
            action="build_plan",
            details=details,
        )

    def decide_next_action(
        self,
        response: AgentResponse,
        context: dict[str, Any] | None = None,
    ) -> StepRecoveryDecision:
        """Decide how the workflow should proceed after an agent response."""

        context = context or {}
        attempt = int(context.get("attempt", response.attempt))
        max_attempts = int(context.get("max_attempts", response.max_attempts))

        if response.status == AgentStatus.FINAL:
            return StepRecoveryDecision(
                decision="continue",
                target_agent=response.agent,
                reason="Final response produced.",
                attempt=attempt,
                max_attempts=max_attempts,
            )

        if response.status == AgentStatus.OK:
            return StepRecoveryDecision(
                decision="continue",
                target_agent=response.agent,
                reason="Stage completed.",
                attempt=attempt,
                max_attempts=max_attempts,
            )

        if response.status == AgentStatus.PARTIAL:
            if response.can_continue:
                return StepRecoveryDecision(
                    decision="continue_partial",
                    target_agent=response.agent,
                    reason=response.message or "Partial artifacts can continue.",
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
            return StepRecoveryDecision(
                decision="fail",
                target_agent=response.agent,
                reason=response.message or "Partial artifacts cannot continue.",
                attempt=attempt,
                max_attempts=max_attempts,
            )

        if response.status == AgentStatus.MISSING_ARTIFACTS:
            if attempt >= max_attempts:
                return StepRecoveryDecision(
                    decision="fail",
                    target_agent=response.agent,
                    reason="Maximum recovery attempts exceeded.",
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
            if response.requested_next_action is not None:
                return StepRecoveryDecision(
                    decision="call_requested_agent",
                    target_agent=response.requested_next_action.target_agent,
                    reason=response.requested_next_action.reason,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
            return StepRecoveryDecision(
                decision="fail",
                target_agent=response.agent,
                reason=response.message or "Missing required artifacts.",
                attempt=attempt,
                max_attempts=max_attempts,
            )

        if response.status == AgentStatus.TOOL_ERROR:
            if response.requires_retry and attempt < max_attempts:
                return StepRecoveryDecision(
                    decision="retry_same_agent",
                    target_agent=response.agent,
                    reason=response.message or "Retryable tool error.",
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
            return StepRecoveryDecision(
                decision="fail",
                target_agent=response.agent,
                reason=response.message or "Tool error could not be recovered.",
                attempt=attempt,
                max_attempts=max_attempts,
            )

        if response.status == AgentStatus.INVALID_INPUT:
            return StepRecoveryDecision(
                decision="fail",
                target_agent=response.agent,
                reason=response.message or "Invalid input.",
                attempt=attempt,
                max_attempts=max_attempts,
            )

        return StepRecoveryDecision(
            decision="fail",
            target_agent=response.agent,
            reason=f"Unhandled agent status: {response.status.value}",
            attempt=attempt,
            max_attempts=max_attempts,
        )

    def parse_and_route(
        self,
        query: str,
    ) -> tuple[ParsedMultiAgentTask, str, MultiAgentStep]:
        """
        Backward-compatible wrapper for callers that still expect a route.

        The returned worker is the role-based workflow marker, not a
        task-specific worker.
        """

        parsed = self.parse(query)
        return parsed, "role_based_workflow", self.build_plan(parsed)

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

        include: list[str] = []

        if "basic_stats" in lower_query or "basic stats" in lower_query:
            include.append("basic_stats")
        if "high_tec" in lower_query or "high tec" in lower_query:
            include.append("high_tec")
        if (
            "stable_intervals" in lower_query
            or "stable intervals" in lower_query
            or "low variability" in lower_query
        ):
            include.append("stable_intervals")

        return include or list(DEFAULT_REPORT_INCLUDE)

    def _default_compare_metrics(self) -> list[str]:
        """Return deterministic metrics used in compare-region primitive chains."""

        return ["mean", "median", "min", "max", "std", "p90", "p95"]


class OrchestratorAgent(RuleBasedOrchestrator):
    """Named role alias for future LLM-backed orchestrator integrations."""


class DataAgent:
    """Role agent responsible only for data loading."""

    allowed_tools = {"tec_get_timeseries", "tec_series_profile"}

    def __init__(self, client: LocalMCPClient, agent_name: str = "data_agent") -> None:
        self.client = client
        self.agent_name = agent_name

    def load_series_for_regions(
        self,
        parsed: ParsedMultiAgentTask,
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Load all required time series before any math stage runs."""

        regions = _regions_for_parsed(parsed)
        if not regions:
            return agent_invalid_input(
                agent=self.agent_name,
                message="No regions were provided.",
                missing_artifacts=["parsed.region_ids"],
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        series_by_region: dict[str, dict[str, Any]] = {}

        for step, region_id in enumerate(regions, start=1):
            response = self.client.call_tool(
                "tec_get_timeseries",
                {
                    "dataset_ref": parsed.dataset_ref,
                    "region_id": region_id,
                    "start": parsed.start,
                    "end": parsed.end,
                },
                agent_name=self.agent_name,
                step=step,
            )
            if response.status != "ok" or response.result is None:
                missing = [f"data.series_by_region.{region_id}"]
                return agent_tool_error(
                    agent=self.agent_name,
                    message=f"tec_get_timeseries failed for {region_id}.",
                    artifacts={
                        "series_by_region": series_by_region,
                        "regions": regions,
                    },
                    missing_artifacts=missing,
                    requested_next_action=RequestedNextAction(
                        target_agent="data_agent",
                        task="retry_load_series",
                        reason="tec_get_timeseries failed",
                        required_artifacts=missing,
                        params={"regions": [region_id]},
                    ),
                    can_continue=False,
                    requires_retry=True,
                    attempt=attempt,
                    max_attempts=MAX_AGENT_RETRIES,
                )

            ts_result = response.result
            series_by_region[region_id] = {
                "series_id": ts_result["series_id"],
                "tool_result": ts_result,
                "metadata": ts_result.get("metadata"),
            }

        artifacts = {
            "series_by_region": series_by_region,
            "regions": regions,
            "dataset_ref": parsed.dataset_ref,
            "start": parsed.start,
            "end": parsed.end,
        }

        return agent_ok(
            agent=self.agent_name,
            artifacts=artifacts,
            message=f"Loaded {len(series_by_region)} series.",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )


class MathAgent:
    """Role agent responsible only for deterministic computations."""

    allowed_tools = {
        "tec_compute_high_threshold",
        "tec_detect_high_intervals",
        "tec_compute_stability_thresholds",
        "tec_detect_stable_intervals",
        "tec_compute_series_stats",
        "tec_compare_stats",
    }

    def __init__(self, client: LocalMCPClient, agent_name: str = "math_agent") -> None:
        self.client = client
        self.agent_name = agent_name

    def compute_for_task(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Dispatch task-specific math while keeping data access separate."""

        if parsed.task_type == "high_tec":
            return self._compute_high_tec(parsed, data_artifacts, attempt=attempt)

        if parsed.task_type == "stable_intervals":
            return self._compute_stable_intervals(parsed, data_artifacts, attempt=attempt)

        if parsed.task_type == "compare_regions":
            return self._compute_compare_regions(parsed, data_artifacts, attempt=attempt)

        if parsed.task_type == "report":
            return self._compute_report_inputs(parsed, data_artifacts, attempt=attempt)

        return agent_invalid_input(
            agent=self.agent_name,
            message=f"Unsupported task type for MathAgent: {parsed.task_type!r}",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _compute_high_tec(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Compute high-TEC thresholds and intervals for one region."""

        region_id = _first_region(parsed)
        try:
            series_id = _series_id_for_region(data_artifacts, region_id)
        except KeyError:
            return _math_missing_series_response(
                self.agent_name,
                region_id,
                attempt=attempt,
            )

        threshold_response = self.client.call_tool(
            "tec_compute_high_threshold",
            {
                "series_id": series_id,
                "method": "quantile",
                "q": parsed.q,
            },
            agent_name=self.agent_name,
            step=1,
        )
        if threshold_response.status != "ok" or threshold_response.result is None:
            return _math_tool_error_response(
                self.agent_name,
                tool_name="tec_compute_high_threshold",
                region_id=region_id,
                missing_artifact=f"math.high_tec.{region_id}.threshold",
                attempt=attempt,
            )
        threshold_result = threshold_response.result

        intervals_response = self.client.call_tool(
            "tec_detect_high_intervals",
            {
                "series_id": series_id,
                "threshold_id": threshold_result["threshold_id"],
                "min_duration_minutes": 0,
                "merge_gap_minutes": 60,
            },
            agent_name=self.agent_name,
            step=2,
        )
        if intervals_response.status != "ok" or intervals_response.result is None:
            return _math_tool_error_response(
                self.agent_name,
                tool_name="tec_detect_high_intervals",
                region_id=region_id,
                missing_artifact=f"math.high_tec.{region_id}.intervals",
                attempt=attempt,
            )
        intervals_result = intervals_response.result

        artifacts = {
            "high_tec": {
                region_id: {
                    "threshold": threshold_result,
                    "intervals": intervals_result,
                }
            }
        }

        return agent_ok(
            agent=self.agent_name,
            artifacts=artifacts,
            message="Computed high TEC artifacts.",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _compute_stable_intervals(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Compute stable interval thresholds and detections for one region."""

        region_id = _first_region(parsed)
        try:
            series_id = _series_id_for_region(data_artifacts, region_id)
        except KeyError:
            return _math_missing_series_response(
                self.agent_name,
                region_id,
                attempt=attempt,
            )

        thresholds_response = self.client.call_tool(
            "tec_compute_stability_thresholds",
            {
                "series_id": series_id,
                "window_minutes": parsed.window_minutes,
                "method": "quantile",
                "q_delta": parsed.q_delta,
                "q_std": parsed.q_std,
            },
            agent_name=self.agent_name,
            step=1,
        )
        if thresholds_response.status != "ok" or thresholds_response.result is None:
            return _math_tool_error_response(
                self.agent_name,
                tool_name="tec_compute_stability_thresholds",
                region_id=region_id,
                missing_artifact=f"math.stable_intervals.{region_id}.thresholds",
                attempt=attempt,
            )
        thresholds_result = thresholds_response.result

        intervals_response = self.client.call_tool(
            "tec_detect_stable_intervals",
            {
                "series_id": series_id,
                "threshold_id": thresholds_result["threshold_id"],
                "min_duration_minutes": parsed.window_minutes,
                "merge_gap_minutes": 60,
            },
            agent_name=self.agent_name,
            step=2,
        )
        if intervals_response.status != "ok" or intervals_response.result is None:
            return _math_tool_error_response(
                self.agent_name,
                tool_name="tec_detect_stable_intervals",
                region_id=region_id,
                missing_artifact=f"math.stable_intervals.{region_id}.intervals",
                attempt=attempt,
            )
        intervals_result = intervals_response.result

        artifacts = {
            "stable_intervals": {
                region_id: {
                    "thresholds": thresholds_result,
                    "intervals": intervals_result,
                }
            }
        }

        return agent_ok(
            agent=self.agent_name,
            artifacts=artifacts,
            message="Computed stable interval artifacts.",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _compute_compare_regions(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Compute all region stats first, then compare the stats handles."""

        regions = _regions_for_parsed(parsed)
        if len(regions) < 2:
            return agent_invalid_input(
                agent=self.agent_name,
                message="Comparison task requires at least two region_ids.",
                missing_artifacts=["parsed.region_ids"],
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        stats_by_region: dict[str, dict[str, Any]] = {}
        stats_ids: list[str] = []

        for step, region_id in enumerate(regions, start=1):
            try:
                series_id = _series_id_for_region(data_artifacts, region_id)
            except KeyError:
                return _math_missing_series_response(
                    self.agent_name,
                    region_id,
                    attempt=attempt,
                )

            stats_response = self.client.call_tool(
                "tec_compute_series_stats",
                {
                    "series_id": series_id,
                    "metrics": parsed.metrics,
                },
                agent_name=self.agent_name,
                step=step,
            )
            if stats_response.status != "ok" or stats_response.result is None:
                return _math_tool_error_response(
                    self.agent_name,
                    tool_name="tec_compute_series_stats",
                    region_id=region_id,
                    missing_artifact=f"math.stats_by_region.{region_id}",
                    attempt=attempt,
                )
            stats_result = stats_response.result

            stats_by_region[region_id] = {
                "stats_id": stats_result["stats_id"],
                "series_id": series_id,
                "stats": stats_result,
                "metrics": stats_result.get("metrics", {}),
            }
            stats_ids.append(stats_result["stats_id"])

        comparison_response = self.client.call_tool(
            "tec_compare_stats",
            {
                "stats_ids": stats_ids,
                "metrics": parsed.metrics,
            },
            agent_name=self.agent_name,
            step=len(regions) + 1,
        )
        if comparison_response.status != "ok" or comparison_response.result is None:
            return _math_tool_error_response(
                self.agent_name,
                tool_name="tec_compare_stats",
                region_id=",".join(regions),
                missing_artifact="math.comparison",
                attempt=attempt,
            )
        comparison_result = comparison_response.result

        artifacts = {
            "stats_by_region": stats_by_region,
            "comparison": comparison_result,
        }

        return agent_ok(
            agent=self.agent_name,
            artifacts=artifacts,
            message="Computed region comparison artifacts.",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _compute_report_inputs(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Compute primitive artifacts needed for a structured TEC report."""

        regions = _regions_for_parsed(parsed)
        if not regions:
            return agent_invalid_input(
                agent=self.agent_name,
                message="Report task requires at least one region.",
                missing_artifacts=["parsed.region_ids"],
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        include = _report_include(parsed.include)
        report_inputs: dict[str, Any] = {}
        step = 1

        if "basic_stats" in include:
            by_region: dict[str, dict[str, Any]] = {}
            stats_ids: list[str] = []

            for region_id in regions:
                try:
                    series_id = _series_id_for_region(data_artifacts, region_id)
                except KeyError:
                    return _math_missing_series_response(
                        self.agent_name,
                        region_id,
                        attempt=attempt,
                    )
                stats_response = self.client.call_tool(
                    "tec_compute_series_stats",
                    {
                        "series_id": series_id,
                        "metrics": parsed.metrics,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1
                if stats_response.status != "ok" or stats_response.result is None:
                    return _math_tool_error_response(
                        self.agent_name,
                        tool_name="tec_compute_series_stats",
                        region_id=region_id,
                        missing_artifact=f"math.report_inputs.basic_stats.{region_id}",
                        attempt=attempt,
                    )

                stats_result = stats_response.result
                by_region[region_id] = {
                    "stats_id": stats_result["stats_id"],
                    "series_id": series_id,
                    "stats": stats_result,
                    "metrics": stats_result.get("metrics", {}),
                }
                stats_ids.append(stats_result["stats_id"])

            comparison = None
            if len(stats_ids) >= 2:
                comparison_response = self.client.call_tool(
                    "tec_compare_stats",
                    {
                        "stats_ids": stats_ids,
                        "metrics": parsed.metrics,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1
                if comparison_response.status != "ok" or comparison_response.result is None:
                    return _math_tool_error_response(
                        self.agent_name,
                        tool_name="tec_compare_stats",
                        region_id=",".join(regions),
                        missing_artifact="math.report_inputs.basic_stats.comparison",
                        attempt=attempt,
                    )
                comparison = comparison_response.result

            report_inputs["basic_stats"] = {
                "by_region": by_region,
                "comparison": comparison,
            }

        if "high_tec" in include:
            by_region = {}

            for region_id in regions:
                try:
                    series_id = _series_id_for_region(data_artifacts, region_id)
                except KeyError:
                    return _math_missing_series_response(
                        self.agent_name,
                        region_id,
                        attempt=attempt,
                    )
                threshold_response = self.client.call_tool(
                    "tec_compute_high_threshold",
                    {
                        "series_id": series_id,
                        "method": "quantile",
                        "q": parsed.q,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1
                if threshold_response.status != "ok" or threshold_response.result is None:
                    return _math_tool_error_response(
                        self.agent_name,
                        tool_name="tec_compute_high_threshold",
                        region_id=region_id,
                        missing_artifact=f"math.report_inputs.high_tec.{region_id}.threshold",
                        attempt=attempt,
                    )
                threshold_result = threshold_response.result

                intervals_response = self.client.call_tool(
                    "tec_detect_high_intervals",
                    {
                        "series_id": series_id,
                        "threshold_id": threshold_result["threshold_id"],
                        "min_duration_minutes": 0,
                        "merge_gap_minutes": 60,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1
                if intervals_response.status != "ok" or intervals_response.result is None:
                    return _math_tool_error_response(
                        self.agent_name,
                        tool_name="tec_detect_high_intervals",
                        region_id=region_id,
                        missing_artifact=f"math.report_inputs.high_tec.{region_id}.intervals",
                        attempt=attempt,
                    )
                intervals_result = intervals_response.result

                by_region[region_id] = {
                    "threshold": threshold_result,
                    "intervals": intervals_result,
                }

            report_inputs["high_tec"] = {"by_region": by_region}

        if "stable_intervals" in include:
            by_region = {}

            for region_id in regions:
                try:
                    series_id = _series_id_for_region(data_artifacts, region_id)
                except KeyError:
                    return _math_missing_series_response(
                        self.agent_name,
                        region_id,
                        attempt=attempt,
                    )
                thresholds_response = self.client.call_tool(
                    "tec_compute_stability_thresholds",
                    {
                        "series_id": series_id,
                        "window_minutes": parsed.window_minutes,
                        "method": "quantile",
                        "q_delta": parsed.q_delta,
                        "q_std": parsed.q_std,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1
                if thresholds_response.status != "ok" or thresholds_response.result is None:
                    return _math_tool_error_response(
                        self.agent_name,
                        tool_name="tec_compute_stability_thresholds",
                        region_id=region_id,
                        missing_artifact=f"math.report_inputs.stable_intervals.{region_id}.thresholds",
                        attempt=attempt,
                    )
                thresholds_result = thresholds_response.result

                intervals_response = self.client.call_tool(
                    "tec_detect_stable_intervals",
                    {
                        "series_id": series_id,
                        "threshold_id": thresholds_result["threshold_id"],
                        "min_duration_minutes": parsed.window_minutes,
                        "merge_gap_minutes": 60,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1
                if intervals_response.status != "ok" or intervals_response.result is None:
                    return _math_tool_error_response(
                        self.agent_name,
                        tool_name="tec_detect_stable_intervals",
                        region_id=region_id,
                        missing_artifact=f"math.report_inputs.stable_intervals.{region_id}.intervals",
                        attempt=attempt,
                    )
                intervals_result = intervals_response.result

                by_region[region_id] = {
                    "thresholds": thresholds_result,
                    "intervals": intervals_result,
                }

            report_inputs["stable_intervals"] = {"by_region": by_region}

        artifacts = {"report_inputs": report_inputs}

        return agent_ok(
            agent=self.agent_name,
            artifacts=artifacts,
            message="Computed report input artifacts.",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )


class AnalysisAgent:
    """Role agent responsible for deterministic artifact interpretation."""

    allowed_tools: set[str] = set()

    def __init__(self, agent_name: str = "analysis_agent") -> None:
        self.agent_name = agent_name

    def analyze(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        math_artifacts: dict[str, Any],
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Create structured findings without calling computational tools."""

        if not math_artifacts:
            return agent_missing_artifacts(
                agent=self.agent_name,
                missing_artifacts=["math"],
                requested_next_action=RequestedNextAction(
                    target_agent="math_agent",
                    task="compute_missing_math_artifacts",
                    reason="AnalysisAgent needs math artifacts",
                    required_artifacts=["math"],
                ),
                message="Cannot analyze without math artifacts.",
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        findings: list[dict[str, Any]] = []

        if parsed.task_type == "compare_regions":
            findings.extend(self._analyze_comparison(math_artifacts))
        elif parsed.task_type == "high_tec":
            findings.extend(self._analyze_high_tec(math_artifacts.get("high_tec", {})))
        elif parsed.task_type == "stable_intervals":
            findings.extend(
                self._analyze_stable_intervals(
                    math_artifacts.get("stable_intervals", {})
                )
            )
        elif parsed.task_type == "report":
            findings.extend(self._analyze_report(math_artifacts))

        summary = self._summary(
            parsed=parsed,
            data_artifacts=data_artifacts,
            math_artifacts=math_artifacts,
            findings=findings,
        )

        artifacts = {
            "findings": findings,
            "summary": summary,
        }

        required = required_artifacts_for_task(parsed.task_type, parsed.include)
        missing_optional = [
            artifact
            for artifact in required.get("optional", [])
            if not _artifact_path_exists(
                {"data": data_artifacts, "math": math_artifacts, "analysis": artifacts},
                artifact,
            )
        ]

        if missing_optional:
            return agent_partial(
                agent=self.agent_name,
                artifacts=artifacts,
                missing_artifacts=missing_optional,
                message="Some optional sections missing; findings generated from available artifacts.",
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        return agent_ok(
            agent=self.agent_name,
            artifacts=artifacts,
            message=f"Generated {len(findings)} findings.",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _analyze_comparison(
        self,
        math_artifacts: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Analyze comparison stats for leader metrics."""

        comparison = math_artifacts.get("comparison") or {}
        items = comparison.get("items") or []
        return [
            _leader_finding(items, metric="mean"),
            _leader_finding(items, metric="p90"),
            _leader_finding(items, metric="max"),
        ]

    def _analyze_high_tec(
        self,
        by_region: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Analyze high-TEC artifacts by region."""

        findings: list[dict[str, Any]] = []

        for region_id, item in by_region.items():
            threshold = item.get("threshold") or {}
            intervals = item.get("intervals") or {}
            interval_records = intervals.get("intervals") or []
            peak = _max_interval_value(interval_records, "peak_value")

            findings.append(
                {
                    "type": "high_tec_interval_count",
                    "region_id": region_id,
                    "value": intervals.get("n_intervals"),
                    "text": (
                        f"{region_id} has {intervals.get('n_intervals')} "
                        "high TEC intervals."
                    ),
                }
            )
            findings.append(
                {
                    "type": "high_tec_threshold",
                    "region_id": region_id,
                    "value": threshold.get("value"),
                    "text": (
                        f"{region_id} high TEC threshold is "
                        f"{_fmt_float(threshold.get('value'))} TECU."
                    ),
                }
            )
            findings.append(
                {
                    "type": "high_tec_global_peak",
                    "region_id": region_id,
                    "value": peak,
                    "text": (
                        f"{region_id} global high TEC peak is "
                        f"{_fmt_float(peak)} TECU."
                    ),
                }
            )

        return findings

    def _analyze_stable_intervals(
        self,
        by_region: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Analyze stable interval artifacts by region."""

        findings: list[dict[str, Any]] = []

        for region_id, item in by_region.items():
            intervals = item.get("intervals") or {}
            interval_records = intervals.get("intervals") or []
            longest = _max_interval_value(interval_records, "duration_minutes")

            findings.append(
                {
                    "type": "stable_interval_count",
                    "region_id": region_id,
                    "value": intervals.get("n_intervals"),
                    "text": (
                        f"{region_id} has {intervals.get('n_intervals')} "
                        "stable TEC intervals."
                    ),
                }
            )
            findings.append(
                {
                    "type": "stable_longest_interval",
                    "region_id": region_id,
                    "value": longest,
                    "text": (
                        f"{region_id} longest stable interval is "
                        f"{_fmt_float(longest)} minutes."
                    ),
                }
            )

        return findings

    def _analyze_report(self, math_artifacts: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze report input artifacts across all included sections."""

        report_inputs = math_artifacts.get("report_inputs") or {}
        findings: list[dict[str, Any]] = []

        basic = report_inputs.get("basic_stats") or {}
        comparison = basic.get("comparison")
        if comparison:
            findings.extend(self._analyze_comparison({"comparison": comparison}))

        high = report_inputs.get("high_tec") or {}
        findings.extend(self._analyze_high_tec(high.get("by_region") or {}))

        stable = report_inputs.get("stable_intervals") or {}
        findings.extend(self._analyze_stable_intervals(stable.get("by_region") or {}))

        return findings

    def _summary(
        self,
        *,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        math_artifacts: dict[str, Any],
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a compact deterministic summary from artifacts."""

        regions = data_artifacts.get("regions") or _regions_for_parsed(parsed)
        primary_region = None

        for finding in findings:
            if finding.get("type") == "comparison_leader":
                primary_region = finding.get("region_id")
                break

        if primary_region is None and regions:
            primary_region = regions[0]

        high_total = 0
        stable_total = 0

        high_by_region = _high_tec_artifacts_by_region(math_artifacts)
        for item in high_by_region.values():
            intervals = item.get("intervals") or {}
            high_total += int(intervals.get("n_intervals") or 0)

        stable_by_region = _stable_artifacts_by_region(math_artifacts)
        for item in stable_by_region.values():
            intervals = item.get("intervals") or {}
            stable_total += int(intervals.get("n_intervals") or 0)

        return {
            "primary_region": primary_region,
            "n_regions": len(regions),
            "n_high_tec_intervals_total": high_total,
            "n_stable_intervals_total": stable_total,
        }


class ReportAgent:
    """Role agent responsible only for final answer formatting."""

    allowed_tools: set[str] = set()

    def __init__(self, agent_name: str = "report_agent") -> None:
        self.agent_name = agent_name

    def format_answer(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        math_artifacts: dict[str, Any],
        analysis_artifacts: dict[str, Any],
        *,
        attempt: int = 1,
    ) -> AgentResponse:
        """Format a deterministic final answer from existing artifacts."""

        context = {
            "data": data_artifacts,
            "math": math_artifacts,
            "analysis": analysis_artifacts,
        }
        artifact_spec = required_artifacts_for_task(parsed.task_type, parsed.include)
        missing_required = [
            path
            for path in artifact_spec.get("required", [])
            if not _artifact_path_exists(context, path)
        ]
        if missing_required:
            target_agent = _target_agent_for_artifact(missing_required[0])
            return agent_missing_artifacts(
                agent=self.agent_name,
                missing_artifacts=missing_required,
                requested_next_action=RequestedNextAction(
                    target_agent=target_agent,
                    task="provide_missing_artifacts",
                    reason="Cannot produce grounded report without required artifacts",
                    required_artifacts=missing_required,
                ),
                message=(
                    "Cannot produce grounded report because required artifacts "
                    "are missing."
                ),
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        missing_optional = [
            path
            for path in artifact_spec.get("optional", [])
            if not _artifact_path_exists(context, path)
        ]

        if parsed.task_type == "high_tec":
            answer = self._format_high_tec_answer(parsed, math_artifacts)
        elif parsed.task_type == "compare_regions":
            answer = self._format_compare_regions_answer(parsed, math_artifacts)
        elif parsed.task_type == "stable_intervals":
            answer = self._format_stable_intervals_answer(parsed, math_artifacts)
        elif parsed.task_type == "report":
            answer = self._format_report_answer(
                parsed=parsed,
                data_artifacts=data_artifacts,
                math_artifacts=math_artifacts,
                analysis_artifacts=analysis_artifacts,
            )
        else:
            return agent_invalid_input(
                agent=self.agent_name,
                message=f"Unsupported task type for ReportAgent: {parsed.task_type!r}",
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        if missing_optional:
            return agent_partial(
                agent=self.agent_name,
                artifacts={"answer": f"[PARTIAL] {answer}"},
                missing_artifacts=missing_optional,
                message="Final answer produced without optional artifacts.",
                attempt=attempt,
                max_attempts=MAX_AGENT_RETRIES,
            )

        return agent_final(
            agent=self.agent_name,
            artifacts={"answer": answer},
            message="Final answer formatted.",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _format_high_tec_answer(
        self,
        parsed: ParsedMultiAgentTask,
        math_artifacts: dict[str, Any],
    ) -> str:
        """Create a compact answer for high-TEC tasks."""

        region_id = _first_region(parsed)
        item = (math_artifacts.get("high_tec") or {})[region_id]
        threshold = item["threshold"]
        intervals = item["intervals"]

        return (
            f"High TEC intervals for {region_id} from {parsed.start} "
            f"to {parsed.end}: threshold={threshold['value']:.3f} TECU "
            f"(q={parsed.q}), detected {intervals['n_intervals']} intervals."
        )

    def _format_compare_regions_answer(
        self,
        parsed: ParsedMultiAgentTask,
        math_artifacts: dict[str, Any],
    ) -> str:
        """Create a compact answer for region comparison."""

        comparison = math_artifacts["comparison"]
        lines = [
            (
                f"TEC statistics comparison for {', '.join(parsed.region_ids)} "
                f"from {parsed.start} to {parsed.end}:"
            )
        ]

        for item in comparison.get("items", []):
            metrics = item.get("metrics") or {}
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
        parsed: ParsedMultiAgentTask,
        math_artifacts: dict[str, Any],
    ) -> str:
        """Create a compact answer for stable interval tasks."""

        region_id = _first_region(parsed)
        item = (math_artifacts.get("stable_intervals") or {})[region_id]
        thresholds = item["thresholds"]
        intervals = item["intervals"]

        return (
            f"Stable TEC intervals for {region_id} from {parsed.start} "
            f"to {parsed.end}: detected {intervals['n_intervals']} intervals "
            f"using window={thresholds['window_minutes']} min, "
            f"max_delta={thresholds['max_delta_threshold']:.3f}, "
            f"rolling_std={thresholds['rolling_std_threshold']:.3f}."
        )

    def _format_report_answer(
        self,
        *,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        math_artifacts: dict[str, Any],
        analysis_artifacts: dict[str, Any],
    ) -> str:
        """Create a compact structured answer for report tasks."""

        regions = data_artifacts.get("regions") or parsed.region_ids
        include = _report_include(parsed.include)
        report_inputs = math_artifacts.get("report_inputs") or {}
        findings = analysis_artifacts.get("findings") or []

        lines = [
            (
                f"TEC report for {', '.join(regions)} from {parsed.start} "
                f"to {parsed.end}."
            ),
            f"Sections: {', '.join(include)}.",
        ]

        if findings:
            lines.append("Main findings:")
            for finding in findings[:5]:
                lines.append(f"- {finding.get('text')}")

        basic = report_inputs.get("basic_stats") or {}
        comparison = basic.get("comparison")
        if comparison:
            lines.append("Basic statistics:")
            for item in comparison.get("items", []):
                metrics = item.get("metrics") or {}
                lines.append(
                    f"- {item['region_id']}: "
                    f"mean={_fmt_float(metrics.get('mean'))}, "
                    f"max={_fmt_float(metrics.get('max'))}, "
                    f"p90={_fmt_float(metrics.get('p90'))}."
                )

        high = report_inputs.get("high_tec") or {}
        high_by_region = high.get("by_region") or {}
        if high_by_region:
            lines.append("High TEC:")
            for region_id, item in high_by_region.items():
                threshold = item.get("threshold") or {}
                intervals = item.get("intervals") or {}
                lines.append(
                    f"- {region_id}: threshold={_fmt_float(threshold.get('value'))}, "
                    f"intervals={intervals.get('n_intervals')}."
                )

        stable = report_inputs.get("stable_intervals") or {}
        stable_by_region = stable.get("by_region") or {}
        if stable_by_region:
            lines.append("Stable intervals:")
            for region_id, item in stable_by_region.items():
                thresholds = item.get("thresholds") or {}
                intervals = item.get("intervals") or {}
                lines.append(
                    f"- {region_id}: intervals={intervals.get('n_intervals')}, "
                    f"window={thresholds.get('window_minutes')} min."
                )

        return "\n".join(lines)


class RuleBasedMultiAgent:
    """
    Deterministic role-based multi-agent baseline.

    The orchestrator parses and plans. DataAgent loads all data first,
    MathAgent computes numerical artifacts, AnalysisAgent creates findings,
    and ReportAgent formats the final answer.
    """

    def __init__(
        self,
        client: LocalMCPClient,
        dataset_ref: str = "default",
    ) -> None:
        self.client = client
        self.dataset_ref = dataset_ref

        self.orchestrator = RuleBasedOrchestrator(dataset_ref=dataset_ref)
        self.data_agent = DataAgent(client=client)
        self.math_agent = MathAgent(client=client)
        self.analysis_agent = AnalysisAgent()
        self.report_agent = ReportAgent()

    def reset(self) -> None:
        """Reset underlying MCP client state if supported."""

        if hasattr(self.client, "reset"):
            self.client.reset()

    def run(self, query: str) -> MultiAgentResult:
        """Run the role-based multi-agent baseline on one user query."""

        orchestration_steps: list[MultiAgentStep] = []

        parsed = self.orchestrator.parse(query)
        plan_step = self.orchestrator.build_plan(parsed)
        orchestration_steps.append(plan_step)

        data_artifacts: dict[str, Any] = {}
        math_artifacts: dict[str, Any] = {}
        analysis_artifacts: dict[str, Any] = {}

        data_response = self._execute_stage(
            stage="data",
            parsed=parsed,
            data_artifacts=data_artifacts,
            math_artifacts=math_artifacts,
            analysis_artifacts=analysis_artifacts,
            orchestration_steps=orchestration_steps,
        )
        if not _response_can_advance(data_response):
            return self._failure_result(parsed, data_response, orchestration_steps)
        data_artifacts.update(data_response.artifacts)

        math_response = self._execute_stage(
            stage="math",
            parsed=parsed,
            data_artifacts=data_artifacts,
            math_artifacts=math_artifacts,
            analysis_artifacts=analysis_artifacts,
            orchestration_steps=orchestration_steps,
        )
        if not _response_can_advance(math_response):
            return self._failure_result(parsed, math_response, orchestration_steps)
        math_artifacts.update(math_response.artifacts)

        analysis_response = self._execute_stage(
            stage="analysis",
            parsed=parsed,
            data_artifacts=data_artifacts,
            math_artifacts=math_artifacts,
            analysis_artifacts=analysis_artifacts,
            orchestration_steps=orchestration_steps,
        )
        if not _response_can_advance(analysis_response):
            return self._failure_result(parsed, analysis_response, orchestration_steps)
        analysis_artifacts.update(analysis_response.artifacts)

        report_response = self._execute_stage(
            stage="report",
            parsed=parsed,
            data_artifacts=data_artifacts,
            math_artifacts=math_artifacts,
            analysis_artifacts=analysis_artifacts,
            orchestration_steps=orchestration_steps,
        )
        if not _response_can_advance(report_response):
            return self._failure_result(parsed, report_response, orchestration_steps)

        answer = str(report_response.artifacts.get("answer", ""))

        return MultiAgentResult(
            answer=answer,
            parsed_task=parsed,
            tool_results={
                "data": data_artifacts,
                "math": math_artifacts,
                "analysis": analysis_artifacts,
            },
            trace=self.client.get_trace(),
            orchestration_steps=orchestration_steps,
        )

    def _execute_stage(
        self,
        *,
        stage: str,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        math_artifacts: dict[str, Any],
        analysis_artifacts: dict[str, Any],
        orchestration_steps: list[MultiAgentStep],
    ) -> AgentResponse:
        """Execute one workflow stage with bounded recovery decisions."""

        for attempt in range(1, MAX_AGENT_RETRIES + 1):
            response = self._call_stage(
                stage=stage,
                parsed=parsed,
                data_artifacts=data_artifacts,
                math_artifacts=math_artifacts,
                analysis_artifacts=analysis_artifacts,
                attempt=attempt,
            )
            decision = self.orchestrator.decide_next_action(
                response,
                {
                    "stage": stage,
                    "attempt": attempt,
                    "max_attempts": MAX_AGENT_RETRIES,
                },
            )
            orchestration_steps.append(
                _step_from_response(
                    response=response,
                    action=_stage_action(stage, parsed.task_type),
                    details=_stage_details(stage, response, parsed),
                    decision=decision,
                )
            )

            if decision.decision in {"continue", "continue_partial"}:
                return response

            if decision.decision == "retry_same_agent":
                continue

            if decision.decision == "call_requested_agent":
                recovery_response = self._call_requested_agent(
                    response=response,
                    parsed=parsed,
                    data_artifacts=data_artifacts,
                    math_artifacts=math_artifacts,
                    analysis_artifacts=analysis_artifacts,
                    attempt=attempt + 1,
                )
                recovery_decision = self.orchestrator.decide_next_action(
                    recovery_response,
                    {
                        "stage": "recovery",
                        "attempt": attempt + 1,
                        "max_attempts": MAX_AGENT_RETRIES,
                    },
                )
                orchestration_steps.append(
                    _step_from_response(
                        response=recovery_response,
                        action="recovery_action",
                        details={"requested_by": response.agent},
                        decision=recovery_decision,
                    )
                )
                if _response_can_advance(recovery_response):
                    _merge_stage_artifacts(
                        recovery_response,
                        data_artifacts=data_artifacts,
                        math_artifacts=math_artifacts,
                        analysis_artifacts=analysis_artifacts,
                    )
                    continue
                return response

            return response

        return response

    def _call_stage(
        self,
        *,
        stage: str,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        math_artifacts: dict[str, Any],
        analysis_artifacts: dict[str, Any],
        attempt: int,
    ) -> AgentResponse:
        """Call the role agent for a workflow stage."""

        if stage == "data":
            return self.data_agent.load_series_for_regions(parsed, attempt=attempt)
        if stage == "math":
            return self.math_agent.compute_for_task(
                parsed,
                data_artifacts,
                attempt=attempt,
            )
        if stage == "analysis":
            return self.analysis_agent.analyze(
                parsed,
                data_artifacts,
                math_artifacts,
                attempt=attempt,
            )
        if stage == "report":
            return self.report_agent.format_answer(
                parsed,
                data_artifacts,
                math_artifacts,
                analysis_artifacts,
                attempt=attempt,
            )
        return agent_invalid_input(
            agent="orchestrator",
            message=f"Unknown workflow stage: {stage}",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _call_requested_agent(
        self,
        *,
        response: AgentResponse,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
        math_artifacts: dict[str, Any],
        analysis_artifacts: dict[str, Any],
        attempt: int,
    ) -> AgentResponse:
        """Call the agent requested by a recovery response."""

        action = response.requested_next_action
        if action is None:
            return response

        target = action.target_agent
        if target == "data_agent":
            return self.data_agent.load_series_for_regions(parsed, attempt=attempt)
        if target == "math_agent":
            return self.math_agent.compute_for_task(
                parsed,
                data_artifacts,
                attempt=attempt,
            )
        if target == "analysis_agent":
            return self.analysis_agent.analyze(
                parsed,
                data_artifacts,
                math_artifacts,
                attempt=attempt,
            )
        return agent_invalid_input(
            agent="orchestrator",
            message=f"Unknown requested target agent: {target}",
            attempt=attempt,
            max_attempts=MAX_AGENT_RETRIES,
        )

    def _failure_result(
        self,
        parsed: ParsedMultiAgentTask,
        response: AgentResponse,
        orchestration_steps: list[MultiAgentStep],
    ) -> MultiAgentResult:
        """Return a structured failure result without crashing the evaluation loop."""

        missing = ", ".join(response.missing_artifacts) or "<none>"
        answer = (
            f"[ERROR] Cannot complete {parsed.task_type} task: "
            f"{response.message or response.status.value}; "
            f"missing required artifacts: {missing}."
        )
        return MultiAgentResult(
            answer=answer,
            parsed_task=parsed,
            tool_results={
                "data": {},
                "math": {},
                "analysis": {},
                "_failure": response.to_dict(),
            },
            trace=self.client.get_trace(),
            orchestration_steps=orchestration_steps,
        )


def _regions_for_parsed(parsed: ParsedMultiAgentTask) -> list[str]:
    """Return task regions in deterministic order."""

    if parsed.region_ids:
        return list(parsed.region_ids)

    if parsed.region_id is not None:
        return [parsed.region_id]

    return []


def _first_region(parsed: ParsedMultiAgentTask) -> str:
    """Return the single primary region for one-region tasks."""

    regions = _regions_for_parsed(parsed)
    if not regions:
        raise ValueError(f"Task {parsed.task_type!r} requires a region")
    return regions[0]


def _series_id_for_region(data_artifacts: dict[str, Any], region_id: str) -> str:
    """Return loaded series_id for a region."""

    try:
        return str(data_artifacts["series_by_region"][region_id]["series_id"])
    except KeyError as exc:
        raise KeyError(f"Missing data artifact for region {region_id!r}") from exc


def _math_missing_series_response(
    agent_name: str,
    region_id: str,
    *,
    attempt: int,
) -> AgentResponse:
    """Return a standard MathAgent missing-series response."""

    missing = [f"data.series_by_region.{region_id}.series_id"]
    return agent_missing_artifacts(
        agent=agent_name,
        missing_artifacts=missing,
        requested_next_action=RequestedNextAction(
            target_agent="data_agent",
            task="load_missing_series",
            reason="MathAgent cannot compute without series_id",
            required_artifacts=missing,
            params={"regions": [region_id]},
        ),
        message=f"Cannot compute math artifacts without series_id for {region_id}.",
        attempt=attempt,
        max_attempts=MAX_AGENT_RETRIES,
    )


def _math_tool_error_response(
    agent_name: str,
    *,
    tool_name: str,
    region_id: str,
    missing_artifact: str,
    attempt: int,
) -> AgentResponse:
    """Return a standard MathAgent retryable tool-error response."""

    return agent_tool_error(
        agent=agent_name,
        message=f"{tool_name} failed for {region_id}.",
        missing_artifacts=[missing_artifact],
        requested_next_action=RequestedNextAction(
            target_agent="math_agent",
            task="retry_compute_step",
            reason="Tool call failed",
            required_artifacts=[missing_artifact],
            params={"tool_name": tool_name, "region": region_id},
        ),
        requires_retry=True,
        attempt=attempt,
        max_attempts=MAX_AGENT_RETRIES,
    )


def _artifact_path_exists(context: dict[str, Any], path: str) -> bool:
    """Return whether a dotted artifact path exists in nested dictionaries."""

    current: Any = context
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
        if current is None:
            return False
    return True


def _target_agent_for_artifact(path: str) -> str:
    """Map an artifact path to the role agent that can provide it."""

    if path.startswith("data."):
        return "data_agent"
    if path.startswith("math."):
        return "math_agent"
    if path.startswith("analysis."):
        return "analysis_agent"
    return "orchestrator"


def _step_from_response(
    *,
    response: AgentResponse,
    action: str,
    details: dict[str, Any] | None = None,
    decision: StepRecoveryDecision | None = None,
) -> MultiAgentStep:
    """Convert an AgentResponse into a serializable orchestration step."""

    response_dict = response.to_dict()
    step_details = dict(details or {})
    step_details["agent_response"] = response_dict
    if decision is not None:
        step_details["orchestrator_decision"] = decision.to_dict()

    return MultiAgentStep(
        node=response.agent,
        action=action,
        details=step_details,
        status=response.status.value,
        missing_artifacts=list(response.missing_artifacts),
        requested_next_action=(
            response.requested_next_action.to_dict()
            if response.requested_next_action is not None
            else None
        ),
        can_continue=response.can_continue,
        requires_retry=response.requires_retry,
        attempt=response.attempt,
        max_attempts=response.max_attempts,
        decision=decision.decision if decision is not None else None,
    )


def _response_can_advance(response: AgentResponse) -> bool:
    """Return True when workflow can continue after a response."""

    return response.status in {
        AgentStatus.OK,
        AgentStatus.PARTIAL,
        AgentStatus.FINAL,
    } and response.can_continue


def _merge_stage_artifacts(
    response: AgentResponse,
    *,
    data_artifacts: dict[str, Any],
    math_artifacts: dict[str, Any],
    analysis_artifacts: dict[str, Any],
) -> None:
    """Merge recovered artifacts into workflow state."""

    if response.agent == "data_agent":
        data_artifacts.update(response.artifacts)
    elif response.agent == "math_agent":
        math_artifacts.update(response.artifacts)
    elif response.agent == "analysis_agent":
        analysis_artifacts.update(response.artifacts)


def _stage_action(stage: str, task_type: str) -> str:
    """Return high-level action name for a stage."""

    if stage == "data":
        return "load_series"
    if stage == "analysis":
        return "analyze_results"
    if stage == "report":
        return "format_final_answer"
    if stage == "math":
        return {
            "high_tec": "compute_high_tec",
            "stable_intervals": "compute_stable_intervals",
            "compare_regions": "compute_region_comparison",
            "report": "compute_report_inputs",
        }.get(task_type, "compute_for_task")
    return stage


def _stage_details(
    stage: str,
    response: AgentResponse,
    parsed: ParsedMultiAgentTask,
) -> dict[str, Any]:
    """Build compact details for a stage step."""

    details: dict[str, Any] = {
        "task_type": parsed.task_type,
        "message": response.message,
    }

    if stage == "data":
        series_by_region = response.artifacts.get("series_by_region") or {}
        details.update(
            {
                "regions": response.artifacts.get("regions") or _regions_for_parsed(parsed),
                "series_ids": {
                    region_id: item.get("series_id")
                    for region_id, item in series_by_region.items()
                    if isinstance(item, dict)
                },
                "n_series": len(series_by_region),
            }
        )

    if stage == "math":
        details["artifact_keys"] = sorted(response.artifacts)

    if stage == "analysis":
        findings = response.artifacts.get("findings") or []
        details["n_findings"] = len(findings)
        details["finding_types"] = sorted(
            {
                str(finding.get("type"))
                for finding in findings
                if isinstance(finding, dict) and finding.get("type") is not None
            }
        )

    if stage == "report":
        answer = str(response.artifacts.get("answer", ""))
        details["answer_length"] = len(answer)

    return details


def _report_include(include: list[str] | None) -> list[str]:
    """Return normalized report include sections."""

    selected = include or DEFAULT_REPORT_INCLUDE
    allowed = set(DEFAULT_REPORT_INCLUDE)
    return [section for section in dict.fromkeys(selected) if section in allowed]


def _fmt_float(value: Any) -> str:
    """Format optional numeric value."""

    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _leader_finding(items: list[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    """Return the region with the maximum available metric value."""

    best_region = None
    best_value = None

    for item in items:
        metrics = item.get("metrics") or {}
        value = metrics.get(metric)
        if value is None:
            continue

        value_float = float(value)
        if best_value is None or value_float > best_value:
            best_value = value_float
            best_region = item.get("region_id")

    if best_region is None:
        return {
            "type": "comparison_leader",
            "metric": metric,
            "region_id": None,
            "value": None,
            "text": f"No {metric} TEC value is available.",
        }

    return {
        "type": "comparison_leader",
        "metric": metric,
        "region_id": best_region,
        "value": best_value,
        "text": f"{best_region} has the highest {metric} TEC.",
    }


def _max_interval_value(intervals: list[dict[str, Any]], field_name: str) -> float | None:
    """Return max numeric interval field value or None."""

    values: list[float] = []
    for item in intervals:
        value = item.get(field_name)
        if value is not None:
            values.append(float(value))

    if not values:
        return None

    return max(values)


def _high_tec_artifacts_by_region(math_artifacts: dict[str, Any]) -> dict[str, Any]:
    """Return high-TEC artifacts from task or report math shapes."""

    if "high_tec" in math_artifacts:
        return math_artifacts.get("high_tec") or {}

    report_inputs = math_artifacts.get("report_inputs") or {}
    high = report_inputs.get("high_tec") or {}
    return high.get("by_region") or {}


def _stable_artifacts_by_region(math_artifacts: dict[str, Any]) -> dict[str, Any]:
    """Return stable interval artifacts from task or report math shapes."""

    if "stable_intervals" in math_artifacts:
        return math_artifacts.get("stable_intervals") or {}

    report_inputs = math_artifacts.get("report_inputs") or {}
    stable = report_inputs.get("stable_intervals") or {}
    return stable.get("by_region") or {}
