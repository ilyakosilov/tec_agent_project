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
from datetime import date
from typing import Any

from dateutil.relativedelta import relativedelta

from tec_agents.data.regions import list_region_ids
from tec_agents.mcp.client import LocalMCPClient


MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


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


@dataclass
class SingleAgentResult:
    """Result of a single-agent run."""

    answer: str
    parsed_task: ParsedSingleAgentTask
    tool_results: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    orchestration_steps: list[SingleAgentStep] = field(default_factory=list)


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

        parsed = self.parse_query(query)

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
                },
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

    def _run_high_tec(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute high-TEC detection through MCP-like tools."""

        if parsed.region_id is None:
            raise ValueError("High-TEC task requires region_id")

        ts_result = self.client.call_tool_result(
            "tec_get_timeseries",
            {
                "dataset_ref": parsed.dataset_ref,
                "region_id": parsed.region_id,
                "start": parsed.start,
                "end": parsed.end,
            },
            agent_name=self.agent_name,
            step=1,
        )

        series_id = ts_result["series_id"]

        threshold_result = self.client.call_tool_result(
            "tec_compute_high_threshold",
            {
                "series_id": series_id,
                "method": "quantile",
                "q": parsed.q,
            },
            agent_name=self.agent_name,
            step=2,
        )

        threshold_id = threshold_result["threshold_id"]

        intervals_result = self.client.call_tool_result(
            "tec_detect_high_intervals",
            {
                "series_id": series_id,
                "threshold_id": threshold_id,
                "min_duration_minutes": 0,
                "merge_gap_minutes": 60,
            },
            agent_name=self.agent_name,
            step=3,
        )

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
            ts_result = self.client.call_tool_result(
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
            timeseries_results.append(ts_result)
            step += 1

            stats_result = self.client.call_tool_result(
                "tec_compute_series_stats",
                {
                    "series_id": ts_result["series_id"],
                    "metrics": parsed.metrics,
                },
                agent_name=self.agent_name,
                step=step,
            )
            stats_results.append(stats_result)
            stats_ids.append(stats_result["stats_id"])
            step += 1

        comparison_result = self.client.call_tool_result(
            "tec_compare_stats",
            {
                "stats_ids": stats_ids,
                "metrics": parsed.metrics,
            },
            agent_name=self.agent_name,
            step=step,
        )

        return {
            "timeseries": timeseries_results,
            "stats": stats_results,
            "comparison": comparison_result,
        }

    def _run_stable_intervals(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute stable-interval detection through MCP-like tools."""

        if parsed.region_id is None:
            raise ValueError("Stable-interval task requires region_id")

        ts_result = self.client.call_tool_result(
            "tec_get_timeseries",
            {
                "dataset_ref": parsed.dataset_ref,
                "region_id": parsed.region_id,
                "start": parsed.start,
                "end": parsed.end,
            },
            agent_name=self.agent_name,
            step=1,
        )

        series_id = ts_result["series_id"]

        thresholds_result = self.client.call_tool_result(
            "tec_compute_stability_thresholds",
            {
                "series_id": series_id,
                "window_minutes": parsed.window_minutes,
                "method": "quantile",
                "q_delta": parsed.q_delta,
                "q_std": parsed.q_std,
            },
            agent_name=self.agent_name,
            step=2,
        )

        intervals_result = self.client.call_tool_result(
            "tec_detect_stable_intervals",
            {
                "series_id": series_id,
                "threshold_id": thresholds_result["threshold_id"],
                "min_duration_minutes": parsed.window_minutes,
                "merge_gap_minutes": 60,
            },
            agent_name=self.agent_name,
            step=3,
        )

        return {
            "timeseries": ts_result,
            "thresholds": thresholds_result,
            "intervals": intervals_result,
        }

    def _run_report(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute structured report generation through one MCP-like tool call."""

        if not parsed.region_ids:
            raise ValueError("Report task requires at least one region")

        return self.client.call_tool_result(
            "tec_build_report",
            {
                "dataset_ref": parsed.dataset_ref,
                "regions": parsed.region_ids,
                "start": parsed.start,
                "end": parsed.end,
                "include": parsed.include,
            },
            agent_name=self.agent_name,
            step=1,
        )

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

        sections = sorted(tool_results.get("sections", {}))
        return (
            f"Built TEC report {tool_results['report_id']} for "
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
            "сравни",
            "сравнение",
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

        year_match = re.search(r"\b(20\d{2}|19\d{2})\b", lower_query)
        if not year_match:
            raise ValueError("Could not find year in query")

        year = int(year_match.group(1))

        month = None
        for month_name, month_number in MONTHS.items():
            if re.search(rf"\b{re.escape(month_name)}\b", lower_query):
                month = month_number
                break

        if month is None:
            raise ValueError("Could not find month in query")

        start_obj = date(year, month, 1)
        end_obj = start_obj + relativedelta(months=1)

        start_date = f"{start_obj.year:04d}-{start_obj.month:02d}-{start_obj.day:02d}"
        end_date = f"{end_obj.year:04d}-{end_obj.month:02d}-{end_obj.day:02d}"

        return start_date, end_date

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
