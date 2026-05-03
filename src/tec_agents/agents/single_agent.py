"""
Simple single-agent baseline.

This agent does not use an LLM yet. It is a deterministic rule-based baseline
that uses the same MCP-like client and the same TEC tools as future LLM agents.

Supported scenario:
- high TEC interval detection for one region and one month.

Example query:
"Find high TEC intervals for midlat_europe in March 2024 with q=0.9"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    region_id: str = "midlat_europe"
    start: str = ""
    end: str = ""
    q: float = 0.9
    raw_query: str = ""


@dataclass
class SingleAgentResult:
    """Result of a single-agent run."""

    answer: str
    parsed_task: ParsedSingleAgentTask
    tool_results: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


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

    def run(self, query: str) -> SingleAgentResult:
        """Run the agent on one user query."""

        parsed = self.parse_query(query)

        if parsed.task_type != "high_tec":
            raise ValueError(
                f"Unsupported task_type={parsed.task_type!r}. "
                "This baseline currently supports only high_tec."
            )

        tool_results = self._run_high_tec(parsed)
        answer = self._format_high_tec_answer(parsed, tool_results)

        return SingleAgentResult(
            answer=answer,
            parsed_task=parsed,
            tool_results=tool_results,
            trace=self.client.get_trace(),
        )

    def parse_query(self, query: str) -> ParsedSingleAgentTask:
        """
        Parse a simple high-TEC query.

        This is intentionally simple. Later, Qwen will replace this parser.
        """

        lower = query.lower()

        task_type = "high_tec" if "high" in lower and "tec" in lower else "unknown"
        region_id = self._extract_region_id(lower)
        start, end = self._extract_month_range(lower)
        q = self._extract_quantile(lower)

        return ParsedSingleAgentTask(
            task_type=task_type,
            dataset_ref=self.dataset_ref,
            region_id=region_id,
            start=start,
            end=end,
            q=q,
            raw_query=query,
        )

    def _run_high_tec(self, parsed: ParsedSingleAgentTask) -> dict[str, Any]:
        """Execute high-TEC detection through MCP-like tools."""

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

    def _format_high_tec_answer(
        self,
        parsed: ParsedSingleAgentTask,
        tool_results: dict[str, Any],
    ) -> str:
        """Create a compact human-readable answer."""

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
            peak_text = (
                f"{peak_value:.3f} TECU at {item['peak_time']}"
                if peak_value is not None
                else "n/a"
            )

            lines.append(
                f"{i}. {item['start']} → {item['end']}; "
                f"duration={item['duration_minutes']:.1f} min; "
                f"peak={peak_text}; "
                f"mean={item['mean_value']:.3f} TECU."
            )

        return "\n".join(lines)

    def _extract_region_id(self, lower_query: str) -> str:
        """Extract region_id from query or use default."""

        for region_id in list_region_ids():
            if region_id.lower() in lower_query:
                return region_id

        aliases = {
            "europe": "midlat_europe",
            "north high latitudes": "highlat_north",
            "northern high latitudes": "highlat_north",
            "high latitude north": "highlat_north",
            "atlantic": "equatorial_atlantic",
            "africa": "equatorial_africa",
            "pacific": "equatorial_pacific",
        }

        for alias, region_id in aliases.items():
            if alias in lower_query:
                return region_id

        return "midlat_europe"

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

        start_date = f"{year:04d}-{month:02d}-01"
        end_dt = relativedelta(months=1)
        # Use pandas-like date arithmetic without adding pandas dependency here.
        from datetime import date

        start_obj = date(year, month, 1)
        end_obj = start_obj + end_dt
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