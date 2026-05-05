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
    ) -> tuple[dict[str, Any], MultiAgentStep]:
        """Load all required time series before any math stage runs."""

        regions = _regions_for_parsed(parsed)
        if not regions:
            raise ValueError(f"Task {parsed.task_type!r} requires at least one region")

        series_by_region: dict[str, dict[str, Any]] = {}

        for step, region_id in enumerate(regions, start=1):
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

        step = MultiAgentStep(
            node=self.agent_name,
            action="load_series",
            details={
                "regions": regions,
                "series_ids": {
                    region_id: item["series_id"]
                    for region_id, item in series_by_region.items()
                },
                "n_series": len(series_by_region),
            },
        )

        return artifacts, step


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
    ) -> tuple[dict[str, Any], MultiAgentStep]:
        """Dispatch task-specific math while keeping data access separate."""

        if parsed.task_type == "high_tec":
            return self._compute_high_tec(parsed, data_artifacts)

        if parsed.task_type == "stable_intervals":
            return self._compute_stable_intervals(parsed, data_artifacts)

        if parsed.task_type == "compare_regions":
            return self._compute_compare_regions(parsed, data_artifacts)

        if parsed.task_type == "report":
            return self._compute_report_inputs(parsed, data_artifacts)

        raise ValueError(f"Unsupported task type for MathAgent: {parsed.task_type!r}")

    def _compute_high_tec(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
    ) -> tuple[dict[str, Any], MultiAgentStep]:
        """Compute high-TEC thresholds and intervals for one region."""

        region_id = _first_region(parsed)
        series_id = _series_id_for_region(data_artifacts, region_id)

        threshold_result = self.client.call_tool_result(
            "tec_compute_high_threshold",
            {
                "series_id": series_id,
                "method": "quantile",
                "q": parsed.q,
            },
            agent_name=self.agent_name,
            step=1,
        )

        intervals_result = self.client.call_tool_result(
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

        artifacts = {
            "high_tec": {
                region_id: {
                    "threshold": threshold_result,
                    "intervals": intervals_result,
                }
            }
        }

        step = MultiAgentStep(
            node=self.agent_name,
            action="compute_high_tec",
            details={
                "regions": [region_id],
                "n_thresholds": 1,
                "n_interval_results": 1,
            },
        )

        return artifacts, step

    def _compute_stable_intervals(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
    ) -> tuple[dict[str, Any], MultiAgentStep]:
        """Compute stable interval thresholds and detections for one region."""

        region_id = _first_region(parsed)
        series_id = _series_id_for_region(data_artifacts, region_id)

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
            step=1,
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
            step=2,
        )

        artifacts = {
            "stable_intervals": {
                region_id: {
                    "thresholds": thresholds_result,
                    "intervals": intervals_result,
                }
            }
        }

        step = MultiAgentStep(
            node=self.agent_name,
            action="compute_stable_intervals",
            details={
                "regions": [region_id],
                "n_thresholds": 1,
                "n_interval_results": 1,
            },
        )

        return artifacts, step

    def _compute_compare_regions(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
    ) -> tuple[dict[str, Any], MultiAgentStep]:
        """Compute all region stats first, then compare the stats handles."""

        regions = _regions_for_parsed(parsed)
        if len(regions) < 2:
            raise ValueError("Comparison task requires at least two region_ids")

        stats_by_region: dict[str, dict[str, Any]] = {}
        stats_ids: list[str] = []

        for step, region_id in enumerate(regions, start=1):
            series_id = _series_id_for_region(data_artifacts, region_id)
            stats_result = self.client.call_tool_result(
                "tec_compute_series_stats",
                {
                    "series_id": series_id,
                    "metrics": parsed.metrics,
                },
                agent_name=self.agent_name,
                step=step,
            )

            stats_by_region[region_id] = {
                "stats_id": stats_result["stats_id"],
                "series_id": series_id,
                "stats": stats_result,
                "metrics": stats_result.get("metrics", {}),
            }
            stats_ids.append(stats_result["stats_id"])

        comparison_result = self.client.call_tool_result(
            "tec_compare_stats",
            {
                "stats_ids": stats_ids,
                "metrics": parsed.metrics,
            },
            agent_name=self.agent_name,
            step=len(regions) + 1,
        )

        artifacts = {
            "stats_by_region": stats_by_region,
            "comparison": comparison_result,
        }

        step = MultiAgentStep(
            node=self.agent_name,
            action="compute_region_comparison",
            details={
                "regions": regions,
                "stats_ids": {
                    region_id: item["stats_id"]
                    for region_id, item in stats_by_region.items()
                },
                "comparison_id": comparison_result["comparison_id"],
                "n_regions": len(regions),
            },
        )

        return artifacts, step

    def _compute_report_inputs(
        self,
        parsed: ParsedMultiAgentTask,
        data_artifacts: dict[str, Any],
    ) -> tuple[dict[str, Any], MultiAgentStep]:
        """Compute primitive artifacts needed for a structured TEC report."""

        regions = _regions_for_parsed(parsed)
        include = _report_include(parsed.include)
        report_inputs: dict[str, Any] = {}
        step = 1

        n_stats = 0
        has_comparison = False
        n_high_results = 0
        n_stable_results = 0

        if "basic_stats" in include:
            by_region: dict[str, dict[str, Any]] = {}
            stats_ids: list[str] = []

            for region_id in regions:
                series_id = _series_id_for_region(data_artifacts, region_id)
                stats_result = self.client.call_tool_result(
                    "tec_compute_series_stats",
                    {
                        "series_id": series_id,
                        "metrics": parsed.metrics,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1

                by_region[region_id] = {
                    "stats_id": stats_result["stats_id"],
                    "series_id": series_id,
                    "stats": stats_result,
                    "metrics": stats_result.get("metrics", {}),
                }
                stats_ids.append(stats_result["stats_id"])

            comparison = None
            if len(stats_ids) >= 2:
                comparison = self.client.call_tool_result(
                    "tec_compare_stats",
                    {
                        "stats_ids": stats_ids,
                        "metrics": parsed.metrics,
                    },
                    agent_name=self.agent_name,
                    step=step,
                )
                step += 1
                has_comparison = True

            report_inputs["basic_stats"] = {
                "by_region": by_region,
                "comparison": comparison,
            }
            n_stats = len(by_region)

        if "high_tec" in include:
            by_region = {}

            for region_id in regions:
                series_id = _series_id_for_region(data_artifacts, region_id)
                threshold_result = self.client.call_tool_result(
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

                intervals_result = self.client.call_tool_result(
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

                by_region[region_id] = {
                    "threshold": threshold_result,
                    "intervals": intervals_result,
                }

            report_inputs["high_tec"] = {"by_region": by_region}
            n_high_results = len(by_region)

        if "stable_intervals" in include:
            by_region = {}

            for region_id in regions:
                series_id = _series_id_for_region(data_artifacts, region_id)
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
                    step=step,
                )
                step += 1

                intervals_result = self.client.call_tool_result(
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

                by_region[region_id] = {
                    "thresholds": thresholds_result,
                    "intervals": intervals_result,
                }

            report_inputs["stable_intervals"] = {"by_region": by_region}
            n_stable_results = len(by_region)

        artifacts = {"report_inputs": report_inputs}

        math_step = MultiAgentStep(
            node=self.agent_name,
            action="compute_report_inputs",
            details={
                "regions": regions,
                "include": include,
                "n_stats": n_stats,
                "has_comparison": has_comparison,
                "n_high_tec_results": n_high_results,
                "n_stable_results": n_stable_results,
            },
        )

        return artifacts, math_step


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
    ) -> tuple[dict[str, Any], MultiAgentStep]:
        """Create structured findings without calling computational tools."""

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

        step = MultiAgentStep(
            node=self.agent_name,
            action="analyze_results",
            details={
                "task_type": parsed.task_type,
                "n_findings": len(findings),
                "finding_types": sorted(
                    {
                        str(finding.get("type"))
                        for finding in findings
                        if finding.get("type") is not None
                    }
                ),
            },
        )

        return artifacts, step

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
    ) -> tuple[str, MultiAgentStep]:
        """Format a deterministic final answer from existing artifacts."""

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
            raise ValueError(f"Unsupported task type for ReportAgent: {parsed.task_type!r}")

        step = MultiAgentStep(
            node=self.agent_name,
            action="format_final_answer",
            details={
                "task_type": parsed.task_type,
                "answer_length": len(answer),
            },
        )

        return answer, step

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

        data_artifacts, data_step = self.data_agent.load_series_for_regions(parsed)
        orchestration_steps.append(data_step)

        math_artifacts, math_step = self.math_agent.compute_for_task(
            parsed,
            data_artifacts,
        )
        orchestration_steps.append(math_step)

        analysis_artifacts, analysis_step = self.analysis_agent.analyze(
            parsed,
            data_artifacts,
            math_artifacts,
        )
        orchestration_steps.append(analysis_step)

        answer, report_step = self.report_agent.format_answer(
            parsed,
            data_artifacts,
            math_artifacts,
            analysis_artifacts,
        )
        orchestration_steps.append(report_step)

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
