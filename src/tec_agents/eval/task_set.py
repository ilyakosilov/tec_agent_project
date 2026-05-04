"""
Evaluation task definitions.

A task contains both a natural-language query and structured expected
parameters. The query is used by agents. The structured fields are used by the
gold runner and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tec_agents.data.regions import list_region_ids


TaskType = Literal[
    "high_tec",
    "compare_regions",
    "stable_intervals",
    "report",
]


@dataclass(frozen=True)
class EvalTask:
    """One evaluation task."""

    task_id: str
    query: str
    task_type: TaskType
    dataset_ref: str
    start: str
    end: str
    region_id: str | None = None
    region_ids: tuple[str, ...] = ()
    q: float = 0.9
    params: dict[str, Any] | None = None
    expected_tool_sequence: tuple[str, ...] = ()
    expected_worker: str | None = None
    description: str = ""


def validate_task(task: EvalTask) -> None:
    """Validate task fields and region IDs."""

    allowed_regions = set(list_region_ids())

    if task.region_id is not None and task.region_id not in allowed_regions:
        raise ValueError(
            f"Unknown region_id={task.region_id!r} in task {task.task_id!r}"
        )

    for region_id in task.region_ids:
        if region_id not in allowed_regions:
            raise ValueError(
                f"Unknown region_id={region_id!r} in task {task.task_id!r}"
            )

    if task.end <= task.start:
        raise ValueError(
            f"Task {task.task_id!r} has invalid date interval: "
            f"start={task.start}, end={task.end}"
        )

    if not (0.0 <= task.q <= 1.0):
        raise ValueError(f"Task {task.task_id!r} has invalid q={task.q}")


def validate_tasks(tasks: list[EvalTask]) -> None:
    """Validate a list of tasks."""

    seen: set[str] = set()

    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"Duplicate task_id={task.task_id!r}")
        seen.add(task.task_id)
        validate_task(task)


def build_smoke_tasks(dataset_ref: str = "smoke") -> list[EvalTask]:
    """
    Build a small task set for local smoke tests.

    These tasks are designed for synthetic data in data/examples, not for the
    final scientific experiment.
    """

    tasks = [
        EvalTask(
            task_id="smoke_hightec_midlat_europe_march_2024",
            query="Find high TEC intervals for midlat_europe in March 2024 with q=0.9",
            task_type="high_tec",
            dataset_ref=dataset_ref,
            region_id="midlat_europe",
            region_ids=("midlat_europe",),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            description="Smoke high-TEC task for the synthetic Europe series.",
        ),
        EvalTask(
            task_id="smoke_hightec_highlat_north_march_2024",
            query="Find high TEC intervals for highlat_north in March 2024 with q=0.9",
            task_type="high_tec",
            dataset_ref=dataset_ref,
            region_id="highlat_north",
            region_ids=("highlat_north",),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            description="Smoke high-TEC task for the synthetic northern high-latitude series.",
        ),
        EvalTask(
            task_id="smoke_compare_europe_highlat_march_2024",
            query=(
                "Compare TEC statistics for midlat_europe and highlat_north "
                "in March 2024"
            ),
            task_type="compare_regions",
            dataset_ref=dataset_ref,
            region_id=None,
            region_ids=("midlat_europe", "highlat_north"),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            description="Smoke comparison task for two synthetic regions.",
        ),
    ]

    validate_tasks(tasks)
    return tasks


def build_default_research_tasks(dataset_ref: str = "default") -> list[EvalTask]:
    """
    Build the initial research task set for processed TEC data.

    This list can be expanded later. The first version focuses on tasks that are
    already supported by deterministic tools.
    """

    tasks = [
        EvalTask(
            task_id="hightec_midlat_europe_march_2024",
            query="Find high TEC intervals for midlat_europe in March 2024 with q=0.9",
            task_type="high_tec",
            dataset_ref=dataset_ref,
            region_id="midlat_europe",
            region_ids=("midlat_europe",),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            expected_tool_sequence=(
                "tec_get_timeseries",
                "tec_compute_high_threshold",
                "tec_detect_high_intervals",
            ),
            expected_worker="high_tec_worker",
            description="High TEC intervals over European mid-latitudes.",
        ),
        EvalTask(
            task_id="hightec_highlat_north_march_2024",
            query="Find high TEC intervals for highlat_north in March 2024 with q=0.9",
            task_type="high_tec",
            dataset_ref=dataset_ref,
            region_id="highlat_north",
            region_ids=("highlat_north",),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            expected_tool_sequence=(
                "tec_get_timeseries",
                "tec_compute_high_threshold",
                "tec_detect_high_intervals",
            ),
            expected_worker="high_tec_worker",
            description="High TEC intervals over northern high latitudes.",
        ),
        EvalTask(
            task_id="hightec_equatorial_atlantic_march_2024",
            query=(
                "Find high TEC intervals for equatorial_atlantic "
                "in March 2024 with q=0.9"
            ),
            task_type="high_tec",
            dataset_ref=dataset_ref,
            region_id="equatorial_atlantic",
            region_ids=("equatorial_atlantic",),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            expected_tool_sequence=(
                "tec_get_timeseries",
                "tec_compute_high_threshold",
                "tec_detect_high_intervals",
            ),
            expected_worker="high_tec_worker",
            description="High TEC intervals over the equatorial Atlantic sector.",
        ),
        EvalTask(
            task_id="compare_midlat_europe_highlat_north_march_2024",
            query=(
                "Compare TEC statistics for midlat_europe and highlat_north "
                "in March 2024"
            ),
            task_type="compare_regions",
            dataset_ref=dataset_ref,
            region_id=None,
            region_ids=("midlat_europe", "highlat_north"),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            expected_tool_sequence=("tec_compare_regions",),
            expected_worker="compare_worker",
            description="Regional TEC comparison between Europe and northern high latitudes.",
        ),
        EvalTask(
            task_id="compare_equatorial_regions_march_2024",
            query=(
                "Compare TEC statistics for equatorial_atlantic, "
                "equatorial_africa and equatorial_pacific in March 2024"
            ),
            task_type="compare_regions",
            dataset_ref=dataset_ref,
            region_id=None,
            region_ids=(
                "equatorial_atlantic",
                "equatorial_africa",
                "equatorial_pacific",
            ),
            start="2024-03-01",
            end="2024-04-01",
            q=0.9,
            expected_tool_sequence=("tec_compare_regions",),
            expected_worker="compare_worker",
            description="Comparison across three equatorial sectors.",
        ),
        EvalTask(
            task_id="stable_midlat_europe_march_2024",
            query="Find stable TEC intervals for midlat_europe in March 2024",
            task_type="stable_intervals",
            dataset_ref=dataset_ref,
            region_id="midlat_europe",
            region_ids=("midlat_europe",),
            start="2024-03-01",
            end="2024-04-01",
            params={
                "window_minutes": 180,
                "q_delta": 0.60,
                "q_std": 0.60,
            },
            expected_tool_sequence=(
                "tec_get_timeseries",
                "tec_compute_stability_thresholds",
                "tec_detect_stable_intervals",
            ),
            expected_worker="stable_worker",
            description="Stable TEC intervals over European mid-latitudes.",
        ),
        EvalTask(
            task_id="stable_highlat_north_march_2024",
            query="Find low variability TEC periods for highlat_north in March 2024",
            task_type="stable_intervals",
            dataset_ref=dataset_ref,
            region_id="highlat_north",
            region_ids=("highlat_north",),
            start="2024-03-01",
            end="2024-04-01",
            params={
                "window_minutes": 180,
                "q_delta": 0.60,
                "q_std": 0.60,
            },
            expected_tool_sequence=(
                "tec_get_timeseries",
                "tec_compute_stability_thresholds",
                "tec_detect_stable_intervals",
            ),
            expected_worker="stable_worker",
            description="Low-variability TEC intervals over northern high latitudes.",
        ),
        EvalTask(
            task_id="report_midlat_europe_highlat_north_march_2024",
            query=(
                "Build a TEC report for midlat_europe and highlat_north "
                "in March 2024"
            ),
            task_type="report",
            dataset_ref=dataset_ref,
            region_id=None,
            region_ids=("midlat_europe", "highlat_north"),
            start="2024-03-01",
            end="2024-04-01",
            params={
                "include": [
                    "basic_stats",
                    "high_tec",
                    "stable_intervals",
                ]
            },
            expected_tool_sequence=("tec_build_report",),
            expected_worker="report_worker",
            description="Structured report for Europe and northern high latitudes.",
        ),
        EvalTask(
            task_id="report_equatorial_regions_march_2024",
            query=(
                "Create a summary report for equatorial_atlantic, "
                "equatorial_africa and equatorial_pacific in March 2024"
            ),
            task_type="report",
            dataset_ref=dataset_ref,
            region_id=None,
            region_ids=(
                "equatorial_atlantic",
                "equatorial_africa",
                "equatorial_pacific",
            ),
            start="2024-03-01",
            end="2024-04-01",
            params={
                "include": [
                    "basic_stats",
                    "high_tec",
                    "stable_intervals",
                ]
            },
            expected_tool_sequence=("tec_build_report",),
            expected_worker="report_worker",
            description="Structured report for three equatorial sectors.",
        ),
    ]

    validate_tasks(tasks)
    return tasks


def task_to_dict(task: EvalTask) -> dict[str, object]:
    """Convert task to a JSON-serializable dictionary."""

    return {
        "task_id": task.task_id,
        "query": task.query,
        "task_type": task.task_type,
        "dataset_ref": task.dataset_ref,
        "start": task.start,
        "end": task.end,
        "region_id": task.region_id,
        "region_ids": list(task.region_ids),
        "q": task.q,
        "params": task.params or {},
        "expected_tool_sequence": list(task.expected_tool_sequence),
        "expected_worker": task.expected_worker,
        "description": task.description,
    }


def tasks_to_dicts(tasks: list[EvalTask]) -> list[dict[str, object]]:
    """Convert tasks to JSON-serializable dictionaries."""

    return [task_to_dict(task) for task in tasks]
