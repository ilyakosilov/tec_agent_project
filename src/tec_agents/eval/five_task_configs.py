"""
Shared five-task evaluation configs for early single/multi-agent comparisons.

These tasks mirror the Qwen single-agent Colab batch notebook. They are a small
experimental stand, not a production benchmark dataset.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tec_agents.eval.task_set import (
    EvalTask,
    primitive_compare_tool_sequence,
    primitive_report_tool_sequence,
    validate_task,
)


DEFAULT_DATASET_REF = "default"
DEFAULT_START = "2024-03-01"
DEFAULT_END = "2024-04-01"
DEFAULT_COMPARE_METRICS = ["mean", "median", "min", "max", "std", "p90", "p95"]
DEFAULT_REPORT_INCLUDE = ["basic_stats", "high_tec", "stable_intervals"]


FIVE_TASK_CONFIGS: list[dict[str, Any]] = [
    {
        "preset_id": "hightec_midlat_europe",
        "task_type": "high_tec",
        "query": (
            "Find high TEC intervals for midlat_europe from 2024-03-01 "
            "to 2024-04-01 using q=0.90 threshold."
        ),
        "region": "midlat_europe",
        "regions": ["midlat_europe"],
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "q": 0.9,
        "dataset_ref": DEFAULT_DATASET_REF,
    },
    {
        "preset_id": "stable_midlat_europe",
        "task_type": "stable_intervals",
        "query": (
            "Find stable TEC intervals for midlat_europe from 2024-03-01 "
            "to 2024-04-01 using the configured stability parameters."
        ),
        "region": "midlat_europe",
        "regions": ["midlat_europe"],
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "window_minutes": 180,
        "q_delta": 0.6,
        "q_std": 0.6,
        "dataset_ref": DEFAULT_DATASET_REF,
    },
    {
        "preset_id": "compare_midlat_europe_highlat_north",
        "task_type": "compare_regions",
        "query": (
            "Compare TEC statistics for midlat_europe and highlat_north "
            "from 2024-03-01 to 2024-04-01."
        ),
        "regions": ["midlat_europe", "highlat_north"],
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "metrics": list(DEFAULT_COMPARE_METRICS),
        "dataset_ref": DEFAULT_DATASET_REF,
    },
    {
        "preset_id": "compare_three_regions",
        "task_type": "compare_regions",
        "query": (
            "Compare TEC statistics for equatorial_atlantic, midlat_europe, "
            "and highlat_north from 2024-03-01 to 2024-04-01."
        ),
        "regions": ["equatorial_atlantic", "midlat_europe", "highlat_north"],
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "metrics": list(DEFAULT_COMPARE_METRICS),
        "dataset_ref": DEFAULT_DATASET_REF,
    },
    {
        "preset_id": "report_midlat_europe",
        "task_type": "report",
        "query": (
            "Build a concise TEC analysis report for midlat_europe from "
            "2024-03-01 to 2024-04-01. Include basic statistics, high TEC "
            "intervals, stable intervals, and a short interpretation based only "
            "on computed artifacts."
        ),
        "region": "midlat_europe",
        "regions": ["midlat_europe"],
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "q": 0.9,
        "window_minutes": 180,
        "q_delta": 0.6,
        "q_std": 0.6,
        "include": list(DEFAULT_REPORT_INCLUDE),
        "metrics": list(DEFAULT_COMPARE_METRICS),
        "dataset_ref": DEFAULT_DATASET_REF,
    },
]


def get_five_task_configs(dataset_ref: str = DEFAULT_DATASET_REF) -> list[dict[str, Any]]:
    """Return five task configs with a caller-selected dataset_ref."""

    configs = deepcopy(FIVE_TASK_CONFIGS)
    for config in configs:
        config["dataset_ref"] = dataset_ref
    return configs


def build_five_task_expected_sequence(config: dict[str, Any]) -> list[str]:
    """Build the primitive expected tool sequence for one five-task config."""

    task_type = str(config["task_type"])

    if task_type == "high_tec":
        return [
            "tec_get_timeseries",
            "tec_compute_high_threshold",
            "tec_detect_high_intervals",
        ]

    if task_type == "stable_intervals":
        return [
            "tec_get_timeseries",
            "tec_compute_stability_thresholds",
            "tec_detect_stable_intervals",
        ]

    if task_type == "compare_regions":
        return list(primitive_compare_tool_sequence(len(config["regions"])))

    if task_type == "report":
        include = list(config.get("include") or DEFAULT_REPORT_INCLUDE)
        return list(primitive_report_tool_sequence(len(config["regions"]), include))

    raise ValueError(f"Unsupported task_type={task_type!r}")


def build_five_task_eval_task(config: dict[str, Any]) -> EvalTask:
    """Build an EvalTask from one five-task config."""

    task_type = str(config["task_type"])
    sequence = tuple(build_five_task_expected_sequence(config))
    regions = tuple(str(region_id) for region_id in config.get("regions", ()))
    params = _params_for_config(config)

    if task_type == "compare_regions":
        region_id = None
    else:
        region_id = str(config.get("region") or (regions[0] if regions else ""))

    task = EvalTask(
        task_id=str(config["preset_id"]),
        query=str(config["query"]),
        task_type=task_type,  # type: ignore[arg-type]
        dataset_ref=str(config.get("dataset_ref") or DEFAULT_DATASET_REF),
        region_id=region_id or None,
        region_ids=regions,
        start=str(config["start"]),
        end=str(config["end"]),
        q=float(config.get("q", 0.9)),
        params=params,
        expected_tool_sequence=sequence,
        expected_worker="role_based_workflow",
        description=f"Five-task baseline: {config['preset_id']}",
    )
    validate_task(task)
    return task


def build_all_five_task_eval_tasks(
    dataset_ref: str = DEFAULT_DATASET_REF,
) -> list[EvalTask]:
    """Return EvalTask objects for all five standard configs."""

    return [
        build_five_task_eval_task(config)
        for config in get_five_task_configs(dataset_ref=dataset_ref)
    ]


def _params_for_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return task params accepted by GoldRunner and metrics."""

    task_type = str(config["task_type"])
    params: dict[str, Any] = {}

    if task_type in {"compare_regions", "report"}:
        params["metrics"] = list(config.get("metrics") or DEFAULT_COMPARE_METRICS)

    if task_type in {"stable_intervals", "report"}:
        params["window_minutes"] = int(config.get("window_minutes", 180))
        params["q_delta"] = float(config.get("q_delta", 0.6))
        params["q_std"] = float(config.get("q_std", 0.6))

    if task_type == "report":
        params["include"] = list(config.get("include") or DEFAULT_REPORT_INCLUDE)

    return params
