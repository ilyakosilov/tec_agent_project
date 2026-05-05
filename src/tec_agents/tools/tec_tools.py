"""
Deterministic TEC analysis tools.

These functions do the actual numeric work. They do not call LLMs and do not
know anything about agents. Agents should access them through ToolExecutor.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tec_agents.data.datasets import get_region_series
from tec_agents.tools.schemas import (
    CompareRegionsInput,
    CompareRegionsOutput,
    CompareStatsInput,
    CompareStatsOutput,
    ComputeHighThresholdInput,
    ComputeHighThresholdOutput,
    ComputeSeriesStatsInput,
    ComputeSeriesStatsOutput,
    ComputeStabilityThresholdsInput,
    ComputeStabilityThresholdsOutput,
    DetectHighIntervalsInput,
    DetectHighIntervalsOutput,
    DetectStableIntervalsInput,
    DetectStableIntervalsOutput,
    FindStableIntervalsDirectInput,
    GetTimeseriesInput,
    GetTimeseriesOutput,
    IntervalRecord,
    RegionStatsRecord,
    SeriesMetadata,
    SeriesProfileInput,
    SeriesProfileOutput,
    StableIntervalRecord,
)


DEFAULT_SERIES_STATS_METRICS = [
    "mean",
    "median",
    "min",
    "max",
    "std",
    "p90",
    "p95",
]

DEFAULT_COMPARE_STATS_METRICS = [
    "mean",
    "median",
    "max",
    "std",
    "p90",
    "p95",
]


@dataclass
class ToolStore:
    """
    In-memory storage for intermediate tool artifacts.

    The store is intentionally explicit and resettable. Each experiment run should
    use a fresh ToolStore to avoid state leaking between runs.
    """

    series: dict[str, pd.Series] = field(default_factory=dict)
    series_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)


def _make_id(prefix: str, payload: Any) -> str:
    """Create a short deterministic ID from a payload."""

    raw = repr(payload).encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _finite_values(series: pd.Series) -> pd.Series:
    """Return finite numeric values from a series."""

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)]


def _safe_float(value: Any) -> float | None:
    """Convert numpy/pandas scalar to a JSON-friendly float or None."""

    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float


def _duration_minutes(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Return interval duration in minutes."""

    return float((end - start).total_seconds() / 60.0)


def _format_timestamp(value: Any) -> str:
    """Return a stable ISO-like timestamp string without timezone guessing."""

    return pd.Timestamp(value).isoformat(sep=" ")


def _estimate_step_minutes(index: pd.DatetimeIndex) -> float | None:
    """Estimate median sampling step in minutes."""

    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return None
    return _safe_float(deltas.median())


def _series_metadata(
    series_id: str,
    dataset_ref: str,
    region_id: str,
    start: str,
    end: str,
    series: pd.Series,
    freq: str | None = None,
) -> SeriesMetadata:
    """Build compact metadata for a TEC series."""

    finite = _finite_values(series)

    return SeriesMetadata(
        series_id=series_id,
        dataset_ref=dataset_ref,
        region_id=region_id,  # type: ignore[arg-type]
        start=start,
        end=end,
        n_points=int(len(series)),
        finite_points=int(len(finite)),
        freq=freq,
        min_value=_safe_float(finite.min()) if len(finite) else None,
        max_value=_safe_float(finite.max()) if len(finite) else None,
        mean_value=_safe_float(finite.mean()) if len(finite) else None,
        approx_q90=_safe_float(finite.quantile(0.9)) if len(finite) else None,
    )


def tec_get_timeseries(
    args: GetTimeseriesInput,
    store: ToolStore,
) -> GetTimeseriesOutput:
    """Load and store a TEC time series for one region."""

    series = get_region_series(
        dataset_ref=args.dataset_ref,
        region_id=args.region_id,
        start=args.start,
        end=args.end,
        freq=args.freq,
    )

    series_id = _make_id(
        "series",
        {
            "dataset_ref": args.dataset_ref,
            "region_id": args.region_id,
            "start": args.start,
            "end": args.end,
            "freq": args.freq,
        },
    )

    store.series[series_id] = series

    metadata = _series_metadata(
        series_id=series_id,
        dataset_ref=args.dataset_ref,
        region_id=args.region_id,
        start=args.start,
        end=args.end,
        series=series,
        freq=args.freq,
    )
    store.series_metadata[series_id] = metadata.model_dump()

    return GetTimeseriesOutput(series_id=series_id, metadata=metadata)


def tec_series_profile(
    args: SeriesProfileInput,
    store: ToolStore,
) -> SeriesProfileOutput:
    """Return descriptive statistics for a stored TEC time series."""

    series = _get_series_or_raise(args.series_id, store)
    finite = _finite_values(series)

    start = str(series.index.min()) if len(series) else ""
    end = str(series.index.max()) if len(series) else ""

    return SeriesProfileOutput(
        series_id=args.series_id,
        n_points=int(len(series)),
        finite_points=int(len(finite)),
        start=start,
        end=end,
        min_value=_safe_float(finite.min()) if len(finite) else None,
        max_value=_safe_float(finite.max()) if len(finite) else None,
        mean_value=_safe_float(finite.mean()) if len(finite) else None,
        median_value=_safe_float(finite.median()) if len(finite) else None,
        std_value=_safe_float(finite.std()) if len(finite) else None,
        q10=_safe_float(finite.quantile(0.10)) if len(finite) else None,
        q25=_safe_float(finite.quantile(0.25)) if len(finite) else None,
        q75=_safe_float(finite.quantile(0.75)) if len(finite) else None,
        q90=_safe_float(finite.quantile(0.90)) if len(finite) else None,
    )


def tec_compute_high_threshold(
    args: ComputeHighThresholdInput,
    store: ToolStore,
) -> ComputeHighThresholdOutput:
    """Compute a high-TEC threshold for a stored series."""

    series = _get_series_or_raise(args.series_id, store)
    finite = _finite_values(series)

    if finite.empty:
        raise ValueError(f"Series {args.series_id!r} has no finite values")

    if args.method == "quantile":
        threshold_value = float(finite.quantile(args.q))
        q_value: float | None = args.q
    elif args.method == "absolute":
        if args.value is None:
            raise ValueError("value is required when method='absolute'")
        threshold_value = float(args.value)
        q_value = None
    else:
        raise ValueError(f"Unsupported threshold method: {args.method!r}")

    threshold_id = _make_id(
        "thr",
        {
            "series_id": args.series_id,
            "method": args.method,
            "q": q_value,
            "value": threshold_value,
        },
    )

    store.thresholds[threshold_id] = {
        "kind": "high",
        "series_id": args.series_id,
        "method": args.method,
        "q": q_value,
        "value": threshold_value,
    }

    return ComputeHighThresholdOutput(
        threshold_id=threshold_id,
        series_id=args.series_id,
        method=args.method,
        q=q_value,
        value=threshold_value,
        n_points_used=int(len(finite)),
    )


def tec_detect_high_intervals(
    args: DetectHighIntervalsInput,
    store: ToolStore,
) -> DetectHighIntervalsOutput:
    """Detect intervals where TEC is greater than or equal to threshold."""

    series = _get_series_or_raise(args.series_id, store)
    threshold = _get_threshold_or_raise(args.threshold_id, store)

    if threshold.get("kind") != "high":
        raise ValueError(f"Threshold {args.threshold_id!r} is not a high-TEC threshold")
    if threshold.get("series_id") != args.series_id:
        raise ValueError(
            f"Threshold {args.threshold_id!r} belongs to series "
            f"{threshold.get('series_id')!r}, not {args.series_id!r}"
        )

    threshold_value = float(threshold["value"])
    finite_series = pd.to_numeric(series, errors="coerce")
    mask = finite_series >= threshold_value

    intervals = _boolean_mask_to_intervals(
        series=finite_series,
        mask=mask.fillna(False),
        min_duration_minutes=args.min_duration_minutes,
        merge_gap_minutes=args.merge_gap_minutes,
        mode="high",
    )

    return DetectHighIntervalsOutput(
        series_id=args.series_id,
        threshold_id=args.threshold_id,
        threshold_value=threshold_value,
        n_intervals=len(intervals),
        intervals=intervals,  # type: ignore[arg-type]
    )


def tec_compute_stability_thresholds(
    args: ComputeStabilityThresholdsInput,
    store: ToolStore,
) -> ComputeStabilityThresholdsOutput:
    """Compute rolling-window thresholds for stable interval detection."""

    series = _get_series_or_raise(args.series_id, store)
    numeric = pd.to_numeric(series, errors="coerce")

    window_points = _window_points_from_minutes(numeric.index, args.window_minutes)
    rolling_std = numeric.rolling(window_points, min_periods=window_points).std()
    rolling_delta = numeric.diff().abs().rolling(
        window_points,
        min_periods=window_points,
    ).max()

    finite_std = rolling_std[np.isfinite(rolling_std)]
    finite_delta = rolling_delta[np.isfinite(rolling_delta)]

    if finite_std.empty or finite_delta.empty:
        raise ValueError(
            f"Not enough finite data to compute stability thresholds for "
            f"series {args.series_id!r}"
        )

    max_std = float(finite_std.quantile(args.q_std))
    max_abs_delta = float(finite_delta.quantile(args.q_delta))
    step_minutes = _estimate_step_minutes(numeric.index)

    threshold_id = _make_id(
        "stab",
        {
            "series_id": args.series_id,
            "method": args.method,
            "window_minutes": args.window_minutes,
            "q_delta": args.q_delta,
            "q_std": args.q_std,
            "max_std": max_std,
            "max_abs_delta": max_abs_delta,
        },
    )

    store.thresholds[threshold_id] = {
        "kind": "stability",
        "series_id": args.series_id,
        "method": args.method,
        "window_minutes": args.window_minutes,
        "window_points": window_points,
        "max_std": max_std,
        "max_abs_delta": max_abs_delta,
        "rolling_std_threshold": max_std,
        "max_delta_threshold": max_abs_delta,
        "estimated_step_minutes": step_minutes,
        "q_delta": args.q_delta,
        "q_std": args.q_std,
    }

    return ComputeStabilityThresholdsOutput(
        threshold_id=threshold_id,
        series_id=args.series_id,
        method=args.method,
        window_minutes=args.window_minutes,
        q_delta=args.q_delta,
        q_std=args.q_std,
        max_delta_threshold=max_abs_delta,
        rolling_std_threshold=max_std,
        estimated_step_minutes=step_minutes,
        window_points=window_points,
        n_points=int(len(numeric)),
        max_abs_delta=max_abs_delta,
        max_std=max_std,
        n_windows_used=int(min(len(finite_std), len(finite_delta))),
    )


def tec_compute_series_stats(
    args: ComputeSeriesStatsInput,
    store: ToolStore,
) -> ComputeSeriesStatsOutput:
    """Compute deterministic statistics for a stored TEC time series."""

    series = _get_series_or_raise(args.series_id, store)
    finite = _finite_values(series)
    metrics = _normalize_stats_metrics(
        args.metrics,
        default=DEFAULT_SERIES_STATS_METRICS,
    )

    values = _compute_metrics_dict(finite=finite, metrics=metrics)
    region_id = _series_region_id(args.series_id, store)

    stats_id = _make_id(
        "stats",
        {
            "series_id": args.series_id,
            "metrics": metrics,
        },
    )

    output = ComputeSeriesStatsOutput(
        stats_id=stats_id,
        series_id=args.series_id,
        region_id=region_id,  # type: ignore[arg-type]
        n_points=int(len(series)),
        finite_points=int(len(finite)),
        metrics=values,
    )

    store.stats[stats_id] = output.model_dump()

    return output


def tec_compare_stats(
    args: CompareStatsInput,
    store: ToolStore,
) -> CompareStatsOutput:
    """Compare previously computed statistics from multiple series."""

    stats_records = [_get_stats_or_raise(stats_id, store) for stats_id in args.stats_ids]

    if args.reference_stats_id is not None:
        if args.reference_stats_id not in args.stats_ids:
            raise ValueError("reference_stats_id must be one of stats_ids")
        _get_stats_or_raise(args.reference_stats_id, store)

    metrics = _normalize_stats_metrics(
        args.metrics,
        default=DEFAULT_COMPARE_STATS_METRICS,
    )

    items = [
        {
            "stats_id": str(record["stats_id"]),
            "series_id": str(record["series_id"]),
            "region_id": record.get("region_id"),
            "metrics": {
                metric: _safe_float((record.get("metrics") or {}).get(metric))
                for metric in metrics
            },
        }
        for record in stats_records
    ]

    pairwise_deltas = _compare_stats_pairwise_deltas(
        items=items,
        metrics=metrics,
        reference_stats_id=args.reference_stats_id,
    )

    comparison_id = _make_id(
        "cmpstats",
        {
            "stats_ids": args.stats_ids,
            "reference_stats_id": args.reference_stats_id,
            "metrics": metrics,
        },
    )

    output = CompareStatsOutput(
        comparison_id=comparison_id,
        stats_ids=list(args.stats_ids),
        reference_stats_id=args.reference_stats_id,
        regions=[
            str(item["region_id"])
            for item in items
            if item.get("region_id") is not None
        ],
        items=items,  # type: ignore[arg-type]
        pairwise_deltas=pairwise_deltas,  # type: ignore[arg-type]
    )

    store.comparisons[comparison_id] = output.model_dump()

    return output


def tec_detect_stable_intervals(
    args: DetectStableIntervalsInput,
    store: ToolStore,
) -> DetectStableIntervalsOutput:
    """Detect stable TEC intervals using a previously computed stability threshold."""

    series = _get_series_or_raise(args.series_id, store)
    threshold = _get_threshold_or_raise(args.threshold_id, store)

    if threshold.get("kind") != "stability":
        raise ValueError(f"Threshold {args.threshold_id!r} is not a stability threshold")
    if threshold.get("series_id") != args.series_id:
        raise ValueError(
            f"Threshold {args.threshold_id!r} belongs to series "
            f"{threshold.get('series_id')!r}, not {args.series_id!r}"
        )

    intervals = _detect_stable_intervals_core(
        series=series,
        window_points=int(threshold["window_points"]),
        max_abs_delta=float(threshold["max_abs_delta"]),
        max_std=float(threshold["max_std"]),
        min_duration_minutes=args.min_duration_minutes,
        merge_gap_minutes=args.merge_gap_minutes,
    )

    return DetectStableIntervalsOutput(
        series_id=args.series_id,
        threshold_id=args.threshold_id,
        n_intervals=len(intervals),
        intervals=intervals,
    )


def tec_find_stable_intervals_direct(
    args: FindStableIntervalsDirectInput,
    store: ToolStore,
) -> DetectStableIntervalsOutput:
    """Detect stable intervals using explicit thresholds without storing them first."""

    series = _get_series_or_raise(args.series_id, store)
    window_points = _window_points_from_minutes(series.index, args.window_minutes)
    max_delta = (
        float(args.max_abs_delta)
        if args.max_abs_delta is not None
        else float(args.max_delta)
    )

    intervals = _detect_stable_intervals_core(
        series=series,
        window_points=window_points,
        max_abs_delta=max_delta,
        max_std=args.max_std,
        min_duration_minutes=args.min_duration_minutes,
        merge_gap_minutes=args.merge_gap_minutes,
    )

    threshold_id = _make_id(
        "stab_direct",
        {
            "series_id": args.series_id,
            "window_minutes": args.window_minutes,
            "max_abs_delta": max_delta,
            "max_std": args.max_std,
        },
    )

    store.thresholds[threshold_id] = {
        "kind": "stability",
        "series_id": args.series_id,
        "window_minutes": args.window_minutes,
        "window_points": window_points,
        "max_abs_delta": max_delta,
        "max_delta_threshold": max_delta,
        "max_std": args.max_std,
        "rolling_std_threshold": args.max_std,
        "direct": True,
    }

    return DetectStableIntervalsOutput(
        series_id=args.series_id,
        threshold_id=threshold_id,
        n_intervals=len(intervals),
        intervals=intervals,
    )


def tec_compare_regions(
    args: CompareRegionsInput,
    store: ToolStore,
) -> CompareRegionsOutput:
    """Compare aggregated TEC statistics across several regions."""

    stats: list[RegionStatsRecord] = []

    for region_id in args.region_ids:
        series = get_region_series(
            dataset_ref=args.dataset_ref,
            region_id=region_id,
            start=args.start,
            end=args.end,
            freq=args.freq,
        )
        finite = _finite_values(series)

        record = _region_stats_record(region_id=region_id, finite=finite)
        stats.append(record)

    return CompareRegionsOutput(
        dataset_ref=args.dataset_ref,
        start=args.start,
        end=args.end,
        stats=stats,
    )


def _get_series_or_raise(series_id: str, store: ToolStore) -> pd.Series:
    """Return a stored series or raise a clear error."""

    try:
        return store.series[series_id]
    except KeyError as exc:
        known = ", ".join(sorted(store.series)) or "<none>"
        raise ValueError(
            f"Unknown series_id: {series_id!r}. Known series IDs: {known}"
        ) from exc


def _get_threshold_or_raise(threshold_id: str, store: ToolStore) -> dict[str, Any]:
    """Return a stored threshold or raise a clear error."""

    try:
        return store.thresholds[threshold_id]
    except KeyError as exc:
        known = ", ".join(sorted(store.thresholds)) or "<none>"
        raise ValueError(
            f"Unknown threshold_id: {threshold_id!r}. Known threshold IDs: {known}"
        ) from exc


def _get_stats_or_raise(stats_id: str, store: ToolStore) -> dict[str, Any]:
    """Return stored series statistics or raise a clear error."""

    try:
        return store.stats[stats_id]
    except KeyError as exc:
        known = ", ".join(sorted(store.stats)) or "<none>"
        raise ValueError(
            f"Unknown stats_id: {stats_id!r}. Known stats IDs: {known}"
        ) from exc


def _series_region_id(series_id: str, store: ToolStore) -> str | None:
    """Return region ID associated with a stored series, if known."""

    metadata = store.series_metadata.get(series_id) or {}
    region_id = metadata.get("region_id")
    if region_id is not None:
        return str(region_id)

    series = store.series.get(series_id)
    if series is not None and series.name is not None:
        return str(series.name)

    return None


def _normalize_stats_metrics(
    metrics: list[str] | None,
    *,
    default: list[str],
) -> list[str]:
    """Return a de-duplicated metric list preserving order."""

    selected = metrics or default
    return list(dict.fromkeys(str(metric) for metric in selected))


def _compute_metrics_dict(
    *,
    finite: pd.Series,
    metrics: list[str],
) -> dict[str, float | None]:
    """Compute selected JSON-friendly numeric metrics for finite values."""

    result: dict[str, float | None] = {}

    for metric in metrics:
        if finite.empty:
            result[metric] = None
        elif metric == "mean":
            result[metric] = _safe_float(finite.mean())
        elif metric == "median":
            result[metric] = _safe_float(finite.median())
        elif metric == "min":
            result[metric] = _safe_float(finite.min())
        elif metric == "max":
            result[metric] = _safe_float(finite.max())
        elif metric == "std":
            result[metric] = _safe_float(finite.std())
        elif metric == "p90":
            result[metric] = _safe_float(finite.quantile(0.90))
        elif metric == "p95":
            result[metric] = _safe_float(finite.quantile(0.95))
        else:
            raise ValueError(f"Unsupported stats metric: {metric!r}")

    return result


def _compare_stats_pairwise_deltas(
    *,
    items: list[dict[str, Any]],
    metrics: list[str],
    reference_stats_id: str | None,
) -> list[dict[str, Any]]:
    """Build pairwise metric deltas for stats comparison output."""

    by_stats_id = {str(item["stats_id"]): item for item in items}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    if reference_stats_id is not None:
        reference = by_stats_id[reference_stats_id]
        pairs = [
            (reference, item)
            for item in items
            if item["stats_id"] != reference_stats_id
        ]
    else:
        for left_idx, left in enumerate(items):
            for right in items[left_idx + 1 :]:
                pairs.append((left, right))

    deltas: list[dict[str, Any]] = []

    for left, right in pairs:
        delta: dict[str, float | None] = {}

        for metric in metrics:
            left_value = (left.get("metrics") or {}).get(metric)
            right_value = (right.get("metrics") or {}).get(metric)

            if left_value is None or right_value is None:
                delta[metric] = None
            else:
                delta[metric] = _safe_float(float(left_value) - float(right_value))

        deltas.append(
            {
                "left_stats_id": left["stats_id"],
                "right_stats_id": right["stats_id"],
                "left_region_id": left.get("region_id"),
                "right_region_id": right.get("region_id"),
                "delta": delta,
            }
        )

    return deltas


def _window_points_from_minutes(
    index: pd.DatetimeIndex,
    window_minutes: int,
) -> int:
    """Convert window duration in minutes to rolling window points."""

    step_minutes = _estimate_step_minutes(index)

    if step_minutes is None or step_minutes <= 0:
        raise ValueError("Cannot estimate sampling step for rolling window")

    points = max(1, int(round(window_minutes / step_minutes)))
    return points


def _boolean_mask_to_intervals(
    series: pd.Series,
    mask: pd.Series,
    min_duration_minutes: float,
    merge_gap_minutes: float,
    mode: str,
) -> list[IntervalRecord | StableIntervalRecord]:
    """Convert a boolean mask into merged time intervals."""

    if len(series) == 0:
        return []

    mask = mask.astype(bool)
    index = series.index

    raw_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_interval = False
    start: pd.Timestamp | None = None
    last_true: pd.Timestamp | None = None

    for ts, flag in mask.items():
        ts = pd.Timestamp(ts)

        if flag and not in_interval:
            in_interval = True
            start = ts

        if flag:
            last_true = ts

        if not flag and in_interval:
            assert start is not None
            assert last_true is not None
            raw_intervals.append((start, last_true))
            in_interval = False
            start = None
            last_true = None

    if in_interval and start is not None and last_true is not None:
        raw_intervals.append((start, last_true))

    merged = _merge_intervals(raw_intervals, merge_gap_minutes)
    filtered = [
        (start_ts, end_ts)
        for start_ts, end_ts in merged
        if _duration_minutes(start_ts, end_ts) >= min_duration_minutes
    ]

    if mode == "high":
        return [_build_high_interval_record(series, start_ts, end_ts) for start_ts, end_ts in filtered]

    raise ValueError(f"Unsupported interval conversion mode: {mode!r}")


def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    merge_gap_minutes: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge intervals separated by gaps no longer than merge_gap_minutes."""

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda item: item[0])
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [intervals[0]]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        gap = _duration_minutes(prev_end, start)

        if gap <= merge_gap_minutes:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _build_high_interval_record(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> IntervalRecord:
    """Build output record for a high-TEC interval."""

    segment = series.loc[(series.index >= start) & (series.index <= end)]
    finite = _finite_values(segment)

    if finite.empty:
        peak_time = None
        peak_value = None
        mean_value = None
        n_points = 0
    else:
        peak_idx = finite.idxmax()
        peak_time = _format_timestamp(peak_idx)
        peak_value = _safe_float(finite.loc[peak_idx])
        mean_value = _safe_float(finite.mean())
        n_points = int(len(finite))

    return IntervalRecord(
        start=_format_timestamp(start),
        end=_format_timestamp(end),
        duration_minutes=_duration_minutes(start, end),
        peak_time=peak_time,
        peak_value=peak_value,
        mean_value=mean_value,
        n_points=n_points,
    )


def _detect_stable_intervals_core(
    series: pd.Series,
    window_points: int,
    max_abs_delta: float,
    max_std: float,
    min_duration_minutes: float,
    merge_gap_minutes: float,
) -> list[StableIntervalRecord]:
    """Detect stable intervals using rolling standard deviation and max delta."""

    numeric = pd.to_numeric(series, errors="coerce")

    rolling_std = numeric.rolling(window_points, min_periods=window_points).std()
    rolling_delta = numeric.diff().abs().rolling(
        window_points,
        min_periods=window_points,
    ).max()

    mask = (rolling_std <= max_std) & (rolling_delta <= max_abs_delta)
    mask = mask.fillna(False)

    raw_intervals = _mask_to_raw_intervals(mask)
    merged = _merge_intervals(raw_intervals, merge_gap_minutes)

    filtered = [
        (start_ts, end_ts)
        for start_ts, end_ts in merged
        if _duration_minutes(start_ts, end_ts) >= min_duration_minutes
    ]

    return [
        _build_stable_interval_record(
            series=numeric,
            start=start_ts,
            end=end_ts,
        )
        for start_ts, end_ts in filtered
    ]


def _mask_to_raw_intervals(
    mask: pd.Series,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert boolean mask to raw intervals without filtering."""

    raw_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_interval = False
    start: pd.Timestamp | None = None
    last_true: pd.Timestamp | None = None

    for ts, flag in mask.items():
        ts = pd.Timestamp(ts)

        if bool(flag) and not in_interval:
            in_interval = True
            start = ts

        if bool(flag):
            last_true = ts

        if not bool(flag) and in_interval:
            assert start is not None
            assert last_true is not None
            raw_intervals.append((start, last_true))
            in_interval = False
            start = None
            last_true = None

    if in_interval and start is not None and last_true is not None:
        raw_intervals.append((start, last_true))

    return raw_intervals


def _build_stable_interval_record(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> StableIntervalRecord:
    """Build output record for a stable TEC interval."""

    segment = series.loc[(series.index >= start) & (series.index <= end)]
    finite = _finite_values(segment)

    if finite.empty:
        return StableIntervalRecord(
            start=_format_timestamp(start),
            end=_format_timestamp(end),
            duration_minutes=_duration_minutes(start, end),
            n_points=0,
        )

    max_abs_delta = _safe_float(finite.diff().abs().max())

    return StableIntervalRecord(
        start=_format_timestamp(start),
        end=_format_timestamp(end),
        duration_minutes=_duration_minutes(start, end),
        mean_value=_safe_float(finite.mean()),
        std_value=_safe_float(finite.std()),
        min_value=_safe_float(finite.min()),
        max_value=_safe_float(finite.max()),
        max_delta=max_abs_delta,
        max_abs_delta=max_abs_delta,
        n_points=int(len(finite)),
    )


def _region_stats_record(
    region_id: str,
    finite: pd.Series,
) -> RegionStatsRecord:
    """Build aggregate statistics for one region."""

    return RegionStatsRecord(
        region_id=region_id,  # type: ignore[arg-type]
        n_points=int(len(finite)),
        finite_points=int(len(finite)),
        mean=_safe_float(finite.mean()) if len(finite) else None,
        median=_safe_float(finite.median()) if len(finite) else None,
        min=_safe_float(finite.min()) if len(finite) else None,
        max=_safe_float(finite.max()) if len(finite) else None,
        std=_safe_float(finite.std()) if len(finite) else None,
        p90=_safe_float(finite.quantile(0.90)) if len(finite) else None,
        p95=_safe_float(finite.quantile(0.95)) if len(finite) else None,
    )
