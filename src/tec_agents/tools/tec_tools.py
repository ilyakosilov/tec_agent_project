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

from tec_agents.data.datasets import get_region_series, load_processed_dataset
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
    IntervalRecord,
    RegionStatsRecord,
    SeriesMetadata,
    SeriesProfileInput,
    SeriesProfileOutput,
    StableIntervalRecord,
)


@dataclass
class ToolStore:
    """
    In-memory storage for intermediate tool artifacts.

    The store is intentionally explicit and resettable. Each experiment run should
    use a fresh ToolStore to avoid state leaking between runs.
    """

    series: dict[str, pd.Series] = field(default_factory=dict)
    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)


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


def tec_build_report(
    args: BuildReportInput,
    store: ToolStore,
) -> BuildReportOutput:
    """Build a structured deterministic report for one or more regions."""

    sections: dict[str, Any] = {}
    region_ids = _report_region_ids(args)
    include = _report_include_sections(args)

    if "basic_stats" in include:
        compare_args = CompareRegionsInput(
            dataset_ref=args.dataset_ref,
            region_ids=region_ids,
            start=args.start,
            end=args.end,
            freq=args.freq,
        )
        comparison = tec_compare_regions(compare_args, store)
        sections["basic_stats"] = {
            item.region_id: item.model_dump()
            for item in comparison.stats
        }

    if "high_tec" in include:
        high_sections: dict[str, Any] = {}

        for region_id in region_ids:
            ts_result = tec_get_timeseries(
                GetTimeseriesInput(
                    dataset_ref=args.dataset_ref,
                    region_id=region_id,
                    start=args.start,
                    end=args.end,
                    freq=args.freq,
                ),
                store,
            )
            threshold = tec_compute_high_threshold(
                ComputeHighThresholdInput(
                    series_id=ts_result.series_id,
                    method="quantile",
                    q=args.q_high,
                ),
                store,
            )
            intervals = tec_detect_high_intervals(
                DetectHighIntervalsInput(
                    series_id=ts_result.series_id,
                    threshold_id=threshold.threshold_id,
                ),
                store,
            )

            interval_records = [item.model_dump() for item in intervals.intervals]
            top_intervals = sorted(
                interval_records,
                key=lambda item: (
                    item.get("peak_value") is not None,
                    float(item.get("peak_value") or float("-inf")),
                ),
                reverse=True,
            )[:5]
            peak_values = [
                float(item["peak_value"])
                for item in interval_records
                if item.get("peak_value") is not None
            ]

            high_sections[region_id] = {
                "series_id": ts_result.series_id,
                "threshold_id": threshold.threshold_id,
                "threshold": threshold.value,
                "threshold_value": threshold.value,
                "q": threshold.q,
                "n_intervals": intervals.n_intervals,
                "global_peak_value": max(peak_values) if peak_values else None,
                "top_intervals": top_intervals,
            }

        sections["high_tec"] = high_sections

    if "stable_intervals" in include:
        stable_sections: dict[str, Any] = {}

        for region_id in region_ids:
            ts_result = tec_get_timeseries(
                GetTimeseriesInput(
                    dataset_ref=args.dataset_ref,
                    region_id=region_id,
                    start=args.start,
                    end=args.end,
                    freq=args.freq,
                ),
                store,
            )
            thresholds = tec_compute_stability_thresholds(
                ComputeStabilityThresholdsInput(
                    series_id=ts_result.series_id,
                    window_minutes=args.window_minutes,
                    method="quantile",
                    q_delta=args.q_delta,
                    q_std=args.q_std,
                ),
                store,
            )
            intervals = tec_detect_stable_intervals(
                DetectStableIntervalsInput(
                    series_id=ts_result.series_id,
                    threshold_id=thresholds.threshold_id,
                    min_duration_minutes=args.window_minutes,
                    merge_gap_minutes=60,
                ),
                store,
            )

            interval_records = [item.model_dump() for item in intervals.intervals]
            top_intervals = sorted(
                interval_records,
                key=lambda item: (
                    float(item.get("duration_minutes") or 0.0),
                    item.get("start") or "",
                ),
                reverse=True,
            )[:5]

            stable_sections[region_id] = {
                "series_id": ts_result.series_id,
                "threshold_id": thresholds.threshold_id,
                "window_minutes": thresholds.window_minutes,
                "q_delta": thresholds.q_delta,
                "q_std": thresholds.q_std,
                "max_delta_threshold": thresholds.max_delta_threshold,
                "rolling_std_threshold": thresholds.rolling_std_threshold,
                "estimated_step_minutes": thresholds.estimated_step_minutes,
                "window_points": thresholds.window_points,
                "n_intervals": intervals.n_intervals,
                "top_intervals": top_intervals,
            }

        sections["stable_intervals"] = stable_sections

    report_id = _make_id(
        "report",
        {
            "dataset_ref": args.dataset_ref,
            "regions": region_ids,
            "start": args.start,
            "end": args.end,
            "freq": args.freq,
            "include": include,
            "q_high": args.q_high,
            "window_minutes": args.window_minutes,
            "q_delta": args.q_delta,
            "q_std": args.q_std,
        },
    )

    output = BuildReportOutput(
        report_id=report_id,
        dataset_ref=args.dataset_ref,
        regions=region_ids,  # type: ignore[arg-type]
        start=args.start,
        end=args.end,
        region_ids=region_ids,  # type: ignore[arg-type]
        sections=sections,
    )

    store.reports[report_id] = output.model_dump()

    return output


def _report_region_ids(args: BuildReportInput) -> list[str]:
    """Return report regions from the current or backward-compatible input field."""

    return list(args.regions or args.region_ids or [])


def _report_include_sections(args: BuildReportInput) -> list[str]:
    """Return normalized report section names."""

    if args.include is not None:
        return list(dict.fromkeys(args.include))

    include = ["basic_stats"]

    if args.include_high_tec is not False:
        include.append("high_tec")

    include_stability = args.include_stability
    if include_stability is None:
        include_stability = True

    if include_stability:
        include.append("stable_intervals")

    return include


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
