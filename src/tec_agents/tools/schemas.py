"""
Pydantic schemas for TEC analysis tools.

These schemas define strict input and output contracts for all tool calls.
They are used by the tool executor, MCP-like layer, and later by LLM agents.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tec_agents.data.regions import list_region_ids


RegionId = Literal[
    "equatorial_atlantic",
    "equatorial_africa",
    "equatorial_pacific",
    "midlat_europe",
    "midlat_usa",
    "midlat_asia",
    "midlat_south_america",
    "midlat_australia",
    "highlat_north",
    "highlat_south",
]

StatsMetric = Literal[
    "mean",
    "median",
    "min",
    "max",
    "std",
    "p90",
    "p95",
]


class ToolError(BaseModel):
    """Structured error returned by a failed tool call."""

    error_type: str
    message: str


class SeriesMetadata(BaseModel):
    """Compact metadata for a TEC time series."""

    series_id: str
    dataset_ref: str
    region_id: RegionId
    start: str
    end: str
    n_points: int
    finite_points: int
    freq: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    approx_q90: float | None = None


class IntervalRecord(BaseModel):
    """One detected time interval."""

    start: str
    end: str
    duration_minutes: float
    peak_time: str | None = None
    peak_value: float | None = None
    mean_value: float | None = None
    n_points: int


class StableIntervalRecord(BaseModel):
    """One detected stable interval with basic variability statistics."""

    start: str
    end: str
    duration_minutes: float
    mean_value: float | None = None
    std_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    max_delta: float | None = None
    max_abs_delta: float | None = None
    n_points: int


class RegionStatsRecord(BaseModel):
    """Aggregated statistics for one region."""

    region_id: RegionId
    n_points: int
    finite_points: int
    mean: float | None = None
    median: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    p90: float | None = None
    p95: float | None = None


# ---------------------------------------------------------------------------
# Dataset / series tools
# ---------------------------------------------------------------------------


class GetTimeseriesInput(BaseModel):
    """Input schema for tec_get_timeseries."""

    dataset_ref: str = Field(default="default")
    region_id: RegionId
    start: str = Field(description="Inclusive start datetime, e.g. 2024-03-01")
    end: str = Field(description="Exclusive end datetime, e.g. 2024-04-01")
    freq: str | None = Field(
        default=None,
        description="Optional pandas resampling frequency, e.g. '1H' or '1D'",
    )


class GetTimeseriesOutput(BaseModel):
    """Output schema for tec_get_timeseries."""

    series_id: str
    metadata: SeriesMetadata


class SeriesProfileInput(BaseModel):
    """Input schema for tec_series_profile."""

    series_id: str


class SeriesProfileOutput(BaseModel):
    """Output schema for tec_series_profile."""

    series_id: str
    n_points: int
    finite_points: int
    start: str
    end: str
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    median_value: float | None = None
    std_value: float | None = None
    q10: float | None = None
    q25: float | None = None
    q75: float | None = None
    q90: float | None = None


class ComputeSeriesStatsInput(BaseModel):
    """Input schema for tec_compute_series_stats."""

    series_id: str
    metrics: list[StatsMetric] | None = None


class ComputeSeriesStatsOutput(BaseModel):
    """Output schema for tec_compute_series_stats."""

    stats_id: str
    series_id: str
    region_id: RegionId | None = None
    n_points: int
    finite_points: int
    metrics: dict[str, float | None]


class CompareStatsInput(BaseModel):
    """Input schema for tec_compare_stats."""

    stats_ids: list[str] = Field(min_length=2)
    reference_stats_id: str | None = None
    metrics: list[StatsMetric] | None = None

    @field_validator("stats_ids")
    @classmethod
    def validate_unique_stats_ids(cls, stats_ids: list[str]) -> list[str]:
        if len(stats_ids) != len(set(stats_ids)):
            raise ValueError("stats_ids must be unique")
        return stats_ids


class CompareStatsItem(BaseModel):
    """One stats item included in a stats comparison."""

    stats_id: str
    series_id: str
    region_id: RegionId | None = None
    metrics: dict[str, float | None]


class PairwiseStatsDeltaRecord(BaseModel):
    """Pairwise metric deltas between two stats handles."""

    left_stats_id: str
    right_stats_id: str
    left_region_id: RegionId | None = None
    right_region_id: RegionId | None = None
    delta: dict[str, float | None]


class CompareStatsOutput(BaseModel):
    """Output schema for tec_compare_stats."""

    comparison_id: str
    stats_ids: list[str]
    reference_stats_id: str | None = None
    regions: list[str]
    items: list[CompareStatsItem]
    pairwise_deltas: list[PairwiseStatsDeltaRecord]


# ---------------------------------------------------------------------------
# High TEC tools
# ---------------------------------------------------------------------------


class ComputeHighThresholdInput(BaseModel):
    """Input schema for tec_compute_high_threshold."""

    series_id: str
    method: Literal["quantile", "absolute"] = "quantile"
    q: float = Field(default=0.9, ge=0.0, le=1.0)
    value: float | None = Field(
        default=None,
        description="Absolute threshold value in TECU if method='absolute'",
    )

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("Absolute TEC threshold must be non-negative")
        return value


class ComputeHighThresholdOutput(BaseModel):
    """Output schema for tec_compute_high_threshold."""

    threshold_id: str
    series_id: str
    method: Literal["quantile", "absolute"]
    q: float | None = None
    value: float
    n_points_used: int


class DetectHighIntervalsInput(BaseModel):
    """Input schema for tec_detect_high_intervals."""

    series_id: str
    threshold_id: str
    min_duration_minutes: float = Field(default=0.0, ge=0.0)
    merge_gap_minutes: float = Field(default=0.0, ge=0.0)


class DetectHighIntervalsOutput(BaseModel):
    """Output schema for tec_detect_high_intervals."""

    series_id: str
    threshold_id: str
    threshold_value: float
    n_intervals: int
    intervals: list[IntervalRecord]


# ---------------------------------------------------------------------------
# Stability tools
# ---------------------------------------------------------------------------


class ComputeStabilityThresholdsInput(BaseModel):
    """Input schema for tec_compute_stability_thresholds."""

    series_id: str
    window_minutes: int = Field(default=180, ge=1)
    method: Literal["quantile"] = "quantile"
    q_delta: float = Field(default=0.6, ge=0.0, le=1.0)
    q_std: float = Field(default=0.6, ge=0.0, le=1.0)


class ComputeStabilityThresholdsOutput(BaseModel):
    """Output schema for tec_compute_stability_thresholds."""

    threshold_id: str
    series_id: str
    method: Literal["quantile"]
    window_minutes: int
    q_delta: float
    q_std: float
    max_delta_threshold: float
    rolling_std_threshold: float
    estimated_step_minutes: float | None = None
    window_points: int
    n_points: int
    max_abs_delta: float
    max_std: float
    n_windows_used: int


class DetectStableIntervalsInput(BaseModel):
    """Input schema for tec_detect_stable_intervals."""

    series_id: str
    threshold_id: str
    min_duration_minutes: float = Field(default=180.0, ge=0.0)
    merge_gap_minutes: float = Field(default=60.0, ge=0.0)


class DetectStableIntervalsOutput(BaseModel):
    """Output schema for tec_detect_stable_intervals."""

    series_id: str
    threshold_id: str
    n_intervals: int
    intervals: list[StableIntervalRecord]


class FindStableIntervalsDirectInput(BaseModel):
    """Input schema for direct stable interval detection with explicit limits."""

    series_id: str
    window_minutes: int = Field(default=180, ge=1)
    max_delta: float | None = Field(default=None, ge=0.0)
    max_abs_delta: float | None = Field(default=None, ge=0.0)
    max_std: float = Field(ge=0.0)
    min_duration_minutes: float = Field(default=180.0, ge=0.0)
    merge_gap_minutes: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_delta_threshold(self) -> "FindStableIntervalsDirectInput":
        if self.max_delta is None and self.max_abs_delta is None:
            raise ValueError("Either max_delta or max_abs_delta is required")
        return self


# Reuses DetectStableIntervalsOutput.


# ---------------------------------------------------------------------------
# Comparison / report tools
# ---------------------------------------------------------------------------


class CompareRegionsInput(BaseModel):
    """Input schema for tec_compare_regions."""

    dataset_ref: str = "default"
    region_ids: list[RegionId] = Field(min_length=2)
    start: str
    end: str
    freq: str | None = None
    metrics: list[Literal["mean", "median", "min", "max", "std", "p90", "p95"]] = Field(
        default_factory=lambda: ["mean", "median", "max", "std", "p90"]
    )

    @field_validator("region_ids")
    @classmethod
    def validate_unique_regions(cls, region_ids: list[RegionId]) -> list[RegionId]:
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region_ids must be unique")
        return region_ids


class CompareRegionsOutput(BaseModel):
    """Output schema for tec_compare_regions."""

    dataset_ref: str
    start: str
    end: str
    stats: list[RegionStatsRecord]


class BuildReportInput(BaseModel):
    """Input schema for tec_build_report."""

    dataset_ref: str = "default"
    regions: list[RegionId] | None = None
    region_ids: list[RegionId] | None = None
    start: str
    end: str
    freq: str | None = None
    include: list[Literal["basic_stats", "high_tec", "stable_intervals"]] | None = None
    include_high_tec: bool | None = True
    include_stability: bool | None = None
    q_high: float = Field(default=0.9, ge=0.0, le=1.0)
    window_minutes: int = Field(default=180, ge=1)
    q_delta: float = Field(default=0.6, ge=0.0, le=1.0)
    q_std: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_regions(self) -> "BuildReportInput":
        selected_regions = self.regions or self.region_ids or []
        if not selected_regions:
            raise ValueError("Build report task requires at least one region")
        if len(selected_regions) != len(set(selected_regions)):
            raise ValueError("Report regions must be unique")
        return self


class BuildReportOutput(BaseModel):
    """Output schema for tec_build_report."""

    report_id: str
    dataset_ref: str
    regions: list[RegionId]
    start: str
    end: str
    region_ids: list[RegionId]
    sections: dict[str, Any]


# ---------------------------------------------------------------------------
# Tool schema export helpers
# ---------------------------------------------------------------------------


def available_region_ids() -> list[str]:
    """Return supported region IDs for external schema generation."""

    return list_region_ids()


def model_json_schema_for_tool(model: type[BaseModel]) -> dict[str, Any]:
    """
    Return JSON schema for a Pydantic model.

    This helper keeps schema generation centralized, which will be useful when
    exposing tools to an OpenAI-compatible API or MCP-like interface.
    """

    return model.model_json_schema()
