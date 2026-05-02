"""
Dataset access helpers for processed TEC regional time series.

This module works with aggregated regional TEC tables, not raw IONEX files.
Raw IONEX files should be downloaded and parsed in Colab notebooks, while
agents and tools should use lightweight processed CSV/Parquet files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from tec_agents.data.regions import get_region, list_region_ids


DatasetFormat = Literal["parquet", "csv"]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a processed TEC dataset."""

    dataset_ref: str
    path: Path
    file_format: DatasetFormat = "parquet"
    time_column: str | None = None


_DATASET_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(
    dataset_ref: str,
    path: str | Path,
    file_format: DatasetFormat | None = None,
    time_column: str | None = None,
) -> DatasetConfig:
    """
    Register a processed TEC dataset.

    Parameters
    ----------
    dataset_ref:
        Short dataset name, for example "default".
    path:
        Path to processed CSV or Parquet file.
    file_format:
        Optional explicit format. If not provided, it is inferred from suffix.
    time_column:
        Optional time column name for CSV/Parquet files where time is not index.
    """

    path = Path(path)

    if file_format is None:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            file_format = "parquet"
        elif suffix == ".csv":
            file_format = "csv"
        else:
            raise ValueError(
                f"Cannot infer dataset format from suffix {suffix!r}. "
                "Use file_format='parquet' or file_format='csv'."
            )

    config = DatasetConfig(
        dataset_ref=dataset_ref,
        path=path,
        file_format=file_format,
        time_column=time_column,
    )
    _DATASET_REGISTRY[dataset_ref] = config
    return config


def get_dataset_config(dataset_ref: str = "default") -> DatasetConfig:
    """Return registered dataset configuration."""

    try:
        return _DATASET_REGISTRY[dataset_ref]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASET_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown dataset_ref: {dataset_ref!r}. Registered datasets: {known}"
        ) from exc


def list_datasets() -> list[dict[str, str]]:
    """Return registered datasets as JSON-serializable records."""

    return [
        {
            "dataset_ref": cfg.dataset_ref,
            "path": str(cfg.path),
            "file_format": cfg.file_format,
            "time_column": cfg.time_column or "",
        }
        for cfg in _DATASET_REGISTRY.values()
    ]


def load_processed_dataset(dataset_ref: str = "default") -> pd.DataFrame:
    """
    Load a processed regional TEC dataset.

    Expected structure:
    - DatetimeIndex or a dedicated time column;
    - one column per region_id;
    - numeric TEC values in TECU.
    """

    cfg = get_dataset_config(dataset_ref)

    if not cfg.path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist for dataset_ref={dataset_ref!r}: {cfg.path}"
        )

    if cfg.file_format == "parquet":
        df = pd.read_parquet(cfg.path)
    elif cfg.file_format == "csv":
        df = pd.read_csv(cfg.path)
    else:
        raise ValueError(f"Unsupported dataset format: {cfg.file_format!r}")

    df = _ensure_datetime_index(df, time_column=cfg.time_column)
    df = df.sort_index()

    return df


def _ensure_datetime_index(
    df: pd.DataFrame,
    time_column: str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with DatetimeIndex."""

    df = df.copy()

    if time_column is not None:
        if time_column not in df.columns:
            raise ValueError(f"time_column={time_column!r} not found in dataset")
        df[time_column] = pd.to_datetime(df[time_column], utc=False)
        df = df.set_index(time_column)
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Common case after CSV export with index=True.
    possible_time_columns = ["time", "datetime", "date", "index", "Unnamed: 0"]
    for col in possible_time_columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
            if parsed.notna().mean() > 0.9:
                df[col] = parsed
                df = df.set_index(col)
                df.index.name = "time"
                return df

    raise ValueError(
        "Could not detect datetime index. Provide time_column explicitly "
        "when registering the dataset."
    )


def validate_dataset_columns(df: pd.DataFrame) -> None:
    """
    Validate that all columns either correspond to known region IDs or are numeric extras.

    For now this function is intentionally permissive: it checks that at least one
    known region column exists.
    """

    known_regions = set(list_region_ids())
    available_regions = known_regions.intersection(df.columns)

    if not available_regions:
        raise ValueError(
            "Dataset has no known region columns. "
            f"Expected at least one of: {sorted(known_regions)}. "
            f"Actual columns: {list(df.columns)}"
        )


def get_region_series(
    dataset_ref: str,
    region_id: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freq: str | None = None,
) -> pd.Series:
    """
    Return a TEC time series for a selected region and half-open time interval.

    The interval convention is [start, end), so end is excluded.
    For March 2024 use start='2024-03-01', end='2024-04-01'.
    """

    get_region(region_id)  # validates region_id

    df = load_processed_dataset(dataset_ref)
    validate_dataset_columns(df)

    if region_id not in df.columns:
        raise ValueError(
            f"Region {region_id!r} is not available in dataset {dataset_ref!r}. "
            f"Available columns: {list(df.columns)}"
        )

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if end_ts <= start_ts:
        raise ValueError(f"end must be greater than start: start={start}, end={end}")

    mask = (df.index >= start_ts) & (df.index < end_ts)
    series = df.loc[mask, region_id].copy()
    series.name = region_id

    if freq is not None:
        series = series.resample(freq).mean()

    return series


def get_available_time_range(dataset_ref: str = "default") -> dict[str, str | int]:
    """Return basic time coverage metadata for a processed dataset."""

    df = load_processed_dataset(dataset_ref)

    if df.empty:
        return {
            "dataset_ref": dataset_ref,
            "n_rows": 0,
            "start": "",
            "end": "",
        }

    return {
        "dataset_ref": dataset_ref,
        "n_rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }


def get_dataset_summary(dataset_ref: str = "default") -> dict[str, object]:
    """Return compact dataset summary useful for tool responses and diagnostics."""

    df = load_processed_dataset(dataset_ref)
    validate_dataset_columns(df)

    region_columns = [col for col in df.columns if col in set(list_region_ids())]

    return {
        "dataset_ref": dataset_ref,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "region_columns": region_columns,
        "start": str(df.index.min()) if len(df) else "",
        "end": str(df.index.max()) if len(df) else "",
        "index_type": type(df.index).__name__,
    }