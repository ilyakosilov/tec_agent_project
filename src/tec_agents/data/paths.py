"""Path helpers for generated processed TEC datasets."""

from __future__ import annotations


def normalize_date_for_filename(value: str) -> str:
    """Return a date string safe for processed dataset filenames."""

    return value.strip().replace("-", "_").replace(":", "_").replace(" ", "_")


def build_processed_dataset_filename(
    start: str,
    end: str,
    freq: str = "hourly",
    prefix: str = "tec_regions",
    suffix: str = "parquet",
) -> str:
    """Build a filename that records the half-open processed data interval."""

    clean_suffix = suffix[1:] if suffix.startswith(".") else suffix
    start_part = normalize_date_for_filename(start)
    end_part = normalize_date_for_filename(end)
    return f"{prefix}_{start_part}_to_{end_part}_{freq}.{clean_suffix}"


__all__ = [
    "build_processed_dataset_filename",
    "normalize_date_for_filename",
]
