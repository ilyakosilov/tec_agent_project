"""
Smoke test for [start, end) date filtering on a generated multi-month dataset.

Set TEC_DATASET_PATH to override the default processed parquet path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.data.datasets import register_dataset
from tec_agents.mcp.client import LocalMCPClient
from tec_agents.mcp.server import build_local_mcp_server


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tec_regions_2024_01_01_to_2024_04_01_hourly.parquet"
)


def main() -> None:
    dataset_path = Path(os.environ.get("TEC_DATASET_PATH", DEFAULT_DATASET_PATH))

    if not dataset_path.exists():
        raise SystemExit(
            "Dataset file not found. Build it with "
            "notebooks/01_build_tec_dataset.ipynb using "
            "START_DATE=2024-01-01, END_DATE=2024-04-01. "
            f"Expected path: {dataset_path}"
        )

    register_dataset(
        dataset_ref="date_filtering_smoke",
        path=dataset_path,
        file_format="parquet",
    )

    server = build_local_mcp_server(run_id="date_filtering_smoke")
    client = LocalMCPClient(server)

    cases = [
        ("jan_2024", "2024-01-01", "2024-02-01", 744),
        ("feb_2024", "2024-02-01", "2024-03-01", 696),
        ("mar_2024", "2024-03-01", "2024-04-01", 744),
        ("jan_to_mar_2024", "2024-01-01", "2024-04-01", 2184),
        ("mar_10_to_20", "2024-03-10", "2024-03-20", 240),
        ("single_day", "2024-03-15", "2024-03-16", 24),
    ]

    for label, start, end, expected_n_points in cases:
        response = client.call_tool(
            "tec_get_timeseries",
            {
                "dataset_ref": "date_filtering_smoke",
                "region_id": "midlat_europe",
                "start": start,
                "end": end,
            },
            agent_name="date_filtering_smoke",
        )

        if response.status != "ok" or response.result is None:
            raise AssertionError(f"{label}: tool failed: {response.error}")

        metadata = response.result["metadata"]
        actual_n_points = metadata["n_points"]
        actual_start = metadata.get("actual_start")
        actual_end = metadata.get("actual_end")

        print(
            f"{label}: [{start}, {end}) -> n_points={actual_n_points}, "
            f"actual_start={actual_start}, actual_end={actual_end}"
        )

        assert metadata.get("interval_convention") == "[start, end)"
        assert metadata.get("requested_start") == start
        assert metadata.get("requested_end") == end
        assert actual_n_points == expected_n_points, (
            f"{label}: expected {expected_n_points}, got {actual_n_points}"
        )

    print("Date filtering smoke test finished successfully.")


if __name__ == "__main__":
    main()
