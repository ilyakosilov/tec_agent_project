"""
Smoke test for GoldRunner.

This test creates a synthetic dataset, registers it, builds smoke evaluation
tasks, and computes deterministic gold results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.data.datasets import register_dataset
from tec_agents.eval.gold_runner import GoldRunner
from tec_agents.eval.task_set import build_smoke_tasks


def build_tiny_dataset(path: Path) -> None:
    """Create a small synthetic regional TEC dataset for local testing."""

    path.parent.mkdir(parents=True, exist_ok=True)

    time_index = pd.date_range(
        start="2024-03-01 00:00:00",
        end="2024-03-31 23:00:00",
        freq="1h",
    )

    hours = np.arange(len(time_index))

    df = pd.DataFrame(
        {
            "time": time_index,
            "midlat_europe": 25.0 + 8.0 * np.sin(hours / 6.0),
            "highlat_north": 12.0 + 3.0 * np.sin(hours / 8.0),
            "equatorial_atlantic": 35.0 + 10.0 * np.sin(hours / 5.0),
        }
    )

    # Add artificial high-TEC events.
    df.loc[120:130, "midlat_europe"] += 20.0
    df.loc[400:408, "midlat_europe"] += 15.0
    df.loc[200:205, "highlat_north"] += 8.0

    df.to_csv(path, index=False)


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "examples" / "smoke_tec_regions_month.csv"
    build_tiny_dataset(dataset_path)

    register_dataset(
        dataset_ref="smoke",
        path=dataset_path,
        file_format="csv",
        time_column="time",
    )

    tasks = build_smoke_tasks(dataset_ref="smoke")
    runner = GoldRunner()

    results = runner.run_many(tasks)

    for result in results:
        print("=" * 80)
        print(f"task_id: {result.task_id}")
        print(f"task_type: {result.task_type}")
        print(f"status: {result.status}")

        if result.error:
            print(f"error: {result.error}")

        if result.result:
            print("result keys:", list(result.result.keys()))

            if result.task_type == "high_tec":
                intervals = result.result["intervals"]
                threshold = result.result["threshold"]
                print(f"threshold: {threshold['value']:.3f}")
                print(f"n_intervals: {intervals['n_intervals']}")

            if result.task_type == "compare_regions":
                comparison = result.result["comparison"]
                print(f"regions: {result.result['region_ids']}")
                print(f"n_stats: {len(comparison['stats'])}")

        print(f"trace calls: {result.trace['n_calls']}")

    assert all(result.status == "success" for result in results)

    high_tec_results = [r for r in results if r.task_type == "high_tec"]
    assert len(high_tec_results) >= 1
    assert high_tec_results[0].result is not None
    assert high_tec_results[0].result["intervals"]["n_intervals"] >= 1

    print("\nGoldRunner smoke test finished successfully.")


if __name__ == "__main__":
    main()
    