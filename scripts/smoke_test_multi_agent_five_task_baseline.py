"""Smoke checks for the deterministic multi-agent five-task baseline configs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.eval.five_task_configs import (
    get_five_task_configs,
    build_five_task_eval_task,
    build_five_task_expected_sequence,
)
from tec_agents.eval.task_set import EvalTask


EXPECTED_PRESETS = {
    "hightec_midlat_europe",
    "stable_midlat_europe",
    "compare_midlat_europe_highlat_north",
    "compare_three_regions",
    "report_midlat_europe",
}


def main() -> None:
    configs = get_five_task_configs()
    assert len(configs) == 5
    assert {config["preset_id"] for config in configs} == EXPECTED_PRESETS

    for config in configs:
        task = build_five_task_eval_task(config)
        sequence = build_five_task_expected_sequence(config)

        assert isinstance(task, EvalTask)
        assert task.query == config["query"]
        assert task.expected_tool_sequence == tuple(sequence)
        assert "tec_build_report" not in sequence

        if config["task_type"] == "compare_regions":
            n_regions = len(config["regions"])
            assert sequence == (
                ["tec_get_timeseries"] * n_regions
                + ["tec_compute_series_stats"] * n_regions
                + ["tec_compare_stats"]
            )
            assert "tec_compute_series_stats" in sequence
            assert "tec_compare_stats" in sequence
            assert "tec_compare_regions" not in sequence

        if config["task_type"] == "report":
            for required_tool in [
                "tec_compute_series_stats",
                "tec_compute_high_threshold",
                "tec_detect_high_intervals",
                "tec_compute_stability_thresholds",
                "tec_detect_stable_intervals",
            ]:
                assert required_tool in sequence
            assert "tec_build_report" not in sequence

    print("Multi-agent five-task baseline smoke test finished successfully.")


if __name__ == "__main__":
    main()
