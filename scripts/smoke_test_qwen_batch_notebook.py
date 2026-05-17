"""Smoke checks for the Qwen single-agent Colab batch notebook.

This test does not run Qwen. It validates the notebook JSON and executes only
the lightweight CONFIG/helper cells that build queries, expected sequences, and
EvalTask objects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tec_agents.eval.task_set import EvalTask


NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_qwen_single_agent_colab.ipynb"


def _source(cell: dict) -> str:
    source = cell.get("source") or ""
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def main() -> None:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "<<<<<<<" not in text
    assert ">>>>>>>" not in text
    assert "=======" not in text

    data = json.loads(text)
    assert data["nbformat"] >= 4
    assert all(not cell.get("outputs") for cell in data["cells"])
    assert all(
        cell.get("execution_count") is None
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    )

    config_cell = next(
        cell for cell in data["cells"] if "TASK_CONFIGS = [" in _source(cell)
    )
    helper_cell = next(
        cell for cell in data["cells"] if "def build_query" in _source(cell)
    )

    namespace: dict[str, object] = {"EvalTask": EvalTask}
    exec(_source(config_cell), namespace)
    exec(_source(helper_cell), namespace)

    task_configs = namespace["TASK_CONFIGS"]
    assert isinstance(task_configs, list)
    assert len(task_configs) == 5

    seen_types = [config["task_type"] for config in task_configs]
    assert seen_types.count("high_tec") == 1
    assert seen_types.count("stable_intervals") == 1
    assert seen_types.count("compare_regions") == 2
    assert seen_types.count("report") == 1

    build_query = namespace["build_query"]
    build_expected_tool_sequence = namespace["build_expected_tool_sequence"]
    build_eval_task = namespace["build_eval_task"]

    for config in task_configs:
        query = build_query(config)
        sequence = build_expected_tool_sequence(config)
        task = build_eval_task(config, query, sequence)

        assert isinstance(query, str) and query
        assert isinstance(sequence, list) and sequence
        assert isinstance(task, EvalTask)
        assert "tec_build_report" not in sequence
        assert "tec_compare_regions" not in sequence

        if config["task_type"] == "compare_regions":
            n_regions = len(config["regions"])
            assert sequence == (
                ["tec_get_timeseries"] * n_regions
                + ["tec_compute_series_stats"] * n_regions
                + ["tec_compare_stats"]
            )

        if config["task_type"] == "report":
            assert sequence == [
                "tec_get_timeseries",
                "tec_compute_series_stats",
                "tec_compute_high_threshold",
                "tec_detect_high_intervals",
                "tec_compute_stability_thresholds",
                "tec_detect_stable_intervals",
            ]

    print("Qwen batch notebook smoke test finished successfully.")


if __name__ == "__main__":
    main()
