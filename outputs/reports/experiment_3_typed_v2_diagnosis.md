# Experiment 3 Typed v2 Diagnosis

## Experiment Metadata

The requested `outputs/metrics/experiment_3/` directory was not present locally. The uploaded real Colab run was found under:

`outputs/metrics/real_runs/multi_agent/experiment_3/`

The batch file is:

`outputs/metrics/real_runs/multi_agent/experiment_3/qwen_multi_agent_typed_v2_batch_colab.json`

| Field | Value |
|---|---|
| Architecture | `qwen_multi_agent_typed_full_llm` |
| Experiment ID | `qwen_multi_agent_typed_v2_batch_colab` |
| Prompt revision | `role_boundaries_tool_schemas_v2` |
| Typed protocol version | `typed_role_contract_v1` |
| Model | `Qwen/Qwen3.5-4B` |
| Dataset ref | `default` |
| Dataset path | `/content/tec_agent_project/data/processed/tec_regions_2024_01_01_to_2024_04_01_hourly.parquet` |
| Period | `[2024-03-01, 2024-04-01)` |
| Tasks | `hightec_midlat_europe`, `stable_midlat_europe`, `compare_midlat_europe_highlat_north`, `compare_three_regions`, `report_midlat_europe` |

## Aggregate Results

| Metric | Value |
|---|---:|
| `n_tasks` | 5 |
| `n_success` | 0 |
| `n_failed` | 5 |
| `success_rate` | 0.0 |
| `overall_ok_rate` | 0.0 |
| `agent_success_rate` | 0.0 |
| `avg_tool_call_count` | 2.8 |
| `avg_tool_error_count` | 0.2 |
| `avg_parse_error_count` | 2.0 |
| `avg_invalid_assignment_count` | 0.0 |
| `avg_invalid_role_response_count` | 0.0 |
| `avg_forbidden_tool_call_count` | 0.4 |
| `avg_premature_role_completion_count` | 0.0 |
| `avg_empty_findings_done_count` | 0.0 |
| `avg_repeated_equivalent_role_assignment_count` | 0.2 |
| `avg_tool_schema_validation_error_count` | 0.0 |
| `stalled_loop_rate` | 0.4 |
| `legacy_report_tool_used_rate` | 0.0 |
| `tool_sequence_match_rate` | null |
| `role_agent_order_match_rate` | null |
| `artifact_flow_valid_rate` | null |
| `required_role_agents_called_rate` | null |

The aggregate diagnosis from the prompt is confirmed. Process metrics such as tool sequence and role order became `null` at the batch level because many per-task metric dictionaries were empty when terminal numerical artifacts were absent.

## Comparison With Previous Typed Run

Previous typed outputs were available locally under:

`outputs/metrics/real_runs/multi_agent/experiment_2/qwen_multi_agent_typed_batch_colab.json`

| Metric | Typed v1 | Typed v2 / experiment 3 |
|---|---:|---:|
| `success_rate` | 0.0 | 0.0 |
| `avg_tool_call_count` | 3.2 | 2.8 |
| `avg_parse_error_count` | 2.4 | 2.0 |
| `avg_forbidden_tool_call_count` | 0.6 | 0.4 |
| `stalled_loop_rate` | 0.6 | 0.4 |
| `tool_sequence_match_rate` | 0.5 | null |
| `avg_tool_error_count` | null | 0.2 |

Typed v2 improved the handoff surface and reduced stalls/protocol violations, but it regressed or failed to expose process metrics for incomplete runs.

## Per-Task Trace Diagnosis

### `hightec_midlat_europe`

Expected primitive sequence:

`tec_get_timeseries -> tec_compute_high_threshold -> tec_detect_high_intervals`

Actual sequence:

`tec_get_timeseries -> tec_compute_series_stats -> tec_compute_high_threshold`

Verified from trace:

- DataAgent loaded `midlat_europe` and returned minimal `role_response done`.
- Orchestrator assigned MathAgent a broader objective requiring `stats_id`, `threshold_id`, and `high_intervals`.
- MathAgent computed `tec_compute_series_stats`, which was not needed for the simple high-TEC interval request.
- MathAgent computed a high threshold but did not call `tec_detect_high_intervals`.
- The role stalled after repeated successful high-threshold calls.
- No final answer was produced.

Conclusion: v2 made the simple high-TEC assignment too broad. The Orchestrator requested intermediate/statistical deliverables that were not necessary for the user request.

### `stable_midlat_europe`

Expected primitive sequence:

`tec_get_timeseries -> tec_compute_stability_thresholds -> tec_detect_stable_intervals`

Actual sequence:

`tec_get_timeseries -> tec_compute_series_stats`

Verified from trace:

- Orchestrator first called MathAgent before any series handle existed.
- The RoleAssignment claimed `available_input_types = ["series_id"]`, but runtime state had no `series_id`.
- MathAgent returned `cannot_complete`.
- DataAgent then loaded the `midlat_europe` series and returned `done`.
- MathAgent later computed stats, not stability thresholds or stable intervals.
- The same or equivalent MathAgent assignment was repeated once.
- No stable interval artifact and no final answer were produced.

Conclusion: Orchestrator-generated input availability is ungrounded. Runtime must be the only source of truth for available handles.

### `compare_midlat_europe_highlat_north`

Expected primitive sequence:

`tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats`

Actual sequence:

`tec_compute_series_stats -> tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats`

Verified from trace:

- Orchestrator first called MathAgent before loading data.
- MathAgent invented a handle: `midlat_europe_20240301_20240401`.
- Tool execution failed with `Unknown series_id`.
- Orchestrator then called DataAgent, which loaded both real series handles.
- MathAgent computed stats for only one real series.
- Subsequent output became malformed/nested protocol output before `tec_compare_stats`.
- No comparison artifact was produced.

Conclusion: the tool schema catalogue alone is not enough. Workers need a runtime-grounded input packet and a distinct `invalid_artifact_handle_count` diagnostic.

### `compare_three_regions`

Expected primitive sequence:

`tec_get_timeseries x3 -> tec_compute_series_stats x3 -> tec_compare_stats`

Actual sequence:

`tec_get_timeseries -> tec_get_timeseries -> tec_get_timeseries`

Verified from trace:

- DataAgent loaded series for all three requested regions.
- State snapshots after the third call contained all three series handles.
- DataAgent did not return `role_response done`.
- Raw output claimed `highlat_north` was still not retrieved and repeated a successful call.
- Runtime stopped the role on repeated identical successful tool call.
- MathAgent never received control.

Conclusion: multi-region progress must be placed in a compact, prominent assignment-status block before the full JSON state.

### `report_midlat_europe`

Expected primitive sequence:

`tec_get_timeseries -> tec_compute_series_stats -> tec_compute_high_threshold -> tec_detect_high_intervals -> tec_compute_stability_thresholds -> tec_detect_stable_intervals`

Actual sequence:

`tec_get_timeseries -> tec_compute_series_stats`

Verified from trace:

- `parsed_task.task_type` was correctly `report`.
- DataAgent loaded the series and returned `done`.
- MathAgent computed basic statistics.
- High-TEC and stable-interval artifacts were not produced.
- MathAgent then produced malformed/nested tool-call syntax.
- No final report was produced.

Conclusion: report classification is fixed, but MathAgent still struggles with long multi-step mathematical assignments and one-block-per-turn discipline.

## Verified Failure Classes

| Failure class | Verified? | Evidence |
|---|---|---|
| Hallucinated available inputs | Yes | Stable and compare-2 assignments claimed `series_id` before data existed. |
| Deliverables/prerequisites confusion | Yes | MathAgent treated requested outputs as already-required input handles. |
| Invented handles | Yes | Compare-2 used `midlat_europe_20240301_20240401`. |
| Multi-block output | Yes | Compare/report raw outputs contained multiple `<tool_call>` blocks; only the first was cleaned/executed. |
| Repeated successful calls | Yes | High repeated high threshold; compare-three repeated `tec_get_timeseries` for an already loaded region. |
| Lost process metrics | Yes | Several per-task `metrics` objects were empty, so aggregate process rates became `null`. |

## Prompt/State Size Check

Approximate token estimates used `characters / 4`; no Qwen tokenizer or model was loaded.

| Message | Chars | Approx tokens |
|---|---:|---:|
| v2 Orchestrator system prompt | 6002 | 1500 |
| v2 DataAgent system prompt | 4112 | 1028 |
| v2 MathAgent system prompt | 5770 | 1442 |
| v2 AnalysisAgent system prompt | 3090 | 772 |
| v2 ReportAgent system prompt | 2513 | 628 |
| Realistic compare-three DataAgent state after 3 loads | 4801 | 1200 |

The realistic DataAgent case is probably inside the 4096-token input limit with its system prompt, but the prompt/state ordering was poor: the scope-covered fact was buried in JSON after other text. v3 moves assignment status and visible inputs to the top of the worker state message and trims recent history windows.

## Changes Implemented For v3

- Split LLM-owned deliverables from runtime-owned inputs.
- Added `deliverables_to_produce` to `RoleAssignment`.
- Kept `available_input_types` backward-compatible but removed it from worker-visible assignment state.
- Added runtime-grounded `available_input_artifacts`.
- Added stable `request_context` derived from parsed user query.
- Added minimal sufficient deliverables rule to Orchestrator prompt.
- Added compact high-priority `CURRENT ASSIGNMENT STATUS` worker message.
- Added one-turn/one-action rule and `multiple_protocol_blocks_in_single_output_count`.
- Added invalid artifact handle validation and `invalid_artifact_handle_count`.
- Added compact invalid-handle feedback without next-tool or next-role hints.
- Updated metrics so process/protocol metrics are computed even when terminal numerical artifacts are absent.
- Prepared notebook outputs for `outputs/metrics/experiment_4/`.

## What Is Intentionally Not Changed

- `role_response` remains a protocol block, not a tool.
- Runtime does not force next-role routing.
- Runtime does not force next-tool routing.
- Runtime does not auto-complete roles.
- Runtime does not inject expected tool sequence, expected role order, GoldRunner output, metrics, verdicts, missing-goal artifacts, evaluator remaining goals, or deterministic baseline trace into prompts/state.
- Historical untyped full LLM multi-agent experiment is not modified.
- Deterministic rule-based multi-agent baseline is not modified.
- Single-agent behavior is not modified in this v3 step.

