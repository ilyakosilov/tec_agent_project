# Experiment 6 Function-Handoff v2 Diagnosis

## Experiment Metadata

| Field | Value |
|---|---|
| Results directory | `outputs/metrics/real_runs/multi_agent/experiment_6_function_handoff_v2/` |
| Batch file | `qwen_multi_agent_function_handoff_v2_batch_colab.json` |
| Architecture | `qwen_multi_agent_function_handoff_full_llm` |
| Protocol | `function_handoff_v1` |
| Prompt revision | `function_handoff_grounded_state_v2` |
| Model | `Qwen/Qwen3.5-4B` |
| Model config | `float16`, no 4-bit, no 8-bit, `max_input_tokens=4096`, `max_new_tokens=512`, `temperature=0.0`, `do_sample=False` |

## Aggregate Results

| Metric | Value |
|---|---:|
| n_tasks | 5 |
| n_success | 0 |
| success_rate | 0.0 |
| overall_ok_rate | 0.0 |
| agent_success_rate | 0.0 |
| tool_sequence_match_rate | 0.4 |
| role_agent_order_match_rate | 0.0 |
| artifact_flow_valid_rate | 0.0 |
| required_role_agents_called_rate | 0.0 |
| avg_tool_call_count | 3.2 |
| avg_tool_error_count | 0.0 |
| avg_parse_error_count | 0.8 |
| avg_invalid_function_name_count | 0.0 |
| avg_forbidden_function_call_count | 0.0 |
| avg_repeated_tool_call_count | 2.0 |
| avg_invalid_artifact_handle_count | 0.0 |
| avg_repeated_role_message_count | 0.0 |
| avg_successful_final_tool_without_return_count | 0.8 |
| stalled_loop_rate | 1.0 |
| legacy_report_tool_used_rate | 0.0 |

## Verified Per-Task Trace Diagnosis

| Preset | Successful TEC tools | Roles called | Terminal numeric artifact | Final answer | Main failure |
|---|---|---|---|---|---|
| `hightec_midlat_europe` | `tec_get_timeseries`, `tec_compute_high_threshold`, `tec_detect_high_intervals` | `data_agent`, `math_agent` | yes | no | MathAgent repeated `tec_detect_high_intervals` after it had already succeeded. |
| `stable_midlat_europe` | `tec_get_timeseries`, `tec_compute_stability_thresholds`, `tec_detect_stable_intervals` | `data_agent`, `math_agent` | yes | no | MathAgent repeated `tec_detect_stable_intervals` after it had already succeeded. |
| `compare_midlat_europe_highlat_north` | `tec_get_timeseries`, `tec_get_timeseries`, `tec_compute_series_stats`, `tec_compute_series_stats` | `data_agent`, `math_agent` | no | no | MathAgent computed two stats but repeated the second stats call instead of returning or comparing. |
| `compare_three_regions` | `tec_get_timeseries` x3 | `data_agent` | no | no | DataAgent covered all three regions but repeated an already successful retrieval instead of returning. |
| `report_midlat_europe` | `tec_get_timeseries`, `tec_compute_series_stats`, `tec_compute_high_threshold` | `data_agent`, `math_agent` | no | no | MathAgent repeated high threshold computation; report workflow did not reach high/stable interval artifacts. |

## Confirmed Improvements Over Earlier Function-Handoff Run

- Initial routing is better: high, stable, compare-2, compare-3, and report start with `data_agent`.
- No invented artifact handles were observed in experiment 6 (`avg_invalid_artifact_handle_count = 0.0`).
- No forbidden function/tool calls were observed (`avg_forbidden_function_call_count = 0.0`).
- The unified `<tool_call>` protocol is mostly stable; parse errors are lower than early typed-role experiments.
- High TEC and stable interval tasks reached the correct terminal numerical artifacts and matched expected primitive tool sequences.

## Remaining Failure Classes

### Worker Does Not Return After Useful Artifact

The dominant failure is not tool schema or JSON protocol. It is assignment finalization inside worker roles:

- after `high_intervals`, MathAgent repeats `tec_detect_high_intervals`;
- after `stable_intervals`, MathAgent repeats `tec_detect_stable_intervals`;
- after per-region stats, MathAgent repeats a stats call;
- after all requested series are present, DataAgent repeats retrieval.

### State Visibility Is Not Diagnosable Enough

Experiment 6 JSON stores raw and cleaned model outputs, artifacts, tool observations, and counters, but it does not store per-call prompt/token diagnostics. Therefore, the trace confirms the repeated actions, but cannot prove whether the model saw the key runtime facts near the end of its prompt.

### Workflow Completion Is Ambiguous

For high and stable tasks, terminal numerical artifacts exist, but the workflow did not reach AnalysisAgent/ReportAgent. Future JSON needs explicit fields separating:

- terminal numeric artifact present;
- workflow completed;
- final answer present;
- analysis/report role called.

## Changes Implemented For v3

- Prompt revision updated to `function_handoff_grounded_completion_state_v3`.
- Next notebook: `notebooks/08_qwen_multi_agent_function_handoff_grounded_completion_v3_qwen35_4b_colab.ipynb`.
- Next output directory: `outputs/metrics/experiment_function_handoff_grounded_completion_v3_qwen35_4b/`.
- First v3 pilot is configured with `RUN_ALL_TASKS = False` and `SELECTED_PRESET = "hightec_midlat_europe"`.
- Worker state now carries role-local runtime facts:
  - successful TEC calls in the current role;
  - observations in the current role;
  - attempted TEC calls in the current role.
- DataAgent state no longer exposes `missing_regions`; it exposes requested regions, covered regions, and `scope_covered`.
- MathAgent prompt emphasizes that deliverables are outputs, not inputs, and that runtime-visible handles are the only valid inputs.
- Runtime records attempted/successful/skipped/failed TEC tool calls separately.
- Runtime records compact per-LLM-call diagnostics:
  - prompt character count;
  - prompt token count before/after truncation when tokenizer is available;
  - prompt truncation flag;
  - generated token count when tokenizer is available;
  - whether output cleaning was applied;
  - whether raw output had prefix text before `<tool_call>`.
- New counters were added for:
  - data retrieval repeated after requested series are already present;
  - MathAgent repeating after terminal artifacts;
  - MathAgent repeating intermediate computations;
  - MathAgent failing to return after assignment artifact is present;
  - MathAgent recomputing existing stats for a series;
  - equivalent orchestrator assignments without state change.

## What Is Intentionally Not Changed

- `role_response` is not reintroduced and is not converted into a tool.
- No separate `RoleAssignment`, `RoleAction`, or `<final_answer>` block is added.
- No runtime forced next role or forced next tool is added.
- No auto-completion of worker roles after artifacts appear is added.
- GoldRunner, expected tool sequences, metrics, verdicts, and evaluator missing goals are not passed to prompts/state.
- Historical untyped, typed, single-agent, deterministic baseline, and experiment 5/6 outputs are not overwritten.
