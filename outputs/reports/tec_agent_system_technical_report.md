# TEC Agent System Technical Report

Generated from local repository state on 2026-05-22 23:35:35.

This markdown is a structured source document for a later HTML report with charts, expandable sections, architecture cards, and comparison tables.

## 1. Краткое резюме проекта

TEC Agent Project is an experimental framework for comparing LLM agent orchestration patterns on ionospheric Total Electron Content (TEC) analysis tasks. TEC is treated as a regional hourly time-series signal. Agents must load regional data, compute thresholds and statistics, detect high or stable intervals, compare regions, and compose reports using deterministic tools. The scientific focus is orchestration under equal data/tools/evaluation rather than raw model quality alone.

The current comparison has three layers: Qwen single-agent, deterministic rule-based multi-agent baseline, and Qwen full LLM multi-agent. GoldRunner is the numerical oracle used only after agent execution. The rule-based multi-agent baseline is the ideal scripted role-orchestration baseline. The full LLM multi-agent network is the real experiment where the orchestrator and all worker roles are model-driven.

## 2. Структура репозитория

| Путь | Назначение | Ключевые файлы |
| --- | --- | --- |
| notebooks/ | Colab/research workflows for dataset building and Qwen experiments | 01_build_tec_dataset.ipynb; 02_qwen_single_agent_colab.ipynb; 03_qwen_multi_agent_colab.ipynb |
| src/tec_agents/agents/ | Agent orchestration implementations | single_agent.py; multi_agent.py; llm_single_agent.py; llm_multi_agent.py; protocol.py |
| src/tec_agents/tools/ | Deterministic TEC tools, schemas, registry and executor | tec_tools.py; schemas.py; registry.py; executor.py |
| src/tec_agents/mcp/ | Local in-process MCP-like client/server layer | client.py; server.py |
| src/tec_agents/llm/ | Qwen wrapper, prompts and textual parser | local_qwen.py; prompts.py; tool_call_parser.py |
| src/tec_agents/eval/ | EvalTask, five-task configs, GoldRunner, metrics, experiment runner | five_task_configs.py; task_set.py; gold_runner.py; metrics.py; experiment_runner.py |
| src/tec_agents/data/ | Dataset registry, date helpers, regions, path helpers | datasets.py; dates.py; regions.py; paths.py |
| scripts/ | Smoke tests, deterministic runners, architecture comparison scripts | smoke_test_*.py; run_five_task_multi_agent_baseline.py; compare_five_task_architectures.py |
| outputs/metrics/ | Local experiment JSON outputs | qwen_single_agent_batch_colab.json; multi_agent_rule_based_five_task_batch.json; qwen_multi_agent_batch_colab.json |
| outputs/reports/ | Generated technical reports and future HTML assets | tec_agent_system_technical_report.md; assets/*.json |
| data/processed/ | Processed TEC regional datasets | tec_regions_2024_01_01_to_2024_04_01_hourly.parquet |

## 3. Данные

Processed datasets contain hourly regional TEC values. Five-task experiments use `dataset_ref = default`, registered at runtime to a processed parquet path. Time intervals use `[start, end)`: start is included and end is excluded. For March 2024 this is `[2024-03-01, 2024-04-01)`, so hourly data should contain 744 points; local outputs confirm `expected_n_points = 744`.

| Dataset path | Rows | Columns/regions | Start | End | Frequency |
| --- | --- | --- | --- | --- | --- |
| data\processed\tec_regions_2024_01_01_to_2024_04_01_hourly.parquet | 2184 | equatorial_atlantic, equatorial_africa, equatorial_pacific, midlat_europe, midlat_usa, midlat_asia, midlat_south_america, midlat_australia, highlat_north, highlat_south | 2024-01-01 00:00:00 | 2024-03-31 23:00:00 | 0 days 01:00:00 |
| data\processed\tec_regions_2024_03_hourly.parquet | 745 | equatorial_atlantic, equatorial_africa, equatorial_pacific, midlat_europe, midlat_usa, midlat_asia, midlat_south_america, midlat_australia, highlat_north, highlat_south | 2024-03-01 00:00:00 | 2024-04-01 00:00:00 | 0 days 01:00:00 |
| data\processed\tec_regions_2024_03_hourly.csv | 745 | equatorial_atlantic, equatorial_africa, equatorial_pacific, midlat_europe, midlat_usa, midlat_asia, midlat_south_america, midlat_australia, highlat_north, highlat_south | 2024-03-01 00:00:00 | 2024-04-01 00:00:00 | 0 days 01:00:00 |

## 4. MCP / tool execution layer

The local MCP-like layer provides the same core idea as MCP without requiring a remote server. Agents emit textual protocol blocks, the parser extracts structured calls, `LocalMCPClient` calls `LocalMCPServer`, and `ToolExecutor` validates schemas, runs deterministic tools, stores artifacts in `ToolStore`, and records a trace.

```text
User query -> Agent -> tool_call text -> parser -> MCP client/server/executor -> TEC tool -> artifact store -> observation/result -> Agent
```

Tools return handles and compact structured artifacts rather than raw arrays. Important handles are `series_id`, `threshold_id`, `stats_id`, and `comparison_id`.

## 5. Инструменты / tools

| Tool | Назначение | Основные аргументы | Возвращаемые artifacts | Используется в сценариях |
| --- | --- | --- | --- | --- |
| tec_compare_regions | Compare aggregated TEC statistics across two or more predefined regions for the same time interval. | dataset_ref, region_ids, start, end, freq, metrics | dataset_ref, start, end, stats | legacy/aggregate compare; not expected primitive workflow |
| tec_compare_stats | Compare two or more previously computed TEC statistics handles and return structured pairwise metric deltas. | stats_ids, reference_stats_id, metrics | comparison_id, stats_ids, reference_stats_id, regions, items, pairwise_deltas | compare_regions |
| tec_compute_high_threshold | Compute a high-TEC threshold for a stored time series. Usually use method='quantile' and q=0.9 unless the user explicitly asks for another threshold. | series_id, method, q, value | threshold_id, series_id, method, q, value, n_points_used | high_tec, report |
| tec_compute_series_stats | Compute selected deterministic statistics for a previously loaded TEC time series and return a stats_id handle. | series_id, metrics | stats_id, series_id, region_id, n_points, finite_points, metrics | compare_regions, report |
| tec_compute_stability_thresholds | Compute rolling-window thresholds for stable interval detection using quantiles of rolling TEC variability. | series_id, window_minutes, method, q_delta, q_std | threshold_id, series_id, method, window_minutes, q_delta, q_std, max_delta_threshold, rolling_std_threshold, estimated_step_minutes, window_points, n_points, max_abs_delta, max_std, n_windows_used | stable_intervals, report |
| tec_detect_high_intervals | Detect intervals where TEC is greater than or equal to a previously computed high-TEC threshold. | series_id, threshold_id, min_duration_minutes, merge_gap_minutes | series_id, threshold_id, threshold_value, n_intervals, intervals | high_tec, report |
| tec_detect_stable_intervals | Detect stable TEC intervals using previously computed stability thresholds. | series_id, threshold_id, min_duration_minutes, merge_gap_minutes | series_id, threshold_id, n_intervals, intervals | stable_intervals, report |
| tec_find_stable_intervals_direct | Detect stable TEC intervals using explicit variability limits without a separate threshold computation step. | series_id, window_minutes, max_delta, max_abs_delta, max_std, min_duration_minutes, merge_gap_minutes | series_id, threshold_id, n_intervals, intervals | direct stable helper; not expected five-task sequence |
| tec_get_timeseries | Load a TEC time series for one predefined geographic region and a half-open time interval [start, end). | dataset_ref, region_id, start, end, freq | series_id, metadata | all scenarios |
| tec_series_profile | Return descriptive statistics for a previously loaded TEC time series. | series_id | series_id, n_points, finite_points, start, end, min_value, max_value, mean_value, median_value, std_value, q10, q25, q75, q90 | optional profiling; sometimes attempted by LLM roles |

`tec_build_report` is absent and should remain absent. `tec_compare_regions` exists as an aggregate helper but is not part of the expected primitive workflow. Compare scenarios should use `tec_get_timeseries -> tec_compute_series_stats -> tec_compare_stats`.

## 6. Пять тестовых задач

The five-task stand is defined in `src/tec_agents/eval/five_task_configs.py` and mirrors the Qwen single-agent batch notebook.

| Preset ID | Task type | Natural-language query | Regions | Period | Expected primitive tool sequence |
| --- | --- | --- | --- | --- | --- |
| hightec_midlat_europe | high_tec | Find high TEC intervals for midlat_europe from 2024-03-01 to 2024-04-01 using q=0.90 threshold. | midlat_europe | [2024-03-01, 2024-04-01) | tec_get_timeseries -> tec_compute_high_threshold -> tec_detect_high_intervals |
| stable_midlat_europe | stable_intervals | Find stable TEC intervals for midlat_europe from 2024-03-01 to 2024-04-01 using the configured stability parameters. | midlat_europe | [2024-03-01, 2024-04-01) | tec_get_timeseries -> tec_compute_stability_thresholds -> tec_detect_stable_intervals |
| compare_midlat_europe_highlat_north | compare_regions | Compare TEC statistics for midlat_europe and highlat_north from 2024-03-01 to 2024-04-01. | midlat_europe, highlat_north | [2024-03-01, 2024-04-01) | tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats |
| compare_three_regions | compare_regions | Compare TEC statistics for equatorial_atlantic, midlat_europe, and highlat_north from 2024-03-01 to 2024-04-01. | equatorial_atlantic, midlat_europe, highlat_north | [2024-03-01, 2024-04-01) | tec_get_timeseries -> tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats |
| report_midlat_europe | report | Build a concise TEC analysis report for midlat_europe from 2024-03-01 to 2024-04-01. Include basic statistics, high TEC intervals, stable intervals, and a short interpretation based only on computed artifacts. | midlat_europe | [2024-03-01, 2024-04-01) | tec_get_timeseries -> tec_compute_series_stats -> tec_compute_high_threshold -> tec_detect_high_intervals -> tec_compute_stability_thresholds -> tec_detect_stable_intervals |

The tasks cover a linear high-TEC chain, a linear stable-interval chain, two-region compare, three-region multi-artifact stress, and one compound report scenario.

## 7. GoldRunner

GoldRunner is the deterministic numerical oracle. It computes the expected structured artifacts for an `EvalTask` using the same primitive tools. It is not an agent, is not visible to prompts, and runs only after the agent for evaluation.

| Task type | Gold artifacts | Gold metrics checked |
| --- | --- | --- |
| high_tec | series, high threshold, high intervals | threshold_abs_error, interval_count_error, global_peak_abs_error, top interval dates |
| stable_intervals | series, stability thresholds, stable intervals | stable_interval_count_error, top stable interval fields |
| compare_regions | series per region, stats per region, compare_stats result | region_set_match, mean/max/p90 errors, pairwise deltas |
| report | basic stats, high TEC section, stable intervals section | required sections/artifacts, high/stable interval count errors, grounding |

## 8. Single-agent architecture

Qwen single-agent is implemented in `src/tec_agents/agents/llm_single_agent.py` and run by `notebooks/02_qwen_single_agent_colab.ipynb`. It receives a single-agent system prompt, user query, compact observations, available artifacts, and completed calls. It emits `<tool_call>` or `<final_answer>` blocks. Runtime cleaning/parsing calls tools through the MCP-like client and logs raw/cleaned outputs, parse counters, repeated calls, stalled loops, and state snapshots.

The single-agent does not receive `expected_tool_sequence`, GoldRunner results, `missing_goal_artifacts`, correct next-tool calls, metrics, or verdict checks. Missing artifacts can remain internal diagnostics but are not prompt-visible.

## 9. Deterministic rule-based multi-agent baseline

`src/tec_agents/agents/multi_agent.py` implements a deterministic role-based workflow. It is not an LLM multi-agent; it is the scripted role-orchestration oracle. `scripts/run_five_task_multi_agent_baseline.py` runs it on the same five configs and saves per-task plus batch JSON outputs.

| Role | Responsibility | Tool access | Output |
| --- | --- | --- | --- |
| orchestrator | Parse/coordinate task and role transitions | none | workflow plan / role decisions |
| data_agent | Load TEC series for requested regions and interval | tec_get_timeseries, optional tec_series_profile | series artifacts |
| math_agent | Compute stats, thresholds, intervals, comparisons | math primitive tools | stats/high/stable/compare artifacts |
| analysis_agent | Transform artifacts into structured findings | none | findings |
| report_agent | Format final answer from artifacts and findings | none | final answer |

GoldRunner is the numerical oracle. Rule-based multi-agent is the orchestration oracle. Qwen full LLM multi-agent is the real LLM role-network experiment.

## 10. Full LLM multi-agent architecture

The full LLM runner is in `src/tec_agents/agents/llm_multi_agent.py` and is used by `notebooks/03_qwen_multi_agent_colab.ipynb`. All roles are LLM-driven: `LLMOrchestratorAgent`, `LLMDataAgent`, `LLMMathAgent`, `LLMAnalysisAgent`, `LLMReportAgent`, coordinated by `LLMFullMultiAgent`. Runtime validates protocol/tool permissions and executes tools, but role and tool decisions come from model output.

| LLM Role | Responsibility | Allowed outputs | Allowed tools | Forbidden actions |
| --- | --- | --- | --- | --- |
| LLMOrchestratorAgent | Choose which role to call or finish from visible state | role_action | none | tool_call, role_response, final_answer |
| LLMDataAgent | Load time series data artifacts | tool_call or role_response | tec_get_timeseries, tec_series_profile only if useful | compute math, call roles as tools, final_answer |
| LLMMathAgent | Compute numerical artifacts | tool_call or role_response | tec_compute_series_stats; tec_compute_high_threshold; tec_detect_high_intervals; tec_compute_stability_thresholds; tec_detect_stable_intervals; tec_compare_stats | data loading, final answer, aggregate compare_regions |
| LLMAnalysisAgent | Produce findings from artifacts | role_response | none | any tool_call, report writing |
| LLMReportAgent | Produce final answer | final_answer or cannot_complete role_response | none | any tool_call, role_action, fabricated numbers |

Latest protocol hardening clarifies that agents are not tools, `role_response` is not a tool, AnalysisAgent/ReportAgent cannot call tools, and DataAgent/MathAgent should return `role_response` after their part is complete. Full LLM metadata uses `full_llm_multi_agent` / `full_llm_role_workflow`, not `role_based_workflow`.

## 11. Protocol blocks

| Block | Who can output | Purpose | Example |
| --- | --- | --- | --- |
| role_action | LLMOrchestratorAgent | Call a role or finish workflow | <role_action>{"action":"call_role","role":"data_agent","message":"retrieve data"}</role_action> |
| tool_call | DataAgent, MathAgent only | Request an allowed TEC tool | <tool_call>{"name":"tec_get_timeseries","arguments":{"region_id":"midlat_europe"}}</tool_call> |
| role_response | DataAgent, MathAgent, AnalysisAgent; ReportAgent for cannot_complete | Return role status/artifacts/findings | <role_response>{"status":"done","message":"data ready","artifacts":{},"findings":[]}</role_response> |
| final_answer | ReportAgent | Final user-facing answer | <final_answer>{"answer":"..."}</final_answer> |

```xml
<role_action>
{"action": "call_role", "role": "data_agent", "message": "retrieve relevant data"}
</role_action>
```

```xml
<tool_call>
{"name": "tec_get_timeseries", "arguments": {"dataset_ref": "default", "region_id": "midlat_europe", "start": "2024-03-01", "end": "2024-04-01"}}
</tool_call>
```

```xml
<role_response>
{"status": "done", "message": "data ready", "artifacts": {}, "findings": []}
</role_response>
```

```xml
<final_answer>
{"answer": "Concise TEC report based only on computed artifacts."}
</final_answer>
```

Protocol invariants: role names cannot be tool names; `role_response` cannot be a tool name; expected sequences, gold outputs, deterministic traces, metrics, and missing-goal hints are not passed to prompts.

## 12. Метрики

| Metric | Applies to | Meaning | Good value |
| --- | --- | --- | --- |
| success / agent_success | all | Agent runner completed according to its own success flag | true |
| overall_ok | all batch records | Notebook/script verdict checks passed | true |
| parse_error_count | LLM agents | Model outputs needing parser repair | 0 |
| invalid_json_count | LLM agents | Invalid JSON inside protocol block | 0 |
| invalid_tool_name_count | tool agents | Tool name not registered | 0 |
| invalid_role_protocol_count | full LLM multi-agent | Role emitted action not allowed by its protocol | 0 |
| forbidden_tool_call_count | full LLM multi-agent | Role attempted a forbidden tool | 0 |
| repeated_tool_call_count | LLM agents | Identical successful tool call repeated | 0 |
| stalled_loop_detected | LLM agents | Runtime stopped because progress stalled | false |
| tool_sequence_match | all | Expected primitive tool chain matched | true |
| role_agent_order_match | multi-agent | Observed role order equals expected role order | true |
| required_role_agents_called | multi-agent | All required roles participated | true |
| artifact_flow_valid | multi-agent | Data/math/analysis/report artifacts are present | true |
| threshold_abs_error | high_tec | Threshold difference versus GoldRunner | 0 |
| interval_count_error | high_tec | High interval count difference | 0 |
| global_peak_abs_error | high_tec | Peak TEC value difference | 0 |
| stable_interval_count_error | stable_intervals | Stable interval count difference | 0 |
| top_stable_duration_abs_error | stable_intervals | Top stable interval duration difference | 0 |
| mean_abs_error_avg / p90_abs_error_avg | compare_regions | Average per-region numerical errors | 0 |
| pairwise_delta_count_match | compare_regions | Pairwise comparison count matches | true |
| required_artifacts_present | report | Required report artifacts exist | true |
| report_grounded_in_artifacts | report | Report can be grounded in computed artifacts | true |

## 13. Current experiment outputs

Key local batch files are present: `qwen_single_agent_batch_colab.json`, `multi_agent_rule_based_five_task_batch.json`, and `qwen_multi_agent_batch_colab.json`.

| File | Kind | Size bytes |
| --- | --- | --- |
| outputs\metrics\multi_agent_rule_based_compare_midlat_europe_highlat_north_five_task.json | Rule-based multi-agent | 59280 |
| outputs\metrics\multi_agent_rule_based_compare_three_regions_five_task.json | Rule-based multi-agent | 83505 |
| outputs\metrics\multi_agent_rule_based_five_task_batch.json | Rule-based multi-agent | 692237 |
| outputs\metrics\multi_agent_rule_based_hightec_midlat_europe_five_task.json | Rule-based multi-agent | 56706 |
| outputs\metrics\multi_agent_rule_based_report_midlat_europe_five_task.json | Rule-based multi-agent | 231026 |
| outputs\metrics\multi_agent_rule_based_stable_midlat_europe_five_task.json | Rule-based multi-agent | 196430 |
| outputs\metrics\qwen_multi_agent_batch_colab.json | Qwen full LLM multi-agent | 263742 |
| outputs\metrics\qwen_multi_agent_compare_midlat_europe_highlat_north_colab.json | Qwen full LLM multi-agent | 36605 |
| outputs\metrics\qwen_multi_agent_compare_three_regions_colab.json | Qwen full LLM multi-agent | 40875 |
| outputs\metrics\qwen_multi_agent_hightec_midlat_europe_colab.json | Qwen full LLM multi-agent | 29430 |
| outputs\metrics\qwen_multi_agent_report_midlat_europe_colab.json | Qwen full LLM multi-agent | 64037 |
| outputs\metrics\qwen_multi_agent_stable_midlat_europe_colab.json | Qwen full LLM multi-agent | 69565 |
| outputs\metrics\qwen_single_agent_batch_colab.json | Qwen single-agent | 665166 |
| outputs\metrics\qwen_single_agent_compare_midlat_europe_highlat_north_colab.json | Qwen single-agent | 47403 |
| outputs\metrics\qwen_single_agent_compare_three_regions_colab.json | Qwen single-agent | 41065 |
| outputs\metrics\qwen_single_agent_hightec_midlat_europe_colab.json | Qwen single-agent | 55117 |
| outputs\metrics\qwen_single_agent_report_midlat_europe_colab.json | Qwen single-agent | 230740 |
| outputs\metrics\qwen_single_agent_smoke_colab.json | Qwen single-agent | 60512 |
| outputs\metrics\qwen_single_agent_smoke_colab_v2.json | Qwen single-agent | 10774 |
| outputs\metrics\qwen_single_agent_smoke_colab_v3.json | Qwen single-agent | 52762 |
| outputs\metrics\qwen_single_agent_smoke_colab_v4_date_checked.json | Qwen single-agent | 59383 |
| outputs\metrics\qwen_single_agent_smoke_colab_v5_date_checked.json | Qwen single-agent | 47481 |
| outputs\metrics\qwen_single_agent_stable_midlat_europe_colab.json | Qwen single-agent | 227563 |
| outputs\metrics\real_multi_agent_rule_based_march_2024.json | Real deterministic multi-agent experiment | 1922906 |
| outputs\metrics\real_single_agent_rule_based_march_2024.json | Real deterministic single-agent experiment | 1793641 |
| outputs\metrics\smoke_single_agent_rule_based.json | Smoke output | 127427 |

## 14. Сводная таблица результатов

### A. Aggregate summary by architecture

| Architecture | Model | n_tasks | success_rate | overall_ok_rate | tool_sequence_match_rate | role_order_match_rate | artifact_flow_valid_rate | stalled_loop_rate | forbidden_tool_call_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen single-agent | Qwen/Qwen3.5-4B | 5 | 40% | 0% | 60% | n/a | n/a | 60% | n/a |
| Rule-based multi-agent | none | 5 | 100% | 100% | 100% | 100% | 100% | 0% | n/a |
| Qwen full LLM multi-agent | Qwen/Qwen3.5-4B | 5 | 0% | 0% | 0% | 0% | 0% | 100% | 100% |

### B. Per-task comparison

| Preset | Task type | Qwen single-agent result | Rule-based multi-agent result | Qwen full LLM multi-agent result | Key difference |
| --- | --- | --- | --- | --- | --- |
| hightec_midlat_europe | high_tec | success=True, overall=False, seq=True, stall=False | success=True, overall=True, seq=True, stall=None | success=False, overall=False, seq=None, stall=True | rule workflow succeeds; full LLM role protocol fails; single-agent computes the simple chain but strict verdict may count parse repairs |
| stable_midlat_europe | stable_intervals | success=True, overall=False, seq=True, stall=False | success=True, overall=True, seq=True, stall=None | success=False, overall=False, seq=None, stall=True | rule workflow succeeds; full LLM role protocol fails; single-agent computes the simple chain but strict verdict may count parse repairs |
| compare_midlat_europe_highlat_north | compare_regions | success=False, overall=False, seq=True, stall=True | success=True, overall=True, seq=True, stall=None | success=False, overall=False, seq=None, stall=True | rule workflow succeeds; full LLM role protocol fails; single-agent struggles with multi-artifact planning/finalization |
| compare_three_regions | compare_regions | success=False, overall=False, seq=None, stall=True | success=True, overall=True, seq=True, stall=None | success=False, overall=False, seq=None, stall=True | rule workflow succeeds; full LLM role protocol fails; single-agent struggles with multi-artifact planning/finalization |
| report_midlat_europe | report | success=False, overall=False, seq=None, stall=True | success=True, overall=True, seq=True, stall=None | success=False, overall=False, seq=None, stall=True | rule workflow succeeds; full LLM role protocol fails; single-agent struggles with multi-artifact planning/finalization |

### C. Tool sequence comparison

| Preset | Expected sequence | Qwen single actual | Rule multi actual | Full LLM multi actual |
| --- | --- | --- | --- | --- |
| hightec_midlat_europe | tec_get_timeseries -> tec_compute_high_threshold -> tec_detect_high_intervals | tec_get_timeseries -> tec_compute_high_threshold -> tec_detect_high_intervals | tec_get_timeseries -> tec_compute_high_threshold -> tec_detect_high_intervals | tec_get_timeseries |
| stable_midlat_europe | tec_get_timeseries -> tec_compute_stability_thresholds -> tec_detect_stable_intervals | tec_get_timeseries -> tec_compute_stability_thresholds -> tec_detect_stable_intervals | tec_get_timeseries -> tec_compute_stability_thresholds -> tec_detect_stable_intervals | tec_get_timeseries -> tec_series_profile |
| compare_midlat_europe_highlat_north | tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats | tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats | tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats | tec_compute_series_stats -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compute_series_stats |
| compare_three_regions | tec_get_timeseries -> tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats | tec_get_timeseries -> tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats | tec_get_timeseries -> tec_get_timeseries -> tec_get_timeseries -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compare_stats | tec_compute_series_stats -> tec_compute_series_stats -> tec_compute_series_stats -> tec_compute_series_stats |
| report_midlat_europe | tec_get_timeseries -> tec_compute_series_stats -> tec_compute_high_threshold -> tec_detect_high_intervals -> tec_compute_stability_thresholds -> tec_detect_stable_intervals | tec_get_timeseries -> tec_compute_series_stats -> tec_compute_stability_thresholds -> tec_detect_stable_intervals -> tec_compute_high_threshold | tec_get_timeseries -> tec_compute_series_stats -> tec_compute_high_threshold -> tec_detect_high_intervals -> tec_compute_stability_thresholds -> tec_detect_stable_intervals | tec_get_timeseries -> tec_series_profile |

### D. Role/protocol comparison

| Preset | Rule role order | Full LLM role order | Full LLM protocol errors | Full LLM forbidden tools | Full LLM stalled? | Full LLM error |
| --- | --- | --- | --- | --- | --- | --- |
| hightec_midlat_europe | orchestrator -> data_agent -> math_agent -> analysis_agent -> report_agent | data_agent | 3 | 3 | true | data_agent failed: Exceeded max_role_steps=8. |
| stable_midlat_europe | orchestrator -> data_agent -> math_agent -> analysis_agent -> report_agent | data_agent | 4 | 4 | true | data_agent failed: Exceeded max_role_steps=8. |
| compare_midlat_europe_highlat_north | orchestrator -> data_agent -> math_agent -> analysis_agent -> report_agent | math_agent | 4 | 4 | true | math_agent failed: Exceeded max_role_steps=8. |
| compare_three_regions | orchestrator -> data_agent -> math_agent -> analysis_agent -> report_agent | math_agent | 4 | 4 | true | math_agent failed: Exceeded max_role_steps=8. |
| report_midlat_europe | orchestrator -> data_agent -> math_agent -> analysis_agent -> report_agent | data_agent | 4 | 4 | true | data_agent failed: Exceeded max_role_steps=8. |

### E. Numeric metric comparison

| Preset | Metric | Gold / expected | Qwen single-agent error | Rule multi-agent error | Full LLM multi-agent error/status |
| --- | --- | --- | --- | --- | --- |
| hightec_midlat_europe | threshold_abs_error | 0 error / true match | 0 | 0 | data_agent failed: Exceeded max_role_steps=8. |
| hightec_midlat_europe | interval_count_error | 0 error / true match | 0 | 0 | data_agent failed: Exceeded max_role_steps=8. |
| hightec_midlat_europe | global_peak_abs_error | 0 error / true match | 0 | 0 | data_agent failed: Exceeded max_role_steps=8. |
| stable_midlat_europe | stable_interval_count_error | 0 error / true match | 0 | 0 | data_agent failed: Exceeded max_role_steps=8. |
| stable_midlat_europe | top_stable_duration_abs_error | 0 error / true match | 0 | 0 | data_agent failed: Exceeded max_role_steps=8. |
| compare_midlat_europe_highlat_north | mean_abs_error_avg | 0 error / true match | 0 | 0 | math_agent failed: Exceeded max_role_steps=8. |
| compare_midlat_europe_highlat_north | p90_abs_error_avg | 0 error / true match | 0 | 0 | math_agent failed: Exceeded max_role_steps=8. |
| compare_midlat_europe_highlat_north | pairwise_delta_count_match | 0 error / true match | true | true | math_agent failed: Exceeded max_role_steps=8. |
| compare_three_regions | mean_abs_error_avg | 0 error / true match | n/a | 0 | math_agent failed: Exceeded max_role_steps=8. |
| compare_three_regions | p90_abs_error_avg | 0 error / true match | n/a | 0 | math_agent failed: Exceeded max_role_steps=8. |
| compare_three_regions | pairwise_delta_count_match | 0 error / true match | n/a | true | math_agent failed: Exceeded max_role_steps=8. |
| report_midlat_europe | required_artifacts_present | 0 error / true match | n/a | true | data_agent failed: Exceeded max_role_steps=8. |
| report_midlat_europe | report_grounded_in_artifacts | 0 error / true match | n/a | true | data_agent failed: Exceeded max_role_steps=8. |
| report_midlat_europe | report_high_tec_interval_count_error_avg | 0 error / true match | n/a | 0 | data_agent failed: Exceeded max_role_steps=8. |
| report_midlat_europe | report_stable_interval_count_error_avg | 0 error / true match | n/a | 0 | data_agent failed: Exceeded max_role_steps=8. |

## 15. Текущие результаты: интерпретация

- Qwen single-agent succeeds on the simple high-TEC and stable-interval tasks, with correct numerical artifacts, but strict `overall_ok` remains false where parse repairs are counted. It reaches the right primitive sequence for two-region compare but stalls before final answer. Three-region compare and report fail on multi-artifact planning/finalization.
- Rule-based multi-agent passes all five tasks with exact tool sequences, valid role order, valid artifact flow, and zero/equivalent numerical errors. This validates tools, data, GoldRunner, metrics, and the role decomposition.
- Qwen full LLM multi-agent currently fails all five local tasks. The batch shows stalled loops and forbidden/protocol violations. Failures are concentrated in role protocol and role selection, not in numerical tools or GoldRunner.
- This should not be read as a general statement that multi-agent is worse. It means the current full LLM role network has not yet learned to hold the protocol, while the deterministic role baseline proves the architecture is computationally viable.

## 16. Графики и визуализации

Suggested charts for HTML report:

1. Bar chart: success_rate by architecture.
2. Bar chart: overall_ok_rate by architecture.
3. Bar chart: stalled_loop_rate by architecture.
4. Grouped bar: per-task success by architecture.
5. Grouped bar: parse/protocol/forbidden errors by architecture.
6. Scatter or line: tool_call_count per task by architecture.
7. Heatmap/table: expected vs actual tool sequence match.
8. Role flow diagram for rule-based multi-agent.
9. Role failure diagram for full LLM multi-agent.
10. TEC timeseries chart for midlat_europe with high threshold, using dataset and high_tec outputs.
11. Stable intervals overlay, using interval records from stable outputs.

No heavy plots were generated here. The JSON assets below are ready for future chart generation.

## 17. Assets for future HTML

- [`outputs/reports/assets/architecture_summary_table.json`](assets/architecture_summary_table.json)
- [`outputs/reports/assets/per_task_comparison_table.json`](assets/per_task_comparison_table.json)
- [`outputs/reports/assets/tool_sequence_comparison_table.json`](assets/tool_sequence_comparison_table.json)
- [`outputs/reports/assets/role_protocol_errors_table.json`](assets/role_protocol_errors_table.json)
- [`outputs/reports/assets/numeric_metrics_comparison_table.json`](assets/numeric_metrics_comparison_table.json)
- [`outputs/reports/assets/dataset_table.json`](assets/dataset_table.json)
- [`outputs/reports/assets/tools_table.json`](assets/tools_table.json)
- [`outputs/reports/assets/five_task_table.json`](assets/five_task_table.json)
- [`outputs/reports/assets/outputs_inventory_table.json`](assets/outputs_inventory_table.json)
- [`outputs/reports/tec_agent_system_technical_report_data.json`](tec_agent_system_technical_report_data.json)

## 18. Проверки

This report was generated by reading source files, notebooks, processed data metadata, and local JSON outputs. No Qwen/LLM inference was run. Code, notebooks, tools, agents, GoldRunner, metrics, datasets, and experiment outputs were not modified. Raw JSON dumps are not embedded.

Smoke-test inventory:

| Script | Area | Checks | GPU? |
| --- | --- | --- | --- |
| smoke_test_tool_call_parser.py | parser | textual tool-call parser | no |
| smoke_test_executor.py | tools/executor | ToolExecutor, registry and trace basics | no |
| smoke_test_mcp_client.py | MCP-like layer | LocalMCPClient/Server list and call tools | no |
| smoke_test_single_agent.py | deterministic single-agent | rule-based single-agent workflows | no |
| smoke_test_llm_single_agent_parser_only.py | LLM single-agent protocol | fake-model parser/tool loop | no |
| smoke_test_qwen_batch_notebook.py | notebook | 02 notebook five-task batch guardrails | no |
| smoke_test_llm_multi_agent_parser_only.py | full LLM multi-agent protocol | fake-model role protocol and restrictions | no |
| smoke_test_qwen_multi_agent_notebook.py | notebook | 03 notebook JSON and forbidden-hint guardrails | no |
| smoke_test_metrics.py | eval/metrics | task-specific and role metrics | no |
| smoke_test_gold_runner.py | GoldRunner | deterministic gold tasks | no |
| smoke_test_multi_agent_role_workflow.py | rule multi-agent | role order and artifact flow | no |
| smoke_test_multi_agent_recovery.py | recovery | AgentResponse recovery/retry | no |
| smoke_test_multi_agent_five_task_baseline.py | five-task baseline | five configs and expected primitive sequences | no |
| smoke_test_no_legacy_report_tool.py | tool policy | no tec_build_report | no |
| smoke_test_primitive_compare_tools.py | primitive compare | series -> stats -> compare_stats | no |
| smoke_test_date_filtering.py | data dates | [start,end) filtering; needs local dataset | no |

Experiment/comparison scripts:

| Script | Purpose | Runs LLM? |
| --- | --- | --- |
| run_smoke_experiment.py | Small deterministic smoke experiment | no |
| run_real_single_agent_experiment.py | Rule-based single-agent on real processed data | no |
| run_real_multi_agent_experiment.py | Rule-based multi-agent on real processed data | no |
| compare_real_experiments.py | Compare deterministic real single/multi outputs | no |
| run_five_task_multi_agent_baseline.py | Rule-based multi-agent baseline for five tasks | no |
| compare_qwen_single_vs_multi_agent_five_task.py | Compare Qwen single and rule multi five-task JSON | no |
| compare_five_task_architectures.py | Compare Qwen single, rule multi and Qwen full LLM multi JSON | no |

Notebook inventory:

| Notebook | Cells | Purpose / headings |
| --- | --- | --- |
| notebooks/01_build_tec_dataset.ipynb | 18 | # Build processed TEC dataset |
| notebooks/02_qwen_single_agent_colab.ipynb | 34 | # Qwen single-agent TEC tool-calling batch experiment<br>## 1. Clean Colab Setup<br>## 2. Import Check<br>## 3. CONFIG<br>## 4. Clone Or Update Repository<br>## 5. Experiment Helpers<br>## Planned test questions<br>## 6. Dataset Setup |
| notebooks/03_qwen_multi_agent_colab.ipynb | 30 | # Qwen full LLM multi-agent TEC experiment<br>## 1. Clean Colab setup<br>## 2. Import check<br>## 3. Clone or update repository<br>## 4. Project imports<br>## 5. CONFIG<br>## Planned test questions<br>## 6. Dataset setup |
| notebooks/TEC_data_QWEN35_clear.ipynb | 67 | # Сбор данных TEC<br>## Визуализация<br># Агенты<br>## Одноагентный режим<br>### Описание инструментов<br>### Функции запуска агента и обработки генерации<br>### Тесты<br>## Многоагентная сеть |
