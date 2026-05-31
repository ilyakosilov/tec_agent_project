# Experiment 5: Qwen3.5-9B-AWQ model ablation

This folder documents the setup for a model-ablation run of the existing typed full LLM multi-agent TEC experiment.

The experiment uses the same architecture, prompts, typed protocol, tools, dataset, five benchmark tasks, GoldRunner evaluation, and metrics as the latest typed v3 experiment. The only intended experimental variable is the model checkpoint:

- model: `QuantTrio/Qwen3.5-9B-AWQ`
- base model metadata: `Qwen/Qwen3.5-9B`
- quantization: pre-quantized AWQ int4 checkpoint
- prompt revision: `grounded_inputs_deliverables_single_block_v3`
- architecture: `qwen_multi_agent_typed_full_llm`

The notebook for this setup is:

- `notebooks/05_qwen_multi_agent_typed_qwen35_9b_awq_colab.ipynb`

Future Colab outputs should be written under:

- `outputs/metrics/experiment_5_qwen35_9b_awq/`

Expected batch output:

- `outputs/metrics/experiment_5_qwen35_9b_awq/qwen_multi_agent_typed_v3_qwen35_9b_awq_batch_colab.json`

The checkpoint is already AWQ-quantized. The notebook should not download the base BF16/FP16 `Qwen/Qwen3.5-9B` weights for inference and should not perform on-the-fly bitsandbytes quantization.

No real experiment results are recorded here yet. Results will appear only after the notebook is run in Colab.
