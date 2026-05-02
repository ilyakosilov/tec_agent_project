# TEC Agent Project

Experimental framework for comparing single-agent and multi-agent LLM orchestration on ionospheric TEC data analysis tasks.

## Project idea

The project separates:

- TEC data preparation;
- deterministic TEC analysis tools;
- MCP-like tool access layer;
- single-agent orchestration;
- multi-agent orchestration;
- evaluation tasks and metrics.

The main research goal is to compare one-agent and multi-agent architectures under the same model, same data, same tools, and same evaluation metrics.

## Structure

```text
src/tec_agents/data      Data loading and region definitions
src/tec_agents/tools     TEC analysis tools and tool registry
src/tec_agents/mcp       Local MCP-like tool interface
src/tec_agents/llm       Local model wrappers and prompts
src/tec_agents/agents    Single-agent and multi-agent orchestration
src/tec_agents/eval      Tasks, gold runner, metrics, experiment runner
src/tec_agents/utils     Shared utilities
notebooks                Colab / research notebooks
configs                  Model, tool and experiment configs
data                     Local small examples only; large data is ignored
outputs                  Experiment outputs, ignored by Git