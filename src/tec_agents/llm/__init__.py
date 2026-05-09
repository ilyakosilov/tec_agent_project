"""Utilities for future LLM-backed TEC agents."""

from tec_agents.llm.prompts import (
    build_all_multi_agent_prompts,
    build_single_agent_system_prompt,
)

__all__ = [
    "build_all_multi_agent_prompts",
    "build_single_agent_system_prompt",
]
