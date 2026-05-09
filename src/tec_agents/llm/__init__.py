"""Utilities for future LLM-backed TEC agents."""

from tec_agents.llm.local_qwen import LocalQwenChatModel
from tec_agents.llm.prompts import (
    build_all_multi_agent_prompts,
    build_single_agent_system_prompt,
)

__all__ = [
    "LocalQwenChatModel",
    "build_all_multi_agent_prompts",
    "build_single_agent_system_prompt",
]
