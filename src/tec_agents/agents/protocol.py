"""
Shared agent response protocol for deterministic and future LLM agents.

The protocol lets role agents report more than a successful artifact payload:
they can describe missing inputs, retryable tool errors, partial results, and
final answers in a structured form that the orchestrator can inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    """Standard status values returned by agents and internal steps."""

    OK = "ok"
    MISSING_ARTIFACTS = "missing_artifacts"
    INVALID_INPUT = "invalid_input"
    TOOL_ERROR = "tool_error"
    PARTIAL = "partial"
    FINAL = "final"


@dataclass
class RequestedNextAction:
    """A structured request for the orchestrator's next action."""

    target_agent: str
    task: str
    reason: str
    required_artifacts: list[str] = field(default_factory=list)
    optional_artifacts: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "target_agent": self.target_agent,
            "task": self.task,
            "reason": self.reason,
            "required_artifacts": list(self.required_artifacts),
            "optional_artifacts": list(self.optional_artifacts),
            "params": dict(self.params),
        }


@dataclass
class AgentResponse:
    """Structured response returned by role agents and recovery helpers."""

    status: AgentStatus
    agent: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    missing_artifacts: list[str] = field(default_factory=list)
    requested_next_action: RequestedNextAction | None = None
    message: str = ""
    can_continue: bool = True
    requires_retry: bool = False
    is_final: bool = False
    attempt: int = 1
    max_attempts: int = 2

    def ok(self) -> bool:
        """Return True when the response can be treated as successful."""

        return self.status in {AgentStatus.OK, AgentStatus.FINAL}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "status": self.status.value,
            "agent": self.agent,
            "artifacts": self.artifacts,
            "missing_artifacts": list(self.missing_artifacts),
            "requested_next_action": (
                self.requested_next_action.to_dict()
                if self.requested_next_action is not None
                else None
            ),
            "message": self.message,
            "can_continue": self.can_continue,
            "requires_retry": self.requires_retry,
            "is_final": self.is_final,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
        }


@dataclass
class StepRecoveryDecision:
    """Orchestrator decision after inspecting an AgentResponse."""

    decision: str
    target_agent: str | None = None
    reason: str = ""
    attempt: int = 1
    max_attempts: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return {
            "decision": self.decision,
            "target_agent": self.target_agent,
            "reason": self.reason,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
        }


def agent_ok(
    *,
    agent: str,
    artifacts: dict[str, Any] | None = None,
    message: str = "",
    attempt: int = 1,
    max_attempts: int = 2,
) -> AgentResponse:
    """Create a successful AgentResponse."""

    return AgentResponse(
        status=AgentStatus.OK,
        agent=agent,
        artifacts=artifacts or {},
        message=message,
        can_continue=True,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def agent_missing_artifacts(
    *,
    agent: str,
    missing_artifacts: list[str],
    requested_next_action: RequestedNextAction | None = None,
    message: str = "",
    attempt: int = 1,
    max_attempts: int = 2,
) -> AgentResponse:
    """Create a missing-artifacts AgentResponse."""

    return AgentResponse(
        status=AgentStatus.MISSING_ARTIFACTS,
        agent=agent,
        missing_artifacts=missing_artifacts,
        requested_next_action=requested_next_action,
        message=message,
        can_continue=False,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def agent_invalid_input(
    *,
    agent: str,
    message: str,
    missing_artifacts: list[str] | None = None,
    attempt: int = 1,
    max_attempts: int = 2,
) -> AgentResponse:
    """Create an invalid-input AgentResponse."""

    return AgentResponse(
        status=AgentStatus.INVALID_INPUT,
        agent=agent,
        missing_artifacts=missing_artifacts or [],
        message=message,
        can_continue=False,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def agent_tool_error(
    *,
    agent: str,
    message: str,
    artifacts: dict[str, Any] | None = None,
    missing_artifacts: list[str] | None = None,
    requested_next_action: RequestedNextAction | None = None,
    can_continue: bool = False,
    requires_retry: bool = True,
    attempt: int = 1,
    max_attempts: int = 2,
) -> AgentResponse:
    """Create a tool-error AgentResponse."""

    return AgentResponse(
        status=AgentStatus.TOOL_ERROR,
        agent=agent,
        artifacts=artifacts or {},
        missing_artifacts=missing_artifacts or [],
        requested_next_action=requested_next_action,
        message=message,
        can_continue=can_continue,
        requires_retry=requires_retry,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def agent_partial(
    *,
    agent: str,
    artifacts: dict[str, Any] | None = None,
    missing_artifacts: list[str] | None = None,
    requested_next_action: RequestedNextAction | None = None,
    message: str = "",
    attempt: int = 1,
    max_attempts: int = 2,
) -> AgentResponse:
    """Create a partial-result AgentResponse."""

    return AgentResponse(
        status=AgentStatus.PARTIAL,
        agent=agent,
        artifacts=artifacts or {},
        missing_artifacts=missing_artifacts or [],
        requested_next_action=requested_next_action,
        message=message,
        can_continue=True,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def agent_final(
    *,
    agent: str,
    artifacts: dict[str, Any],
    message: str = "",
    attempt: int = 1,
    max_attempts: int = 2,
) -> AgentResponse:
    """Create a final AgentResponse."""

    return AgentResponse(
        status=AgentStatus.FINAL,
        agent=agent,
        artifacts=artifacts,
        message=message,
        can_continue=True,
        is_final=True,
        attempt=attempt,
        max_attempts=max_attempts,
    )


def required_artifacts_for_task(
    task_type: str,
    include: list[str] | None = None,
) -> dict[str, list[str]]:
    """Return required and optional artifact paths for a task."""

    if task_type == "high_tec":
        return {
            "required": [
                "data.series_by_region",
                "math.high_tec",
                "analysis.findings",
            ],
            "optional": [],
        }

    if task_type == "stable_intervals":
        return {
            "required": [
                "data.series_by_region",
                "math.stable_intervals",
                "analysis.findings",
            ],
            "optional": [],
        }

    if task_type == "compare_regions":
        return {
            "required": [
                "data.series_by_region",
                "math.stats_by_region",
                "math.comparison",
                "analysis.findings",
            ],
            "optional": [],
        }

    if task_type == "report":
        selected = include or ["basic_stats", "high_tec", "stable_intervals"]
        required = [
            "data.series_by_region",
            "math.report_inputs.basic_stats",
            "analysis.findings",
        ]
        optional: list[str] = []

        for section in ["high_tec", "stable_intervals"]:
            artifact_path = f"math.report_inputs.{section}"
            if section in selected:
                required.append(artifact_path)
            else:
                optional.append(artifact_path)

        return {"required": required, "optional": optional}

    return {"required": [], "optional": []}
