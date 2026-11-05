from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile

__all__ = ("SubagentExecutionContext", "current_subagent_execution")


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentExecutionContext:
    """Identity and policy context for the active child-agent execution."""

    run_id: str
    profile: AgentProfile
    conversation_id: str


_current_subagent_execution: ContextVar[SubagentExecutionContext | None] = ContextVar(
    "kedi_current_subagent_execution",
    default=None,
)


def current_subagent_execution() -> SubagentExecutionContext | None:
    """Return the child execution active in the current async task, if any."""

    return _current_subagent_execution.get()


@contextmanager
def _subagent_execution_scope(
    context: SubagentExecutionContext,
) -> Generator[None, None, None]:
    token = _current_subagent_execution.set(context)
    try:
        yield
    finally:
        _current_subagent_execution.reset(token)
