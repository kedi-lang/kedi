from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

from .models import AgentRunRequest

_CURRENT_AGENT_RUN_REQUEST: ContextVar[AgentRunRequest | None] = ContextVar(
    "kedi_current_agent_run_request",
    default=None,
)


@contextmanager
def agent_run_request_scope(request: AgentRunRequest) -> Generator[None, None, None]:
    """Expose one immutable semantic request while its adapter call is active."""

    token = _CURRENT_AGENT_RUN_REQUEST.set(request)
    try:
        yield
    finally:
        _CURRENT_AGENT_RUN_REQUEST.reset(token)


def current_agent_run_request() -> AgentRunRequest | None:
    """Return the semantic request currently being lowered, when available."""

    return _CURRENT_AGENT_RUN_REQUEST.get()


__all__ = ["agent_run_request_scope", "current_agent_run_request"]
