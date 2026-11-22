"""Retry model requests outside admission middleware, never replaying tools."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from kedi.agent_adapter.adapters.langchain.fallback import (
    response_started,
    retain_fallback_continuation_reset,
)
from kedi.agent_adapter.request_retry import wait_for_request_retry


@dataclass
class _RequestRetryState:
    reset_continuation: bool = False


_REQUEST_RETRY: ContextVar[_RequestRetryState | None] = ContextVar(
    "kedi_langchain_request_retry", default=None
)


def retain_retry_continuation_reset(reset: bool) -> bool:
    """Keep a consumed history reset until this logical request finishes."""
    reset = retain_fallback_continuation_reset(reset)
    state = _REQUEST_RETRY.get()
    if state is None:
        return reset
    state.reset_continuation = state.reset_continuation or reset
    return state.reset_continuation


class RequestRetryMiddleware(AgentMiddleware):
    def __init__(self, retries: int) -> None:
        self.retries = retries

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        token = _REQUEST_RETRY.set(_RequestRetryState())
        try:
            attempt = 1
            while True:
                try:
                    return await handler(request)
                except Exception as error:
                    if response_started():
                        raise
                    if not await wait_for_request_retry(
                        error, attempt=attempt, retries=self.retries
                    ):
                        raise
                    attempt += 1
        finally:
            _REQUEST_RETRY.reset(token)
