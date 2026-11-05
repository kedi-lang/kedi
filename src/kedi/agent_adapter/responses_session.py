from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar
from typing import Any, cast

_ACTIVE_MODELS: ContextVar[frozenset[tuple[int, int]]] = ContextVar(
    "kedi_active_responses_session_models", default=frozenset()
)


@asynccontextmanager
async def model_responses_session(model: Any) -> AsyncGenerator[None, None]:
    session_factory = getattr(model, "responses_session", None)
    if not callable(session_factory):
        yield
        return
    owner = getattr(model, "responses_session_owner", model)
    ownership = (id(owner), id(asyncio.current_task()))
    active = _ACTIVE_MODELS.get()
    if ownership in active:
        yield
        return
    token = _ACTIVE_MODELS.set(active | {ownership})
    try:
        session = (
            session_factory(isolate=True)
            if getattr(model, "supports_responses_session_isolation", False)
            else session_factory()
        )
        async with cast(AbstractAsyncContextManager[object], session):
            yield
    finally:
        _ACTIVE_MODELS.reset(token)


@asynccontextmanager
async def model_responses_turn(model: Any) -> AsyncGenerator[None, None]:
    """Separate one agent run's routing state from a reusable model connection."""
    async with model_responses_session(model):
        turn_factory = getattr(model, "responses_turn", None)
        if not callable(turn_factory):
            yield
            return
        async with cast(AbstractAsyncContextManager[object], turn_factory()):
            yield
