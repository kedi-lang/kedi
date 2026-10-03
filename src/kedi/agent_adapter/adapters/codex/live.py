"""Fence native turn startup and drain cancellation before returning to the owner."""

from __future__ import annotations

import asyncio
from threading import Event
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Callable, TypeVar

from kedi.agent_adapter.subagents.control import (
    LiveInput,
    RunInterrupted,
    controlled_attempt,
    current_live_input,
)

T = TypeVar("T")

if TYPE_CHECKING:
    from .app_server import CodexTurnResult


def ensure_turn_current(control: LiveInput | None, revision: int, stop: Event | None) -> None:
    if stop is not None and stop.is_set():
        raise asyncio.CancelledError
    if control is not None and control.revision != revision:
        raise RunInterrupted


async def run_cancellable_turn(
    fn: Callable[[Event], CodexTurnResult],
) -> CodexTurnResult:
    stop_event = Event()
    task = asyncio.create_task(asyncio.to_thread(fn, stop_event))
    task.add_done_callback(_consume_exception)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stop_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), 5.0)
        except asyncio.CancelledError:
            pass
        raise


async def run_cancellable_binding(operation: Awaitable[T]) -> T:
    control = current_live_input()
    if control is None:
        return await operation
    return await controlled_attempt(control, operation, quiesce_tools=False)


def _consume_exception(task: asyncio.Future[CodexTurnResult]) -> None:
    if not task.cancelled():
        task.exception()
