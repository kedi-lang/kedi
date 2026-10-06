"""Fence native turn startup and drain cancellation before returning to the owner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import contextmanager
from threading import Condition, Event
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from kedi.agent_adapter.subagents.control import (
    LiveInput,
    RunInterrupted,
    UnconfirmedStop,
    controlled_attempt,
    current_live_input,
)

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .app_server import CodexTurnResult


class TurnBindings:
    """Own dynamic tools and approvals until their native turn has fully stopped."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._closed = False
        self._active = 0
        self._tasks: dict[asyncio.Future[Any], bool] = {}

    @contextmanager
    def binding(self) -> Iterator[None]:
        with self._condition:
            if self._closed:
                raise asyncio.CancelledError
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    async def resolve(self, operation: Awaitable[T]) -> T:
        task = asyncio.ensure_future(operation)
        with self._condition:
            self._tasks[task] = False
            if self._closed:
                self._cancel_locked(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with self._condition:
                self._cancel_locked(task)
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            if not task.cancelled():
                task.exception()
            raise
        finally:
            with self._condition:
                self._tasks.pop(task, None)

    def _cancel_locked(self, task: asyncio.Future[Any]) -> None:
        # Redirect and owner cancellation share one cancellation request so an
        # asynchronous finally block cannot receive a second cancel.
        if not self._tasks[task]:
            self._tasks[task] = True
            task.get_loop().call_soon_threadsafe(task.cancel)

    def close(self, *, timeout: float = 5.0) -> None:
        # Called from the blocking turn thread, never from an owned tool's loop.
        with self._condition:
            if not self._closed:
                self._closed = True
                for task in tuple(self._tasks):
                    self._cancel_locked(task)
            if not self._condition.wait_for(lambda: self._active == 0, timeout=timeout):
                raise UnconfirmedStop("Codex turn bindings did not stop before cleanup deadline")


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

        async def drain() -> CodexTurnResult:
            return await asyncio.wait_for(asyncio.shield(task), 5.0)

        cleanup = asyncio.create_task(drain())
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        try:
            cleanup.result()
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
