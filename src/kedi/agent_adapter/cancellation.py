"""Thread-safe cancellation ownership for synchronous embedding surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any


class RunCancelled(InterruptedError):
    """The owner cancelled adapter work and its asynchronous cleanup finished."""


class RunCancellation:
    """Cancel only calls registered under this owner, including copied threads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cancelled = False
        self._tasks: set[asyncio.Future[Any]] = set()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            tasks = tuple(self._tasks)
        for task in tasks:
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # The task can finish and its owner loop close after the snapshot.
                if not task.get_loop().is_closed():
                    raise

    def register(self, task: asyncio.Future[Any]) -> None:
        with self._lock:
            if self._cancelled:
                task.cancel()
            else:
                self._tasks.add(task)

    def unregister(self, task: asyncio.Future[Any]) -> None:
        with self._lock:
            self._tasks.discard(task)


_RUN_CANCELLATIONS: ContextVar[tuple[RunCancellation, ...]] = ContextVar(
    "kedi_run_cancellations", default=()
)


def current_run_cancellations() -> tuple[RunCancellation, ...]:
    return _RUN_CANCELLATIONS.get()


def check_run_cancelled() -> None:
    """Stop synchronous cooperative work before admitting another unit of work."""
    if any(owner.cancelled for owner in current_run_cancellations()):
        raise RunCancelled("adapter execution was interrupted")


@contextmanager
def use_run_cancellation(owner: RunCancellation) -> Generator[None, None, None]:
    """Nested owners cannot disable cancellation requested by an outer owner."""
    parents = _RUN_CANCELLATIONS.get()
    token = _RUN_CANCELLATIONS.set(parents if owner in parents else (*parents, owner))
    try:
        yield
    finally:
        _RUN_CANCELLATIONS.reset(token)
