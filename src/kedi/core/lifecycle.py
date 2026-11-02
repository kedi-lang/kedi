"""Admission and cancellation ownership for a runtime's outer invocations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Condition

from kedi.agent_adapter.cancellation import RunCancellation, use_run_cancellation
from kedi.agent_adapter.execution import _operation_event_loop, run_in_worker


class RuntimeLifecycle:
    def __init__(self) -> None:
        self._condition = Condition()
        self._closed = False
        self._owners: set[RunCancellation] = set()
        self._current: ContextVar[RunCancellation | None] = ContextVar(
            "kedi_runtime_admission", default=None
        )

    @contextmanager
    def admission(self) -> Iterator[None]:
        owner = RunCancellation()
        with self._condition:
            nested = self._current.get() in self._owners
            if not nested:
                if self._closed:
                    raise RuntimeError("KediRuntime is closed")
                self._owners.add(owner)
        if nested:
            yield
            return
        token = self._current.set(owner)
        try:
            # One event loop for the whole operation: concurrent model calls share provider
            # clients, which support a single running loop at a time.
            with use_run_cancellation(owner), _operation_event_loop():
                yield
        finally:
            self._current.reset(token)
            with self._condition:
                self._owners.remove(owner)
                self._condition.notify_all()

    async def aclose(self) -> None:
        with self._condition:
            if self._current.get() in self._owners:
                raise RuntimeError("Cannot close KediRuntime from an active invocation")
            self._closed = True
            owners = tuple(self._owners)
        for owner in owners:
            owner.cancel()
        await run_in_worker(self._wait)

    def _wait(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._owners)
