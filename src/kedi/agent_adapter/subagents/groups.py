"""Cancellation and descendant ownership for structured result processing."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .native import NativeSubagentOwner, SubagentTask


_regions: ContextVar[tuple[ProcessingRegion, ...]] = ContextVar(
    "kedi_processing_regions", default=()
)


class ProcessingRegion:
    """Track work across copied contexts; cancellation never abandons running work."""

    def __init__(self, owner: NativeSubagentOwner) -> None:
        self.owner = owner
        self._lock = Lock()
        self._cancelled = False
        self._children: dict[str, SubagentTask] = {}
        self._futures: set[Future[Any]] = set()

    def check(self) -> None:
        with self._lock:
            cancelled = self._cancelled
        if cancelled:
            raise CancelledError("Task group processing was cancelled")

    def add_child(self, task: SubagentTask) -> None:
        if task._owner is not self.owner:
            return
        with self._lock:
            self._children[task.run_id] = task
            cancelled = self._cancelled
        if cancelled:
            self.owner.runtime._cancel_subagent(task.run_id)

    def add_future(self, future: Future[Any]) -> None:
        with self._lock:
            self._futures.add(future)
            cancelled = self._cancelled
        if cancelled:
            future.cancel()

    def cancel(self) -> None:
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            children = tuple(self._children)
            futures = tuple(self._futures)
        for run_id in children:
            self.owner.runtime._cancel_subagent(run_id)
        for future in futures:
            future.cancel()

    def drain(self, primary_error: BaseException | None) -> None:
        with self._lock:
            children = tuple(self._children)
        future = self.owner.submit(
            lambda: self.owner.drain_tasks(children, primary_error), enforce_deadline=False
        )
        self.owner.runtime._engine.wait(future)

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _regions.set((*_regions.get(), self))
        try:
            self.check()
            yield
        finally:
            _regions.reset(token)


def check_process_cancelled() -> None:
    for region in _regions.get():
        region.check()


def register_process_child(task: SubagentTask) -> None:
    for region in _regions.get():
        region.add_child(task)


def register_process_future(future: Future[Any]) -> None:
    for region in _regions.get():
        region.add_future(future)
