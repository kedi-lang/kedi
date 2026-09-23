from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Iterator, Literal

ExecutionState = Literal["not_started", "running", "returned", "raised", "interrupted"]
ResultState = Literal["not_ready", "ready", "handed_off", "failed", "unknown"]
FailureStage = Literal[
    "execution", "result_validation", "post_hook", "artifact", "serialization", "handoff"
]


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    tool_call_id: str
    attempt: int
    execution_state: ExecutionState
    result_state: ResultState = "not_ready"
    failure_stage: FailureStage | None = None
    output_available: bool = False
    output: Any = None
    destination: Literal["framework", "sandbox"] | None = None

    def __repr__(self) -> str:
        return (
            "ToolCallRecord("
            f"tool_call_id={self.tool_call_id!r}, attempt={self.attempt}, "
            f"execution_state={self.execution_state!r}, result_state={self.result_state!r}, "
            f"failure_stage={self.failure_stage!r}, output_available={self.output_available})"
        )


class ToolCallTracker:
    def __init__(self, tool_call_id: str) -> None:
        self.tool_call_id = tool_call_id
        self._records: deque[ToolCallRecord] = deque(maxlen=32)

    @property
    def records(self) -> tuple[ToolCallRecord, ...]:
        return tuple(self._records)

    @property
    def latest(self) -> ToolCallRecord | None:
        return self._records[-1] if self._records else None

    def start(self) -> ToolCallRecord:
        record = ToolCallRecord(
            tool_call_id=self.tool_call_id,
            attempt=(self._records[-1].attempt + 1 if self._records else 1),
            execution_state="running",
        )
        self._records.append(record)
        return record

    def update(self, **changes: Any) -> ToolCallRecord:
        if not self._records:
            raise RuntimeError("Tool call has not started")
        record = replace(self._records[-1], **changes)
        self._records[-1] = record
        return record

    def returned(self, output: Any) -> ToolCallRecord:
        return self.update(execution_state="returned", output_available=True, output=output)

    def body_failed(self, error: BaseException) -> ToolCallRecord:
        interrupted = isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))
        return self.update(
            execution_state="interrupted" if interrupted else "raised",
            result_state="unknown" if interrupted else "failed",
            failure_stage="execution",
        )

    def result_failed(self, stage: FailureStage) -> ToolCallRecord | None:
        if self.latest is None or self.latest.execution_state != "returned":
            return None
        return self.update(result_state="failed", failure_stage=stage)

    def ready(self) -> ToolCallRecord | None:
        if self.latest is None or self.latest.execution_state != "returned":
            return None
        return self.update(result_state="ready")

    def handed_off(self) -> ToolCallRecord | None:
        if self.latest is None or self.latest.result_state != "ready":
            return None
        return self.update(result_state="handed_off")


class ToolResultHandoff:
    """Defer nested-tool delivery until the sandbox has accepted the execution."""

    def __init__(self) -> None:
        self.tracker: ToolCallTracker | None = None
        self.observer: Callable[[BaseException, ToolCallRecord], Awaitable[None]] | None = None

    async def fail(self, error: BaseException, stage: FailureStage) -> None:
        if self.tracker is None:
            return
        record = self.tracker.result_failed(stage)
        if record is not None:
            attach_tool_call(error, record)
            if self.observer is not None:
                await self.observer(error, record)

    def finish(self, *, accepted: bool) -> None:
        if self.tracker is not None and self.tracker.latest is not None:
            if self.tracker.latest.result_state == "ready":
                if accepted:
                    self.tracker.handed_off()
                else:
                    self.tracker.update(result_state="unknown")


_handoff: ContextVar[ToolResultHandoff | None] = ContextVar(
    "kedi_tool_result_handoff", default=None
)


@contextmanager
def defer_tool_handoff(handoff: ToolResultHandoff) -> Iterator[None]:
    token = _handoff.set(handoff)
    try:
        yield
    finally:
        _handoff.reset(token)


def register_tool_handoff(
    tracker: ToolCallTracker,
    observer: Callable[[BaseException, ToolCallRecord], Awaitable[None]],
) -> bool:
    handoff = _handoff.get()
    if handoff is None or tracker.latest is None:
        return False
    tracker.update(destination="sandbox")
    handoff.tracker = tracker
    handoff.observer = observer
    return True


_tracker: ContextVar[ToolCallTracker | None] = ContextVar("kedi_tool_call_tracker", default=None)


@contextmanager
def track_tool_call(tool_call_id: str) -> Iterator[ToolCallTracker]:
    tracker = ToolCallTracker(tool_call_id)
    token = _tracker.set(tracker)
    try:
        yield tracker
    finally:
        _tracker.reset(token)


def current_tool_call_tracker() -> ToolCallTracker | None:
    return _tracker.get()


@contextmanager
def bind_tool_call_tracker(tracker: ToolCallTracker) -> Iterator[None]:
    token = _tracker.set(tracker)
    try:
        yield
    finally:
        _tracker.reset(token)


def attach_tool_call(error: BaseException, record: ToolCallRecord | None) -> None:
    if record is not None:
        try:
            setattr(error, "tool_call", record)
        except (AttributeError, TypeError):
            pass
