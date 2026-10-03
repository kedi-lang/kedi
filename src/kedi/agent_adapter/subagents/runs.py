"""Subagent run state, events, and registry."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, replace
from threading import Lock
from time import time
from typing import Literal, TypeAlias

SubagentRunStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
]

_TRANSITIONS: dict[SubagentRunStatus, frozenset[SubagentRunStatus]] = {
    "pending": frozenset({"running"}),
    "running": frozenset({"completed", "failed", "cancelled", "timed_out"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "timed_out": frozenset(),
}
_SECRET_VALUE = re.compile(r"(?i)\b(api[_ -]?key|token|secret|password)\b\s*[:=]\s*[^\s,;]+")
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
MAX_SUBAGENT_ERROR_MESSAGE = 500


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentRun:
    run_id: str
    parent_run_id: str | None
    root_run_id: str
    previous_run_id: str | None
    conversation_id: str
    owner_profile_name: str
    profile_name: str
    depth: int
    status: SubagentRunStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    adapter_shortname: str | None = None
    model: str | None = None
    structured_output: bool = False
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class SubagentStarted:
    run: SubagentRun


@dataclass(frozen=True, slots=True)
class SubagentCompleted:
    run: SubagentRun


@dataclass(frozen=True, slots=True)
class SubagentFailed:
    run: SubagentRun


@dataclass(frozen=True, slots=True)
class SubagentCancelled:
    run: SubagentRun


@dataclass(frozen=True, slots=True)
class SubagentInputEvent:
    run: SubagentRun
    message_id: str
    sender_id: str
    revision: int
    phase: Literal["admitted", "delivered", "undelivered"]
    interrupt: bool


SubagentEvent: TypeAlias = (
    SubagentStarted | SubagentCompleted | SubagentFailed | SubagentCancelled | SubagentInputEvent
)


class SubagentEventLog:
    def __init__(self, *, max_events: int) -> None:
        if max_events < 1:
            raise ValueError("Subagent event retention must be at least 1")
        self._events: deque[SubagentEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def append(self, event: SubagentEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[SubagentEvent, ...]:
        with self._lock:
            return tuple(self._events)


class SubagentRunRegistry:
    """Thread-safe lifecycle records for child invocations."""

    def __init__(self) -> None:
        self._runs: dict[str, SubagentRun] = {}
        self._lock = Lock()

    def create(
        self,
        *,
        run_id: str,
        parent_run_id: str | None,
        root_run_id: str,
        profile_name: str,
        depth: int,
        previous_run_id: str | None = None,
        conversation_id: str | None = None,
        owner_profile_name: str = "",
    ) -> SubagentRun:
        run = SubagentRun(
            run_id=run_id,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id,
            previous_run_id=previous_run_id,
            conversation_id=conversation_id or run_id,
            owner_profile_name=owner_profile_name,
            profile_name=profile_name,
            depth=depth,
            status="pending",
            created_at=time(),
        )
        with self._lock:
            if run_id in self._runs:
                raise ValueError(f"Duplicate subagent run ID: {run_id}")
            self._runs[run_id] = run
        return run

    def transition(
        self,
        run_id: str,
        status: SubagentRunStatus,
        *,
        error: BaseException | None = None,
    ) -> SubagentRun:
        if error is not None and status not in {"failed", "cancelled", "timed_out"}:
            raise ValueError("Only failed, cancelled, or timed-out runs may record an error")
        now = time()
        with self._lock:
            current = self._runs[run_id]
            if status not in _TRANSITIONS[current.status]:
                raise RuntimeError(f"Invalid subagent run transition: {current.status} -> {status}")
            updated = replace(
                current,
                status=status,
                started_at=now if status == "running" else current.started_at,
                finished_at=(
                    now if status in {"completed", "failed", "cancelled", "timed_out"} else None
                ),
                error_type=type(error).__name__ if error is not None else None,
                error_message=safe_subagent_error_message(error),
            )
            self._runs[run_id] = updated
            return updated

    def get(self, run_id: str) -> SubagentRun:
        with self._lock:
            return self._runs[run_id]

    def describe(
        self,
        run_id: str,
        *,
        adapter_shortname: str,
        model: str | None,
        structured_output: bool,
    ) -> SubagentRun:
        with self._lock:
            updated = replace(
                self._runs[run_id],
                adapter_shortname=adapter_shortname,
                model=model,
                structured_output=structured_output,
            )
            self._runs[run_id] = updated
            return updated

    def add_usage(
        self,
        run_id: str,
        *,
        requests: int,
        tool_calls: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> SubagentRun:
        with self._lock:
            current = self._runs[run_id]
            updated = replace(
                current,
                requests=current.requests + requests,
                tool_calls=current.tool_calls + tool_calls,
                input_tokens=current.input_tokens + input_tokens,
                output_tokens=current.output_tokens + output_tokens,
                total_tokens=current.total_tokens + total_tokens,
            )
            self._runs[run_id] = updated
            return updated

    def snapshot(self) -> tuple[SubagentRun, ...]:
        with self._lock:
            return tuple(self._runs.values())

    def restore(self, run: SubagentRun) -> None:
        with self._lock:
            if run.run_id in self._runs:
                raise ValueError(f"Duplicate subagent run ID: {run.run_id}")
            self._runs[run.run_id] = run

    def discard(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)


def safe_subagent_error_message(error: BaseException | None) -> str | None:
    if error is None:
        return None
    message = getattr(error, "message", None)
    if not isinstance(message, str):
        message = str(error)
    message = " ".join(message.split())
    message = _SECRET_VALUE.sub(lambda match: f"{match.group(1)}=<redacted>", message)
    message = _BEARER_VALUE.sub("Bearer <redacted>", message)
    return message[:MAX_SUBAGENT_ERROR_MESSAGE]
