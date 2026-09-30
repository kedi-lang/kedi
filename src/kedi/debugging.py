"""Opt-in, debugger-neutral execution boundaries.

Observers run in the executing lane. Only ``safe`` events may suspend that lane;
async callers are observation-only so a synchronous observer cannot park an event loop.
The optional DAP package owns inspection, transport, and pause policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DebugFrame:
    name: str
    runtime: Any
    env: dict[str, Any]
    span: Any = None


@dataclass(frozen=True, slots=True)
class DebugEvent:
    kind: str
    runtime: Any
    frames: tuple[DebugFrame, ...]
    env: dict[str, Any]
    span: Any
    safe: bool
    payload: Any = None


_observer: ContextVar[Callable[[DebugEvent], None] | None] = ContextVar(
    "kedi_debug_observer", default=None
)
_frames: ContextVar[tuple[DebugFrame, ...]] = ContextVar("kedi_debug_frames", default=())
_observation_only: ContextVar[bool] = ContextVar("kedi_debug_observation_only", default=False)


def execution_observed() -> bool:
    return _observer.get() is not None


@contextmanager
def observe_execution(observer: Callable[[DebugEvent], None]) -> Iterator[None]:
    """Observe this execution and its context-propagating workers, not unrelated runs."""
    token = _observer.set(observer)
    try:
        yield
    finally:
        _observer.reset(token)


@contextmanager
def observation_only() -> Iterator[None]:
    """Keep nested execution observable without parking a transaction owner."""
    if _observer.get() is None:
        yield
        return
    token = _observation_only.set(True)
    try:
        yield
    finally:
        _observation_only.reset(token)


def checkpoint(
    runtime: Any,
    kind: str,
    *,
    env: dict[str, Any] | None = None,
    span: Any = None,
    payload: Any = None,
    safe: bool = True,
) -> None:
    observer = _observer.get()
    if observer is None:
        return
    safe = safe and not _observation_only.get()
    if safe:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            safe = False
    if env is None:
        env = runtime._current_env.get()
        if env is None:
            env = {}
    if span is None:
        for trace in reversed(runtime.current_trace_frames()):
            if trace.span is not None:
                span = trace.span
                break
    observer(DebugEvent(kind, runtime, _frames.get(), env, span, safe, payload))


@contextmanager
def execution_frame(
    runtime: Any, name: str, env: dict[str, Any], span: Any = None
) -> Iterator[None]:
    if _observer.get() is None:
        yield
        return
    frame = DebugFrame(name, runtime, env, span)
    token = _frames.set((*_frames.get(), frame))
    try:
        yield
    finally:
        try:
            checkpoint(runtime, "frame_exit", env=env, span=span, safe=False)
        finally:
            _frames.reset(token)


@contextmanager
def waiting_for_capture(name: str | None) -> Iterator[None]:
    """Observe an existing value dependency; do not resolve or pause it here."""
    frames = _frames.get() if _observer.get() is not None else ()
    if not frames:
        yield
        return
    runtime = frames[-1].runtime
    checkpoint(runtime, "waiting", payload={"capture": name}, safe=False)
    try:
        yield
    finally:
        checkpoint(runtime, "wait_complete", safe=False)
