from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InvocationUsage:
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


InvocationObserver = Callable[[InvocationUsage], None]
_INVOCATION_OBSERVER: ContextVar[InvocationObserver | None] = ContextVar(
    "kedi_invocation_observer",
    default=None,
)


@contextmanager
def observe_invocations(observer: InvocationObserver) -> Iterator[None]:
    token = _INVOCATION_OBSERVER.set(observer)
    try:
        yield
    finally:
        _INVOCATION_OBSERVER.reset(token)


def invocation_observer() -> InvocationObserver | None:
    return _INVOCATION_OBSERVER.get()


def report_invocation(usage: InvocationUsage) -> None:
    observer = _INVOCATION_OBSERVER.get()
    if observer is not None:
        observer(usage)
