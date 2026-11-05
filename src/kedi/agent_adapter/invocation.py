from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class InvocationUsage:
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Decimal | None = None

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cache_read_tokens)

    def __add__(self, other: InvocationUsage) -> InvocationUsage:
        return InvocationUsage(
            requests=self.requests + other.requests,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=_add_cost(self.cost_usd, other.cost_usd),
        )


InvocationObserver = Callable[[InvocationUsage], None]
_INVOCATION_OBSERVER: ContextVar[InvocationObserver | None] = ContextVar(
    "kedi_invocation_observer",
    default=None,
)


@contextmanager
def observe_invocations(observer: InvocationObserver) -> Generator[None, None, None]:
    previous = _INVOCATION_OBSERVER.get()

    if previous is None:
        token = _INVOCATION_OBSERVER.set(observer)
    else:

        def composed(usage: InvocationUsage, /) -> None:
            try:
                observer(usage)
            finally:
                previous(usage)

        token = _INVOCATION_OBSERVER.set(composed)
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


def _add_cost(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right
