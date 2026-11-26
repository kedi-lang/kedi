from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class InvocationUsage:
    """What one or more model invocations used.

    A token count is ``None`` when the backend did not measure it, and 0 only when it
    measured zero. A sum is ``None`` when any part of it is, so a total over unmeasured
    invocations never passes for an exact one.
    """

    requests: int = 0
    tool_calls: int = 0
    input_tokens: int | None = 0
    cache_read_tokens: int | None = 0
    cache_write_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0
    cost_usd: Decimal | None = None

    @classmethod
    def unmeasured(cls, *, requests: int = 1, tool_calls: int = 0) -> InvocationUsage:
        """Invocations whose backend reported no token counts."""

        return cls(
            requests=requests,
            tool_calls=tool_calls,
            input_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )

    @property
    def uncached_input_tokens(self) -> int | None:
        # An unknown cache count is taken as no cache hits: never fewer tokens than sent.
        if self.input_tokens is None:
            return None
        return max(0, self.input_tokens - (self.cache_read_tokens or 0))

    def __add__(self, other: InvocationUsage) -> InvocationUsage:
        return InvocationUsage(
            requests=self.requests + other.requests,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=_add_count(self.input_tokens, other.input_tokens),
            cache_read_tokens=_add_count(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens=_add_count(self.cache_write_tokens, other.cache_write_tokens),
            output_tokens=_add_count(self.output_tokens, other.output_tokens),
            total_tokens=_add_count(self.total_tokens, other.total_tokens),
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


def _add_count(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


def _add_cost(left: Decimal | None, right: Decimal | None) -> Decimal | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right
