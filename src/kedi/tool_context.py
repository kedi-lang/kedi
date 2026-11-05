"""Runtime context for calls that enter Kedi through an agent tool."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

_KEDI_TOOL_CALL_DEPTH: ContextVar[int] = ContextVar(
    "kedi_tool_call_depth",
    default=0,
)


def inside_kedi_tool_call() -> bool:
    """Return whether the current execution is inside a registered Kedi tool."""

    return _KEDI_TOOL_CALL_DEPTH.get() > 0


@contextmanager
def kedi_tool_call_context() -> Generator[None, None, None]:
    """Mark nested execution as originating from a registered Kedi tool."""

    token = _KEDI_TOOL_CALL_DEPTH.set(_KEDI_TOOL_CALL_DEPTH.get() + 1)
    try:
        yield
    finally:
        _KEDI_TOOL_CALL_DEPTH.reset(token)
