"""Read nested payload segments from ``CallArg`` / ``LiteralArg`` nodes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ast import CallArg, CallSegment, PythonExprSegment, Segment


def call_arg_native_backtick(arg: CallArg) -> str | None:
    from .ast import LiteralArg

    if isinstance(arg, LiteralArg) and arg.native_backtick is not None:
        return arg.native_backtick
    legacy = arg.__dict__.get("__native_backtick__")
    return legacy if isinstance(legacy, str) else None


def call_arg_call_segment(arg: CallArg) -> CallSegment | None:
    from .ast import CallSegment, LiteralArg

    if isinstance(arg, LiteralArg) and arg.call_segment is not None:
        return arg.call_segment
    legacy = arg.__dict__.get("__call_segment__")
    return legacy if isinstance(legacy, CallSegment) else None


def call_arg_python_segment(arg: CallArg) -> PythonExprSegment | None:
    from .ast import LiteralArg, PythonExprSegment

    if isinstance(arg, LiteralArg) and arg.python_segment is not None:
        return arg.python_segment
    legacy = arg.__dict__.get("__python_segment__")
    return legacy if isinstance(legacy, PythonExprSegment) else None


def call_arg_render_segments(arg: CallArg) -> list[Segment]:
    from .ast import LiteralArg, Segment

    if isinstance(arg, LiteralArg) and arg.render_segments:
        return list(arg.render_segments)
    legacy = arg.__dict__.get("__render_segments__")
    if not isinstance(legacy, list):
        return []
    return [segment for segment in legacy if isinstance(segment, Segment)]


def iter_call_arg_nested(arg: CallArg) -> Iterator[Segment | CallSegment]:
    from .ast import CallSegment, Segment

    seen: set[int] = set()

    def emit(value: object) -> Iterator[Segment | CallSegment]:
        if not isinstance(value, (Segment, CallSegment)):
            return
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)
        yield value

    call = call_arg_call_segment(arg)
    yield from emit(call)
    legacy_call = arg.__dict__.get("__call_segment__")
    if legacy_call is not None and legacy_call is not call:
        yield from emit(legacy_call)
    python = call_arg_python_segment(arg)
    yield from emit(python)
    legacy_python = arg.__dict__.get("__python_segment__")
    if legacy_python is not None and legacy_python is not python:
        yield from emit(legacy_python)
    for segment in call_arg_render_segments(arg):
        yield from emit(segment)
