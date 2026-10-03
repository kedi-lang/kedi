"""Track actual callable completion, including cancelled thread-backed tools."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import replace
from functools import wraps
from typing import ParamSpec, TypeVar, cast

from kedi.agent_adapter.subagents.control import current_live_input, current_live_revision
from kedi.agent_adapter.tools.specs import ToolSpec

P = ParamSpec("P")
R = TypeVar("R")


def with_live_tool_tracking(tool: ToolSpec) -> ToolSpec:
    if inspect.iscoroutinefunction(tool.fn):
        return replace(tool, fn=_track_async(tool.fn))
    return replace(tool, fn=_track_sync(tool.fn))


def _track_async(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    @wraps(fn)
    async def async_call(*args: P.args, **kwargs: P.kwargs) -> R:
        control = current_live_input()
        if control is not None:
            control.tool_started(revision=current_live_revision(control))
        try:
            return await fn(*args, **kwargs)
        finally:
            if control is not None:
                control.tool_finished()

    return async_call


def _track_sync(fn: Callable[P, R | Awaitable[R]]) -> Callable[P, R | Awaitable[R]]:
    @wraps(fn)
    def sync_call(*args: P.args, **kwargs: P.kwargs) -> R | Awaitable[R]:
        control = current_live_input()
        if control is not None:
            control.tool_started(revision=current_live_revision(control))
        try:
            result = fn(*args, **kwargs)
        except BaseException:
            if control is not None:
                control.tool_finished()
            raise
        if inspect.isawaitable(result):
            awaitable = cast(Awaitable[R], result)

            async def resolve() -> R:
                try:
                    return await awaitable
                finally:
                    if control is not None:
                        control.tool_finished()

            return resolve()
        if control is not None:
            control.tool_finished()
        return result

    return sync_call
