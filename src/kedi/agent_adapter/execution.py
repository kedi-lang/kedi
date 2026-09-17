from __future__ import annotations

import asyncio
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    TypeVar,
    cast,
)

_T = TypeVar("_T")

_RUN_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "kedi_run_deadline", default=None
)


class RunDeadlineExceeded(TimeoutError):
    """The enclosing execution deadline expired after cancelling async work."""


@contextmanager
def use_run_deadline(deadline_monotonic: float | None) -> Iterator[None]:
    """Apply an absolute deadline to adapter calls, including copied worker contexts."""
    parent = _RUN_DEADLINE.get()
    deadline = (
        min(parent, deadline_monotonic)
        if parent is not None and deadline_monotonic is not None
        else parent
        if deadline_monotonic is None
        else deadline_monotonic
    )
    token = _RUN_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _RUN_DEADLINE.reset(token)


def run_sync(awaitable: Awaitable[_T]) -> _T:
    """Run an awaitable to completion from sync code, even inside a running loop.

    When no event loop is running this is just :func:`asyncio.run`. When called
    from within a running loop (``asyncio.run`` would raise there) the coroutine
    is executed on its own loop in a dedicated worker thread.
    """

    async def resolve() -> _T:
        deadline = _RUN_DEADLINE.get()
        if deadline is None:
            return await awaitable
        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.wait_for(
                task,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError as exc:
            if task.cancelled():
                raise RunDeadlineExceeded("adapter execution deadline elapsed") from exc
            raise

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_awaitable(awaitable, resolve)
    # Copy the caller's context so a dspy.context(lm=...) override reaches the worker.
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        # cast: Context.run's TypeVar doesn't unify with run_sync's via submit().
        return cast(
            "_T", executor.submit(lambda: ctx.run(_run_awaitable, awaitable, resolve)).result()
        )


def _run_awaitable(
    awaitable: Awaitable[_T],
    resolve: Callable[[], Coroutine[Any, Any, _T]],
) -> _T:
    try:
        return asyncio.run(resolve())
    except RuntimeError as exc:
        if "different loop" not in str(exc) and "attached to a different loop" not in str(exc):
            raise
        raise RuntimeError(
            "run_sync cannot execute an awaitable bound to another event loop; "
            "await it from its owner loop instead"
        ) from exc
