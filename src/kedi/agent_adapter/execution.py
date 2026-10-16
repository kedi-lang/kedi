from __future__ import annotations

import asyncio
import contextvars
import inspect
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Thread
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    ParamSpec,
    TypeVar,
    cast,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")

_RUN_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "kedi_run_deadline", default=None
)
_ASYNC_OWNER_LOOP: contextvars.ContextVar[asyncio.AbstractEventLoop | None] = (
    contextvars.ContextVar("kedi_async_owner_loop", default=None)
)


def current_async_owner_loop() -> asyncio.AbstractEventLoop | None:
    """Return the host loop borrowed by synchronous worker-thread execution."""
    return _ASYNC_OWNER_LOOP.get()


@contextmanager
def use_async_owner_loop() -> Iterator[None]:
    """Route worker-thread sync calls to an async host's existing event loop.

    Enter this in the host coroutine, then run synchronous Kedi code with
    asyncio.to_thread. The host still owns and closes its clients and loop.
    """
    loop = asyncio.get_running_loop()
    token = _ASYNC_OWNER_LOOP.set(loop)
    try:
        yield
    finally:
        _ASYNC_OWNER_LOOP.reset(token)


async def run_in_worker(fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Offload sync work, cancelling nested async calls and joining cleanup on cancellation.

    Python threads cannot be forcibly interrupted. A synchronous handler must return
    before cancellation completes, but it never blocks the host event loop.
    """
    from .cancellation import (
        RunCancellation,
        check_run_cancelled,
        current_run_cancellations,
        use_run_cancellation,
    )

    cancellation = RunCancellation()
    # An enclosing synchronous worker can be waiting on this host loop. Its
    # executor may be saturated, so nested work needs an independent worker.
    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="kedi-nested")
        if current_run_cancellations()
        else None
    )

    def invoke() -> _T:
        check_run_cancelled()
        return fn(*args, **kwargs)

    try:
        with use_async_owner_loop(), use_run_cancellation(cancellation):
            worker = asyncio.get_running_loop().run_in_executor(
                executor, contextvars.copy_context().run, invoke
            )
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancellation.cancel()
                while True:
                    try:
                        result = await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        if worker.done():
                            break
                    except BaseException:
                        break
                    else:
                        if inspect.iscoroutine(result):
                            result.close()
                        break
                raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


class RunDeadlineExceeded(TimeoutError):
    """The enclosing execution deadline expired after cancelling async work."""


class OwnedAsyncLoop:
    """Keep one event loop alive for background work in a synchronous invocation."""

    def __init__(self) -> None:
        self._ready = Event()
        self._thread = Thread(target=self._serve, name="kedi-invocation-loop", daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._startup_error: BaseException | None = None
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("Invocation event loop did not start")
        if self._startup_error is not None:
            self._thread.join()
            raise RuntimeError(
                f"Invocation event loop failed to start: {self._startup_error}"
            ) from self._startup_error

    def _serve(self) -> None:
        async def wait_for_close() -> None:
            self._loop = asyncio.get_running_loop()
            self._stop = asyncio.Event()
            self._ready.set()
            await self._stop.wait()

        try:
            asyncio.run(wait_for_close())
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._ready.set()

    def run(self, awaitable: Awaitable[_T], *, enforce_deadline: bool = True) -> _T:
        return self.submit(awaitable, enforce_deadline=enforce_deadline).result()

    def submit(self, awaitable: Awaitable[_T], *, enforce_deadline: bool = True) -> Future[_T]:
        loop = self._loop
        if loop is None or self._stop is None or loop.is_closed():
            raise RuntimeError("Invocation event loop is closed")
        if _running_loop() is loop:
            raise RuntimeError("Cannot synchronously wait on the invocation event loop")

        async def resolve() -> _T:
            return await awaitable

        coroutine = _resolve_deadline(awaitable) if enforce_deadline else resolve()
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    def close(self) -> None:
        loop = self._loop
        stop = self._stop
        if loop is None or stop is None:
            return
        self._stop = None
        loop.call_soon_threadsafe(stop.set)
        self._thread.join()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


async def _resolve_deadline(awaitable: Awaitable[_T]) -> _T:
    from .cancellation import RunCancelled, current_run_cancellations

    deadline = _RUN_DEADLINE.get()
    owners = current_run_cancellations()
    if deadline is None and not owners:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    for owner in owners:
        owner.register(task)
    try:
        result = (
            await task
            if deadline is None
            else await asyncio.wait_for(task, timeout=max(0.0, deadline - time.monotonic()))
        )
        if any(owner.cancelled for owner in owners):
            raise RunCancelled("adapter execution was interrupted")
        return result
    except asyncio.CancelledError as exc:
        if any(owner.cancelled for owner in owners):
            raise RunCancelled("adapter execution was interrupted") from exc
        raise
    except asyncio.TimeoutError as exc:
        if task.cancelled():
            raise RunDeadlineExceeded("adapter execution deadline elapsed") from exc
        raise
    finally:
        for owner in owners:
            owner.unregister(task)


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

    owner = current_async_owner_loop()
    if owner is not None:
        if owner is _running_loop() or owner.is_closed() or not owner.is_running():
            # The awaitable has not been scheduled. Close a newly-created
            # coroutine so rejecting an invalid bridge does not leak it.
            if isinstance(awaitable, Coroutine):
                awaitable.close()
            raise RuntimeError(
                "The async owner loop must be running in another thread; "
                "run synchronous Kedi code with asyncio.to_thread"
            )
        return asyncio.run_coroutine_threadsafe(_resolve_deadline(awaitable), owner).result()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_awaitable(awaitable, lambda: _resolve_deadline(awaitable))
    # Copy the caller's context so a dspy.context(lm=...) override reaches the worker.
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        # cast: Context.run's TypeVar doesn't unify with run_sync's via submit().
        return cast(
            "_T",
            executor.submit(
                lambda: ctx.run(_run_awaitable, awaitable, lambda: _resolve_deadline(awaitable))
            ).result(),
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
