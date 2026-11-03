"""Run async adapter work from synchronous Kedi code.

Which event loop that work runs on is a recorded design decision with invariants that
tests enforce: read docs/EVENT_LOOPS.md before changing it.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import os
import time
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from contextlib import contextmanager, nullcontext
from threading import Condition, Event, Lock, Thread
from typing import (
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


# Seconds an interrupted waiter gives the cancelled task to finish its cleanup.
_INTERRUPT_CLEANUP_SECONDS = 10.0


class _SharedLoop:
    """The event loop that synchronous Kedi work overlapping in time shares.

    Provider clients (an HTTP connection pool) bind to the loop that uses them and support one
    running loop at a time, so concurrent work in a process (operations started on any thread,
    direct adapter calls, Codex tool calls) must not each run its own loop. The loop starts when
    first held and, like ``asyncio.run``, is closed once nothing holds it. Cleanup of clients
    shared on it is deferred until then.
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        # Also run in a forked child, which has no loop thread: holds from before the fork are stale.
        self._lock = Condition()
        self._owned: OwnedAsyncLoop | None = None
        self._holders = 0
        self._closing = False
        self._cleanups: dict[int, Callable[[], Awaitable[None]]] = {}

    def runs(self, loop: asyncio.AbstractEventLoop | None) -> bool:
        owned = self._owned
        return owned is not None and owned._loop is loop

    def hold(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            # A provider keeps its client while the loop it last ran on still runs, so no new
            # loop starts before a closing one has stopped.
            self._lock.wait_for(lambda: not self._closing)
            if self._owned is None:
                self._owned = OwnedAsyncLoop()
            self._holders += 1
            loop = self._owned._loop
        assert loop is not None
        return loop

    def release(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            owned = self._owned
            if owned is None or owned._loop is not loop:
                return
            self._holders -= 1
            if self._holders:
                return
            self._closing = True
            cleanups = list(self._cleanups.values())
            self._cleanups.clear()
        if _running_loop() is loop:
            # Released by work running on the loop itself, which cannot wait for it to stop.
            Thread(target=self._close, args=(owned, cleanups), daemon=True).start()
        else:
            self._close(owned, cleanups)

    def _close(self, owned: OwnedAsyncLoop, cleanups: list[Callable[[], Awaitable[None]]]) -> None:
        try:
            if cleanups:
                assert owned._loop is not None
                asyncio.run_coroutine_threadsafe(_run_cleanups(cleanups), owned._loop).result()
        finally:
            try:
                owned.close()
            finally:
                with self._lock:
                    self._owned = None
                    self._closing = False
                    self._lock.notify_all()

    def defer(self, key: int, cleanup: Callable[[], Awaitable[None]]) -> bool:
        """Keep ``cleanup`` until the shared loop closes, if called on it."""
        running = _running_loop()
        with self._lock:
            if self._closing or not self.runs(running):
                return False
            self._cleanups.setdefault(key, cleanup)
            return True


_SHARED_LOOP = _SharedLoop()
if hasattr(os, "register_at_fork"):  # pragma: no branch - not on Windows
    os.register_at_fork(after_in_child=_SHARED_LOOP._reset)


class _HostLoops:
    """Async hosts' loops that synchronous Kedi work is running on.

    Cleanup of clients shared on such a loop waits until no operation or call uses it.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._users: dict[asyncio.AbstractEventLoop, int] = {}
        self._cleanups: dict[
            asyncio.AbstractEventLoop, dict[int, Callable[[], Awaitable[None]]]
        ] = {}

    @contextmanager
    def use(self, loop: asyncio.AbstractEventLoop) -> Iterator[None]:
        with self._lock:
            self._users[loop] = self._users.get(loop, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._users[loop] -= 1
                last = not self._users[loop]
                if last:
                    del self._users[loop]
                cleanups = list(self._cleanups.pop(loop, {}).values()) if last else []
            if cleanups:
                asyncio.run_coroutine_threadsafe(_run_cleanups(cleanups), loop).result()

    def defer(self, key: int, cleanup: Callable[[], Awaitable[None]]) -> bool:
        """Keep ``cleanup`` until no Kedi work uses the host loop it is called on."""
        running = _running_loop()
        with self._lock:
            if running not in self._users:
                return False
            self._cleanups.setdefault(running, {}).setdefault(key, cleanup)
            return True


_HOST_LOOPS = _HostLoops()


class _OperationLoop:
    """One synchronous operation's hold on the loop its async work runs on, from first use to end.

    That is the shared loop, unless the operation gets a loop of its own (see
    :func:`_operation_event_loop`).
    """

    def __init__(self, own_loop: bool = False) -> None:
        self._lock = Lock()
        self._loops = _SharedLoop() if own_loop else _SHARED_LOOP
        self._held: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("Kedi operation event loop is closed")
            if self._held is None:
                self._held = self._loops.hold()
            return self._held

    def held(self) -> asyncio.AbstractEventLoop | None:
        return self._held

    def defer(self, key: int, cleanup: Callable[[], Awaitable[None]]) -> bool:
        return self._loops.defer(key, cleanup)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            held, self._held = self._held, None
        if held is not None:
            self._loops.release(held)


async def _run_cleanups(cleanups: list[Callable[[], Awaitable[None]]]) -> None:
    for cleanup in cleanups:
        await cleanup()


_OPERATION_LOOP: contextvars.ContextVar[_OperationLoop | None] = contextvars.ContextVar(
    "kedi_operation_loop", default=None
)
_BLOCKED_LOOPS: contextvars.ContextVar[frozenset[asyncio.AbstractEventLoop]] = (
    contextvars.ContextVar("kedi_blocked_loops", default=frozenset())
)


def _blocked_loops() -> frozenset[asyncio.AbstractEventLoop]:
    """Loops whose thread synchronous code in this call chain holds while it waits."""
    running = _running_loop()
    blocked = _BLOCKED_LOOPS.get()
    return blocked if running is None else blocked | {running}


@contextmanager
def _blocking() -> Iterator[frozenset[asyncio.AbstractEventLoop]]:
    """Record that this thread's running loop is blocked by the synchronous wait inside."""
    blocked = _blocked_loops()
    token = _BLOCKED_LOOPS.set(blocked)
    try:
        yield blocked
    finally:
        _BLOCKED_LOOPS.reset(token)


@contextmanager
def _operation_event_loop() -> Iterator[None]:
    """Run one synchronous operation's async work on a single event loop.

    A no-op inside an enclosing operation. Under an async host the host's loop is used;
    otherwise the operation holds the shared loop, which closes once nothing holds it. Sync
    code waiting on an operation, such as a sync call inside a coroutine, blocks the loop it
    runs on; an operation started there gets a loop of its own instead of deadlocking.
    """
    enclosing = _OPERATION_LOOP.get()
    blocked = _blocked_loops()
    if enclosing is not None and enclosing.held() not in blocked:
        yield
        return
    host = _ASYNC_OWNER_LOOP.get()
    scope = _OperationLoop(own_loop=any(_SHARED_LOOP.runs(loop) for loop in blocked))
    token = _OPERATION_LOOP.set(scope)
    try:
        with _blocking(), _HOST_LOOPS.use(host) if host is not None else nullcontext():
            yield
    finally:
        _OPERATION_LOOP.reset(token)
        scope.close()


def _defer_loop_cleanup(key: int, cleanup: Callable[[], Awaitable[None]]) -> bool:
    """Run ``cleanup`` once Kedi stops using the loop it is called on, if Kedi shares that loop."""
    scope = _OPERATION_LOOP.get()
    return (
        (scope is not None and scope.defer(key, cleanup))
        or _SHARED_LOOP.defer(key, cleanup)
        or _HOST_LOOPS.defer(key, cleanup)
    )


def _shared_execution_loop() -> asyncio.AbstractEventLoop | None:
    """The async host's loop, else the current operation's loop, else ``None``."""
    owner = current_async_owner_loop()
    if owner is not None:
        return owner
    scope = _OPERATION_LOOP.get()
    return scope.loop() if scope is not None else None


def _wait_on_loop(loop: asyncio.AbstractEventLoop, awaitable: Awaitable[_T]) -> _T:
    future = asyncio.run_coroutine_threadsafe(_resolve_deadline(awaitable), loop)
    try:
        return future.result()
    except FutureCancelledError as exc:
        # As with asyncio.run, a cancelled coroutine raises asyncio's CancelledError.
        raise asyncio.CancelledError from exc
    except BaseException:
        if not future.done():
            # The waiting thread was interrupted: stop the task it owns and let it clean up.
            future.cancel()
            wait_futures([future], timeout=_INTERRUPT_CLEANUP_SECONDS)
        raise


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

    An async host's owner loop runs it when one is active; otherwise the current
    operation's loop or, outside one, Kedi's shared loop held for this call. If sync code
    waiting on this call blocks that loop (a sync call inside a coroutine running on it),
    the coroutine runs on its own loop in a dedicated worker thread instead.
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
        with _HOST_LOOPS.use(owner):
            return _wait_on_loop(owner, awaitable)
    with _blocking() as blocked:
        scope = _OPERATION_LOOP.get()
        if scope is not None:
            loop = scope.loop()
            if loop not in blocked:
                return _translate_foreign_loop(lambda: _wait_on_loop(loop, awaitable))
        elif not any(_SHARED_LOOP.runs(loop) for loop in blocked):
            shared = _SHARED_LOOP.hold()
            try:
                return _translate_foreign_loop(lambda: _wait_on_loop(shared, awaitable))
            finally:
                _SHARED_LOOP.release(shared)
        # Sync code waiting on this call blocks that loop, so waiting on it would deadlock:
        # the call keeps its own loop in a worker thread. Copy the caller's context so a
        # dspy.context(lm=...) override reaches the worker.
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            # cast: Context.run's TypeVar doesn't unify with run_sync's via submit().
            return cast(
                "_T",
                executor.submit(
                    lambda: ctx.run(
                        _translate_foreign_loop, lambda: asyncio.run(_resolve_deadline(awaitable))
                    )
                ).result(),
            )


def _translate_foreign_loop(run: Callable[[], _T]) -> _T:
    try:
        return run()
    except RuntimeError as exc:
        if "different loop" not in str(exc) and "attached to a different loop" not in str(exc):
            raise
        raise RuntimeError(
            "run_sync cannot execute an awaitable bound to another event loop; "
            "await it from its owner loop instead"
        ) from exc
