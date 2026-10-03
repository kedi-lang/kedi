"""Invocation-local, thread-safe input admission for a logical agent task."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Literal, TypedDict, TypeVar
from uuid import uuid4

MAX_PENDING_INPUTS = 32
MAX_INPUT_BYTES = 32 * 1024
MAX_PENDING_INPUT_BYTES = 128 * 1024
INTERRUPT_CLEANUP_SECONDS = 5.0

InputRejection = Literal[
    "task_finished", "not_authorized", "unsupported", "busy", "queue_full", "input_too_large"
]
InputPhase = Literal["admitted", "delivered", "undelivered"]
InputStatus = Literal["queued", "interrupt_requested", "rejected"]


class InputReceiptResult(TypedDict):
    message_id: str | None
    target_id: str
    revision: int
    status: InputStatus
    reason: InputRejection | None


@dataclass(frozen=True, slots=True)
class InputReceipt:
    message_id: str | None
    target_id: str
    revision: int
    status: InputStatus
    reason: InputRejection | None = None

    def as_tool_result(self) -> InputReceiptResult:
        return {
            "message_id": self.message_id,
            "target_id": self.target_id,
            "revision": self.revision,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AgentInput:
    message_id: str
    target_id: str
    sender_id: str
    revision: int
    message: str
    interrupt: bool = False

    def render(self) -> str:
        # JSON separates user-provided content from runtime-owned identity and routing.
        return "Kedi agent input:\n" + json.dumps(asdict(self), ensure_ascii=True)


class RunInterrupted(BaseException):
    """The physical attempt stopped; the logical task must continue."""


class UnconfirmedStop(RuntimeError):
    """A native backend ended without confirming its physical attempt stopped."""


class LiveInput:
    def __init__(
        self,
        target_id: str,
        *,
        parent: LiveInput | None = None,
        observe: Callable[[AgentInput, InputPhase], None] | None = None,
    ) -> None:
        self.target_id = target_id
        self.parent = parent
        self._observe = observe
        self._lock = Lock()
        self._queue: deque[AgentInput] = deque()
        self._claimed: dict[str, AgentInput] = {}
        self._bytes = 0
        self._revision = 0
        self._sequence = 0
        self._input_sequence = 0
        self._interrupting = False
        self._finished = False
        self._supported = False
        self._interrupt_supported = False
        self._active_tools = 0
        self._waiters: dict[asyncio.Future[None], asyncio.AbstractEventLoop] = {}
        self._delivered: deque[str] = deque(maxlen=1024)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def pending(self) -> bool:
        with self._lock:
            return bool(self._queue)

    @property
    def waiting_input(self) -> bool:
        with self._lock:
            return bool(self._queue or self._claimed)

    @property
    def input_sequence(self) -> int:
        with self._lock:
            return self._input_sequence

    @property
    def delivered(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._delivered)

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def configure(self, *, input: bool, interrupt: bool = False) -> None:
        with self._lock:
            self._supported = input
            self._interrupt_supported = interrupt

    def reject(self, reason: InputRejection) -> InputReceipt:
        return InputReceipt(None, self.target_id, self.revision, "rejected", reason)

    def admit(self, message: str, *, sender_id: str, interrupt: bool = False) -> InputReceipt:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Agent input must be a nonempty string")
        size = len(message.encode("utf-8"))
        with self._lock:
            reason: InputRejection | None = None
            if self._finished:
                reason = "task_finished"
            elif not self._supported or (interrupt and not self._interrupt_supported):
                reason = "unsupported"
            elif interrupt and self._interrupting:
                reason = "busy"
            elif size > MAX_INPUT_BYTES:
                reason = "input_too_large"
            elif (
                len(self._queue) + len(self._claimed) >= MAX_PENDING_INPUTS
                or self._bytes + size > MAX_PENDING_INPUT_BYTES
            ):
                reason = "queue_full"
            if reason is not None:
                return InputReceipt(None, self.target_id, self._revision, "rejected", reason)
            if interrupt:
                self._revision += 1
                self._interrupting = True
            item = AgentInput(
                uuid4().hex, self.target_id, sender_id, self._revision, message, interrupt
            )
            self._queue.append(item)
            self._bytes += size
            self._input_sequence += 1
            if self._observe is not None:
                self._observe(item, "admitted")
            self._notify_locked()
            return InputReceipt(
                item.message_id,
                self.target_id,
                self._revision,
                "interrupt_requested" if interrupt else "queued",
            )

    def begin_attempt(self) -> int:
        with self._lock:
            self._interrupting = False
            self._queue.extendleft(reversed(tuple(self._claimed.values())))
            self._claimed.clear()
            return self._revision

    def take_pending(self, *, revision: int | None = None) -> tuple[AgentInput, ...]:
        with self._lock:
            if revision is not None and self._revision != revision:
                return ()
            items = tuple(self._queue)
            self._queue.clear()
            self._claimed.update((item.message_id, item) for item in items)
            return items

    def mark_delivered(self, items: tuple[AgentInput, ...]) -> None:
        with self._lock:
            items = tuple(
                item for item in items if self._claimed.pop(item.message_id, None) is not None
            )
            self._bytes -= sum(len(item.message.encode("utf-8")) for item in items)
            self._delivered.extend(item.message_id for item in items)
            if self._observe is not None:
                for item in items:
                    self._observe(item, "delivered")

    def confirm_contents(self, contents: tuple[str, ...]) -> None:
        with self._lock:
            items = tuple(self._claimed.values())
        self.mark_delivered(
            tuple(item for item in items if any(item.render() in content for content in contents))
        )

    def restore_pending(self, items: tuple[AgentInput, ...]) -> None:
        with self._lock:
            restored = tuple(
                item for item in items if self._claimed.pop(item.message_id, None) is not None
            )
            self._queue.extendleft(reversed(restored))
            self._notify_locked()

    def try_finish(self, revision: int, publish: Callable[[], None]) -> bool:
        # Admission and final publication have a single linearization point.
        with self._lock:
            if self._revision != revision or self._queue or self._claimed or self._interrupting:
                return False
            publish()
            self._finished = True
            self._notify_locked()
            return True

    def close(self) -> None:
        with self._lock:
            self._finished = True
            if self._observe is not None:
                for item in (*self._claimed.values(), *self._queue):
                    self._observe(item, "undelivered")
            self._claimed.clear()
            self._queue.clear()
            self._bytes = 0
            self._notify_locked()

    def tool_started(self, *, revision: int | None = None) -> None:
        with self._lock:
            if self._finished or (revision is not None and revision != self._revision):
                raise RuntimeError("A finished or superseded agent attempt cannot start tools")
            self._active_tools += 1

    def tool_finished(self) -> None:
        with self._lock:
            self._active_tools -= 1
            self._notify_locked()

    async def wait_quiescent(self) -> None:
        async def wait() -> None:
            while True:
                with self._lock:
                    if not self._active_tools:
                        return
                    sequence = self._sequence
                await self.wait_changed(sequence)

        try:
            await asyncio.wait_for(wait(), INTERRUPT_CLEANUP_SECONDS)
        except TimeoutError as exc:
            raise RuntimeError(
                "Agent tool cleanup did not reach a terminal state; task was not restarted"
            ) from exc

    async def wait_changed(self, sequence: int) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        with self._lock:
            if self._sequence != sequence:
                return
            self._waiters[future] = loop
        try:
            await future
        finally:
            with self._lock:
                self._waiters.pop(future, None)

    def _notify_locked(self) -> None:
        self._sequence += 1
        for future, loop in tuple(self._waiters.items()):
            if not loop.is_closed():
                loop.call_soon_threadsafe(_wake, future)


def _wake(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


_LIVE_INPUT: ContextVar[LiveInput | None] = ContextVar("kedi_live_agent_input", default=None)
_LIVE_REVISION: ContextVar[int | None] = ContextVar("kedi_live_agent_revision", default=None)


def current_live_input() -> LiveInput | None:
    return _LIVE_INPUT.get()


def current_live_revision(control: LiveInput) -> int:
    revision = _LIVE_REVISION.get()
    return (
        revision if current_live_input() is control and revision is not None else control.revision
    )


def sender_is_current(control: LiveInput) -> bool:
    return not control.finished and (
        current_live_input() is not control or _LIVE_REVISION.get() == control.revision
    )


@contextmanager
def live_input_scope(control: LiveInput | None, *, revision: int | None = None) -> Iterator[None]:
    pinned_revision = (
        (current_live_revision(control) if revision is None else revision)
        if control is not None
        else None
    )
    token = _LIVE_INPUT.set(control)
    revision_token = _LIVE_REVISION.set(pinned_revision)
    try:
        yield
    finally:
        _LIVE_REVISION.reset(revision_token)
        _LIVE_INPUT.reset(token)


T = TypeVar("T")


async def controlled_attempt(
    control: LiveInput,
    operation: Awaitable[T],
    *,
    deliver: Callable[[tuple[AgentInput, ...]], Awaitable[bool | None]] | None = None,
    stop: Callable[[], Awaitable[None]] | None = None,
    quiesce_tools: bool = True,
) -> T:
    """Stop a physical attempt; bound tools leave the shared barrier to their owner."""
    revision = current_live_revision(control)
    if control.revision != revision:
        if inspect.iscoroutine(operation):
            operation.close()
        elif isinstance(operation, asyncio.Future):
            operation.cancel()
        raise RunInterrupted
    task = asyncio.ensure_future(operation)

    async def monitor() -> None:
        while not task.done():
            sequence = control.sequence
            if control.revision != revision:
                return
            if deliver is not None and control.pending:
                items = control.take_pending(revision=revision)
                if await deliver(items) is not False:
                    control.mark_delivered(items)
            await control.wait_changed(sequence)

    watcher = asyncio.create_task(monitor())
    cleanup_task: asyncio.Task[None] | None = None

    async def cleanup() -> None:
        if task.done():
            if not task.cancelled():
                error = task.exception()
                if isinstance(error, UnconfirmedStop):
                    raise error
            return
        try:
            if stop is None:
                task.cancel()
            else:
                await asyncio.wait_for(stop(), INTERRUPT_CLEANUP_SECONDS)
            await asyncio.wait_for(asyncio.shield(task), INTERRUPT_CLEANUP_SECONDS)
        except asyncio.CancelledError as error:
            if not task.cancelled():
                task.cancel()
                task.add_done_callback(_consume_exception)
                await asyncio.wait((task,), timeout=INTERRUPT_CLEANUP_SECONDS)
                raise UnconfirmedStop(
                    "Native interrupt was cancelled before terminal confirmation"
                ) from error
        except TimeoutError as exc:
            task.cancel()
            task.add_done_callback(_consume_exception)
            raise RuntimeError(
                "Agent interrupt cleanup did not reach a terminal state; task was not restarted"
            ) from exc
        except Exception as error:
            if not task.done():
                task.cancel()
                task.add_done_callback(_consume_exception)
                await asyncio.wait((task,), timeout=INTERRUPT_CLEANUP_SECONDS)
                raise
            if isinstance(error, UnconfirmedStop):
                raise
            # Native SDK cancellation can finish the physical task with its own exception.

    async def settle() -> None:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(cleanup())
        # A logical cancel must not cancel an already-running redirect cleanup.
        await asyncio.shield(cleanup_task)

    try:
        done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            watcher.result()
        if control.revision != revision:
            await settle()
            if quiesce_tools:
                await control.wait_quiescent()
            raise RunInterrupted
        result = await task
        if control.revision != revision:
            if quiesce_tools:
                await control.wait_quiescent()
            raise RunInterrupted
        return result
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await settle()
        if quiesce_tools:
            await control.wait_quiescent()


def _consume_exception(task: asyncio.Future[T]) -> None:
    if not task.cancelled():
        task.exception()


__all__ = ["AgentInput", "InputReceipt"]
