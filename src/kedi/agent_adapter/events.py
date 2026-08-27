from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from threading import Condition, Lock, RLock, Thread
from typing import Literal, Protocol, TypeAlias, cast
from uuid import uuid4

AgentMessagePhase: TypeAlias = Literal["commentary", "final"]
AgentRunState: TypeAlias = Literal["started", "completed", "failed", "cancelled"]

_LOGGER = logging.getLogger(__name__)
_COMMENTARY_CAPACITY = 256
_SINK_FAILURE_LIMIT = 3
_CLOSE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class AgentMessageEvent:
    run_id: str
    message_id: str
    adapter: str
    sequence: int
    phase: AgentMessagePhase
    content: str
    parent_run_id: str | None = None
    segment_index: int = 0
    segment_count: int = 1
    agent_name: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunStateEvent:
    run_id: str
    adapter: str
    sequence: int
    state: AgentRunState
    parent_run_id: str | None = None
    agent_name: str | None = None


AgentStreamEvent: TypeAlias = AgentMessageEvent | AgentRunStateEvent


class AgentEventSink(Protocol):
    def emit(self, event: AgentStreamEvent) -> None: ...


AgentEventCallback: TypeAlias = Callable[[AgentStreamEvent], None]


@dataclass(frozen=True, slots=True)
class AgentRunIdentity:
    run_id: str
    parent_run_id: str | None = None
    agent_name: str | None = None


@dataclass(slots=True)
class _MessageState:
    message_id: str
    chunks: list[str]
    content: str | None = None
    explicit_phase: AgentMessagePhase | None = None
    published: bool = False


class AgentEventDispatcher:
    """Deliver low-volume semantic events without blocking an adapter run."""

    def __init__(self, sink: AgentEventSink | AgentEventCallback) -> None:
        self._emit_event = cast(AgentEventCallback, sink) if callable(sink) else sink.emit
        self._condition = Condition(Lock())
        self._commentary: deque[tuple[int, AgentStreamEvent]] = deque()
        self._control: deque[tuple[int, AgentStreamEvent]] = deque()
        self._arrival = 0
        self._sequences: dict[str, int] = {}
        self._closing = False
        self._detached = False
        self._sink_failures = 0
        self._worker = Thread(
            target=self._deliver,
            name=f"kedi-agent-events-{id(self)}",
            daemon=True,
        )
        self._worker.start()

    @property
    def active(self) -> bool:
        with self._condition:
            return not self._closing and not self._detached

    def publish(self, event: AgentStreamEvent) -> None:
        with self._condition:
            if self._closing or self._detached:
                return
            sequence = self._sequences.get(event.run_id, 0) + 1
            self._sequences[event.run_id] = sequence
            event = replace(event, sequence=sequence)
            self._arrival += 1
            item = (self._arrival, event)
            if isinstance(event, AgentMessageEvent) and event.phase == "commentary":
                if len(self._commentary) >= _COMMENTARY_CAPACITY:
                    self._commentary.popleft()
                    _LOGGER.warning("Kedi agent commentary event queue overflowed")
                self._commentary.append(item)
            else:
                self._control.append(item)
            self._condition.notify()

    def close(self, *, timeout: float = _CLOSE_TIMEOUT_SECONDS) -> None:
        with self._condition:
            if self._closing:
                return
            self._closing = True
            self._condition.notify_all()
        self._worker.join(timeout=max(timeout, 0.0))
        if self._worker.is_alive():
            with self._condition:
                self._detached = True
                self._commentary.clear()
                self._control.clear()
                self._condition.notify_all()
            _LOGGER.warning("Kedi agent event sink did not drain before timeout")

    def _deliver(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._commentary
                    and not self._control
                    and not self._closing
                    and not self._detached
                ):
                    self._condition.wait()
                if self._detached:
                    return
                if not self._commentary and not self._control:
                    return
                event = self._pop_next()
            try:
                self._emit(event)
            except Exception:
                _LOGGER.exception("Kedi agent event sink failed")
                with self._condition:
                    self._sink_failures += 1
                    if self._sink_failures >= _SINK_FAILURE_LIMIT:
                        self._detached = True
                        self._commentary.clear()
                        self._control.clear()
                        self._condition.notify_all()
                        return
            else:
                with self._condition:
                    self._sink_failures = 0

    def _pop_next(self) -> AgentStreamEvent:
        if not self._commentary:
            return self._control.popleft()[1]
        if not self._control:
            return self._commentary.popleft()[1]
        if self._commentary[0][0] < self._control[0][0]:
            return self._commentary.popleft()[1]
        return self._control.popleft()[1]

    def _emit(self, event: AgentStreamEvent) -> None:
        self._emit_event(event)


_EVENT_DISPATCHER: ContextVar[AgentEventDispatcher | None] = ContextVar(
    "kedi_agent_event_dispatcher",
    default=None,
)
_BOUND_RUN_IDENTITY: ContextVar[AgentRunIdentity | None] = ContextVar(
    "kedi_bound_agent_run_identity",
    default=None,
)
_ACTIVE_RUN_IDENTITY: ContextVar[AgentRunIdentity | None] = ContextVar(
    "kedi_active_agent_run_identity",
    default=None,
)


@contextmanager
def observe_agent_events(
    sink: AgentEventSink | AgentEventCallback,
) -> Iterator[None]:
    dispatcher = AgentEventDispatcher(sink)
    try:
        with bind_agent_event_dispatcher(dispatcher):
            yield
    finally:
        dispatcher.close()


@contextmanager
def bind_agent_event_dispatcher(
    dispatcher: AgentEventDispatcher,
) -> Iterator[None]:
    """Bind a caller-owned dispatcher to the current task without closing it."""

    if not dispatcher.active:
        raise RuntimeError("Cannot bind a closed Kedi agent event dispatcher")
    token = _EVENT_DISPATCHER.set(dispatcher)
    try:
        yield
    finally:
        _EVENT_DISPATCHER.reset(token)


def current_agent_event_dispatcher() -> AgentEventDispatcher | None:
    return _EVENT_DISPATCHER.get()


def current_agent_run_identity() -> AgentRunIdentity | None:
    return _ACTIVE_RUN_IDENTITY.get()


def current_bound_agent_run_identity() -> AgentRunIdentity | None:
    return _BOUND_RUN_IDENTITY.get()


@contextmanager
def bind_agent_run(
    run_id: str,
    *,
    parent_run_id: str | None = None,
    agent_name: str | None = None,
) -> Iterator[None]:
    token = _BOUND_RUN_IDENTITY.set(AgentRunIdentity(run_id, parent_run_id, agent_name))
    try:
        yield
    finally:
        _BOUND_RUN_IDENTITY.reset(token)


class AgentMessageResolver:
    """Resolve provider message boundaries into immutable Kedi events."""

    def __init__(
        self,
        dispatcher: AgentEventDispatcher,
        *,
        adapter: str,
        identity: AgentRunIdentity,
    ) -> None:
        self.dispatcher = dispatcher
        self.adapter = adapter
        self.identity = identity
        self._messages: dict[str, _MessageState] = {}
        self._completed_order: list[str] = []
        self._terminal = False
        self._active_token: Token[AgentRunIdentity | None] | None = None
        self._lock = RLock()

    @property
    def run_id(self) -> str:
        return self.identity.run_id

    @property
    def parent_run_id(self) -> str | None:
        return self.identity.parent_run_id

    def activate(self) -> None:
        self._active_token = _ACTIVE_RUN_IDENTITY.set(self.identity)
        self._publish_state("started")

    def deactivate(self) -> None:
        if self._active_token is not None:
            _ACTIVE_RUN_IDENTITY.reset(self._active_token)
            self._active_token = None

    def append_delta(
        self,
        message_id: str,
        text: str,
        *,
        explicit_phase: AgentMessagePhase | None = None,
    ) -> None:
        with self._lock:
            if not text or self._terminal:
                return
            state = self._message(message_id)
            state.chunks.append(text)
            if explicit_phase is not None:
                state.explicit_phase = explicit_phase

    def complete_message(
        self,
        message_id: str,
        text: str | None = None,
        *,
        explicit_phase: AgentMessagePhase | None = None,
    ) -> None:
        with self._lock:
            if self._terminal:
                return
            state = self._message(message_id)
            content = text if text is not None else "".join(state.chunks)
            content = content.strip()
            if not content:
                self._messages.pop(message_id, None)
                return
            state.content = content
            if explicit_phase is not None:
                state.explicit_phase = explicit_phase
            if message_id not in self._completed_order:
                self._completed_order.append(message_id)
            if state.explicit_phase == "commentary":
                self._publish_message(state, "commentary")

    def mark_tool_boundary(self) -> None:
        with self._lock:
            if not self._terminal:
                self._publish_pending_commentary()

    def mark_message_boundary(self) -> None:
        with self._lock:
            if self._terminal:
                return
            pending = self._pending_states()
            for state in pending[:-1]:
                self._publish_message(state, "commentary")

    def complete_run(self, *, final_text: str | None = None) -> None:
        with self._lock:
            if self._terminal:
                return
            if final_text is not None:
                normalized = _normalize_message(final_text)
                candidates = self._pending_states()
                final_state = next(
                    (
                        state
                        for state in reversed(candidates)
                        if state.content is not None
                        and _normalize_message(state.content) == normalized
                    ),
                    None,
                )
                if final_state is None:
                    message_id = f"{self.run_id}:final"
                    final_state = _MessageState(
                        message_id=message_id,
                        chunks=[],
                        content=final_text,
                        explicit_phase="final",
                    )
                for state in candidates:
                    if (
                        state is not final_state
                        and state.explicit_phase != "final"
                        and state.content is not None
                        and _normalize_message(state.content) != normalized
                    ):
                        self._publish_message(state, "commentary")
                final_state.content = final_text
                self._publish_message(final_state, "final")
            self._terminal = True
            self._publish_state("completed")
            self._discard_private_state()

    def fail_run(self) -> None:
        self._finish_unsuccessfully("failed")

    def cancel_run(self) -> None:
        self._finish_unsuccessfully("cancelled")

    def _finish_unsuccessfully(self, state: Literal["failed", "cancelled"]) -> None:
        with self._lock:
            if self._terminal:
                return
            self._terminal = True
            self._discard_private_state()
            self._publish_state(state)

    def _message(self, message_id: str) -> _MessageState:
        state = self._messages.get(message_id)
        if state is None:
            state = _MessageState(message_id=message_id, chunks=[])
            self._messages[message_id] = state
        return state

    def _pending_states(self) -> list[_MessageState]:
        return [
            self._messages[message_id]
            for message_id in self._completed_order
            if message_id in self._messages and not self._messages[message_id].published
        ]

    def _publish_pending_commentary(self) -> None:
        for state in self._pending_states():
            if state.explicit_phase != "final":
                self._publish_message(state, "commentary")

    def _publish_message(self, state: _MessageState, phase: AgentMessagePhase) -> None:
        if state.published or state.content is None:
            return
        state.published = True
        self.dispatcher.publish(
            AgentMessageEvent(
                run_id=self.run_id,
                parent_run_id=self.parent_run_id,
                message_id=state.message_id,
                adapter=self.adapter,
                sequence=0,
                phase=phase,
                content=state.content,
                agent_name=self.identity.agent_name,
            )
        )

    def _publish_state(self, state: AgentRunState) -> None:
        self.dispatcher.publish(
            AgentRunStateEvent(
                run_id=self.run_id,
                parent_run_id=self.parent_run_id,
                adapter=self.adapter,
                sequence=0,
                state=state,
                agent_name=self.identity.agent_name,
            )
        )

    def _discard_private_state(self) -> None:
        self._messages.clear()
        self._completed_order.clear()


@contextmanager
def agent_event_run(adapter: str) -> Iterator[AgentMessageResolver | None]:
    dispatcher = current_agent_event_dispatcher()
    if dispatcher is None:
        yield None
        return

    identity = _BOUND_RUN_IDENTITY.get() or AgentRunIdentity(f"agent_{uuid4().hex}")
    resolver = AgentMessageResolver(dispatcher, adapter=adapter, identity=identity)
    resolver.activate()
    try:
        yield resolver
    except asyncio.CancelledError:
        resolver.cancel_run()
        raise
    except BaseException:
        resolver.fail_run()
        raise
    else:
        resolver.complete_run()
    finally:
        resolver.deactivate()


class AsyncAgentEventQueue:
    """Thread-safe sink adapter for asyncio consumers."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.loop = loop or asyncio.get_running_loop()
        self.queue: asyncio.Queue[AgentStreamEvent] = asyncio.Queue()

    def emit(self, event: AgentStreamEvent) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)


def _normalize_message(content: str) -> str:
    return " ".join(content.split())
