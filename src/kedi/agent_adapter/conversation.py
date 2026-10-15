from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator, Iterable
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from threading import Condition, RLock, local
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Protocol, runtime_checkable

from kedi.artifacts import ArtifactHistory, ArtifactManager

if TYPE_CHECKING:
    from .request import AdapterRunOutcome

ConversationRole = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationTurn:
    role: ConversationRole
    content: str


@dataclass(slots=True, kw_only=True)
class AdapterConversationContext:
    session_id: str = ""
    cache_epoch: int = 0
    cache_key: str = ""
    resume_state: dict[str, Any] = field(default_factory=dict)
    result_state: dict[str, Any] = field(default_factory=dict)
    cache_prefix_changed: bool = False
    cleanup: Callable[[], None] | None = None
    lane_identity: str = ""
    turn_sequence: int = 0
    _exchange_prompt: str | None = field(default=None, repr=False)
    _exchange_result: Any = field(default=None, repr=False)
    _has_exchange: bool = field(default=False, repr=False)
    _requires_explicit_completion: bool = field(default=False, repr=False)
    _completed: bool = field(default=False, repr=False)

    def record_exchange(self, prompt: str, result: Any) -> None:
        self._exchange_prompt = prompt
        self._exchange_result = result
        self._has_exchange = True

    def apply_outcome(self, prompt: str, outcome: AdapterRunOutcome) -> None:
        """Stage one completed adapter outcome for atomic transaction commit."""

        self.result_state = dict(outcome.continuation_state)
        self.cleanup = outcome.cleanup
        self.cache_prefix_changed = outcome.cache_prefix_changed
        self.record_exchange(prompt, outcome.portable_result)
        self.mark_completed()

    def require_explicit_completion(self) -> None:
        """Commit this context only after its adapter reports a completed native run."""

        self._requires_explicit_completion = True

    def mark_completed(self) -> None:
        """Mark native continuation changes as safe to commit."""

        self._completed = True

    def advance_cache_generation(self) -> int:
        """Stage a provider-prefix generation change for this active turn."""

        self.cache_epoch += 1
        self.cache_prefix_changed = True
        if self.session_id and self.lane_identity:
            self.cache_key = _conversation_cache_key(
                self.session_id,
                self.lane_identity,
                self.cache_epoch,
            )
        return self.cache_epoch


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationTurnTicket:
    sequence: int


class _ConversationTurnCancelled(RuntimeError):
    """Internal signal for a reserved turn retired before adapter execution."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterStateSnapshot:
    """Immutable lane coordinates plus a shallow copy of adapter-owned state."""

    adapter_shortname: str
    lane_identity: str
    cache_epoch: int
    state: dict[str, Any]


@runtime_checkable
class SupportsConversationState(Protocol):
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> AbstractContextManager[None]:
        """Bind continuation state and mark it completed only after a terminal result."""
        ...


class ConversationState:
    """Explicit process-local conversation and artifact session state."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        artifacts: ArtifactManager | None = None,
    ) -> None:
        if artifacts is not None and session_id is not None and session_id != artifacts.session_id:
            raise ValueError("conversation and artifact session IDs must match")
        self.session_id = (
            artifacts.session_id
            if artifacts is not None
            else session_id or f"conversation_{secrets.token_urlsafe(18)}"
        )
        self._owns_artifacts = artifacts is None
        self.history = artifacts.history if artifacts is not None else ArtifactHistory()
        self.artifacts = artifacts or ArtifactManager(
            session_id=self.session_id,
            history=self.history,
        )
        self._lock = RLock()
        self._turn_condition = Condition(self._lock)
        self._turns: list[ConversationTurn] = []
        self._adapter_states: dict[str, dict[str, Any]] = {}
        self._adapter_lane_identities: dict[str, str] = {}
        self._adapter_cleanups: dict[str, Callable[[], None]] = {}
        self._cache_epoch = 0
        self._closed = False
        self._next_ticket = 0
        self._next_turn = 0
        self._active_ticket: int | None = None
        self._cancelled_tickets: set[int] = set()
        self._sync_turn_local = local()
        self._async_turn_owner: ContextVar[asyncio.Task[Any] | None] = ContextVar(
            f"kedi_conversation_turn_owner_{id(self)}",
            default=None,
        )

    @property
    def cache_epoch(self) -> int:
        with self._lock:
            return self._cache_epoch

    @contextmanager
    def adapter_turn(
        self,
        adapter_shortname: str,
        *,
        ticket: ConversationTurnTicket | None = None,
        lane_identity: str | None = None,
    ) -> Iterator[AdapterConversationContext]:
        current_task = _current_async_task()
        if getattr(self._sync_turn_local, "active", False) or (
            current_task is not None and self._async_turn_owner.get() is current_task
        ):
            raise RuntimeError("nested turns on the same conversation are not supported")
        active_ticket = ticket or self.reserve_turn()
        lane = lane_identity or adapter_shortname
        context = self._begin_turn(active_ticket, adapter_shortname, lane)
        self._sync_turn_local.active = True
        succeeded = False
        try:
            yield context
            succeeded = True
        finally:
            self._sync_turn_local.active = False
            _run_cleanups(
                self._finish_turn(
                    context,
                    adapter_shortname,
                    lane,
                    succeeded=succeeded,
                )
            )

    @asynccontextmanager
    async def adapter_turn_async(
        self,
        adapter_shortname: str,
        *,
        ticket: ConversationTurnTicket | None = None,
        lane_identity: str | None = None,
    ) -> AsyncIterator[AdapterConversationContext]:
        """Task-safe async form that never blocks the event-loop thread while queued."""

        current_task = asyncio.current_task()
        if current_task is None:  # pragma: no cover - async context managers always have a task
            raise RuntimeError("async conversation turns require a running task")
        if self._async_turn_owner.get() is current_task or getattr(
            self._sync_turn_local, "active", False
        ):
            raise RuntimeError("nested turns on the same conversation are not supported")
        active_ticket = ticket or self.reserve_turn()
        lane = lane_identity or adapter_shortname
        begin_task = asyncio.create_task(
            asyncio.to_thread(
                self._begin_turn,
                active_ticket,
                adapter_shortname,
                lane,
            )
        )
        try:
            context = await asyncio.shield(begin_task)
        except asyncio.CancelledError as cancellation:
            self.cancel_turn(active_ticket)
            try:
                context = await asyncio.shield(begin_task)
            except _ConversationTurnCancelled:
                pass
            else:
                _run_cleanups(
                    self._finish_turn(
                        context,
                        adapter_shortname,
                        lane,
                        succeeded=False,
                    )
                )
            raise cancellation
        owner_token = self._async_turn_owner.set(current_task)
        succeeded = False
        try:
            yield context
            succeeded = True
        finally:
            try:
                cleanups = self._finish_turn(
                    context,
                    adapter_shortname,
                    lane,
                    succeeded=succeeded,
                )
                _run_cleanups(cleanups)
            finally:
                self._async_turn_owner.reset(owner_token)

    def reserve_turn(self) -> ConversationTurnTicket:
        """Reserve source order before a possibly deferred model job is scheduled."""

        with self._turn_condition:
            self._ensure_open()
            ticket = ConversationTurnTicket(sequence=self._next_ticket)
            self._next_ticket += 1
            return ticket

    def cancel_turn(self, ticket: ConversationTurnTicket) -> None:
        """Retire a reserved turn that will never enter adapter execution."""

        with self._turn_condition:
            if ticket.sequence < self._next_turn:
                return
            if ticket.sequence == self._active_ticket:
                return
            self._cancelled_tickets.add(ticket.sequence)
            self._advance_cancelled_tickets()
            self._turn_condition.notify_all()

    def advance_cache_epoch(self) -> int:
        """Start a fresh provider-continuation epoch without changing the session."""

        with self._turn_condition:
            self._ensure_open()
            if self._active_ticket is not None:
                raise RuntimeError("cannot advance cache epoch during an active conversation turn")
            self._cache_epoch += 1
            self._adapter_states.clear()
            self._adapter_lane_identities.clear()
            cleanups = tuple(self._adapter_cleanups.values())
            self._adapter_cleanups.clear()
            cache_epoch = self._cache_epoch
        _run_cleanups(cleanups)
        return cache_epoch

    def adapter_state_snapshot(self, adapter_shortname: str) -> AdapterStateSnapshot:
        """Read one continuation lane without entering or committing an adapter turn."""

        with self._lock:
            self._ensure_open()
            return AdapterStateSnapshot(
                adapter_shortname=adapter_shortname,
                lane_identity=self._adapter_lane_identities.get(
                    adapter_shortname,
                    adapter_shortname,
                ),
                cache_epoch=self._cache_epoch,
                state=dict(self._adapter_states.get(adapter_shortname, {})),
            )

    def record_exchange(self, prompt: str, result: Any) -> None:
        assistant = _portable_content(result)
        with self._turn_condition:
            self._ensure_open()
            if self._active_ticket is not None:
                raise RuntimeError(
                    "record an active conversation exchange on AdapterConversationContext"
                )
            self._turns.extend(
                (
                    _conversation_turn("user", prompt),
                    _conversation_turn("assistant", assistant),
                )
            )

    def snapshot(self) -> tuple[ConversationTurn, ...]:
        with self._lock:
            return tuple(self._turns)

    def as_jsonable(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "cache_epoch": self._cache_epoch,
                "turns": [asdict(turn) for turn in self._turns],
                "artifact_history": self.history.as_jsonable(),
            }

    def export_checkpoint(self, adapter_shortname: str) -> dict[str, Any]:
        """Export a completed native lane, excluding live transport and cleanup objects."""

        with self._turn_condition:
            self._ensure_open()
            if self._active_ticket is not None:
                raise RuntimeError("cannot checkpoint during an active conversation turn")
            other_lanes = self._adapter_states.keys() - {adapter_shortname}
            if other_lanes:
                raise ValueError("A2A checkpoint cannot contain another adapter lane")
            state = _checkpoint_lane_state(
                adapter_shortname, self._adapter_states.get(adapter_shortname, {})
            )
            checkpoint = {
                "session_id": self.session_id,
                "cache_epoch": self._cache_epoch,
                "turns": [asdict(turn) for turn in self._turns],
                "lane_identity": self._adapter_lane_identities.get(
                    adapter_shortname, adapter_shortname
                ),
                "state": state,
            }
            json.dumps(checkpoint, allow_nan=False)
            return checkpoint

    @classmethod
    def from_checkpoint(
        cls,
        data: dict[str, Any],
        *,
        adapter_shortname: str,
        artifacts: ArtifactManager,
    ) -> ConversationState:
        if not isinstance(data, dict) or data.get("session_id") != artifacts.session_id:
            raise ValueError("conversation checkpoint session ID is invalid")
        cache_epoch = data.get("cache_epoch")
        lane_identity = data.get("lane_identity")
        if type(cache_epoch) is not int or cache_epoch < 0:
            raise ValueError("conversation checkpoint cache epoch is invalid")
        if not isinstance(lane_identity, str) or not lane_identity:
            raise ValueError("conversation checkpoint lane identity is invalid")
        turns = data.get("turns")
        if not isinstance(turns, list):
            raise ValueError("conversation checkpoint turns must be a list")
        parsed_turns = []
        for value in turns:
            if not isinstance(value, dict) or value.get("role") not in {"user", "assistant"}:
                raise ValueError("conversation checkpoint turn is invalid")
            if not isinstance(value.get("content"), str):
                raise ValueError("conversation checkpoint turn content is invalid")
            parsed_turns.append(ConversationTurn(role=value["role"], content=value["content"]))
        state = _restore_lane_state(adapter_shortname, data.get("state"))
        conversation = cls(artifacts=artifacts)
        conversation._turns = parsed_turns
        conversation._cache_epoch = cache_epoch
        if state:
            conversation._adapter_states[adapter_shortname] = state
            conversation._adapter_lane_identities[adapter_shortname] = lane_identity
        return conversation

    def close(self) -> None:
        with self._turn_condition:
            if self._closed:
                return
            if self._active_ticket is not None:
                raise RuntimeError("cannot close conversation state during an active turn")
            self._closed = True
            self._adapter_states.clear()
            self._adapter_lane_identities.clear()
            cleanups = tuple(self._adapter_cleanups.values())
            self._adapter_cleanups.clear()
            self._turns.clear()
            self._turn_condition.notify_all()
        _run_cleanups(cleanups)
        if self._owns_artifacts:
            self.artifacts.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("conversation state is closed")

    def _cache_key(self, adapter_shortname: str) -> str:
        return _conversation_cache_key(
            self.session_id,
            adapter_shortname,
            self._cache_epoch,
        )

    def _begin_turn(
        self,
        ticket: ConversationTurnTicket,
        adapter_shortname: str,
        lane: str,
    ) -> AdapterConversationContext:
        with self._turn_condition:
            self._ensure_open()
            while ticket.sequence != self._next_turn:
                if ticket.sequence in self._cancelled_tickets:
                    raise _ConversationTurnCancelled(
                        "conversation turn was cancelled before adapter execution"
                    )
                if ticket.sequence < self._next_turn:
                    raise _ConversationTurnCancelled(
                        "conversation turn ticket has already been retired"
                    )
                self._turn_condition.wait()
                self._ensure_open()
            self._active_ticket = ticket.sequence
            stored_lane = self._adapter_lane_identities.get(adapter_shortname)
            same_lane = stored_lane in {None, adapter_shortname, lane}
            return AdapterConversationContext(
                session_id=self.session_id,
                cache_epoch=self._cache_epoch,
                cache_key=self._cache_key(lane),
                resume_state=(
                    dict(self._adapter_states.get(adapter_shortname, {})) if same_lane else {}
                ),
                cleanup=(self._adapter_cleanups.get(adapter_shortname) if same_lane else None),
                lane_identity=lane,
                turn_sequence=ticket.sequence,
            )

    def _finish_turn(
        self,
        context: AdapterConversationContext,
        adapter_shortname: str,
        lane: str,
        *,
        succeeded: bool,
    ) -> tuple[Callable[[], None], ...]:
        retired_cleanups: list[Callable[[], None]] = []
        with self._turn_condition:
            previous_cleanup = self._adapter_cleanups.get(adapter_shortname)
            should_commit = succeeded and (
                not context._requires_explicit_completion or context._completed
            )
            if should_commit:
                if context._has_exchange:
                    assert context._exchange_prompt is not None
                    self._turns.extend(
                        (
                            _conversation_turn("user", context._exchange_prompt),
                            _conversation_turn(
                                "assistant",
                                _portable_content(context._exchange_result),
                            ),
                        )
                    )
                if context.cache_prefix_changed:
                    self._cache_epoch = max(self._cache_epoch + 1, context.cache_epoch)
                    self._adapter_states.clear()
                    self._adapter_lane_identities.clear()
                    retired_cleanups.extend(
                        cleanup
                        for cleanup in self._adapter_cleanups.values()
                        if cleanup is not context.cleanup
                    )
                    self._adapter_cleanups.clear()
                else:
                    if previous_cleanup is not None and previous_cleanup is not context.cleanup:
                        retired_cleanups.append(previous_cleanup)
                self._adapter_states[adapter_shortname] = dict(context.result_state)
                self._adapter_lane_identities[adapter_shortname] = lane
                if context.cleanup is None:
                    self._adapter_cleanups.pop(adapter_shortname, None)
                else:
                    self._adapter_cleanups[adapter_shortname] = context.cleanup
            elif context.cleanup is not None and context.cleanup is not previous_cleanup:
                # The failed turn may have opened a replacement native session before raising.
                # Keep the last committed session, but retire the uncommitted replacement.
                retired_cleanups.append(context.cleanup)
            self._active_ticket = None
            self._next_turn += 1
            self._advance_cancelled_tickets()
            self._turn_condition.notify_all()
        return tuple(retired_cleanups)

    def _advance_cancelled_tickets(self) -> None:
        while self._active_ticket is None and self._next_turn in self._cancelled_tickets:
            self._cancelled_tickets.remove(self._next_turn)
            self._next_turn += 1


def _conversation_turn(role: ConversationRole, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content)


def _conversation_cache_key(session_id: str, lane_identity: str, cache_epoch: int) -> str:
    material = f"{session_id}\0{lane_identity}\0{cache_epoch}".encode()
    digest = sha256(material).hexdigest()[:24]
    return f"kedi_cache_{digest}_e{cache_epoch}"


def _current_async_task() -> asyncio.Task[Any] | None:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _portable_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _run_cleanups(cleanups: Iterable[Callable[[], None]]) -> None:
    for cleanup in cleanups:
        try:
            cleanup()
        except Exception:
            # Native harness cleanup is best-effort and must not replace a completed model result.
            continue


def _lineage_adapter() -> Any:
    from pydantic import TypeAdapter

    from kedi.agent_adapter.history_provenance import HistoryLineage
    from kedi.lang.ast import Span

    adapter = TypeAdapter(HistoryLineage)
    adapter.rebuild(_types_namespace={"Span": Span})
    return adapter


def _checkpoint_lane_state(adapter_shortname: str, state: dict[str, Any]) -> dict[str, Any]:
    from .request_budget import HISTORY_USAGE_KEY, encode_history_usage_state

    native_key = {
        "pydantic": "pydantic_messages",
        "langchain": "langchain_messages",
    }.get(adapter_shortname)
    if native_key is None:
        raise ValueError(f"A2A durable history does not support adapter {adapter_shortname!r}")
    allowed = {
        native_key,
        "history_processor_origins",
        "history_processor_rewritten",
        HISTORY_USAGE_KEY,
    }
    if adapter_shortname == "langchain":
        allowed.add("langchain_instruction_digest")
    if state.keys() - allowed:
        raise ValueError("A2A native history has unsupported continuation fields")
    result = encode_history_usage_state(state)
    if "langchain_instruction_digest" in result:
        digest = result["langchain_instruction_digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("A2A native history instruction digest is invalid")
    if native_key in result:
        if adapter_shortname == "pydantic":
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            messages = ModelMessagesTypeAdapter.validate_python(result[native_key])
            result[native_key] = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        else:
            from langchain_core.messages import message_to_dict, messages_from_dict

            result[native_key] = [
                message_to_dict(message) for message in messages_from_dict(result[native_key])
            ]
    if "history_processor_origins" in result:
        result["history_processor_origins"] = _lineage_adapter().dump_python(
            result["history_processor_origins"], mode="json"
        )
    if (
        "history_processor_rewritten" in result
        and type(result["history_processor_rewritten"]) is not bool
    ):
        raise ValueError("A2A native history rewrite marker is invalid")
    return result


def _restore_lane_state(adapter_shortname: str, state: Any) -> dict[str, Any]:
    from .request_budget import decode_history_usage_state

    if not isinstance(state, dict):
        raise ValueError("A2A native history state must be an object")
    result = _checkpoint_lane_state(adapter_shortname, state)
    if "history_processor_origins" in result:
        result["history_processor_origins"] = _lineage_adapter().validate_python(
            result["history_processor_origins"]
        )
    return decode_history_usage_state(result)
