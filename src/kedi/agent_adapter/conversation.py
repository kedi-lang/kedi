from __future__ import annotations

import json
import secrets
from hashlib import sha256
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Iterator, Literal, Protocol, runtime_checkable

from kedi.artifacts import ArtifactHistory, ArtifactManager

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


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterStateSnapshot:
    """Immutable lane coordinates plus a shallow copy of adapter-owned state."""

    adapter_shortname: str
    cache_epoch: int
    state: dict[str, Any]


@runtime_checkable
class SupportsConversationState(Protocol):
    def conversation_scope(
        self,
        conversation: AdapterConversationContext,
    ) -> AbstractContextManager[None]:
        """Bind process-local continuation state for one adapter call."""
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
        self._turns: list[ConversationTurn] = []
        self._adapter_states: dict[str, dict[str, Any]] = {}
        self._cache_epoch = 0
        self._closed = False

    @property
    def cache_epoch(self) -> int:
        with self._lock:
            return self._cache_epoch

    @contextmanager
    def adapter_turn(
        self,
        adapter_shortname: str,
    ) -> Iterator[AdapterConversationContext]:
        with self._lock:
            self._ensure_open()
            resume_state = dict(self._adapter_states.get(adapter_shortname, {}))
            context = AdapterConversationContext(
                session_id=self.session_id,
                cache_epoch=self._cache_epoch,
                cache_key=self._cache_key(adapter_shortname),
                resume_state=resume_state,
            )
            try:
                yield context
            except BaseException:
                raise
            else:
                if context.cache_prefix_changed:
                    self._cache_epoch += 1
                    self._adapter_states.clear()
                self._adapter_states[adapter_shortname] = dict(context.result_state)

    def advance_cache_epoch(self) -> int:
        """Start a fresh provider-continuation epoch without changing the session."""

        with self._lock:
            self._ensure_open()
            self._cache_epoch += 1
            self._adapter_states.clear()
            return self._cache_epoch

    def adapter_state_snapshot(self, adapter_shortname: str) -> AdapterStateSnapshot:
        """Read one continuation lane without entering or committing an adapter turn."""

        with self._lock:
            self._ensure_open()
            return AdapterStateSnapshot(
                adapter_shortname=adapter_shortname,
                cache_epoch=self._cache_epoch,
                state=dict(self._adapter_states.get(adapter_shortname, {})),
            )

    def record_exchange(self, prompt: str, result: Any) -> None:
        assistant = _portable_content(result)
        with self._lock:
            self._ensure_open()
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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._adapter_states.clear()
            self._turns.clear()
        if self._owns_artifacts:
            self.artifacts.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("conversation state is closed")

    def _cache_key(self, adapter_shortname: str) -> str:
        material = f"{self.session_id}\0{adapter_shortname}\0{self._cache_epoch}".encode()
        digest = sha256(material).hexdigest()[:24]
        return f"kedi_cache_{digest}_e{self._cache_epoch}"


def _conversation_turn(role: ConversationRole, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content)


def _portable_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
