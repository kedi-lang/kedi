from __future__ import annotations

import json
import secrets
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
    resume_state: dict[str, Any] = field(default_factory=dict)
    result_state: dict[str, Any] = field(default_factory=dict)


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

    def __init__(self, *, session_id: str | None = None) -> None:
        self.session_id = session_id or f"conversation_{secrets.token_urlsafe(18)}"
        self.history = ArtifactHistory()
        self.artifacts = ArtifactManager(
            session_id=self.session_id,
            history=self.history,
        )
        self._lock = RLock()
        self._turns: list[ConversationTurn] = []
        self._adapter_states: dict[str, dict[str, Any]] = {}
        self._closed = False

    @contextmanager
    def adapter_turn(
        self,
        adapter_shortname: str,
    ) -> Iterator[AdapterConversationContext]:
        with self._lock:
            self._ensure_open()
            resume_state = dict(self._adapter_states.get(adapter_shortname, {}))
            context = AdapterConversationContext(
                resume_state=resume_state,
            )
            try:
                yield context
            except BaseException:
                raise
            else:
                self._adapter_states[adapter_shortname] = dict(context.result_state)

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
        self.artifacts.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("conversation state is closed")


def _conversation_turn(role: ConversationRole, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content)


def _portable_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
