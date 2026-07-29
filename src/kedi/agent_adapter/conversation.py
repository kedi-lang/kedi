from __future__ import annotations

import json
import re
import secrets
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Iterator, Literal, Mapping, Protocol, runtime_checkable

from kedi.artifacts import ArtifactHistory, ArtifactManager
from kedi.artifacts import telemetry as artifact_telemetry

ConversationRole = Literal["user", "assistant"]
_ARTIFACT_REF_RE = re.compile(r"(?<![A-Za-z0-9_-])art_[A-Za-z0-9_-]+(?![A-Za-z0-9_-])")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConversationTurn:
    role: ConversationRole
    content: str
    ref_ids: tuple[str, ...] = ()


@dataclass(slots=True, kw_only=True)
class AdapterConversationContext:
    resume_state: dict[str, Any] = field(default_factory=dict)
    result_state: dict[str, Any] = field(default_factory=dict)
    replay_required: bool = False


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
        self._processed_tombstone_sequence = 0
        self._closed = False

    @contextmanager
    def adapter_turn(
        self,
        adapter_shortname: str,
    ) -> Iterator[AdapterConversationContext]:
        with self._lock:
            self._ensure_open()
            with artifact_telemetry.artifact_span(
                "checkpoint rotate",
                session_id=self.session_id,
                adapter=adapter_shortname,
            ):
                replay_required = self._apply_pending_compaction_locked()
            resume_state = (
                {} if replay_required else dict(self._adapter_states.get(adapter_shortname, {}))
            )
            context = AdapterConversationContext(
                resume_state=resume_state,
                replay_required=replay_required,
            )
            try:
                yield context
            except BaseException:
                raise
            else:
                self._adapter_states[adapter_shortname] = dict(context.result_state)
                if replay_required:
                    self.history.mark_checkpoint_fresh()

    def replay_prompt(self, prompt: str) -> str:
        with self._lock:
            if not self._turns:
                return prompt
            transcript = [{"role": turn.role, "content": turn.content} for turn in self._turns]
        return (
            "Continue from this compact Kedi conversation transcript:\n"
            f"{json.dumps(transcript, ensure_ascii=False)}\n\n"
            f"Current request:\n{prompt}"
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

    def _apply_pending_compaction_locked(self) -> bool:
        tombstones = [
            event
            for event in self.history.snapshot()
            if event.kind == "artifact_tombstone"
            and event.sequence > self._processed_tombstone_sequence
            and event.ref_id is not None
        ]
        if not tombstones and not self.history.checkpoint_stale:
            return False
        for event in tombstones:
            ref_id = event.ref_id
            assert ref_id is not None
            self._turns = [_compact_turn(turn, ref_id) for turn in self._turns]
            self._processed_tombstone_sequence = max(
                self._processed_tombstone_sequence,
                event.sequence,
            )
        self._adapter_states.clear()
        return True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("conversation state is closed")


def _conversation_turn(role: ConversationRole, content: str) -> ConversationTurn:
    return ConversationTurn(
        role=role,
        content=content,
        ref_ids=tuple(dict.fromkeys(_ARTIFACT_REF_RE.findall(content))),
    )


def _portable_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _compact_turn(turn: ConversationTurn, ref_id: str) -> ConversationTurn:
    if ref_id not in turn.ref_ids:
        return turn
    content = turn.content
    try:
        portable = json.loads(content)
    except (TypeError, ValueError):
        content = content.replace(ref_id, f"[unavailable artifact {ref_id}]")
    else:
        content = json.dumps(
            _compact_portable_value(portable, ref_id),
            ensure_ascii=False,
            sort_keys=True,
        )
    return ConversationTurn(
        role=turn.role,
        content=content,
        ref_ids=tuple(item for item in turn.ref_ids if item != ref_id),
    )


def _compact_portable_value(value: Any, ref_id: str) -> Any:
    if isinstance(value, Mapping):
        if value.get("ref_id") == ref_id:
            return {
                "ref_id": f"[unavailable artifact {ref_id}]",
                "status": "unavailable",
            }
        return {str(key): _compact_portable_value(item, ref_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_portable_value(item, ref_id) for item in value]
    if isinstance(value, str):
        return value.replace(ref_id, f"[unavailable artifact {ref_id}]")
    return value
