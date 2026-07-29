from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal, TypeAlias

from . import telemetry


@dataclass(frozen=True, kw_only=True)
class ArtifactHistoryEvent:
    sequence: int
    kind: str
    created_at: datetime
    ref_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


EventKind: TypeAlias = Literal[
    "tool_call",
    "tool_result",
    "artifact_created",
    "artifact_read",
    "artifact_released",
    "artifact_expired",
    "artifact_tombstone",
    "adapter_checkpoint_invalidated",
]


class ArtifactHistory:
    """Thread-safe portable artifact lifecycle log."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._events: list[ArtifactHistoryEvent] = []
        self._next_sequence = 1
        self._checkpoint_stale = False
        self._compacted_refs: set[str] = set()

    @property
    def checkpoint_stale(self) -> bool:
        with self._lock:
            return self._checkpoint_stale

    def append(
        self,
        kind: EventKind,
        *,
        ref_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> ArtifactHistoryEvent:
        with self._lock:
            event = ArtifactHistoryEvent(
                sequence=self._next_sequence,
                kind=kind,
                created_at=datetime.now(timezone.utc),
                ref_id=ref_id,
                data=dict(data or {}),
            )
            self._next_sequence += 1
            self._events.append(event)
            return event

    def compact(self, ref_id: str, *, reason: str) -> int:
        with telemetry.artifact_span("history compact", ref_id=ref_id, reason=reason):
            with self._lock:
                if ref_id in self._compacted_refs:
                    return 0
                linked_call_ids = {
                    event.data.get("call_id")
                    for event in self._events
                    if event.ref_id == ref_id and isinstance(event.data.get("call_id"), str)
                }
                retained: list[ArtifactHistoryEvent] = []
                removed = 0
                for event in self._events:
                    call_id = event.data.get("call_id")
                    linked_tool_event = (
                        event.kind in {"tool_call", "tool_result"} and call_id in linked_call_ids
                    )
                    if (
                        event.ref_id == ref_id and event.kind == "artifact_read"
                    ) or linked_tool_event:
                        removed += 1
                        continue
                    retained.append(event)
                retained.append(
                    ArtifactHistoryEvent(
                        sequence=self._next_sequence,
                        kind="artifact_tombstone",
                        created_at=datetime.now(timezone.utc),
                        ref_id=ref_id,
                        data={"reason": reason, "removed_events": removed},
                    )
                )
                self._next_sequence += 1
                self._checkpoint_stale = True
                retained.append(
                    ArtifactHistoryEvent(
                        sequence=self._next_sequence,
                        kind="adapter_checkpoint_invalidated",
                        created_at=datetime.now(timezone.utc),
                        ref_id=ref_id,
                        data={"reason": reason},
                    )
                )
                self._next_sequence += 1
                self._events = retained
                self._compacted_refs.add(ref_id)
                telemetry.record_history_compaction(removed_events=removed)
                return removed

    def mark_checkpoint_fresh(self) -> None:
        with self._lock:
            self._checkpoint_stale = False

    def snapshot(self) -> tuple[ArtifactHistoryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def as_jsonable(self) -> list[dict[str, Any]]:
        with self._lock:
            events = [asdict(event) for event in self._events]
        for event in events:
            event["created_at"] = event["created_at"].isoformat()
        return events
