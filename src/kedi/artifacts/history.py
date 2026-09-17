from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal, TypeAlias


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
    "artifact_query",
    "artifact_released",
    "artifact_expired",
]


class ArtifactHistory:
    """Thread-safe portable artifact lifecycle log."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._events: list[ArtifactHistoryEvent] = []
        self._next_sequence = 1

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

    def snapshot(self) -> tuple[ArtifactHistoryEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def as_jsonable(self) -> list[dict[str, Any]]:
        with self._lock:
            events = [asdict(event) for event in self._events]
        for event in events:
            event["created_at"] = event["created_at"].isoformat()
        return events
