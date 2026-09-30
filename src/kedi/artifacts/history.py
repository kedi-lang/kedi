from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Literal, TypeAlias, cast


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
    """Thread-safe recent lifecycle log with lifetime-monotonic sequence IDs."""

    DEFAULT_MAX_EVENTS = 4096

    def __init__(self, *, max_events: int | None = DEFAULT_MAX_EVENTS) -> None:
        if max_events is not None and (type(max_events) is not int or max_events < 1):
            raise ValueError("artifact history max_events must be a positive integer or None")
        self._lock = RLock()
        self._events: deque[ArtifactHistoryEvent] = deque(maxlen=max_events)
        self._next_sequence = 1

    @property
    def max_events(self) -> int | None:
        return self._events.maxlen

    @property
    def retained_events(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._next_sequence - 1 - len(self._events)

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

    def export_checkpoint(self) -> dict[str, Any]:
        """Export retention and event positions atomically, not provider history."""
        with self._lock:
            return {
                "events": self.as_jsonable(),
                "max_events": self.max_events,
                "first_sequence": self._next_sequence - len(self._events),
            }

    @classmethod
    def from_checkpoint(cls, value: object) -> ArtifactHistory:
        if isinstance(value, list):
            return cls.from_jsonable(value)
        if not isinstance(value, dict):
            raise ValueError("artifact history checkpoint is invalid")
        checkpoint = cast(dict[str, Any], value)
        if set(checkpoint) != {"events", "max_events", "first_sequence"}:
            raise ValueError("artifact history checkpoint is invalid")
        return cls.from_jsonable(
            checkpoint["events"],
            max_events=checkpoint["max_events"],
            first_sequence=checkpoint["first_sequence"],
        )

    @classmethod
    def from_jsonable(
        cls,
        values: object,
        *,
        max_events: int | None = DEFAULT_MAX_EVENTS,
        first_sequence: int = 1,
    ) -> ArtifactHistory:
        history = cls(max_events=max_events)
        if type(first_sequence) is not int or first_sequence < 1:
            raise ValueError("artifact history sequence is invalid")
        if not isinstance(values, list):
            raise ValueError("artifact history must be a list")
        for index, item in enumerate(cast(list[object], values), start=first_sequence):
            if not isinstance(item, dict):
                raise ValueError("artifact history event is invalid")
            value = cast(dict[str, Any], item)
            if type(value.get("sequence")) is not int or value.get("sequence") != index:
                raise ValueError("artifact history sequence is invalid")
            kind = value.get("kind")
            ref_id = value.get("ref_id")
            data = value.get("data")
            if not isinstance(kind, str) or not isinstance(data, dict):
                raise ValueError("artifact history event is invalid")
            if ref_id is not None and not isinstance(ref_id, str):
                raise ValueError("artifact history reference is invalid")
            timestamp = value.get("created_at")
            if not isinstance(timestamp, str):
                raise ValueError("artifact history timestamp is invalid")
            created_at = datetime.fromisoformat(timestamp)
            if created_at.tzinfo is None:
                raise ValueError("artifact history timestamp must include a timezone")
            history._events.append(
                ArtifactHistoryEvent(
                    sequence=index,
                    kind=kind,
                    created_at=created_at,
                    ref_id=ref_id,
                    data=data,
                )
            )
        history._next_sequence = first_sequence + len(values)
        return history
