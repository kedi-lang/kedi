from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import metrics, trace

_SCOPE = "kedi.artifacts"
_tracer = trace.get_tracer(_SCOPE)
_meter = metrics.get_meter(_SCOPE)

_created = _meter.create_counter("kedi.artifact.created")
_released = _meter.create_counter("kedi.artifact.released")
_expired = _meter.create_counter("kedi.artifact.expired")
_reads = _meter.create_counter("kedi.artifact.reads")
_read_bytes = _meter.create_counter("kedi.artifact.read_bytes", unit="By")
_quota_rejections = _meter.create_counter("kedi.artifact.quota_rejections")
_context_bytes_avoided = _meter.create_counter(
    "kedi.artifact.context_bytes_avoided",
    unit="By",
)
_history_events_removed = _meter.create_counter("kedi.artifact.history_events_removed")
_active_artifacts = _meter.create_up_down_counter("kedi.artifact.active")
_active_bytes = _meter.create_up_down_counter("kedi.artifact.active_bytes", unit="By")
_cleanup_duration = _meter.create_histogram(
    "kedi.artifact.cleanup_duration",
    unit="s",
)


@contextmanager
def artifact_span(action: str, **attributes: object) -> Iterator[trace.Span]:
    """Create one payload-free artifact span."""

    safe_attributes = {
        key: value
        for key, value in attributes.items()
        if isinstance(value, (bool, str, int, float))
    }
    with _tracer.start_as_current_span(
        f"artifact {action}",
        attributes=safe_attributes,
    ) as span:
        yield span


def record_created(*, size_bytes: int, context_bytes_avoided: int) -> None:
    _created.add(1)
    record_active(size_bytes=size_bytes)
    _context_bytes_avoided.add(max(0, context_bytes_avoided))


def record_active(*, size_bytes: int) -> None:
    _active_artifacts.add(1)
    _active_bytes.add(size_bytes)


def record_payload_removed(*, size_bytes: int) -> None:
    _active_artifacts.add(-1)
    _active_bytes.add(-size_bytes)


def record_released() -> None:
    _released.add(1)


def record_expired() -> None:
    _expired.add(1)


def record_read(*, bytes_returned: int) -> None:
    _reads.add(1)
    _read_bytes.add(bytes_returned)


def record_quota_rejection() -> None:
    _quota_rejections.add(1)


def record_history_compaction(*, removed_events: int) -> None:
    _history_events_removed.add(removed_events)


def record_cleanup(*, duration_seconds: float) -> None:
    _cleanup_duration.record(max(0.0, duration_seconds))
