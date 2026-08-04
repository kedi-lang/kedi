from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ..telemetry import (
    TelemetrySpan,
    add_counter,
    add_up_down_counter,
    current_span,
    record_histogram,
    span,
)


def _safe_attributes(attributes: dict[str, object]) -> dict[str, bool | str | int | float]:
    return {
        key: value
        for key, value in attributes.items()
        if isinstance(value, (bool, str, int, float))
    }


def _artifact_attributes(
    action: str,
    attributes: dict[str, object],
) -> dict[str, bool | str | int | float]:
    safe = _safe_attributes(attributes)
    safe.pop("session_id", None)
    normalized: dict[str, bool | str | int | float] = {}
    special_names = {
        "has_query": "kedi.artifact.search.pattern_used",
        "limit": "kedi.artifact.search.limit",
        "offset": "kedi.artifact.read.offset",
        "offset_from": "kedi.artifact.read.offset_from",
        "outcome": "kedi.outcome",
    }
    for key, value in safe.items():
        if key.startswith(("kedi.", "error.")):
            name = key
        else:
            name = special_names.get(key, f"kedi.artifact.{key}")
        normalized[name] = value
    return normalized


def _context_attributes(
    attributes: dict[str, object],
) -> dict[str, bool | str | int | float]:
    return {
        key if key.startswith(("kedi.", "error.")) else f"kedi.context.{key}": value
        for key, value in _safe_attributes(attributes).items()
    }


@contextmanager
def artifact_span(action: str, **attributes: object) -> Iterator[TelemetrySpan]:
    """Trace meaningful artifact I/O; model lightweight lifecycle work as events."""

    safe_attributes = _artifact_attributes(action, attributes)
    if action == "create":
        with span(
            "artifacts",
            "store artifact",
            "store_artifact",
            attributes=safe_attributes,
        ) as active:
            yield active
        return
    if action == "cleanup":
        with span(
            "artifacts",
            "cleanup artifacts",
            "cleanup_artifacts",
            attributes=safe_attributes,
        ) as active:
            yield active
        return

    active = current_span()
    event_action = {
        "create": "created",
        "search": "searched",
        "release": "released",
        "expire": "expired",
    }.get(action, action)
    active.add_event(f"artifact {event_action}", safe_attributes)
    yield active


@contextmanager
def model_context_preflight_span(**attributes: object) -> Iterator[TelemetrySpan]:
    """Attach one payload-free request-admission decision to the active agent span."""

    active = current_span()
    active.add_event("context preflight", _context_attributes(attributes))
    yield active


def record_created(*, size_bytes: int, context_bytes_avoided: int) -> None:
    add_counter("artifacts", "kedi.artifact.created")
    record_active(size_bytes=size_bytes)
    add_counter(
        "artifacts",
        "kedi.artifact.context_bytes_avoided",
        max(0, context_bytes_avoided),
        unit="By",
    )


def record_active(*, size_bytes: int) -> None:
    add_up_down_counter("artifacts", "kedi.artifact.active", 1)
    add_up_down_counter("artifacts", "kedi.artifact.active_bytes", size_bytes, unit="By")


def record_payload_removed(*, size_bytes: int) -> None:
    add_up_down_counter("artifacts", "kedi.artifact.active", -1)
    add_up_down_counter("artifacts", "kedi.artifact.active_bytes", -size_bytes, unit="By")


def record_released() -> None:
    add_counter("artifacts", "kedi.artifact.released")


def record_expired() -> None:
    add_counter("artifacts", "kedi.artifact.expired")


def record_read(*, bytes_returned: int) -> None:
    add_counter("artifacts", "kedi.artifact.reads")
    add_counter("artifacts", "kedi.artifact.read_bytes", bytes_returned, unit="By")


def record_quota_rejection() -> None:
    add_counter("artifacts", "kedi.artifact.quota_rejections")


def record_output_admission(*, decision: str, size_bytes: int) -> None:
    add_counter(
        "artifacts",
        "kedi.tool.output_admissions",
        attributes={"decision": decision},
    )
    record_histogram(
        "artifacts",
        "kedi.tool.output_size",
        size_bytes,
        unit="By",
        attributes={"decision": decision},
    )


def record_cleanup(*, duration_seconds: float) -> None:
    record_histogram(
        "artifacts",
        "kedi.artifact.cleanup_duration",
        max(0.0, duration_seconds),
        unit="s",
    )
