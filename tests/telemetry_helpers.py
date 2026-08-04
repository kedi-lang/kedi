from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

from kedi.telemetry import (
    AttributeValue,
    Attributes,
    CaptureKind,
    SpanKind,
    SpanLevel,
    TelemetryBackend,
    TelemetryScope,
    install_backend,
    restore_backend,
)


@dataclass(slots=True)
class RecordedEvent:
    name: str
    attributes: dict[str, AttributeValue]


@dataclass(slots=True)
class RecordedSpan:
    scope: TelemetryScope
    name: str
    operation: str
    kind: SpanKind
    level: SpanLevel
    attributes: dict[str, AttributeValue]
    parent: RecordedSpan | None
    events: list[RecordedEvent] = field(default_factory=list)
    exceptions: list[BaseException] = field(default_factory=list)
    recording: bool = True

    def is_recording(self) -> bool:
        return self.recording

    def get_attribute(self, name: str) -> AttributeValue | None:
        return self.attributes.get(name)

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        self.attributes[name] = value

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        self.events.append(RecordedEvent(name, dict(attributes or {})))

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def update_name(self, name: str) -> None:
        self.name = name


_NOOP_SPAN = RecordedSpan(
    scope="runtime",
    name="noop",
    operation="noop",
    kind="internal",
    level="lifecycle",
    attributes={},
    parent=None,
    recording=False,
)


@dataclass(frozen=True, slots=True)
class RecordedMetric:
    kind: Literal["counter", "up_down_counter", "histogram"]
    scope: TelemetryScope
    name: str
    value: int | float
    unit: str
    attributes: dict[str, AttributeValue]


class RecordingTelemetryBackend:
    def __init__(
        self,
        *,
        native_span_owners: Mapping[str, frozenset[str]] | None = None,
        capture: frozenset[CaptureKind] = frozenset(),
    ) -> None:
        self.started_spans: list[RecordedSpan] = []
        self.finished_spans: list[RecordedSpan] = []
        self.metrics: list[RecordedMetric] = []
        self._native_span_owners = dict(native_span_owners or {})
        self._capture = capture
        self._active: ContextVar[RecordedSpan | None] = ContextVar(
            f"recording_telemetry_{id(self)}",
            default=None,
        )

    @contextmanager
    def start_span(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        operation: str,
        kind: SpanKind,
        level: SpanLevel,
        attributes: Attributes | None,
    ) -> Iterator[RecordedSpan]:
        parent = self._active.get()
        recorded_attributes = dict(attributes or {})
        recorded_attributes["kedi.operation.name"] = operation
        semantic_operation = {
            "run_agent": "invoke_agent",
            "chat": "chat",
            "call_tool": "execute_tool",
        }.get(operation)
        if semantic_operation is not None:
            recorded_attributes["gen_ai.operation.name"] = semantic_operation
        recorded = RecordedSpan(
            scope=scope,
            name=name,
            operation=operation,
            kind=kind,
            level=level,
            attributes=recorded_attributes,
            parent=parent,
        )
        self.started_spans.append(recorded)
        token = self._active.set(recorded)
        try:
            yield recorded
        except BaseException as exc:
            recorded.record_exception(exc)
            raise
        finally:
            self._active.reset(token)
            self.finished_spans.append(recorded)

    def current_span(self) -> RecordedSpan:
        return self._active.get() or _NOOP_SPAN

    def enabled(self, scope: TelemetryScope, level: SpanLevel) -> bool:
        return True

    def add_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._record_metric("counter", scope, name, value, unit, attributes)

    def add_up_down_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._record_metric("up_down_counter", scope, name, value, unit, attributes)

    def record_histogram(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self._record_metric("histogram", scope, name, value, unit, attributes)

    def owns_native_spans(self, adapter_shortname: str, operation: str) -> bool:
        return operation in self._native_span_owners.get(adapter_shortname, ())

    def capture_enabled(self, kind: CaptureKind) -> bool:
        return kind in self._capture

    def _record_metric(
        self,
        kind: Literal["counter", "up_down_counter", "histogram"],
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        self.metrics.append(RecordedMetric(kind, scope, name, value, unit, dict(attributes or {})))


@contextmanager
def installed_recording_backend(
    backend: RecordingTelemetryBackend | None = None,
) -> Iterator[RecordingTelemetryBackend]:
    active = backend or RecordingTelemetryBackend()
    previous: TelemetryBackend = install_backend(active)
    try:
        yield active
    finally:
        restore_backend(previous)
