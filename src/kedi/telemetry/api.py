from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from threading import RLock
from typing import Literal, Protocol

AttributeScalar = bool | str | int | float
AttributeValue = AttributeScalar | Sequence[AttributeScalar]
Attributes = Mapping[str, AttributeValue]
CaptureKind = Literal["content", "source_path", "source_snippet"]
SpanKind = Literal["internal", "client"]
SpanLevel = Literal["lifecycle", "detailed"]
TelemetryScope = Literal["runtime", "agent", "agentic", "artifacts"]


class TelemetrySpan(Protocol):
    """Small span surface used by Kedi without importing OpenTelemetry."""

    def is_recording(self) -> bool: ...

    def get_attribute(self, name: str) -> AttributeValue | None: ...

    def set_attribute(self, name: str, value: AttributeValue) -> None: ...

    def add_event(self, name: str, attributes: Attributes | None = None) -> None: ...

    def record_exception(self, exc: BaseException) -> None: ...

    def update_name(self, name: str) -> None: ...


class TelemetryBackend(Protocol):
    """Backend contract implemented by the optional Kedi OTel package."""

    def start_span(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        operation: str,
        kind: SpanKind,
        level: SpanLevel,
        attributes: Attributes | None,
    ) -> AbstractContextManager[TelemetrySpan]: ...

    def current_span(self) -> TelemetrySpan: ...

    def enabled(self, scope: TelemetryScope, level: SpanLevel) -> bool: ...

    def add_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None: ...

    def add_up_down_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None: ...

    def record_histogram(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None: ...

    def owns_native_spans(self, adapter_shortname: str, operation: str) -> bool: ...

    def capture_enabled(self, kind: CaptureKind) -> bool: ...


class _NoOpSpan:
    __slots__ = ()

    def is_recording(self) -> bool:
        return False

    def get_attribute(self, name: str) -> AttributeValue | None:
        return None

    def set_attribute(self, name: str, value: AttributeValue) -> None:
        return None

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def update_name(self, name: str) -> None:
        return None


_NOOP_SPAN = _NoOpSpan()


class _NoOpSpanContext(AbstractContextManager[TelemetrySpan]):
    __slots__ = ()

    def __enter__(self) -> TelemetrySpan:
        return _NOOP_SPAN

    def __exit__(self, *args: object) -> None:
        return None


_NOOP_CONTEXT = _NoOpSpanContext()


class NoOpTelemetryBackend:
    """Default backend; importing and running Kedi emits no telemetry."""

    __slots__ = ()

    def start_span(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        operation: str,
        kind: SpanKind,
        level: SpanLevel,
        attributes: Attributes | None,
    ) -> AbstractContextManager[TelemetrySpan]:
        return _NOOP_CONTEXT

    def current_span(self) -> TelemetrySpan:
        return _NOOP_SPAN

    def enabled(self, scope: TelemetryScope, level: SpanLevel) -> bool:
        return False

    def add_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        return None

    def add_up_down_counter(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        return None

    def record_histogram(
        self,
        *,
        scope: TelemetryScope,
        name: str,
        value: int | float,
        unit: str,
        attributes: Attributes | None,
    ) -> None:
        return None

    def owns_native_spans(self, adapter_shortname: str, operation: str) -> bool:
        return False

    def capture_enabled(self, kind: CaptureKind) -> bool:
        return False


_NOOP_BACKEND = NoOpTelemetryBackend()
_backend: TelemetryBackend = _NOOP_BACKEND
_backend_lock = RLock()


def install_backend(backend: TelemetryBackend) -> TelemetryBackend:
    """Install one process-wide backend and return the previous backend."""

    global _backend
    with _backend_lock:
        previous = _backend
        _backend = backend
        return previous


def restore_backend(backend: TelemetryBackend) -> None:
    global _backend
    with _backend_lock:
        _backend = backend


def restore_backend_if_current(
    *,
    expected: TelemetryBackend,
    replacement: TelemetryBackend,
) -> bool:
    """Restore a backend only while the caller still owns the active slot."""

    global _backend
    with _backend_lock:
        if _backend is not expected:
            return False
        _backend = replacement
        return True


def reset_backend() -> None:
    restore_backend(_NOOP_BACKEND)


def get_backend() -> TelemetryBackend:
    return _backend


def span(
    scope: TelemetryScope,
    name: str,
    operation: str,
    *,
    kind: SpanKind = "internal",
    level: SpanLevel = "lifecycle",
    attributes: Attributes | None = None,
) -> AbstractContextManager[TelemetrySpan]:
    return _backend.start_span(
        scope=scope,
        name=name,
        operation=operation,
        kind=kind,
        level=level,
        attributes=attributes,
    )


def current_span() -> TelemetrySpan:
    return _backend.current_span()


def telemetry_enabled(scope: TelemetryScope, level: SpanLevel = "lifecycle") -> bool:
    return _backend.enabled(scope, level)


def add_counter(
    scope: TelemetryScope,
    name: str,
    value: int | float = 1,
    *,
    unit: str = "1",
    attributes: Attributes | None = None,
) -> None:
    _backend.add_counter(
        scope=scope,
        name=name,
        value=value,
        unit=unit,
        attributes=attributes,
    )


def add_up_down_counter(
    scope: TelemetryScope,
    name: str,
    value: int | float,
    *,
    unit: str = "1",
    attributes: Attributes | None = None,
) -> None:
    _backend.add_up_down_counter(
        scope=scope,
        name=name,
        value=value,
        unit=unit,
        attributes=attributes,
    )


def record_histogram(
    scope: TelemetryScope,
    name: str,
    value: int | float,
    *,
    unit: str = "1",
    attributes: Attributes | None = None,
) -> None:
    _backend.record_histogram(
        scope=scope,
        name=name,
        value=value,
        unit=unit,
        attributes=attributes,
    )


def owns_native_spans(adapter_shortname: str, operation: str) -> bool:
    return _backend.owns_native_spans(adapter_shortname, operation)


def capture_enabled(kind: CaptureKind) -> bool:
    return _backend.capture_enabled(kind)
