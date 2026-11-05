from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from time import perf_counter

from ...telemetry import TelemetrySpan, add_counter, record_histogram, span


@contextmanager
def codemode_span(
    name: str,
    operation: str,
    attributes: Mapping[str, bool | str | int | float],
) -> Generator[TelemetrySpan, None, None]:
    """Trace a payload-free CodeMode operation and its operational cost."""

    started_at = perf_counter()
    outcome = "success"
    with span(
        "agent",
        name,
        operation,
        level="detailed",
        attributes=attributes,
    ) as active:
        try:
            yield active
        except BaseException as exc:
            outcome = "cancelled" if type(exc).__name__ == "CancelledError" else "error"
            active.set_attribute("error.type", type(exc).__name__)
            raise
        finally:
            dimensions = {"operation": operation, "outcome": outcome}
            add_counter("agent", "kedi.codemode.operations", attributes=dimensions)
            record_histogram(
                "agent",
                "kedi.codemode.duration",
                max(0.0, perf_counter() - started_at),
                unit="s",
                attributes=dimensions,
            )


__all__ = ["codemode_span"]
