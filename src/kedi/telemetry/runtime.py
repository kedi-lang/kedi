from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from .api import (
    Attributes,
    AttributeScalar,
    SpanLevel,
    TelemetrySpan,
    add_counter,
    add_up_down_counter,
    record_histogram,
    span,
    telemetry_enabled,
)

RuntimeOperation = Literal[
    "run_program",
    "parse_program",
    "compile_program",
    "compile_module",
    "call_procedure",
    "map_stage",
    "run_python",
]


@dataclass(frozen=True, slots=True)
class _RuntimeMetrics:
    counter: str | None
    duration: str | None
    active: str | None = None


_METRICS: Mapping[RuntimeOperation, _RuntimeMetrics] = {
    "run_program": _RuntimeMetrics("kedi.run.count", "kedi.run.duration", "kedi.run.active"),
    "parse_program": _RuntimeMetrics("kedi.parse.count", "kedi.parse.duration"),
    "compile_program": _RuntimeMetrics("kedi.compile.count", "kedi.compile.duration"),
    "compile_module": _RuntimeMetrics(None, None),
    "call_procedure": _RuntimeMetrics("kedi.procedure.calls", "kedi.procedure.duration"),
    "map_stage": _RuntimeMetrics("kedi.map.stages", "kedi.map.duration"),
    "run_python": _RuntimeMetrics(
        "kedi.python.execution.count",
        "kedi.python.execution.duration",
    ),
}


@contextmanager
def runtime_span(
    name: str,
    operation: RuntimeOperation,
    *,
    attributes: Attributes | None = None,
    metric_attributes: Mapping[str, AttributeScalar] | None = None,
    level: SpanLevel = "lifecycle",
) -> Iterator[TelemetrySpan]:
    """Trace and measure one bounded runtime operation."""

    metrics = _METRICS[operation]
    dimensions = dict(metric_attributes or {})
    enabled = telemetry_enabled("runtime", level)
    started_at = perf_counter()
    outcome = "success"
    if enabled and metrics.active is not None:
        add_up_down_counter("runtime", metrics.active, 1, attributes=dimensions)

    with span(
        "runtime",
        name,
        operation,
        level=level,
        attributes=attributes,
    ) as active:
        try:
            yield active
        except asyncio.CancelledError as exc:
            outcome = "cancelled"
            active.set_attribute("error.type", type(exc).__name__)
            active.add_event("run cancelled")
            raise
        except BaseException as exc:
            outcome = "error"
            active.set_attribute("error.type", type(exc).__name__)
            raise
        finally:
            measured = {**dimensions, "outcome": outcome}
            if enabled and metrics.counter is not None:
                add_counter("runtime", metrics.counter, attributes=measured)
            if enabled and metrics.duration is not None:
                record_histogram(
                    "runtime",
                    metrics.duration,
                    max(0.0, perf_counter() - started_at),
                    unit="s",
                    attributes=measured,
                )
            if enabled and metrics.active is not None:
                add_up_down_counter("runtime", metrics.active, -1, attributes=dimensions)
