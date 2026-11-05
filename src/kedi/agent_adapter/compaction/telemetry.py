from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from ...telemetry import TelemetrySpan, add_counter, record_histogram, span
from .models import CompactionOutcome

CompactionMode = Literal["kedi", "native"]

_METRIC_TRIGGERS = frozenset(
    {
        "checkpoint",
        "context_limit",
        "manual",
        "provider",
        "threshold",
        "unknown",
    }
)


@dataclass(slots=True)
class CompactionTelemetry:
    """Mutable observations for one real compaction attempt."""

    span: TelemetrySpan
    mode: CompactionMode
    trigger: str
    started_at: float
    applied: bool | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def complete(
        self,
        outcome: CompactionOutcome,
        *,
        input_messages: int | None = None,
        output_messages: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        selected_tokens: int | None = None,
        prefix_preserved: bool | None = None,
        cache_epoch_changed: bool | None = None,
        checkpoint_valid: bool | None = None,
    ) -> None:
        self.applied = outcome.applied
        self.trigger = _span_trigger(outcome.trigger or self.trigger)
        self.input_tokens = input_tokens if input_tokens is not None else outcome.source_tokens
        self.output_tokens = output_tokens if output_tokens is not None else outcome.result_tokens

        self.span.set_attribute("kedi.history.compaction.applied", outcome.applied)
        self.span.set_attribute("kedi.history.compaction.trigger", self.trigger)
        _set_optional(self.span, "kedi.history.compaction.input_messages", input_messages)
        _set_optional(self.span, "kedi.history.compaction.output_messages", output_messages)
        _set_optional(self.span, "kedi.history.compaction.input_tokens", self.input_tokens)
        _set_optional(self.span, "kedi.history.compaction.output_tokens", self.output_tokens)
        _set_optional(self.span, "kedi.history.compaction.selected_tokens", selected_tokens)
        _set_optional(self.span, "kedi.history.compaction.prefix_preserved", prefix_preserved)
        _set_optional(
            self.span,
            "kedi.history.compaction.cache_epoch_changed",
            cache_epoch_changed,
        )
        _set_optional(self.span, "kedi.history.compaction.checkpoint_valid", checkpoint_valid)

        reduction_ratio = _reduction_ratio(self.input_tokens, self.output_tokens)
        _set_optional(
            self.span,
            "kedi.history.compaction.reduction_ratio",
            reduction_ratio,
        )
        if outcome.usage is not None:
            _set_optional(
                self.span,
                "gen_ai.usage.input_tokens",
                outcome.usage.input_tokens,
            )
            _set_optional(
                self.span,
                "gen_ai.usage.output_tokens",
                outcome.usage.output_tokens,
            )
            _set_optional(
                self.span,
                "gen_ai.usage.cache_read_tokens",
                outcome.usage.cache_read_tokens,
            )
            _set_optional(
                self.span,
                "gen_ai.usage.cache_write_tokens",
                outcome.usage.cache_write_tokens,
            )

    def checkpoint_validated(self) -> None:
        self.span.add_event("checkpoint validated")

    def checkpoint_committed(self) -> None:
        self.span.add_event("checkpoint committed")

    def cache_epoch_advanced(self) -> None:
        self.span.add_event("cache epoch advanced")


@contextmanager
def compact_history_span(
    *,
    mode: CompactionMode,
    trigger: str,
    adapter_name: str | None = None,
    input_messages: int | None = None,
    input_tokens: int | None = None,
) -> Generator[CompactionTelemetry, None, None]:
    """Trace one attempted native or Kedi-owned history compaction."""

    normalized_trigger = _span_trigger(trigger)
    attributes: dict[str, bool | str | int | float] = {
        "kedi.history.compaction.mode": mode,
        "kedi.history.compaction.trigger": normalized_trigger,
    }
    if adapter_name is not None:
        attributes["kedi.adapter.name"] = adapter_name
    if input_messages is not None:
        attributes["kedi.history.compaction.input_messages"] = input_messages
    if input_tokens is not None:
        attributes["kedi.history.compaction.input_tokens"] = input_tokens

    started_at = perf_counter()
    result = "error"
    with span(
        "agent",
        "compact history",
        "compact_history",
        level="detailed",
        attributes=attributes,
    ) as current:
        telemetry = CompactionTelemetry(
            span=current,
            mode=mode,
            trigger=normalized_trigger,
            started_at=started_at,
            input_tokens=input_tokens,
        )
        current.add_event(
            "compaction triggered",
            {
                "kedi.history.compaction.mode": mode,
                "kedi.history.compaction.trigger": normalized_trigger,
            },
        )
        try:
            yield telemetry
        except asyncio.CancelledError as exc:
            result = "cancelled"
            current.set_attribute("error.type", type(exc).__name__)
            current.add_event(
                "compaction rejected",
                {"kedi.history.compaction.rejection_reason": "cancelled"},
            )
            raise
        except BaseException as exc:
            current.set_attribute("error.type", type(exc).__name__)
            current.add_event(
                "compaction rejected",
                {"kedi.history.compaction.rejection_reason": "error"},
            )
            raise
        else:
            result = "applied" if telemetry.applied else "not_applied"
        finally:
            _record_compaction_metrics(telemetry, result=result)


def _record_compaction_metrics(
    telemetry: CompactionTelemetry,
    *,
    result: str,
) -> None:
    dimensions = {
        "mode": telemetry.mode,
        "trigger": _metric_trigger(telemetry.trigger),
        "result": result,
    }
    add_counter(
        "agent",
        "kedi.history.compactions",
        attributes=dimensions,
    )
    record_histogram(
        "agent",
        "kedi.history.compaction.duration",
        max(0.0, perf_counter() - telemetry.started_at),
        unit="s",
        attributes=dimensions,
    )
    if telemetry.input_tokens is not None:
        record_histogram(
            "agent",
            "kedi.history.compaction.input_tokens",
            telemetry.input_tokens,
            unit="token",
            attributes=dimensions,
        )
    if telemetry.output_tokens is not None:
        record_histogram(
            "agent",
            "kedi.history.compaction.output_tokens",
            telemetry.output_tokens,
            unit="token",
            attributes=dimensions,
        )
    if (ratio := _reduction_ratio(telemetry.input_tokens, telemetry.output_tokens)) is not None:
        record_histogram(
            "agent",
            "kedi.history.compaction.reduction_ratio",
            ratio,
            unit="1",
            attributes=dimensions,
        )
    if result == "error":
        add_counter(
            "agent",
            "kedi.history.compaction.failures",
            attributes={
                "mode": telemetry.mode,
                "trigger": _metric_trigger(telemetry.trigger),
            },
        )


def _metric_trigger(trigger: str) -> str:
    normalized = _span_trigger(trigger)
    return normalized if normalized in _METRIC_TRIGGERS else "other"


def _span_trigger(trigger: str) -> str:
    normalized = trigger.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized[:64] or "unknown"


def _reduction_ratio(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None or input_tokens <= 0:
        return None
    return max(0.0, min(1.0, (input_tokens - output_tokens) / input_tokens))


def _set_optional(span: TelemetrySpan, name: str, value: bool | int | float | None) -> None:
    if value is not None:
        span.set_attribute(name, value)
