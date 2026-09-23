from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from time import monotonic
from types import TracebackType

from ...telemetry import TelemetrySpan, add_counter, record_histogram, span
from .models import HookEvent

_EVENT_LABELS = {
    "user_prompt_submit": "user prompt submit",
    "pre_tool_use": "pre tool use",
    "post_tool_use": "post tool use",
    "post_tool_use_failure": "post tool use failure",
    "post_tool_result_failure": "post tool result failure",
}


@dataclass(slots=True)
class HookChainTelemetry(AbstractContextManager["HookChainTelemetry"]):
    event: HookEvent
    handler_count: int
    _started: float = field(default=0.0, init=False)
    _context: AbstractContextManager[TelemetrySpan] | None = field(default=None, init=False)
    _span: TelemetrySpan | None = field(default=None, init=False)
    _outcome: str = field(default="continued", init=False)

    def __enter__(self) -> HookChainTelemetry:
        attributes: dict[str, bool | str | int] = {
            "kedi.hook.event": self.event.event,
            "kedi.hook.handler_count": self.handler_count,
            "kedi.agent.adapter": self.event.adapter_shortname,
        }
        origin = getattr(self.event, "origin", None)
        if isinstance(origin, str):
            attributes["kedi.tool.origin"] = origin
        self._started = monotonic()
        self._context = span(
            "agent",
            f"run {_EVENT_LABELS[self.event.event]} hooks",
            "run_hooks",
            attributes=attributes,
        )
        self._span = self._context.__enter__()
        return self

    def complete(self, *, edited: bool = False, denied: bool = False) -> None:
        if denied:
            self._outcome = "denied"
        elif edited:
            self._outcome = "edited"

    def fail(self, *, timed_out: bool = False) -> None:
        self._outcome = "timed_out" if timed_out else "failed"

    def cancel(self) -> None:
        self._outcome = "cancelled"

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        elapsed = max(monotonic() - self._started, 0.0)
        dimensions = {
            "event": self.event.event,
            "adapter": self.event.adapter_shortname,
            "outcome": self._outcome,
        }
        origin = getattr(self.event, "origin", None)
        if isinstance(origin, str):
            dimensions["origin"] = origin
        if self._span is not None:
            self._span.set_attribute("kedi.hook.outcome", self._outcome)
            self._span.set_attribute("kedi.hook.edited", self._outcome == "edited")
            self._span.set_attribute("kedi.hook.denied", self._outcome == "denied")
        add_counter("agent", "kedi.hook.calls", attributes=dimensions)
        record_histogram(
            "agent",
            "kedi.hook.duration",
            elapsed,
            unit="s",
            attributes=dimensions,
        )
        metric = {
            "edited": "kedi.hook.edits",
            "denied": "kedi.hook.denials",
            "failed": "kedi.hook.failures",
            "timed_out": "kedi.hook.timeouts",
        }.get(self._outcome)
        if metric is not None:
            add_counter("agent", metric, attributes=dimensions)
        if self._context is None:
            return None
        return self._context.__exit__(exc_type, exc, traceback)


__all__ = ["HookChainTelemetry"]
