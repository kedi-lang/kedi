from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace

from .approval import ApprovalDecision, ApprovalRequest

_tracer = trace.get_tracer("kedi.approvals")


@contextmanager
def approval_span(request: ApprovalRequest) -> Iterator[trace.Span]:
    """Trace one risky approval decision without recording tool arguments."""

    attributes: dict[str, bool | str] = {
        "approval.tool_name": request.tool_name,
        "approval.risk": request.risk,
        "approval.state": "requested",
    }
    if request.adapter_shortname is not None:
        attributes["approval.adapter"] = request.adapter_shortname
    with _tracer.start_as_current_span("approval request", attributes=attributes) as span:
        span.add_event("approval.requested")
        yield span


def record_pending(span: trace.Span) -> None:
    span.set_attribute("approval.state", "pending")
    span.add_event("approval.pending")


def record_decision(span: trace.Span, decision: ApprovalDecision) -> None:
    outcome = "denied" if decision.decision == "deny" else "allowed"
    span.set_attribute("approval.state", outcome)
    span.set_attribute("approval.arguments_edited", decision.decision == "edit")
    span.add_event(f"approval.{outcome}")


def record_cancelled(span: trace.Span) -> None:
    span.set_attribute("approval.state", "cancelled")
    span.add_event("approval.cancelled")
