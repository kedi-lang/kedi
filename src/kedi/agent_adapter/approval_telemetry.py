from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from ..telemetry import (
    TelemetrySpan,
    add_counter,
    current_span,
    record_histogram,
    span,
)
from .approval import ApprovalDecision, ApprovalRequest

ApprovalMechanism = Literal["cli", "python_handler", "native_adapter"]
ImmediateApprovalMode = Literal[
    "automatic_read_only",
    "default_deny",
    "static_allow",
    "static_deny",
]


@dataclass(slots=True)
class ApprovalTelemetry:
    span: TelemetrySpan
    parent_tool_span: TelemetrySpan
    mechanism: ApprovalMechanism
    risk: str
    started_at: float
    decision: str = "requested"


def approval_mechanism(handler: object) -> ApprovalMechanism:
    mechanism = getattr(handler, "telemetry_mechanism", None)
    if mechanism == "cli":
        return "cli"
    if mechanism == "native_adapter":
        return "native_adapter"
    return "python_handler"


@contextmanager
def approval_span(
    request: ApprovalRequest,
    *,
    mechanism: ApprovalMechanism,
) -> Iterator[ApprovalTelemetry]:
    """Trace one blocking approval decision without recording tool arguments."""

    parent = current_span()
    _mark_parent_requested(parent, request)
    attributes: dict[str, bool | str] = {
        "kedi.approval.tool.name": request.tool_name,
        "kedi.approval.risk": request.risk,
        "kedi.approval.mechanism": mechanism,
        "kedi.approval.state": "requested",
    }
    if request.adapter_shortname is not None:
        attributes["kedi.adapter.name"] = request.adapter_shortname
    started_at = perf_counter()
    with span(
        "agentic",
        f"await approval {request.tool_name}",
        "await_approval",
        attributes=attributes,
    ) as active:
        telemetry = ApprovalTelemetry(
            span=active,
            parent_tool_span=parent,
            mechanism=mechanism,
            risk=request.risk,
            started_at=started_at,
        )
        active.add_event("kedi.approval.requested")
        try:
            yield telemetry
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                telemetry.decision = "cancelled"
            elif telemetry.decision == "requested":
                telemetry.decision = "error"
            active.set_attribute("error.type", type(exc).__name__)
            raise
        finally:
            dimensions = {
                "mechanism": mechanism,
                "risk": request.risk,
                "decision": telemetry.decision,
            }
            add_counter("agentic", "kedi.approval.requests", attributes=dimensions)
            record_histogram(
                "agentic",
                "kedi.approval.wait.duration",
                max(0.0, perf_counter() - started_at),
                unit="s",
                attributes=dimensions,
            )


def record_immediate_decision(
    request: ApprovalRequest,
    decision: ApprovalDecision,
    *,
    mode: ImmediateApprovalMode,
) -> None:
    parent = current_span()
    _mark_parent_decision(parent, request, decision)
    parent.add_event(
        "kedi.tool.approval",
        {
            "kedi.approval.mode": mode,
            "kedi.approval.decision": decision.decision,
            "kedi.approval.arguments_edited": False,
        },
    )


def record_pending(telemetry: ApprovalTelemetry) -> None:
    telemetry.span.set_attribute("kedi.approval.state", "pending")
    telemetry.span.add_event("kedi.approval.pending")


def record_decision(telemetry: ApprovalTelemetry, decision: ApprovalDecision) -> None:
    state = (
        "edited"
        if decision.decision == "edit"
        else ("denied" if decision.decision == "deny" else "allowed")
    )
    telemetry.decision = state
    telemetry.span.set_attribute("kedi.approval.state", state)
    telemetry.span.set_attribute(
        "kedi.approval.arguments_edited",
        decision.decision == "edit",
    )
    telemetry.span.add_event(
        "kedi.approval.decided",
        {"kedi.approval.decision": decision.decision},
    )
    _mark_parent_decision(telemetry.parent_tool_span, None, decision)
    if decision.decision == "edit":
        add_counter(
            "agentic",
            "kedi.approval.edits",
            attributes={"mechanism": telemetry.mechanism, "risk": telemetry.risk},
        )


def record_cancelled(telemetry: ApprovalTelemetry) -> None:
    telemetry.decision = "cancelled"
    telemetry.span.set_attribute("kedi.approval.state", "cancelled")
    telemetry.span.add_event("kedi.approval.cancelled")
    telemetry.parent_tool_span.set_attribute("kedi.tool.approval.decision", "cancelled")


def _mark_parent_requested(parent: TelemetrySpan, request: ApprovalRequest) -> None:
    if parent.get_attribute("kedi.operation.name") != "call_tool":
        return
    parent.set_attribute("kedi.tool.approval.required", True)
    parent.set_attribute("kedi.tool.risk", request.risk)


def _mark_parent_decision(
    parent: TelemetrySpan,
    request: ApprovalRequest | None,
    decision: ApprovalDecision,
) -> None:
    if parent.get_attribute("kedi.operation.name") != "call_tool":
        return
    if request is not None:
        parent.set_attribute("kedi.tool.risk", request.risk)
        parent.set_attribute("kedi.tool.approval.required", request.risk != "read_only")
    parent.set_attribute("kedi.tool.approval.decision", decision.decision)
    parent.set_attribute("kedi.tool.arguments.edited", decision.decision == "edit")
