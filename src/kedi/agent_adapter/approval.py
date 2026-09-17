"""Compatibility exports for the documented approval API."""

from .approvals.policy import (
    ApprovalCancelled,
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalHandler,
    ApprovalMode,
    ApprovalPolicy,
    ApprovalRequest,
    InvalidApprovalDecision,
    SupportsAsyncApprovalResolution,
    ToolApprovalDenied,
    ToolRisk,
    resolve_tool_approval,
    resolve_tool_approval_async,
)

__all__ = [
    "ApprovalCancelled",
    "ApprovalDecision",
    "ApprovalDecisionKind",
    "ApprovalHandler",
    "ApprovalMode",
    "ApprovalPolicy",
    "ApprovalRequest",
    "InvalidApprovalDecision",
    "SupportsAsyncApprovalResolution",
    "ToolApprovalDenied",
    "ToolRisk",
    "resolve_tool_approval",
    "resolve_tool_approval_async",
]
