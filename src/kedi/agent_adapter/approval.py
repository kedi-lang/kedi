from __future__ import annotations

import copy
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeAlias, cast

from ..errors import KediExecutionError

ApprovalMode: TypeAlias = Literal["allow", "deny"]
ApprovalDecisionKind: TypeAlias = Literal["allow", "deny", "edit"]
ToolRisk: TypeAlias = Literal["read_only", "mutating", "sensitive"]
EditArgs: TypeAlias = Mapping[str, Any]
ApprovalHandler: TypeAlias = Callable[
    ["ApprovalRequest"],
    "ApprovalDecision | Awaitable[ApprovalDecision]",
]


class ToolApprovalDenied(KediExecutionError):
    """Raised when an approval policy refuses a tool invocation."""


class InvalidApprovalDecision(ValueError):
    """Raised when an approval handler returns an invalid decision."""


@dataclass(frozen=True)
class ApprovalRequest:
    """Immutable context supplied to a dynamic approval handler."""

    tool_name: str
    arguments: Mapping[str, Any]
    risk: ToolRisk
    adapter_shortname: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        risk: ToolRisk,
        adapter_shortname: str | None,
        description: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> ApprovalRequest:
        return cls(
            tool_name=tool_name,
            arguments=MappingProxyType(copy.deepcopy(dict(arguments))),
            risk=risk,
            adapter_shortname=adapter_shortname,
            description=description,
            metadata=(
                MappingProxyType(copy.deepcopy(dict(metadata))) if metadata is not None else None
            ),
        )


@dataclass(frozen=True)
class ApprovalDecision:
    """A dynamic handler's allow, deny, or edited-arguments decision."""

    decision: ApprovalDecisionKind
    edit_arguments: EditArgs | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.decision not in ("allow", "deny", "edit"):
            raise InvalidApprovalDecision("approval decision must be 'allow', 'deny', or 'edit'")
        if self.decision == "edit":
            if self.edit_arguments is None:
                raise InvalidApprovalDecision(
                    "ApprovalDecision(decision='edit') requires edit_arguments"
                )
            object.__setattr__(
                self,
                "edit_arguments",
                MappingProxyType(copy.deepcopy(dict(self.edit_arguments))),
            )
            return
        if self.edit_arguments is not None:
            raise InvalidApprovalDecision(
                "edit_arguments may only be provided when decision='edit'"
            )

    @classmethod
    def allow(cls, *, reason: str | None = None) -> ApprovalDecision:
        return cls(decision="allow", reason=reason)

    @classmethod
    def deny(cls, *, reason: str | None = None) -> ApprovalDecision:
        return cls(decision="deny", reason=reason)

    @classmethod
    def edit(
        cls,
        edit_arguments: EditArgs,
        *,
        reason: str | None = None,
    ) -> ApprovalDecision:
        return cls(decision="edit", edit_arguments=edit_arguments, reason=reason)


@dataclass(frozen=True)
class ApprovalPolicy:
    """A static policy or a dynamic callable handler for tool invocations."""

    mode: ApprovalMode | None = None
    handler: ApprovalHandler | None = None

    def __post_init__(self) -> None:
        if self.mode is not None and self.handler is not None:
            raise ValueError("approval mode and approval handler are mutually exclusive")
        if self.mode not in (None, "allow", "deny"):
            raise ValueError("approval mode must be 'allow' or 'deny'")
        if self.handler is not None and not callable(self.handler):
            raise TypeError("approval handler must be callable")

    @classmethod
    def allow(cls) -> ApprovalPolicy:
        return cls(mode="allow")

    @classmethod
    def deny(cls) -> ApprovalPolicy:
        return cls(mode="deny")

    @classmethod
    def dynamic(cls, handler: ApprovalHandler) -> ApprovalPolicy:
        return cls(handler=handler)


def resolve_tool_approval(
    policy: ApprovalPolicy | None,
    request: ApprovalRequest,
) -> ApprovalDecision:
    """Resolve one tool invocation according to Kedi's default-safe policy."""

    if request.risk == "read_only":
        return ApprovalDecision.allow(reason="read-only tool")
    if policy is None:
        return ApprovalDecision.deny(reason="no approval policy configured")
    if policy.mode == "allow":
        return ApprovalDecision.allow(reason="static allow policy")
    if policy.mode == "deny":
        return ApprovalDecision.deny(reason="static deny policy")
    assert policy.handler is not None
    result = policy.handler(request)
    if inspect.isawaitable(result):
        from .utils import run_sync

        result = run_sync(cast(Awaitable[ApprovalDecision], result))
    if not isinstance(result, ApprovalDecision):
        raise InvalidApprovalDecision(
            f"approval handler must return ApprovalDecision, got {type(result).__name__}"
        )
    return result
