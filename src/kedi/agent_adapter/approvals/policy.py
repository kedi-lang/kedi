from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterator, Literal, Protocol, TypeAlias, cast, runtime_checkable

from ...errors import KediExecutionError

ApprovalMode: TypeAlias = Literal["allow", "deny"]
ApprovalDecisionKind: TypeAlias = Literal["allow", "deny", "edit"]
ToolRisk: TypeAlias = Literal["read_only", "mutating", "sensitive"]
EditArgs: TypeAlias = Mapping[str, Any]
ApprovalHandler: TypeAlias = Callable[
    ["ApprovalRequest"],
    "ApprovalDecision | Awaitable[ApprovalDecision]",
]


@runtime_checkable
class SupportsAsyncApprovalResolution(Protocol):
    """Optional async path for handlers that also support synchronous prompting."""

    async def resolve_async(self, request: ApprovalRequest) -> ApprovalDecision: ...


class ToolApprovalDenied(KediExecutionError):
    """Raised when an approval policy refuses a tool invocation."""


class ApprovalCancelled(KediExecutionError):
    """Raised when an interactive approval prompt is cancelled."""


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
    reason: str | None = None
    tool_reason_enabled: bool = False

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
        reason: str | None = None,
        tool_reason_enabled: bool = False,
    ) -> ApprovalRequest:
        return cls(
            tool_name=tool_name,
            arguments=MappingProxyType(copy.deepcopy(dict(arguments))),
            risk=risk,
            adapter_shortname=adapter_shortname,
            description=description,
            reason=reason,
            tool_reason_enabled=tool_reason_enabled,
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


def _default_deny_reason(request: ApprovalRequest) -> str:
    # The caller, human or model, must be able to tell a classification problem from a refusal.
    return (
        f"no approval policy configured; tool {request.tool_name!r} is classified "
        f"{request.risk!r}. A tool that only reads can declare "
        '`__kedi_tool_risk__ = "read_only"`; otherwise configure approval with `> approval:`.'
    )


def resolve_tool_approval(
    policy: ApprovalPolicy | None,
    request: ApprovalRequest,
) -> ApprovalDecision:
    """Resolve one tool invocation according to Kedi's default-safe policy."""

    from .telemetry import (
        approval_mechanism,
        approval_span,
        record_cancelled,
        record_decision,
        record_immediate_decision,
        record_pending,
    )

    if request.risk == "read_only":
        decision = ApprovalDecision.allow(reason="read-only tool")
        record_immediate_decision(request, decision, mode="automatic_read_only")
        return decision
    if policy is None:
        decision = ApprovalDecision.deny(reason=_default_deny_reason(request))
        record_immediate_decision(request, decision, mode="default_deny")
        return decision
    if policy.mode is not None:
        decision = (
            ApprovalDecision.allow(reason="static allow policy")
            if policy.mode == "allow"
            else ApprovalDecision.deny(reason="static deny policy")
        )
        immediate_mode = "static_allow" if policy.mode == "allow" else "static_deny"
        record_immediate_decision(request, decision, mode=immediate_mode)
        return decision

    assert policy.handler is not None
    with approval_span(
        request,
        mechanism=approval_mechanism(policy.handler),
    ) as telemetry:
        try:
            record_pending(telemetry)
            with _approval_handler_agent_scope():
                result = policy.handler(request)
                if inspect.isawaitable(result):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        decision = asyncio.run(
                            _resolve_awaitable(cast(Awaitable[ApprovalDecision], result))
                        )
                    else:
                        if inspect.iscoroutine(result):
                            result.close()
                        raise RuntimeError(
                            "async approval handlers require an async tool wrapper when called "
                            "from an event loop"
                        )
                else:
                    decision = _validated_decision(result)
        except ApprovalCancelled:
            record_cancelled(telemetry)
            raise
        record_decision(telemetry, decision)
        return decision


async def resolve_tool_approval_async(
    policy: ApprovalPolicy | None,
    request: ApprovalRequest,
) -> ApprovalDecision:
    """Resolve one tool invocation without leaving the caller's event loop."""

    from .telemetry import (
        approval_mechanism,
        approval_span,
        record_cancelled,
        record_decision,
        record_immediate_decision,
        record_pending,
    )

    if request.risk == "read_only":
        decision = ApprovalDecision.allow(reason="read-only tool")
        record_immediate_decision(request, decision, mode="automatic_read_only")
        return decision
    if policy is None:
        decision = ApprovalDecision.deny(reason=_default_deny_reason(request))
        record_immediate_decision(request, decision, mode="default_deny")
        return decision
    if policy.mode is not None:
        decision = (
            ApprovalDecision.allow(reason="static allow policy")
            if policy.mode == "allow"
            else ApprovalDecision.deny(reason="static deny policy")
        )
        immediate_mode = "static_allow" if policy.mode == "allow" else "static_deny"
        record_immediate_decision(request, decision, mode=immediate_mode)
        return decision

    assert policy.handler is not None
    with approval_span(
        request,
        mechanism=approval_mechanism(policy.handler),
    ) as telemetry:
        try:
            record_pending(telemetry)
            with _approval_handler_agent_scope():
                handler = policy.handler
                if isinstance(handler, SupportsAsyncApprovalResolution):
                    result = handler.resolve_async(request)
                elif inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(
                    getattr(handler, "__call__", None)
                ):
                    result = handler(request)
                else:
                    result = await _run_sync_handler(handler, request)
                if inspect.isawaitable(result):
                    result = await result
                decision = _validated_decision(result)
        except ApprovalCancelled:
            record_cancelled(telemetry)
            raise
        record_decision(telemetry, decision)
        return decision


async def _run_sync_handler(
    handler: ApprovalHandler, request: ApprovalRequest
) -> ApprovalDecision | Awaitable[ApprovalDecision]:
    from ..execution import run_in_worker

    return await run_in_worker(handler, request)


async def _resolve_awaitable(result: Awaitable[ApprovalDecision]) -> ApprovalDecision:
    return _validated_decision(await result)


@contextmanager
def _approval_handler_agent_scope() -> Iterator[None]:
    """Keep backend selection while isolating a nested approval-model call."""

    from ...agent_profile.profile import AgentProfile
    from ...agent_profile.scope import (
        current_block_agent_profile,
        replace_block_agent_profile_scope,
    )
    from ...artifacts import ArtifactPolicy

    current = current_block_agent_profile()
    isolated = AgentProfile(
        model=current.model,
        adapter_kind=current.adapter_kind,
        adapter_shortname=current.adapter_shortname,
        agent_command=current.agent_command,
        a2a_connection=current.a2a_connection,
        effort=current.effort,
        settings=current.settings,
        artifacts=ArtifactPolicy.patch({"enabled": False}),
    )
    with replace_block_agent_profile_scope(isolated):
        yield


def _validated_decision(result: object) -> ApprovalDecision:
    if not isinstance(result, ApprovalDecision):
        raise InvalidApprovalDecision(
            f"approval handler must return ApprovalDecision, got {type(result).__name__}"
        )
    return result
