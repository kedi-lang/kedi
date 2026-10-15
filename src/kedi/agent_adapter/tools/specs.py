from __future__ import annotations

from collections.abc import Awaitable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..approvals.policy import (
    ApprovalPolicy,
    ToolRisk,
)
from .retry import ToolRetryPolicy

if TYPE_CHECKING:
    from kedi.agent_profile.capabilities import CapabilityToolPolicy
    from kedi.lang.ast import ProcedureDef


ToolArgumentValidator = Callable[[Mapping[str, Any]], dict[str, Any]]

ToolArtifactMode = Literal["auto", "never"]

ToolResultAdmitter = Callable[[Any, str], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    fn: Callable[..., Any]
    name: str
    description: str | None = None
    json_schema: dict[str, Any] | None = None
    return_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    proc: ProcedureDef | None = None
    risk: ToolRisk = "mutating"
    risk_resolver: Callable[[Mapping[str, Any]], ToolRisk] | None = None
    argument_validator: ToolArgumentValidator | None = None
    artifact_result: ToolArtifactMode = "auto"
    sequential: bool = False
    result_admitter: ToolResultAdmitter | None = field(default=None, repr=False, compare=False)
    retry_policy: ToolRetryPolicy = field(default_factory=ToolRetryPolicy)
    approval: ApprovalPolicy | None = None
    capability_policy: CapabilityToolPolicy | None = None
    capability_policy_resolver: (
        Callable[[Mapping[str, Any]], CapabilityToolPolicy | None] | None
    ) = field(default=None, repr=False, compare=False)
    deferred_capability: str | None = None


def tool_capability_policy(
    tool: ToolSpec, arguments: Mapping[str, Any]
) -> CapabilityToolPolicy | None:
    if tool.capability_policy_resolver is not None:
        return tool.capability_policy_resolver(arguments)
    return tool.capability_policy


def tool_approval_policy(
    tool: ToolSpec, default: ApprovalPolicy | None, arguments: Mapping[str, Any] | None = None
) -> ApprovalPolicy | None:
    """Resolve local policy without weakening separately applied approval ceilings."""

    if tool.approval is not None:
        return tool.approval
    policy = (
        tool.capability_policy if arguments is None else tool_capability_policy(tool, arguments)
    )
    if policy is not None and policy.approval is not None:
        return policy.approval
    return default


def tool_has_artifact_boundary(tool: ToolSpec) -> bool:
    """Whether a Kedi tool already admits its own result before native history."""

    return (tool.metadata or {}).get("kedi_artifact_wrapped") is True


@runtime_checkable
class SupportsToolRegistration(Protocol):
    def register_tool(self, tool: ToolSpec) -> None:
        """Register one adapter-native tool for the active agent run scope."""
        ...

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        """Create a temporary tool registration scope for one agent run."""
        ...
