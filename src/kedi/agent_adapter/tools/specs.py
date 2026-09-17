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
    ToolRisk,
)

if TYPE_CHECKING:
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
