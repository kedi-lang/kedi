from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .approval import (
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalDenied,
    ToolRisk,
    resolve_tool_approval,
)

if TYPE_CHECKING:
    from kedi.lang.ast import ProcedureDef


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


def with_approval(
    tool: ToolSpec,
    *,
    policy: ApprovalPolicy | None,
    adapter_shortname: str | None,
) -> ToolSpec:
    """Return a signature-preserving ToolSpec that enforces one approval policy."""

    @wraps(tool.fn)
    def approved_tool(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")
        risk = tool.risk_resolver(kwargs) if tool.risk_resolver is not None else tool.risk
        if risk not in ("read_only", "mutating", "sensitive"):
            raise ValueError(f"invalid Kedi tool risk {risk!r} for {tool.name!r}")
        request = ApprovalRequest.create(
            tool_name=tool.name,
            arguments=kwargs,
            risk=risk,
            adapter_shortname=adapter_shortname,
            description=tool.description,
            metadata=tool.metadata,
        )
        decision = resolve_tool_approval(policy, request)
        if decision.decision == "deny":
            suffix = f": {decision.reason}" if decision.reason else ""
            raise ToolApprovalDenied(f"Tool {tool.name!r} was denied{suffix}")
        final_kwargs = (
            kwargs if decision.decision == "allow" else dict(decision.edit_arguments or {})
        )
        return tool.fn(**final_kwargs)

    return replace(tool, fn=approved_tool)


@runtime_checkable
class SupportsToolRegistration(Protocol):
    def register_tool(self, tool: ToolSpec) -> None:
        """Register one adapter-native tool for the active agent run scope."""
        ...

    def tool_scope(self, tools: Sequence[Any] = ()) -> AbstractContextManager[None]:
        """Create a temporary tool registration scope for one agent run."""
        ...


class ToolScope:
    def __init__(self) -> None:
        self._scoped_tools: ContextVar[tuple[Any, ...]] = ContextVar(
            "kedi_scoped_tools", default=()
        )

    @contextmanager
    def tool_scope(self, tools: Sequence[Any] = ()) -> Iterator[None]:
        token = self._scoped_tools.set(())
        try:
            for tool in tools:
                self.register_tool(tool)
            yield
        finally:
            self._scoped_tools.reset(token)

    def register_tool(self, tool: Any) -> None:
        current = self._scoped_tools.get()
        name = getattr(tool, "name", None)
        if name is not None:
            current = tuple(
                registered for registered in current if getattr(registered, "name", None) != name
            )
        self._scoped_tools.set((*current, tool))

    @property
    def scoped_tools(self) -> tuple[Any, ...]:
        return self._scoped_tools.get()
