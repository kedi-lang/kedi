from __future__ import annotations

import inspect
import secrets
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

from .approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalDenied,
    ToolRisk,
    resolve_tool_approval,
    resolve_tool_approval_async,
)

if TYPE_CHECKING:
    from kedi.lang.ast import ProcedureDef


ToolArgumentValidator = Callable[[Mapping[str, Any]], dict[str, Any]]
ToolArtifactMode = Literal["auto", "never"]

_RISK_ORDER: dict[ToolRisk, int] = {
    "read_only": 0,
    "mutating": 1,
    "sensitive": 2,
}


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


def with_approval(
    tool: ToolSpec,
    *,
    policy: ApprovalPolicy | None,
    adapter_shortname: str | None,
) -> ToolSpec:
    """Return a signature-preserving ToolSpec that enforces one approval policy."""

    is_async = _is_async_callable(tool.fn) or _has_async_approval_handler(policy)

    if is_async:

        @wraps(tool.fn)
        async def approved_async_tool(*args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")
            request = approval_request_for_tool(tool, kwargs, adapter_shortname)
            decision = await resolve_tool_approval_async(policy, request)
            final_kwargs = arguments_after_approval(tool, kwargs, decision)
            result = tool.fn(**final_kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return replace(tool, fn=approved_async_tool)

    @wraps(tool.fn)
    def approved_tool(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")
        request = approval_request_for_tool(tool, kwargs, adapter_shortname)
        decision = resolve_tool_approval(policy, request)
        final_kwargs = arguments_after_approval(tool, kwargs, decision)
        return tool.fn(**final_kwargs)

    return replace(tool, fn=approved_tool)


def with_artifact_result(
    tool: ToolSpec,
    *,
    manager: Any,
    policy: Any,
) -> ToolSpec:
    """Return a signature-preserving tool that exposes large results as compact refs."""

    metadata = dict(tool.metadata or {})
    if tool.artifact_result == "never" or metadata.get("kedi_artifact_wrapped") is True:
        return tool
    metadata["kedi_artifact_wrapped"] = True

    def compact_result(result: Any, *, call_id: str) -> Any:
        from kedi.artifacts import ArtifactHandle

        manager.history.append(
            "tool_call",
            data={"tool_name": tool.name, "call_id": call_id},
        )
        stored = manager.artifactize(
            result,
            policy=policy,
            source=f"tool:{tool.name}",
        )
        if not isinstance(stored, ArtifactHandle):
            manager.history.append(
                "tool_result",
                data={
                    "tool_name": tool.name,
                    "call_id": call_id,
                    "artifacted": False,
                },
            )
            return stored
        ref = stored.as_ref()
        manager.history.append(
            "tool_result",
            ref_id=ref.ref_id,
            data={
                "tool_name": tool.name,
                "call_id": call_id,
                "artifacted": True,
            },
        )
        return ref.model_dump(mode="json")

    is_async = _is_async_callable(tool.fn)
    if is_async:

        @wraps(tool.fn)
        async def artifact_async_tool(*args: Any, **kwargs: Any) -> Any:
            call_id = f"call_{secrets.token_urlsafe(10)}"
            result = await tool.fn(*args, **kwargs)
            return compact_result(result, call_id=call_id)

        return replace(tool, fn=artifact_async_tool, metadata=metadata)

    @wraps(tool.fn)
    def artifact_tool(*args: Any, **kwargs: Any) -> Any:
        call_id = f"call_{secrets.token_urlsafe(10)}"
        result = tool.fn(*args, **kwargs)
        if inspect.isawaitable(result):

            async def await_and_compact() -> Any:
                return compact_result(await result, call_id=call_id)

            return await_and_compact()
        return compact_result(result, call_id=call_id)

    return replace(tool, fn=artifact_tool, metadata=metadata)


def approval_request_for_tool(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    adapter_shortname: str | None,
) -> ApprovalRequest:
    static_risk = _validated_risk(tool.risk, tool.name)
    resolved_risk = (
        _validated_risk(tool.risk_resolver(arguments), tool.name)
        if tool.risk_resolver is not None
        else static_risk
    )
    risk = max((static_risk, resolved_risk), key=_RISK_ORDER.__getitem__)
    return ApprovalRequest.create(
        tool_name=tool.name,
        arguments=arguments,
        risk=risk,
        adapter_shortname=adapter_shortname,
        description=tool.description,
        metadata=tool.metadata,
    )


def arguments_after_approval(
    tool: ToolSpec,
    original_arguments: Mapping[str, Any],
    decision: ApprovalDecision,
) -> dict[str, Any]:
    if decision.decision == "deny":
        suffix = f": {decision.reason}" if decision.reason else ""
        raise ToolApprovalDenied(f"Tool {tool.name!r} was denied{suffix}")
    arguments = (
        dict(original_arguments)
        if decision.decision == "allow"
        else dict(decision.edit_arguments or {})
    )
    if tool.argument_validator is not None:
        return tool.argument_validator(arguments)
    return arguments


def arguments_target_secret_file(value: object) -> bool:
    """Return whether nested tool arguments target a dotenv-style secret file."""

    if isinstance(value, Mapping):
        return any(arguments_target_secret_file(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(arguments_target_secret_file(item) for item in value)
    if not isinstance(value, str):
        return False
    name = Path(value).name
    return name == ".env" or name.startswith(".env.")


def _validated_risk(risk: object, tool_name: str) -> ToolRisk:
    if risk not in _RISK_ORDER:
        raise ValueError(f"invalid Kedi tool risk {risk!r} for {tool_name!r}")
    return cast(ToolRisk, risk)


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


def _has_async_approval_handler(policy: ApprovalPolicy | None) -> bool:
    return policy is not None and policy.handler is not None and _is_async_callable(policy.handler)


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
