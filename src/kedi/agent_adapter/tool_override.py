from __future__ import annotations

import asyncio
import inspect
import secrets
from collections.abc import Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field, replace
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
    sequential: bool = False


@dataclass(slots=True)
class RequiredToolCalls:
    """Track tools that must return before an adapter response can be accepted."""

    names: frozenset[str]
    attempted: set[str] = field(default_factory=set)
    returned: set[str] = field(default_factory=set)


class RequiredToolCallError(RuntimeError):
    """Raised when an adapter returns before a required tool completes."""


_required_tool_calls: ContextVar[RequiredToolCalls | None] = ContextVar(
    "kedi_required_tool_calls",
    default=None,
)


def current_required_tool_calls() -> RequiredToolCalls | None:
    """Return the required-tool state for the active adapter turn."""

    return _required_tool_calls.get()


@contextmanager
def required_tool_call_scope(tools: Sequence[ToolSpec]) -> Iterator[None]:
    """Isolate required-tool tracking while allowing an adapter's nested scope to reuse it."""

    names = frozenset(
        tool.name for tool in tools if (tool.metadata or {}).get("required_before_output") is True
    )
    current = _required_tool_calls.get()
    if current is not None and current.names == names:
        yield
        return

    token = _required_tool_calls.set(RequiredToolCalls(names) if names else None)
    try:
        yield
    finally:
        _required_tool_calls.reset(token)


def validate_required_tool_calls() -> None:
    """Reject an adapter result if a required tool did not return successfully."""

    state = _required_tool_calls.get()
    if state is None or state.names <= state.returned:
        return
    missing_names = state.names - state.returned
    missing = ", ".join(sorted(missing_names))
    if missing_names & state.attempted:
        raise RequiredToolCallError(
            f"Required tool {missing} did not complete successfully; the adapter response "
            "cannot be accepted"
        )
    raise RequiredToolCallError(
        f"Required tool {missing} was not called; the adapter response cannot be accepted"
    )


def with_required_before_output(tool: ToolSpec) -> ToolSpec:
    """Track successful completion of a required tool without changing its signature."""

    metadata = dict(tool.metadata or {})
    if (
        metadata.get("required_before_output") is not True
        or metadata.get("kedi_required_tool_wrapped") is True
    ):
        return tool
    metadata["kedi_required_tool_wrapped"] = True

    def mark_attempted() -> RequiredToolCalls | None:
        state = _required_tool_calls.get()
        if state is not None and tool.name in state.names:
            state.attempted.add(tool.name)
            return state
        return None

    def mark_returned(state: RequiredToolCalls | None) -> None:
        if state is not None:
            state.returned.add(tool.name)

    if _is_async_callable(tool.fn):

        @wraps(tool.fn)
        async def required_async_tool(*args: Any, **kwargs: Any) -> Any:
            state = mark_attempted()
            result = tool.fn(*args, **kwargs)
            result = await result
            mark_returned(state)
            return result

        return replace(tool, fn=required_async_tool, metadata=metadata)

    @wraps(tool.fn)
    def required_tool(*args: Any, **kwargs: Any) -> Any:
        state = mark_attempted()
        result = tool.fn(*args, **kwargs)
        if inspect.isawaitable(result):

            async def await_required_result() -> Any:
                resolved = await result
                mark_returned(state)
                return resolved

            return await_required_result()
        mark_returned(state)
        return result

    return replace(tool, fn=required_tool, metadata=metadata)


def with_tool_telemetry(tool: ToolSpec, *, adapter_shortname: str) -> ToolSpec:
    """Trace a logical tool call without changing the callable's public signature."""

    metadata = dict(tool.metadata or {})
    if metadata.get("kedi_telemetry_wrapped") is True:
        return tool
    metadata["kedi_telemetry_wrapped"] = True

    if _is_async_callable(tool.fn):

        @wraps(tool.fn)
        async def telemetry_async_tool(*args: Any, **kwargs: Any) -> Any:
            from .telemetry import tool_call_span

            with tool_call_span(
                tool,
                adapter_shortname=adapter_shortname,
                argument_count=len(args) + len(kwargs),
            ) as telemetry:
                result = tool.fn(*args, **kwargs)
                result = await result
                telemetry.complete(result)
                return result

        return replace(tool, fn=telemetry_async_tool, metadata=metadata)

    @wraps(tool.fn)
    def telemetry_tool(*args: Any, **kwargs: Any) -> Any:
        from .telemetry import tool_call_span

        call_context = copy_context()
        context = tool_call_span(
            tool,
            adapter_shortname=adapter_shortname,
            argument_count=len(args) + len(kwargs),
        )

        def invoke() -> tuple[Any, Any | None]:
            telemetry = context.__enter__()
            try:
                result = tool.fn(*args, **kwargs)
            except BaseException as exc:
                context.__exit__(type(exc), exc, exc.__traceback__)
                raise
            if inspect.isawaitable(result):
                return result, telemetry
            telemetry.complete(result)
            context.__exit__(None, None, None)
            return result, None

        result, pending_telemetry = call_context.run(invoke)
        if pending_telemetry is not None:

            async def await_telemetry_result() -> Any:
                try:
                    future = call_context.run(asyncio.ensure_future, result)
                    resolved = await future
                except BaseException as exc:
                    call_context.run(
                        context.__exit__,
                        type(exc),
                        exc,
                        exc.__traceback__,
                    )
                    raise

                def finish() -> None:
                    pending_telemetry.complete(resolved)
                    context.__exit__(None, None, None)

                call_context.run(finish)
                return resolved

            return await_telemetry_result()
        return result

    return replace(tool, fn=telemetry_tool, metadata=metadata)


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

    def record_admission_failure(*, call_id: str, error: str) -> None:
        manager.history.append(
            "tool_result",
            data={
                "tool_name": tool.name,
                "call_id": call_id,
                "artifacted": False,
                "error": error,
            },
        )

    def quota_error(exc: Exception) -> dict[str, object]:
        return {
            "ok": False,
            "error": "artifact_quota_exceeded",
            "message": str(exc),
            "recovery": (
                "The tool output was not retained. Retry the source tool with a narrower "
                "query or bounded output, and release artifacts that are no longer needed."
            ),
        }

    def compact_result(result: Any, *, call_id: str) -> Any:
        from kedi.artifacts import ArtifactHandle, ArtifactStream
        from kedi.artifacts.errors import (
            ArtifactQuotaExceededError,
            ToolOutputTooLargeError,
        )

        manager.history.append(
            "tool_call",
            data={"tool_name": tool.name, "call_id": call_id},
        )
        try:
            if isinstance(result, ArtifactStream):
                stored = manager.artifactize_stream(
                    result,
                    policy=policy,
                    source=f"tool:{tool.name}",
                )
            else:
                stored = manager.admit_tool_result(
                    result,
                    policy=policy,
                    source=f"tool:{tool.name}",
                )
        except ToolOutputTooLargeError:
            record_admission_failure(call_id=call_id, error="tool_output_too_large")
            raise
        except ArtifactQuotaExceededError as exc:
            record_admission_failure(call_id=call_id, error="artifact_quota_exceeded")
            return quota_error(exc)
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

    async def compact_result_async(result: Any, *, call_id: str) -> Any:
        from kedi.artifacts import ArtifactHandle, ArtifactStream
        from kedi.artifacts.errors import (
            ArtifactQuotaExceededError,
            ToolOutputTooLargeError,
        )

        if not isinstance(result, ArtifactStream):
            return compact_result(result, call_id=call_id)
        manager.history.append(
            "tool_call",
            data={"tool_name": tool.name, "call_id": call_id},
        )
        try:
            stored = await manager.artifactize_stream_async(
                result,
                policy=policy,
                source=f"tool:{tool.name}",
            )
        except ToolOutputTooLargeError:
            record_admission_failure(call_id=call_id, error="tool_output_too_large")
            raise
        except ArtifactQuotaExceededError as exc:
            record_admission_failure(call_id=call_id, error="artifact_quota_exceeded")
            return quota_error(exc)
        if not isinstance(stored, ArtifactHandle):
            manager.history.append(
                "tool_result",
                data={"tool_name": tool.name, "call_id": call_id, "artifacted": False},
            )
            return stored
        ref = stored.as_ref()
        manager.history.append(
            "tool_result",
            ref_id=ref.ref_id,
            data={"tool_name": tool.name, "call_id": call_id, "artifacted": True},
        )
        return ref.model_dump(mode="json")

    is_async = _is_async_callable(tool.fn)
    if is_async:

        @wraps(tool.fn)
        async def artifact_async_tool(*args: Any, **kwargs: Any) -> Any:
            call_id = f"call_{secrets.token_urlsafe(10)}"
            result = await tool.fn(*args, **kwargs)
            return await compact_result_async(result, call_id=call_id)

        return replace(tool, fn=artifact_async_tool, metadata=metadata)

    @wraps(tool.fn)
    def artifact_tool(*args: Any, **kwargs: Any) -> Any:
        call_id = f"call_{secrets.token_urlsafe(10)}"
        result = tool.fn(*args, **kwargs)
        if inspect.isawaitable(result):

            async def await_and_compact() -> Any:
                return await compact_result_async(await result, call_id=call_id)

            return await_and_compact()
        from kedi.artifacts import ArtifactStream

        if isinstance(result, ArtifactStream) and result.is_async:

            async def compact_async_stream() -> Any:
                return await compact_result_async(result, call_id=call_id)

            return compact_async_stream()
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
