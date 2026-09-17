from __future__ import annotations

import asyncio
import inspect
import secrets
from collections.abc import Mapping
from contextvars import copy_context
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, cast

from ..approvals.policy import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalDenied,
    ToolRisk,
    resolve_tool_approval,
    resolve_tool_approval_async,
)
from .scope import (
    RequiredToolCalls,
    _defer_artifact_admission,
    _required_tool_calls,
    current_artifact_admission,
)
from .specs import ToolSpec

_RISK_ORDER: dict[ToolRisk, int] = {
    "read_only": 0,
    "mutating": 1,
    "sensitive": 2,
}


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
            from ..telemetry import tool_call_span

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
        from ..telemetry import tool_call_span

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
            final_kwargs = await resolve_tool_arguments_async(
                tool,
                kwargs,
                policy=policy,
                adapter_shortname=adapter_shortname,
            )
            result = tool.fn(**final_kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return replace(tool, fn=approved_async_tool)

    @wraps(tool.fn)
    def approved_tool(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")
        final_kwargs = resolve_tool_arguments(
            tool,
            kwargs,
            policy=policy,
            adapter_shortname=adapter_shortname,
        )
        return tool.fn(**final_kwargs)

    return replace(tool, fn=approved_tool)


def _record_artifact_admission_failure(
    manager: Any,
    *,
    tool_name: str,
    call_id: str,
    error: str,
) -> None:
    manager.history.append(
        "tool_result",
        data={
            "tool_name": tool_name,
            "call_id": call_id,
            "artifacted": False,
            "error": error,
        },
    )


def _artifact_quota_error(exc: Exception) -> dict[str, object]:
    return {
        "ok": False,
        "error": "artifact_quota_exceeded",
        "message": str(exc),
        "recovery": (
            "The tool output was not retained. Retry the source tool with a narrower "
            "query or bounded output, and release artifacts that are no longer needed."
        ),
    }


def admit_artifact_result(
    result: Any,
    *,
    manager: Any,
    policy: Any,
    tool_name: str,
    call_id: str,
) -> Any:
    """Admit one successful materialized result while preserving its native call ID."""

    from kedi.artifacts import ArtifactHandle, ArtifactStream
    from kedi.artifacts.errors import ArtifactQuotaExceededError, ToolOutputTooLargeError

    manager.history.append(
        "tool_call",
        data={"tool_name": tool_name, "call_id": call_id},
    )
    try:
        if isinstance(result, ArtifactStream):
            stored = manager.artifactize_stream(
                result,
                policy=policy,
                source=f"tool:{tool_name}",
            )
        else:
            stored = manager.admit_tool_result(
                result,
                policy=policy,
                source=f"tool:{tool_name}",
            )
    except ToolOutputTooLargeError:
        _record_artifact_admission_failure(
            manager,
            tool_name=tool_name,
            call_id=call_id,
            error="tool_output_too_large",
        )
        raise
    except ArtifactQuotaExceededError as exc:
        _record_artifact_admission_failure(
            manager,
            tool_name=tool_name,
            call_id=call_id,
            error="artifact_quota_exceeded",
        )
        return _artifact_quota_error(exc)
    if not isinstance(stored, ArtifactHandle):
        manager.history.append(
            "tool_result",
            data={"tool_name": tool_name, "call_id": call_id, "artifacted": False},
        )
        return stored
    ref = stored.as_ref()
    manager.history.append(
        "tool_result",
        ref_id=ref.ref_id,
        data={"tool_name": tool_name, "call_id": call_id, "artifacted": True},
    )
    return ref.model_dump(mode="json")


async def admit_artifact_result_async(
    result: Any,
    *,
    manager: Any,
    policy: Any,
    tool_name: str,
    call_id: str,
) -> Any:
    """Admit materialized and streamed results through one async adapter boundary."""

    from kedi.artifacts import ArtifactHandle, ArtifactStream
    from kedi.artifacts.errors import ArtifactQuotaExceededError, ToolOutputTooLargeError

    if not isinstance(result, ArtifactStream) or not result.is_async:
        return admit_artifact_result(
            result,
            manager=manager,
            policy=policy,
            tool_name=tool_name,
            call_id=call_id,
        )
    manager.history.append(
        "tool_call",
        data={"tool_name": tool_name, "call_id": call_id},
    )
    try:
        stored = await manager.artifactize_stream_async(
            result,
            policy=policy,
            source=f"tool:{tool_name}",
        )
    except ToolOutputTooLargeError:
        _record_artifact_admission_failure(
            manager,
            tool_name=tool_name,
            call_id=call_id,
            error="tool_output_too_large",
        )
        raise
    except ArtifactQuotaExceededError as exc:
        _record_artifact_admission_failure(
            manager,
            tool_name=tool_name,
            call_id=call_id,
            error="artifact_quota_exceeded",
        )
        return _artifact_quota_error(exc)
    if not isinstance(stored, ArtifactHandle):
        manager.history.append(
            "tool_result",
            data={"tool_name": tool_name, "call_id": call_id, "artifacted": False},
        )
        return stored
    ref = stored.as_ref()
    manager.history.append(
        "tool_result",
        ref_id=ref.ref_id,
        data={"tool_name": tool_name, "call_id": call_id, "artifacted": True},
    )
    return ref.model_dump(mode="json")


async def admit_current_artifact_result(
    result: Any,
    *,
    tool_name: str,
    call_id: str,
) -> Any:
    """Artifact-admit a native result when a Kedi execution scope is active."""

    context = current_artifact_admission()
    if context is None or _defer_artifact_admission.get():
        return result
    manager, policy = context
    return await admit_artifact_result_async(
        result,
        manager=manager,
        policy=policy,
        tool_name=tool_name,
        call_id=call_id,
    )


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

    async def admit_result(result: Any, tool_name: str) -> Any:
        return await admit_artifact_result_async(
            result,
            manager=manager,
            policy=policy,
            tool_name=tool_name,
            call_id=f"call_{secrets.token_urlsafe(10)}",
        )

    setattr(admit_result, "_kedi_artifact_boundary_key", (id(manager), id(policy)))

    is_async = _is_async_callable(tool.fn)
    if is_async:

        @wraps(tool.fn)
        async def artifact_async_tool(*args: Any, **kwargs: Any) -> Any:
            call_id = f"call_{secrets.token_urlsafe(10)}"
            result = await tool.fn(*args, **kwargs)
            if _defer_artifact_admission.get():
                return result
            return await admit_artifact_result_async(
                result,
                manager=manager,
                policy=policy,
                tool_name=tool.name,
                call_id=call_id,
            )

        return replace(
            tool,
            fn=artifact_async_tool,
            metadata=metadata,
            result_admitter=admit_result,
        )

    @wraps(tool.fn)
    def artifact_tool(*args: Any, **kwargs: Any) -> Any:
        call_id = f"call_{secrets.token_urlsafe(10)}"
        result = tool.fn(*args, **kwargs)
        if inspect.isawaitable(result):

            async def await_and_compact() -> Any:
                resolved = await result
                if _defer_artifact_admission.get():
                    return resolved
                return await admit_artifact_result_async(
                    resolved,
                    manager=manager,
                    policy=policy,
                    tool_name=tool.name,
                    call_id=call_id,
                )

            return await_and_compact()
        if _defer_artifact_admission.get():
            return result
        from kedi.artifacts import ArtifactStream

        if isinstance(result, ArtifactStream) and result.is_async:

            async def compact_async_stream() -> Any:
                return await admit_artifact_result_async(
                    result,
                    manager=manager,
                    policy=policy,
                    tool_name=tool.name,
                    call_id=call_id,
                )

            return compact_async_stream()
        return admit_artifact_result(
            result,
            manager=manager,
            policy=policy,
            tool_name=tool.name,
            call_id=call_id,
        )

    return replace(
        tool,
        fn=artifact_tool,
        metadata=metadata,
        result_admitter=admit_result,
    )


def approval_request_for_tool(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    adapter_shortname: str | None,
) -> ApprovalRequest:
    return _approval_request_for_canonical_arguments(
        tool,
        validate_tool_arguments(tool, arguments),
        adapter_shortname,
    )


def _approval_request_for_canonical_arguments(
    tool: ToolSpec,
    arguments: dict[str, Any],
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
    return validate_tool_arguments(tool, arguments)


def validate_tool_arguments(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical validated/defaulted arguments used for approval and execution."""

    values = dict(arguments)
    if tool.argument_validator is not None:
        return tool.argument_validator(values)
    return values


def resolve_tool_arguments(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    policy: ApprovalPolicy | None,
    adapter_shortname: str | None,
) -> dict[str, Any]:
    """Validate, approve, and reapprove edits that increase a tool call's risk."""

    request = approval_request_for_tool(tool, arguments, adapter_shortname)
    while True:
        decision = resolve_tool_approval(policy, request)
        approved = arguments_after_approval(tool, request.arguments, decision)
        if decision.decision != "edit":
            return approved
        edited_request = _approval_request_for_canonical_arguments(
            tool,
            approved,
            adapter_shortname,
        )
        if not _risk_increased(request, edited_request):
            return approved
        request = edited_request


async def resolve_tool_arguments_async(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    policy: ApprovalPolicy | None,
    adapter_shortname: str | None,
) -> dict[str, Any]:
    """Async counterpart of :func:`resolve_tool_arguments`."""

    request = approval_request_for_tool(tool, arguments, adapter_shortname)
    while True:
        decision = await resolve_tool_approval_async(policy, request)
        approved = arguments_after_approval(tool, request.arguments, decision)
        if decision.decision != "edit":
            return approved
        edited_request = _approval_request_for_canonical_arguments(
            tool,
            approved,
            adapter_shortname,
        )
        if not _risk_increased(request, edited_request):
            return approved
        request = edited_request


def _risk_increased(before: ApprovalRequest, after: ApprovalRequest) -> bool:
    return _RISK_ORDER[after.risk] > _RISK_ORDER[before.risk]


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
