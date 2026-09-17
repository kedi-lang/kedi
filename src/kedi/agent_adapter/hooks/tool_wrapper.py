from __future__ import annotations

import asyncio
import inspect
import secrets
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from time import monotonic
from typing import Any

from ..approvals.policy import ToolApprovalDenied
from ..events import current_agent_run_identity, current_bound_agent_run_identity
from ..tools.execution import validate_tool_arguments
from ..tools.specs import ToolSpec
from .dispatcher import current_hook_dispatcher
from .models import (
    HookDeniedError,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreToolUseDecision,
    PreToolUseRequest,
    ToolOrigin,
)


def new_tool_call_id() -> str:
    return f"call_{secrets.token_urlsafe(10)}"


async def dispatch_pre_tool_use(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin = "kedi",
    native_tool_name: str | None = None,
) -> dict[str, Any]:
    canonical = validate_tool_arguments(tool, arguments)
    dispatcher = current_hook_dispatcher()
    if dispatcher is None or not dispatcher.handlers_for("pre_tool_use"):
        return canonical
    result = await dispatcher.dispatch(
        _pre_event(
            tool,
            canonical,
            adapter_shortname=adapter_shortname,
            tool_call_id=tool_call_id,
            origin=origin,
            native_tool_name=native_tool_name,
        )
    )
    return _resolve_pre_decision(tool, result, canonical)


def dispatch_pre_tool_use_sync(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin = "kedi",
    native_tool_name: str | None = None,
) -> dict[str, Any]:
    canonical = validate_tool_arguments(tool, arguments)
    dispatcher = current_hook_dispatcher()
    if dispatcher is None or not dispatcher.handlers_for("pre_tool_use"):
        return canonical
    result = dispatcher.dispatch_sync(
        _pre_event(
            tool,
            canonical,
            adapter_shortname=adapter_shortname,
            tool_call_id=tool_call_id,
            origin=origin,
            native_tool_name=native_tool_name,
        )
    )
    return _resolve_pre_decision(tool, result, canonical)


async def dispatch_post_tool_use(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    result: Any,
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin = "kedi",
    native_tool_name: str | None = None,
) -> None:
    dispatcher = current_hook_dispatcher()
    if dispatcher is None or not dispatcher.handlers_for("post_tool_use"):
        return
    await dispatcher.dispatch(
        PostToolUseEvent(
            **_tool_event_fields(
                tool,
                arguments,
                adapter_shortname=adapter_shortname,
                tool_call_id=tool_call_id,
                origin=origin,
                native_tool_name=native_tool_name,
            ),
            result=result,
        )
    )


def dispatch_post_tool_use_sync(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    result: Any,
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin = "kedi",
    native_tool_name: str | None = None,
) -> None:
    dispatcher = current_hook_dispatcher()
    if dispatcher is None or not dispatcher.handlers_for("post_tool_use"):
        return
    dispatcher.dispatch_sync(
        PostToolUseEvent(
            **_tool_event_fields(
                tool,
                arguments,
                adapter_shortname=adapter_shortname,
                tool_call_id=tool_call_id,
                origin=origin,
                native_tool_name=native_tool_name,
            ),
            result=result,
        )
    )


async def dispatch_post_tool_use_failure(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    error: BaseException,
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin = "kedi",
    native_tool_name: str | None = None,
    duration_seconds: float | None = None,
    interrupted: bool | None = None,
) -> None:
    dispatcher = current_hook_dispatcher()
    if dispatcher is None or not dispatcher.handlers_for("post_tool_use_failure"):
        return
    await dispatcher.dispatch(
        _failure_event(
            tool,
            arguments,
            error,
            adapter_shortname=adapter_shortname,
            tool_call_id=tool_call_id,
            origin=origin,
            native_tool_name=native_tool_name,
            duration_seconds=duration_seconds,
            interrupted=interrupted,
        )
    )


def dispatch_post_tool_use_failure_sync(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    error: BaseException,
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin = "kedi",
    native_tool_name: str | None = None,
    duration_seconds: float | None = None,
    interrupted: bool | None = None,
) -> None:
    dispatcher = current_hook_dispatcher()
    if dispatcher is None or not dispatcher.handlers_for("post_tool_use_failure"):
        return
    dispatcher.dispatch_sync(
        _failure_event(
            tool,
            arguments,
            error,
            adapter_shortname=adapter_shortname,
            tool_call_id=tool_call_id,
            origin=origin,
            native_tool_name=native_tool_name,
            duration_seconds=duration_seconds,
            interrupted=interrupted,
        )
    )


def with_hook_lifecycle(
    tool: ToolSpec,
    *,
    adapter_shortname: str,
    origin: ToolOrigin = "kedi",
) -> ToolSpec:
    """Wrap a final ToolSpec boundary when its adapter has no native lifecycle seam."""

    metadata = dict(tool.metadata or {})
    if metadata.get("kedi_hook_wrapped") is True:
        return tool
    metadata["kedi_hook_wrapped"] = True

    if _is_async_callable(tool.fn):

        @wraps(tool.fn)
        async def hooked_async_tool(*args: Any, **kwargs: Any) -> Any:
            _require_keyword_arguments(tool, args)
            call_id = new_tool_call_id()
            final_arguments = await dispatch_pre_tool_use(
                tool,
                kwargs,
                adapter_shortname=adapter_shortname,
                tool_call_id=call_id,
                origin=origin,
            )
            started = monotonic()
            try:
                result = await tool.fn(**final_arguments)
            except BaseException as exc:
                if _is_execution_failure(exc):
                    await dispatch_post_tool_use_failure(
                        tool,
                        final_arguments,
                        exc,
                        adapter_shortname=adapter_shortname,
                        tool_call_id=call_id,
                        origin=origin,
                        duration_seconds=monotonic() - started,
                        interrupted=is_tool_execution_interrupted(exc),
                    )
                raise
            await dispatch_post_tool_use(
                tool,
                final_arguments,
                result,
                adapter_shortname=adapter_shortname,
                tool_call_id=call_id,
                origin=origin,
            )
            return result

        return replace(tool, fn=hooked_async_tool, metadata=metadata)

    @wraps(tool.fn)
    def hooked_tool(*args: Any, **kwargs: Any) -> Any:
        _require_keyword_arguments(tool, args)
        call_id = new_tool_call_id()
        final_arguments = dispatch_pre_tool_use_sync(
            tool,
            kwargs,
            adapter_shortname=adapter_shortname,
            tool_call_id=call_id,
            origin=origin,
        )
        started = monotonic()
        try:
            result = tool.fn(**final_arguments)
        except BaseException as exc:
            if _is_execution_failure(exc):
                dispatch_post_tool_use_failure_sync(
                    tool,
                    final_arguments,
                    exc,
                    adapter_shortname=adapter_shortname,
                    tool_call_id=call_id,
                    origin=origin,
                    duration_seconds=monotonic() - started,
                    interrupted=is_tool_execution_interrupted(exc),
                )
            raise
        if inspect.isawaitable(result):

            async def await_hooked_result() -> Any:
                try:
                    resolved = await result
                except BaseException as exc:
                    if _is_execution_failure(exc):
                        await dispatch_post_tool_use_failure(
                            tool,
                            final_arguments,
                            exc,
                            adapter_shortname=adapter_shortname,
                            tool_call_id=call_id,
                            origin=origin,
                            duration_seconds=monotonic() - started,
                            interrupted=is_tool_execution_interrupted(exc),
                        )
                    raise
                await dispatch_post_tool_use(
                    tool,
                    final_arguments,
                    resolved,
                    adapter_shortname=adapter_shortname,
                    tool_call_id=call_id,
                    origin=origin,
                )
                return resolved

            return await_hooked_result()
        dispatch_post_tool_use_sync(
            tool,
            final_arguments,
            result,
            adapter_shortname=adapter_shortname,
            tool_call_id=call_id,
            origin=origin,
        )
        return result

    return replace(tool, fn=hooked_tool, metadata=metadata)


def _pre_event(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin,
    native_tool_name: str | None,
) -> PreToolUseRequest:
    return PreToolUseRequest(
        **_tool_event_fields(
            tool,
            arguments,
            adapter_shortname=adapter_shortname,
            tool_call_id=tool_call_id,
            origin=origin,
            native_tool_name=native_tool_name,
        )
    )


def _failure_event(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    error: BaseException,
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin,
    native_tool_name: str | None,
    duration_seconds: float | None,
    interrupted: bool | None,
) -> PostToolUseFailureEvent:
    return PostToolUseFailureEvent(
        **_tool_event_fields(
            tool,
            arguments,
            adapter_shortname=adapter_shortname,
            tool_call_id=tool_call_id,
            origin=origin,
            native_tool_name=native_tool_name,
        ),
        error_type=type(error).__name__,
        error_message=str(error),
        error=error,
        duration_seconds=duration_seconds,
        interrupted=(is_tool_execution_interrupted(error) if interrupted is None else interrupted),
    )


def _tool_event_fields(
    tool: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    adapter_shortname: str,
    tool_call_id: str,
    origin: ToolOrigin,
    native_tool_name: str | None,
) -> dict[str, Any]:
    identity = current_bound_agent_run_identity() or current_agent_run_identity()
    metadata = {
        key: value for key, value in (tool.metadata or {}).items() if not key.startswith("kedi_")
    }
    return {
        "run_id": identity.run_id if identity is not None else "unbound",
        "parent_run_id": identity.parent_run_id if identity is not None else None,
        "agent_name": identity.agent_name if identity is not None else None,
        "sequence": 0,
        "adapter_shortname": adapter_shortname,
        "tool_call_id": tool_call_id,
        "tool_name": tool.name,
        "native_tool_name": native_tool_name,
        "origin": origin,
        "arguments": arguments,
        "description": tool.description,
        "metadata": metadata,
    }


def _resolve_pre_decision(
    tool: ToolSpec,
    result: Any,
    original: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, PreToolUseDecision):
        raise TypeError("pre_tool_use dispatch returned an invalid decision")
    if result.decision == "deny":
        assert result.reason is not None
        raise HookDeniedError(event="pre_tool_use", reason=result.reason)
    if result.decision == "edit":
        assert result.edit_arguments is not None
        return validate_tool_arguments(tool, result.edit_arguments)
    return dict(original)


def _require_keyword_arguments(tool: ToolSpec, args: tuple[Any, ...]) -> None:
    if args:
        raise TypeError(f"Kedi tool {tool.name!r} only accepts keyword arguments")


def _is_execution_failure(error: BaseException) -> bool:
    return not isinstance(error, (HookDeniedError, ToolApprovalDenied))


def is_tool_execution_interrupted(error: BaseException) -> bool:
    return isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))


def _is_async_callable(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


__all__ = [
    "dispatch_post_tool_use",
    "dispatch_post_tool_use_failure",
    "dispatch_post_tool_use_failure_sync",
    "is_tool_execution_interrupted",
    "dispatch_post_tool_use_sync",
    "dispatch_pre_tool_use",
    "dispatch_pre_tool_use_sync",
    "new_tool_call_id",
    "with_hook_lifecycle",
]
