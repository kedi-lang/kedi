from __future__ import annotations

import builtins
import dataclasses
import functools
import inspect
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    cast,
    overload,
)

from kedi.agent_adapter.approvals.policy import (
    ApprovalHandler,
    ApprovalMode,
    ApprovalPolicy,
    ToolRisk,
)
from kedi.agent_adapter.hooks import HookEventName, HookHandler, validate_hook_event_names
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.api.state import ClassT, P, R, _config_var

_TOOL_NAME_ATTR = "__kedi_tool_name__"


_TOOL_DESCRIPTION_ATTR = "__kedi_tool_description__"


_TOOL_RISK_ATTR = "__kedi_tool_risk__"


_TYPE_INJECT_ATTR = "__kedi_type_inject__"


_TYPE_REGISTRY: dict[str, dict[str, builtins.type[Any]]] = {}


def _callable_name(fn: Callable[..., Any]) -> str:
    tool_name = getattr(fn, _TOOL_NAME_ATTR, None)
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    name = getattr(fn, "__name__", None)
    return name if isinstance(name, str) else builtins.type(fn).__name__


def _tool_mapping(tools: Sequence[Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    return {_callable_name(tool): tool for tool in tools}


def _is_existing_type_surface(cls: builtins.type[Any]) -> bool:
    from pydantic import BaseModel
    from pydantic.dataclasses import is_pydantic_dataclass

    return issubclass(cls, BaseModel) or is_pydantic_dataclass(cls) or dataclasses.is_dataclass(cls)


def _register_type(cls: ClassT, *, inject: bool) -> ClassT:
    if not isinstance(cls, builtins.type):
        raise TypeError("@kedi.type can only decorate classes.")

    registered = cls if _is_existing_type_surface(cls) else dataclasses.dataclass(cls)
    setattr(registered, _TYPE_INJECT_ATTR, inject)

    module_registry = _TYPE_REGISTRY.setdefault(registered.__module__, {})
    if inject:
        module_registry[registered.__name__] = registered
    else:
        module_registry.pop(registered.__name__, None)
    return cast(ClassT, registered)


def _auto_type_env(fn: Callable[..., Any]) -> dict[str, Any]:
    module = getattr(fn, "__module__", None)
    if not isinstance(module, str):
        return {}
    return dict(_TYPE_REGISTRY.get(module, {}))


def _coerce_approval(
    value: ApprovalPolicy | ApprovalMode | ApprovalHandler | None,
) -> ApprovalPolicy | None:
    if value is None:
        return None
    if isinstance(value, ApprovalPolicy):
        return value
    if isinstance(value, str):
        if value == "allow":
            return ApprovalPolicy.allow()
        if value == "deny":
            return ApprovalPolicy.deny()
        raise TypeError("approval must be 'allow', 'deny', ApprovalPolicy, or an approval handler")
    if callable(value):
        return ApprovalPolicy.dynamic(value)
    raise TypeError("approval must be 'allow', 'deny', ApprovalPolicy, or an approval handler")


def approval(handler: ApprovalHandler) -> ApprovalHandler:
    """Register ``handler`` as the current Python API's default approval policy.

    Explicit ``approval=`` values on :func:`query`, :func:`bind`, or
    :func:`context` take precedence for their own scope.
    """

    policy = ApprovalPolicy.dynamic(handler)
    config = _config_var.get()
    _config_var.set(
        dataclasses.replace(
            config,
            profile=config.profile.merge(AgentProfile(approval=policy)),
        )
    )
    return handler


def on(
    events: HookEventName | str | Sequence[HookEventName | str],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a hook handler in the current Python API configuration."""

    event_names = validate_hook_event_names(events)

    def register(handler: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(handler):
            raise TypeError("Kedi hook handler must be callable")
        config = _config_var.get()
        settings = HookSettings()
        for event in event_names:
            settings = settings.append(event, cast(HookHandler, handler))
        _config_var.set(
            dataclasses.replace(
                config,
                profile=config.profile.merge(
                    AgentProfile(
                        hooks=settings,
                        _hook_enforcement=settings,
                    )
                ),
            )
        )
        return handler

    return register


@overload
def tool(fn: Callable[P, R]) -> Callable[P, R]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int = 0,
    risk: ToolRisk = "mutating",
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def tool(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    retries: int = 0,
    risk: ToolRisk = "mutating",
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    if retries < 0:
        raise ValueError("tool retries must be >= 0")
    if risk not in ("read_only", "mutating", "sensitive"):
        raise ValueError("tool risk must be 'read_only', 'mutating', or 'sensitive'")

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        tool_name = name or _callable_name(inner)
        tool_description = description or inspect.getdoc(inner)

        if _is_async_callable(inner):

            @functools.wraps(inner)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                attempts = retries + 1
                last_error: Exception | None = None
                for _ in range(attempts):
                    try:
                        return await cast(Any, inner)(*args, **kwargs)
                    except Exception as exc:
                        last_error = exc
                assert last_error is not None
                raise last_error

            wrapped = cast(Callable[P, R], async_wrapper)
        else:

            @functools.wraps(inner)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                attempts = retries + 1
                last_error: Exception | None = None
                for _ in range(attempts):
                    try:
                        return inner(*args, **kwargs)
                    except Exception as exc:
                        last_error = exc
                assert last_error is not None
                raise last_error

            wrapped = sync_wrapper

        setattr(wrapped, _TOOL_NAME_ATTR, tool_name)
        setattr(wrapped, _TOOL_RISK_ATTR, risk)
        if tool_description is not None:
            setattr(wrapped, _TOOL_DESCRIPTION_ATTR, tool_description)
            wrapped.__doc__ = tool_description
        return wrapped

    if fn is not None:
        return decorate(fn)
    return decorate


def _is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    )


@overload
def type(cls: ClassT) -> ClassT: ...


@overload
def type(*, inject: bool = True) -> Callable[[ClassT], ClassT]: ...


def type(
    cls: ClassT | None = None,
    *,
    inject: bool = True,
) -> ClassT | Callable[[ClassT], ClassT]:
    def decorate(inner: ClassT) -> ClassT:
        return _register_type(inner, inject=inject)

    if cls is not None:
        return decorate(cls)
    return decorate
