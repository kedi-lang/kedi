from __future__ import annotations

import asyncio
import builtins
import copy
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar, cast

from kedi.errors import KediExecutionError, KediLoopIterationLimitError

_TOOL_RETRY_POLICY_ATTR = "__kedi_tool_retry_policy__"
_TOOL_BODY_ATTR = "__kedi_tool_body__"

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ToolRetryPolicy:
    """Normalized retry policy for one tool callable body."""

    retries: int = 0
    retry_on: tuple[type[Exception], ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.retries, bool) or not isinstance(self.retries, int):
            raise TypeError("tool retries must be an integer")
        if self.retries < 0:
            raise ValueError("tool retries must be >= 0")
        if self.retry_on is not None:
            object.__setattr__(self, "retry_on", normalize_retry_on(self.retry_on))

    def matches(self, exc: BaseException) -> bool:
        candidate = _retry_candidate(exc)
        if candidate is None:
            return False
        expected = (Exception,) if self.retry_on is None else self.retry_on
        return bool(expected) and isinstance(candidate, expected)


def normalize_retry_on(
    retry_on: Sequence[type[Exception]] | None,
) -> tuple[type[Exception], ...] | None:
    if retry_on is None:
        return None
    if isinstance(retry_on, (str, bytes)) or not isinstance(retry_on, Sequence):
        raise TypeError("tool retry_on must be a sequence of Exception classes")
    normalized: list[type[Exception]] = []
    seen: set[type[Exception]] = set()
    for exception_type in retry_on:
        if not isinstance(exception_type, type) or not issubclass(exception_type, Exception):
            raise TypeError("tool retry_on must contain only Exception classes")
        if exception_type not in seen:
            seen.add(exception_type)
            normalized.append(exception_type)
    return tuple(normalized)


def resolve_retry_on(
    names: tuple[str, ...] | None,
    env: Mapping[str, Any] | None,
) -> tuple[type[Exception], ...] | None:
    """Resolve native exception names without evaluating user expressions."""

    if names is None:
        return None
    resolved: list[type[Exception]] = []
    for name in names:
        value = env.get(name) if env is not None and name in env else getattr(builtins, name, None)
        if value is None:
            raise KediExecutionError(f"Unknown retry exception: {name!r}")
        if not isinstance(value, type) or not issubclass(value, Exception):
            raise KediExecutionError(f"Retry target {name!r} must resolve to an Exception class")
        resolved.append(value)
    return cast(tuple[type[Exception], ...], normalize_retry_on(resolved))


def tool_retry_policy(value: Callable[..., Any]) -> ToolRetryPolicy:
    policy = getattr(value, _TOOL_RETRY_POLICY_ATTR, None)
    return policy if isinstance(policy, ToolRetryPolicy) else ToolRetryPolicy()


def tool_body(value: Callable[..., Any]) -> Callable[..., Any]:
    body = getattr(value, _TOOL_BODY_ATTR, None)
    return body if callable(body) else value


def with_body_retry(
    fn: Callable[..., T],
    policy: ToolRetryPolicy,
    *,
    isolate_arguments: bool,
) -> Callable[..., T]:
    """Wrap only callable-body failures; callers own approval and result handoff."""

    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)
    ):

        @wraps(fn)
        async def async_retry(*args: Any, **kwargs: Any) -> Any:
            inspect.signature(fn).bind(*args, **kwargs)
            frozen = _freeze_arguments(args, kwargs) if isolate_arguments else None
            attempt = 0
            while True:
                call_args, call_kwargs = (
                    _fresh_arguments(frozen) if frozen is not None else (args, kwargs)
                )
                try:
                    return await cast(Any, fn)(*call_args, **call_kwargs)
                except BaseException as exc:
                    if attempt >= policy.retries or not policy.matches(exc):
                        raise
                    attempt += 1

        wrapped = async_retry
    else:

        @wraps(fn)
        def sync_retry(*args: Any, **kwargs: Any) -> Any:
            inspect.signature(fn).bind(*args, **kwargs)
            frozen = _freeze_arguments(args, kwargs) if isolate_arguments else None
            attempt = 0
            while True:
                call_args, call_kwargs = (
                    _fresh_arguments(frozen) if frozen is not None else (args, kwargs)
                )
                try:
                    return fn(*call_args, **call_kwargs)
                except BaseException as exc:
                    if attempt >= policy.retries or not policy.matches(exc):
                        raise
                    attempt += 1

        wrapped = sync_retry

    setattr(wrapped, _TOOL_RETRY_POLICY_ATTR, policy)
    setattr(wrapped, _TOOL_BODY_ATTR, fn)
    return cast(Callable[..., T], wrapped)


def _freeze_arguments(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return copy.deepcopy((args, kwargs))


def _fresh_arguments(
    frozen: tuple[tuple[Any, ...], dict[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    return copy.deepcopy(frozen)


def _retry_candidate(exc: BaseException) -> Exception | None:
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
        return None
    if isinstance(exc, KediExecutionError):
        from kedi.agent_adapter.hooks.models import HookExecutionError

        if isinstance(exc, (HookExecutionError, KediLoopIterationLimitError)):
            return None
        original = exc.original
        return original if isinstance(original, Exception) else None
    return exc if isinstance(exc, Exception) else None


__all__ = [
    "ToolRetryPolicy",
    "normalize_retry_on",
    "resolve_retry_on",
    "tool_body",
    "tool_retry_policy",
    "with_body_retry",
]
