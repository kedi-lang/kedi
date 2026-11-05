from __future__ import annotations

import functools
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar, cast
from uuid import uuid4

from ..events import (
    AgentRunIdentity,
    bind_agent_run,
    current_agent_run_identity,
    current_bound_agent_run_identity,
)
from .dispatcher import (
    HookDispatcher,
    HookDispatcherSnapshot,
    bind_hook_dispatcher,
    current_hook_dispatcher,
)
from .models import HookDeniedError, UserPromptSubmitDecision, UserPromptSubmitRequest
from .registration import HookRegistrationMixin

P = ParamSpec("P")
R = TypeVar("R")

_ACTIVE_HOOK_RUNS: ContextVar[frozenset[tuple[int, str]]] = ContextVar(
    "kedi_active_hook_runs",
    default=frozenset(),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedAgentPrompt:
    content: str
    identity: AgentRunIdentity


def hooked_agent_run(prompt_argument: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Apply user-prompt hooks and bind their dispatcher for one adapter run."""

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)
        if prompt_argument not in signature.parameters:
            name = getattr(fn, "__qualname__", type(fn).__qualname__)
            raise TypeError(f"{name} has no {prompt_argument!r} argument")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                adapter = _adapter_from_args(args)
                bound = signature.bind(*args, **kwargs)
                prompt = _prompt_from_bound(bound, prompt_argument)
                async with prepared_agent_hook_run_async(adapter, prompt) as prepared:
                    bound.arguments[prompt_argument] = prepared.content
                    return await cast(Callable[..., Awaitable[Any]], fn)(
                        *bound.args,
                        **bound.kwargs,
                    )

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            adapter = _adapter_from_args(args)
            bound = signature.bind(*args, **kwargs)
            prompt = _prompt_from_bound(bound, prompt_argument)
            with prepared_agent_hook_run(adapter, prompt) as prepared:
                bound.arguments[prompt_argument] = prepared.content
                return fn(*bound.args, **bound.kwargs)

        return cast(Callable[P, R], sync_wrapper)

    return decorate


@contextmanager
def prepared_agent_hook_run(
    adapter: HookRegistrationMixin,
    prompt: str,
) -> Generator[PreparedAgentPrompt, None, None]:
    """Resolve prompt mutation before request assembly and suppress nested redispatch."""

    dispatcher, identity = _dispatcher_for(adapter)
    run_key = (id(adapter), identity.run_id)
    if dispatcher is None or run_key in _ACTIVE_HOOK_RUNS.get():
        yield PreparedAgentPrompt(content=prompt, identity=identity)
        return
    bridge_token = _ACTIVE_HOOK_RUNS.set(_ACTIVE_HOOK_RUNS.get() | {run_key})
    try:
        with (
            bind_agent_run(
                identity.run_id,
                parent_run_id=identity.parent_run_id,
                agent_name=identity.agent_name,
            ),
            bind_hook_dispatcher(dispatcher),
        ):
            yield PreparedAgentPrompt(
                content=_dispatch_prompt_sync(adapter, dispatcher, identity, prompt),
                identity=identity,
            )
    finally:
        _ACTIVE_HOOK_RUNS.reset(bridge_token)


@asynccontextmanager
async def prepared_agent_hook_run_async(
    adapter: HookRegistrationMixin,
    prompt: str,
) -> AsyncGenerator[PreparedAgentPrompt, None]:
    """Async counterpart of :func:`prepared_agent_hook_run`."""

    dispatcher, identity = _dispatcher_for(adapter)
    run_key = (id(adapter), identity.run_id)
    if dispatcher is None or run_key in _ACTIVE_HOOK_RUNS.get():
        yield PreparedAgentPrompt(content=prompt, identity=identity)
        return
    bridge_token = _ACTIVE_HOOK_RUNS.set(_ACTIVE_HOOK_RUNS.get() | {run_key})
    try:
        with (
            bind_agent_run(
                identity.run_id,
                parent_run_id=identity.parent_run_id,
                agent_name=identity.agent_name,
            ),
            bind_hook_dispatcher(dispatcher),
        ):
            yield PreparedAgentPrompt(
                content=await _dispatch_prompt(adapter, dispatcher, identity, prompt),
                identity=identity,
            )
    finally:
        _ACTIVE_HOOK_RUNS.reset(bridge_token)


@asynccontextmanager
async def native_agent_hook_run(
    adapter: HookRegistrationMixin,
    prompt: Any,
) -> AsyncGenerator[Any, None]:
    """Bind hooks for a native adapter run that did not enter a decorated Kedi method."""

    if current_hook_dispatcher() is not None or not isinstance(prompt, str):
        yield prompt
        return
    async with prepared_agent_hook_run_async(adapter, prompt) as prepared:
        yield prepared.content


def _adapter_from_args(args: tuple[Any, ...]) -> HookRegistrationMixin:
    if not args or not isinstance(args[0], HookRegistrationMixin):
        raise TypeError("hooked_agent_run requires a HookRegistrationMixin adapter method")
    return args[0]


def _prompt_from_bound(bound: inspect.BoundArguments, name: str) -> str:
    prompt = bound.arguments.get(name)
    if not isinstance(prompt, str):
        raise TypeError(f"Kedi hook prompt argument {name!r} must be a string")
    return prompt


def _dispatcher_for(
    adapter: HookRegistrationMixin,
) -> tuple[HookDispatcher | None, AgentRunIdentity]:
    profile = getattr(adapter, "active_profile", None)
    settings = getattr(profile, "hooks", None)
    snapshot = HookDispatcherSnapshot.compose(adapter.hook_snapshot(), settings)
    identity = _run_identity()
    owned_hooks = any(
        tool.capability_policy is not None and tool.capability_policy.hook_contributions
        for tool in getattr(adapter, "scoped_tool_specs", ())
    )
    if not snapshot.active and not getattr(profile, "capabilities", ()) and not owned_hooks:
        return None, identity
    return HookDispatcher(snapshot), identity


def _run_identity() -> AgentRunIdentity:
    bound = current_bound_agent_run_identity()
    if bound is not None:
        return bound
    parent = current_agent_run_identity()
    return AgentRunIdentity(
        run_id=f"agent_{uuid4().hex}",
        parent_run_id=None if parent is None else parent.run_id,
        agent_name=None if parent is None else parent.agent_name,
    )


async def _dispatch_prompt(
    adapter: HookRegistrationMixin,
    dispatcher: HookDispatcher,
    identity: AgentRunIdentity,
    prompt: str,
) -> str:
    result = await dispatcher.dispatch(_prompt_event(adapter, identity, prompt))
    return _resolve_prompt_decision(result, prompt)


def _dispatch_prompt_sync(
    adapter: HookRegistrationMixin,
    dispatcher: HookDispatcher,
    identity: AgentRunIdentity,
    prompt: str,
) -> str:
    result = dispatcher.dispatch_sync(_prompt_event(adapter, identity, prompt))
    return _resolve_prompt_decision(result, prompt)


def _prompt_event(
    adapter: HookRegistrationMixin,
    identity: AgentRunIdentity,
    prompt: str,
) -> UserPromptSubmitRequest:
    profile = getattr(adapter, "active_profile", None)
    return UserPromptSubmitRequest(
        run_id=identity.run_id,
        parent_run_id=identity.parent_run_id,
        agent_name=identity.agent_name,
        profile_name=getattr(profile, "name", None),
        adapter_shortname=cast(str, getattr(adapter, "shortname", type(adapter).__name__)),
        sequence=0,
        content=prompt,
    )


def _resolve_prompt_decision(result: Any, original: str) -> str:
    if not isinstance(result, UserPromptSubmitDecision):
        raise TypeError("user_prompt_submit dispatch returned an invalid decision")
    if result.decision == "deny":
        assert result.reason is not None
        raise HookDeniedError(event="user_prompt_submit", reason=result.reason)
    if result.decision == "edit":
        assert result.edited_content is not None
        return result.edited_content
    return original


__all__ = [
    "PreparedAgentPrompt",
    "hooked_agent_run",
    "native_agent_hook_run",
    "prepared_agent_hook_run",
    "prepared_agent_hook_run_async",
]
