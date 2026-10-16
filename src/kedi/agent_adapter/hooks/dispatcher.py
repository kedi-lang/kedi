from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterator, cast

from ..execution import run_in_worker
from .models import (
    HOOK_EVENT_NAMES,
    HookEvent,
    HookEventName,
    HookExecutionError,
    HookHandler,
    HookResult,
    HookTimeoutError,
    PostToolResultFailureEvent,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreToolUseDecision,
    PreToolUseRequest,
    UserPromptSubmitDecision,
    UserPromptSubmitRequest,
)
from .registration import AdapterHookSnapshot, handler_identity
from .telemetry import HookChainTelemetry

if TYPE_CHECKING:
    from ...agent_profile.hooks import HookSettings

_ACTIVE_HOOK_DISPATCHER: ContextVar[HookDispatcher | None] = ContextVar(
    "kedi_active_hook_dispatcher",
    default=None,
)
_HOOK_DISPATCH_SUSPENDED: ContextVar[bool] = ContextVar(
    "kedi_hook_dispatch_suspended",
    default=False,
)


@dataclass(frozen=True, slots=True)
class HookDispatcherSnapshot:
    handlers: Mapping[HookEventName, tuple[HookHandler, ...]]
    timeout: float | None = None

    @classmethod
    def compose(
        cls,
        adapter: AdapterHookSnapshot | None,
        profile: HookSettings | None,
        *,
        timeout: float | None = None,
    ) -> HookDispatcherSnapshot:
        if timeout is not None and timeout <= 0:
            raise ValueError("hook timeout must be greater than zero")
        adapter = adapter or AdapterHookSnapshot.empty()
        handlers: dict[HookEventName, tuple[HookHandler, ...]] = {}
        for event in HOOK_EVENT_NAMES:
            chain = adapter.handlers_for(event)
            if profile is not None:
                chain += profile.handlers_for(event)
            if chain:
                handlers[event] = chain
        return cls(handlers=MappingProxyType(handlers), timeout=timeout)

    @property
    def active(self) -> bool:
        return bool(self.handlers)


class HookDispatcher:
    def __init__(self, snapshot: HookDispatcherSnapshot) -> None:
        self.snapshot = snapshot
        self._sequence_lock = Lock()
        self._sequences: dict[str, int] = {}

    def handlers_for(self, event: HookEventName) -> tuple[HookHandler, ...]:
        return self.snapshot.handlers.get(event, ())

    def with_tool_hooks(self, hooks: HookSettings) -> HookDispatcher:
        """Create a tool-local chain while preserving this run's event sequence."""

        handlers: dict[HookEventName, tuple[HookHandler, ...]] = {
            event: self.handlers_for(event)
            + (() if event == "user_prompt_submit" else hooks.handlers_for(event))
            for event in HOOK_EVENT_NAMES
        }
        child = HookDispatcher(
            HookDispatcherSnapshot(
                handlers=MappingProxyType(handlers), timeout=self.snapshot.timeout
            )
        )
        child._sequence_lock = self._sequence_lock
        child._sequences = self._sequences
        return child

    async def dispatch(self, event: HookEvent) -> HookResult:
        if _HOOK_DISPATCH_SUSPENDED.get():
            return _default_result(event)
        current = self._with_sequence(event)
        handlers = self.handlers_for(current.event)
        if not handlers:
            return _default_result(current)
        with HookChainTelemetry(current, len(handlers)) as telemetry:
            edited = False
            try:
                for handler in handlers:
                    result = await self._call_async(handler, current)
                    current, decision, changed = _apply_result(current, result)
                    edited = edited or changed
                    if _is_denied(decision):
                        telemetry.complete(denied=True)
                        return decision
            except asyncio.CancelledError:
                telemetry.cancel()
                raise
            except HookTimeoutError:
                telemetry.fail(timed_out=True)
                raise
            except BaseException:
                telemetry.fail()
                raise
            telemetry.complete(edited=edited)
            return _final_result(current, edited)

    def dispatch_sync(self, event: HookEvent) -> HookResult:
        if _HOOK_DISPATCH_SUSPENDED.get():
            return _default_result(event)
        current = self._with_sequence(event)
        handlers = self.handlers_for(current.event)
        if not handlers:
            return _default_result(current)
        with HookChainTelemetry(current, len(handlers)) as telemetry:
            edited = False
            try:
                for handler in handlers:
                    result = self._call_sync(handler, current)
                    current, decision, changed = _apply_result(current, result)
                    edited = edited or changed
                    if _is_denied(decision):
                        telemetry.complete(denied=True)
                        return decision
            except HookTimeoutError:
                telemetry.fail(timed_out=True)
                raise
            except BaseException:
                telemetry.fail()
                raise
            telemetry.complete(edited=edited)
            return _final_result(current, edited)

    async def _call_async(self, handler: HookHandler, event: HookEvent) -> HookResult:
        identity = handler_identity(handler)
        token = _HOOK_DISPATCH_SUSPENDED.set(True)
        try:

            async def invoke() -> HookResult:
                result = (
                    handler(event)
                    if inspect.iscoroutinefunction(handler)
                    or inspect.iscoroutinefunction(getattr(handler, "__call__", None))
                    else await run_in_worker(handler, event)
                )
                if inspect.isawaitable(result):
                    return await cast(Awaitable[HookResult], result)
                return cast(HookResult, result)

            if self.snapshot.timeout is None:
                return await invoke()
            try:
                return await asyncio.wait_for(invoke(), timeout=self.snapshot.timeout)
            except asyncio.TimeoutError as exc:
                raise HookTimeoutError(
                    event=event.event,
                    handler_name=identity,
                    timeout=self.snapshot.timeout,
                ) from exc
        except HookExecutionError:
            raise
        except Exception as exc:
            raise HookExecutionError(
                event=event.event,
                handler_name=identity,
                cause=exc,
            ) from exc
        finally:
            _HOOK_DISPATCH_SUSPENDED.reset(token)

    def _call_sync(self, handler: HookHandler, event: HookEvent) -> HookResult:
        identity = handler_identity(handler)
        started = monotonic()
        token = _HOOK_DISPATCH_SUSPENDED.set(True)
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(
                        self._await_sync_result(
                            cast(Awaitable[HookResult], result),
                            event,
                            identity,
                        )
                    )
                if inspect.iscoroutine(result):
                    result.close()
                raise RuntimeError(
                    "A synchronous Kedi hook boundary cannot await a handler on an active "
                    "event loop; use the adapter's async API"
                )
            self._check_elapsed(event, identity, started)
            return cast(HookResult, result)
        except HookExecutionError:
            raise
        except Exception as exc:
            raise HookExecutionError(
                event=event.event,
                handler_name=identity,
                cause=exc,
            ) from exc
        finally:
            _HOOK_DISPATCH_SUSPENDED.reset(token)

    async def _await_sync_result(
        self,
        result: Awaitable[HookResult],
        event: HookEvent,
        identity: str,
    ) -> HookResult:
        if self.snapshot.timeout is None:
            return await result
        try:
            return await asyncio.wait_for(result, timeout=self.snapshot.timeout)
        except asyncio.TimeoutError as exc:
            raise HookTimeoutError(
                event=event.event,
                handler_name=identity,
                timeout=self.snapshot.timeout,
            ) from exc

    def _check_elapsed(self, event: HookEvent, identity: str, started: float) -> None:
        timeout = self.snapshot.timeout
        if timeout is not None and monotonic() - started > timeout:
            raise HookTimeoutError(
                event=event.event,
                handler_name=identity,
                timeout=timeout,
            )

    def _with_sequence(self, event: HookEvent) -> HookEvent:
        with self._sequence_lock:
            sequence = self._sequences.get(event.run_id, 0) + 1
            self._sequences[event.run_id] = sequence
        return replace(event, sequence=sequence)


def _apply_result(
    event: HookEvent,
    result: HookResult,
) -> tuple[HookEvent, HookResult, bool]:
    if isinstance(event, UserPromptSubmitRequest):
        if result is None:
            return event, UserPromptSubmitDecision.continue_(), False
        if not isinstance(result, UserPromptSubmitDecision):
            raise TypeError(
                "user_prompt_submit handlers must return UserPromptSubmitDecision or None"
            )
        if result.decision == "edit":
            assert result.edited_content is not None
            return replace(event, content=result.edited_content), result, True
        return event, result, False
    if isinstance(event, PreToolUseRequest):
        if result is None:
            return event, PreToolUseDecision.continue_(), False
        if not isinstance(result, PreToolUseDecision):
            raise TypeError("pre_tool_use handlers must return PreToolUseDecision or None")
        if result.decision == "edit":
            assert result.edit_arguments is not None
            return replace(event, arguments=result.edit_arguments), result, True
        return event, result, False
    if not isinstance(
        event, (PostToolUseEvent, PostToolUseFailureEvent, PostToolResultFailureEvent)
    ):
        raise TypeError(f"Unsupported Kedi hook event object {type(event).__name__}")
    if result is not None:
        raise TypeError(f"{event.event} handlers must return None")
    return event, None, False


def _is_denied(result: HookResult) -> bool:
    return isinstance(result, (UserPromptSubmitDecision, PreToolUseDecision)) and (
        result.decision == "deny"
    )


def _default_result(event: HookEvent) -> HookResult:
    if isinstance(event, UserPromptSubmitRequest):
        return UserPromptSubmitDecision.continue_()
    if isinstance(event, PreToolUseRequest):
        return PreToolUseDecision.continue_()
    return None


def _final_result(event: HookEvent, edited: bool) -> HookResult:
    if isinstance(event, UserPromptSubmitRequest):
        return (
            UserPromptSubmitDecision.edit(event.content)
            if edited
            else UserPromptSubmitDecision.continue_()
        )
    if isinstance(event, PreToolUseRequest):
        return (
            PreToolUseDecision.edit(event.arguments) if edited else PreToolUseDecision.continue_()
        )
    return None


@contextmanager
def bind_hook_dispatcher(dispatcher: HookDispatcher) -> Iterator[None]:
    token: Token[HookDispatcher | None] = _ACTIVE_HOOK_DISPATCHER.set(dispatcher)
    try:
        yield
    finally:
        _ACTIVE_HOOK_DISPATCHER.reset(token)


def current_hook_dispatcher() -> HookDispatcher | None:
    return _ACTIVE_HOOK_DISPATCHER.get()


__all__ = [
    "HookDispatcher",
    "HookDispatcherSnapshot",
    "bind_hook_dispatcher",
    "current_hook_dispatcher",
]
