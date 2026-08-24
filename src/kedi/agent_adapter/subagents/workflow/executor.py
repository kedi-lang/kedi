from __future__ import annotations

import asyncio
from collections.abc import Callable, Container, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from pydantic_monty import (
    ExternalException,
    ExternalReturnValue,
    ExternalSettledResult,
    FunctionSnapshot,
    FutureSnapshot,
    MontyComplete,
    NameLookupSnapshot,
)

WorkflowDispatch: TypeAlias = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]
WorkflowState: TypeAlias = FunctionSnapshot | FutureSnapshot | NameLookupSnapshot | MontyComplete
PendingCall: TypeAlias = asyncio.Task[Any] | Coroutine[Any, Any, Any]


class WorkflowAbortError(RuntimeError):
    """Abort the complete workflow instead of exposing an error to sandbox code."""


@dataclass
class WorkflowExecutor:
    """Drive one Monty snapshot session and dispatch host-side child calls."""

    dispatch: WorkflowDispatch
    valid_names: Container[str]
    _pending: dict[int, PendingCall] = field(default_factory=dict, init=False)

    async def run(self, state: WorkflowState) -> MontyComplete:
        try:
            while not isinstance(state, MontyComplete):
                if isinstance(state, NameLookupSnapshot):
                    state = state.resume()
                elif isinstance(state, FunctionSnapshot):
                    state = self._dispatch(state)
                else:
                    state = await self._resolve(state)
        finally:
            tasks: list[asyncio.Task[Any]] = []
            for call in self._pending.values():
                if isinstance(call, asyncio.Task):
                    call.cancel()
                    tasks.append(call)
                else:
                    call.close()
            self._pending.clear()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        return state

    def _dispatch(self, snapshot: FunctionSnapshot) -> WorkflowState:
        if snapshot.is_os_function:
            return snapshot.resume_auto()
        name = snapshot.function_name
        if name not in self.valid_names:
            return snapshot.resume({"exception": NameError(f"Unknown function: {name}")})
        if snapshot.args:
            return snapshot.resume(
                {
                    "exception": TypeError(
                        f"{name}() does not accept positional arguments; use keyword arguments"
                    )
                }
            )
        self._pending[snapshot.call_id] = asyncio.create_task(self.dispatch(name, snapshot.kwargs))
        return snapshot.resume({"future": ...})

    async def _resolve(self, snapshot: FutureSnapshot) -> WorkflowState:
        call_ids = list(snapshot.pending_call_ids)
        settled = await asyncio.gather(
            *(self._pending[call_id] for call_id in call_ids),
            return_exceptions=True,
        )
        results: dict[int, ExternalSettledResult] = {}
        for call_id, outcome in zip(call_ids, settled):
            del self._pending[call_id]
            if isinstance(outcome, WorkflowAbortError):
                raise outcome
            if isinstance(outcome, Exception):
                results[call_id] = ExternalException(exception=_sandbox_exception(outcome))
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                results[call_id] = ExternalReturnValue(return_value=outcome)
        return snapshot.resume(results)


def _sandbox_exception(exc: Exception) -> RuntimeError:
    """Create a traceback-free exception safe to serialize into Monty."""

    message = str(exc).strip() or type(exc).__name__
    return RuntimeError(message)
