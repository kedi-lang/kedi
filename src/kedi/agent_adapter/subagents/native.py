"""Program-controlled subagent tasks backed by the existing coordinator."""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Lock
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

from pydantic import TypeAdapter

from kedi.agent_adapter.execution import (
    OwnedAsyncLoop,
    _resolve_deadline,
    current_async_owner_loop,
)
from kedi.agent_adapter.subagents import (
    _NO_TYPED_OUTPUT,
    DelegationResult,
    SubagentOutputState,
)
from kedi.agent_adapter.subagents.attempts import track_delegation_attempt
from kedi.agent_adapter.subagents.budget import DelegationBudget
from kedi.agent_adapter.subagents.control import (
    AgentInput,
    InputReceipt,
    LiveInput,
    current_live_input,
    live_input_scope,
)
from kedi.agent_profile import current_block_agent_profile
from kedi.errors import KediExecutionError

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile
    from kedi.core import KediRuntime

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class SubagentResult:
    output: object | None
    run_id: str
    subagent: str
    task_summary: str


class SubagentTask:
    def __init__(
        self,
        owner: NativeSubagentOwner,
        run_id: str,
        output_state: SubagentOutputState | None = None,
    ) -> None:
        self._owner = owner
        self.run_id = run_id
        self._result: SubagentResult | None = None
        self._output_state = output_state

    @property
    def task_id(self) -> str:
        return self.run_id

    async def send(self, message: str, *, interrupt: bool = False) -> InputReceipt:
        self._owner._check_owner()
        self._owner._bind_execution_loop()
        sender = current_live_input() or self._owner.inbox
        return self._owner.runtime._subagent_coordinator.send_input(
            self.run_id, message, interrupt=interrupt, sender=sender
        )

    async def wait(self) -> SubagentResult:
        self._owner._check_owner()
        self._owner._bind_execution_loop()
        if self._result is None:
            try:
                raw = await self._owner.runtime._wait_subagent(self.run_id)
                # Another waiter may have published the result while this one was suspended.
                if self._result is None:
                    result = _result_value(raw)
                    if self._output_state is not None:
                        if self._output_state.typed_output is _NO_TYPED_OUTPUT:
                            raise RuntimeError(
                                "Subagent completed without native output validation"
                            )
                        result = replace(result, output=deepcopy(self._output_state.typed_output))
                    self._result = result
            except asyncio.CancelledError:
                # Cancelling a waiter is not the same as observing a cancelled child.
                if self._owner.runtime._subagent_status(self.run_id) == "cancelled":
                    self._owner._observed.add(self.run_id)
                raise
            except Exception:
                if self._owner.runtime._subagent_status(self.run_id) != "running":
                    self._owner._observed.add(self.run_id)
                raise
        self._owner._observed.add(self.run_id)
        return self._result


def _result_value(raw: DelegationResult) -> SubagentResult:
    return SubagentResult(
        output=deepcopy(raw.final_result),
        run_id=raw.run_id,
        subagent=raw.subagent,
        task_summary=raw.task_summary,
    )


class NativeSubagentOwner:
    """Own background runs and their execution loop for one root invocation."""

    def __init__(self, runtime: KediRuntime) -> None:
        self.runtime = runtime
        self._loop: OwnedAsyncLoop | None = None
        self._budgets: dict[tuple[str | None, str], DelegationBudget] = {}
        self._budget_lock = Lock()
        self._concurrency = runtime._new_subagent_concurrency()
        self._tasks: dict[str, SubagentTask] = {}
        self._observed: set[str] = set()
        self._execution_loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = Lock()
        self._groups: dict[Future[None], Callable[[], None]] = {}
        self._closing = False
        self._closed = False
        self.inbox = LiveInput("owner-" + uuid4().hex)
        self.inbox.configure(input=True)

    async def receive_messages(self) -> tuple[AgentInput, ...]:
        self._check_owner()
        self._bind_execution_loop()
        while True:
            sequence = self.inbox.sequence
            items = self.inbox.take_pending()
            if items:
                self.inbox.mark_delivered(items)
                return items
            if self.inbox.finished:
                return ()
            await self.inbox.wait_changed(sequence)

    def budget_for(self, profile: AgentProfile) -> DelegationBudget:
        name = profile.name or "<anonymous>"
        stack = self.runtime._subagent_stack.get()
        key = (stack[-1][0] if stack else None, name)
        with self._budget_lock:
            budget = self._budgets.get(key)
            if budget is None:
                budget = DelegationBudget.for_profile(
                    profile_name=name, max_agents=profile.max_agents
                )
                self._budgets[key] = budget
            elif profile.max_agents is not None:
                budget.limit = min(budget.limit, profile.max_agents)
            return budget

    @property
    def concurrency(self) -> Any:
        return self._concurrency

    def _check_owner(self) -> None:
        if self._closed or self.runtime._native_subagent_owner.get() is not self:
            raise KediExecutionError("Subagent task belongs to another or closed invocation")

    def _bind_execution_loop(self) -> None:
        current = asyncio.get_running_loop()
        with self._state_lock:
            if self._execution_loop is None:
                self._execution_loop = current
            elif self._execution_loop is not current:
                raise KediExecutionError("Subagent task belongs to another event loop")

    async def start(
        self,
        subagent: str,
        *,
        task: str,
        parent_profile: AgentProfile,
        final_schema: dict[str, Any] | None = None,
        output_adapter: TypeAdapter[Any] | None = None,
    ) -> SubagentTask:
        from .groups import check_process_cancelled, register_process_child

        self._check_owner()
        self._bind_execution_loop()
        check_process_cancelled()
        if self._closing:
            raise KediExecutionError("Subagent invocation is closing")
        if subagent not in parent_profile.subagents:
            raise KediExecutionError(
                f"Subagent profile {subagent!r} is not allowed in the current profile"
            )
        adapter = self.runtime._resolve_adapter(parent_profile)
        capabilities = getattr(adapter, "capabilities", None)
        if capabilities is None or not capabilities.supports("background_subagents"):
            raise KediExecutionError("The parent adapter does not support background subagents")
        definition = dict(parent_profile.subagent_bindings).get(subagent)
        if definition is None:
            definition = self.runtime._profile_registry.get(subagent)
        child_profile = None
        if definition is not None:
            child_profile = self.runtime._evaluate_profile_definition(definition)
            if final_schema is not None and child_profile.output_type is not None:
                raise KediExecutionError(
                    f"Subagent {subagent!r} cannot combine task captures with > output:"
                )
            if output_adapter is None and child_profile.output_type is not None:
                output_adapter = TypeAdapter(child_profile.output_type)
        output_state = SubagentOutputState(output_adapter) if output_adapter is not None else None
        with (
            live_input_scope(current_live_input() or self.inbox),
            track_delegation_attempt(requested_profile=subagent, task=task),
        ):
            handle = self.runtime._start_subagent(
                allowed_subagents=tuple(parent_profile.subagents),
                subagent=subagent,
                task=task,
                final_schema=final_schema,
                output_state=output_state,
                parent_profile=parent_profile,
                parent_budget=self.budget_for(parent_profile),
                parent_concurrency=self._concurrency,
                subagent_definitions=dict(parent_profile.subagent_bindings),
                evaluated_profile=child_profile,
                retain_for_native_owner=True,
            )
        native_task = SubagentTask(self, handle.run_id, output_state)
        self._tasks[handle.run_id] = native_task
        register_process_child(native_task)
        return native_task

    def start_sync(
        self,
        subagent: str,
        *,
        task: str,
        parent_profile: AgentProfile,
        final_schema: dict[str, Any] | None = None,
        output_adapter: TypeAdapter[Any] | None = None,
    ) -> SubagentTask:
        self._check_owner()
        return self._run_sync(
            lambda: self.start(
                subagent,
                task=task,
                parent_profile=parent_profile,
                final_schema=final_schema,
                output_adapter=output_adapter,
            )
        )

    def wait_sync(self, task: SubagentTask) -> SubagentResult:
        self._check_owner()
        if task._owner is not self:
            raise KediExecutionError("Subagent task belongs to another invocation")
        if self._loop is None and self._execution_loop is None:
            raise KediExecutionError("Subagent task has no active invocation loop")
        return self._run_sync(task.wait)

    def _run_sync(self, factory: Callable[[], Awaitable[_T]]) -> _T:
        return self.runtime._engine.wait(self.submit(factory))

    def submit(
        self, factory: Callable[[], Awaitable[_T]], *, enforce_deadline: bool = True
    ) -> Future[_T]:
        with self._state_lock:
            if self._loop is None and self._execution_loop is None:
                self._execution_loop = current_async_owner_loop()
                if self._execution_loop is None:
                    self._loop = OwnedAsyncLoop()
            owned_loop = self._loop
            loop = self._execution_loop
        if owned_loop is not None:
            return owned_loop.submit(factory(), enforce_deadline=enforce_deadline)
        assert loop is not None
        if loop.is_closed():
            raise KediExecutionError("Subagent owner event loop is closed")
        if not loop.is_running():
            raise KediExecutionError("Subagent owner event loop is not running")
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is loop:
            raise KediExecutionError("Cannot synchronously wait on the subagent owner event loop")

        async def resolve() -> _T:
            return await factory()

        awaitable = _resolve_deadline(factory()) if enforce_deadline else resolve()
        return asyncio.run_coroutine_threadsafe(awaitable, loop)

    def register_group(self, finished: Future[None], cancel: Callable[[], None]) -> None:
        self._check_owner()
        with self._state_lock:
            if self._closing:
                raise KediExecutionError("Subagent invocation is closing")
            self._groups[finished] = cancel

    def release_group(self, finished: Future[None]) -> None:
        with self._state_lock:
            self._groups.pop(finished, None)

    async def close(self, primary_error: BaseException | None = None) -> None:
        if self._closed:
            return
        self.inbox.close()
        self._bind_execution_loop()
        with self._state_lock:
            self._closing = True
            groups = tuple(self._groups.items())
        for _, cancel in groups:
            cancel()
        try:
            if groups:
                await asyncio.gather(
                    *(asyncio.wrap_future(finished) for finished, _ in groups),
                    return_exceptions=True,
                )
            await self.drain_tasks(tuple(self._tasks), primary_error)
        finally:
            self._closed = True
            self.runtime._subagent_coordinator.release_native_runs(tuple(self._tasks))
            self._tasks.clear()
            self._observed.clear()

    async def drain_tasks(
        self, pending: tuple[str, ...], primary_error: BaseException | None = None
    ) -> None:
        self._bind_execution_loop()
        unobserved = set(pending).difference(self._observed)
        cancelled_on_close: set[str] = set()
        for run_id in pending:
            if self.runtime._subagent_status(run_id) == "running":
                if self.runtime._cancel_subagent(run_id):
                    cancelled_on_close.add(run_id)
        results = await asyncio.gather(
            *(self._tasks[run_id].wait() for run_id in pending),
            return_exceptions=True,
        )
        if primary_error is not None:
            return
        for run_id, result in zip(pending, results):
            if run_id not in unobserved:
                continue
            if run_id in cancelled_on_close:
                raise KediExecutionError(f"Subagent task {run_id!r} was not awaited")
            if isinstance(result, BaseException):
                raise result
            raise KediExecutionError(f"Subagent task {run_id!r} was not awaited")

    def close_sync(self, primary_error: BaseException | None = None) -> None:
        try:
            if self._loop is not None or self._execution_loop is not None:
                self.runtime._engine.wait(
                    self.submit(lambda: self.close(primary_error), enforce_deadline=False)
                )
            else:
                self._closed = True
        finally:
            if self._loop is not None:
                self._loop.close()


class SubagentScope:
    def __init__(self, runtime: KediRuntime, parent: str) -> None:
        self.runtime = runtime
        self.parent = parent
        self._owner: NativeSubagentOwner | None = None
        self._token: Any = None
        self._owns_lifetime = False
        self._active = False
        self._invocation_token: Any = None
        self._request_budget_token: Any = None
        self._run_budget_scope: Any = None

    async def __aenter__(self) -> SubagentScope:
        if self._active:
            raise KediExecutionError("Subagent scope is already active")
        existing = self.runtime._native_subagent_owner.get()
        owner = existing or NativeSubagentOwner(self.runtime)
        owner._bind_execution_loop()
        self._owner = owner
        self._owns_lifetime = existing is None
        if self._owns_lifetime:
            from kedi.agent_adapter.request_budget import ModelRequestBudget
            from kedi.run_budget import run_budget_scope

            self._run_budget_scope = run_budget_scope(self.runtime.run_budget)
            self._run_budget_scope.__enter__()
            self._invocation_token = self.runtime._invocation_id.set(object())
            self._request_budget_token = self.runtime._request_budget.set(ModelRequestBudget())
        self._token = self.runtime._native_subagent_owner.set(self._owner)
        self._active = True
        return self

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        try:
            if self._owns_lifetime and self._owner is not None:
                cleanup = asyncio.create_task(self._owner.close(exc))
                interrupted: asyncio.CancelledError | None = None
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError as cancellation:
                        interrupted = cancellation
                    except BaseException:
                        pass
                cleanup_error: BaseException | None = None
                try:
                    cleanup.result()
                except BaseException as failure:
                    cleanup_error = failure
                if exc is not None:
                    if cleanup_error is not None:
                        add_note = getattr(exc, "add_note", None)
                        if add_note is not None:
                            add_note(f"Subagent cleanup also failed: {cleanup_error}")
                        else:
                            warnings.warn(
                                f"Subagent cleanup also failed: {cleanup_error}",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                elif interrupted is not None:
                    raise interrupted from cleanup_error
                elif cleanup_error is not None:
                    raise cleanup_error
        finally:
            self.runtime._native_subagent_owner.reset(self._token)
            if self._request_budget_token is not None:
                self.runtime._request_budget.reset(self._request_budget_token)
                self._request_budget_token = None
            if self._invocation_token is not None:
                self.runtime._invocation_id.reset(self._invocation_token)
                self._invocation_token = None
            if self._run_budget_scope is not None:
                self._run_budget_scope.__exit__(exc_type, exc, tb)
                self._run_budget_scope = None
            self._token = None
            self._active = False
            self._owner = None

    async def start(self, subagent: str, *, task: str) -> SubagentTask:
        if not self._active or self._owner is None:
            raise KediExecutionError("Subagent scope is not active")
        definition = self.runtime._profile_registry.get(self.parent)
        if definition is None:
            raise KediExecutionError(f"Unknown parent profile: {self.parent!r}")
        profile = self.runtime._evaluate_profile_definition(definition)
        lexical = current_block_agent_profile()
        if lexical.name and lexical.name != profile.name:
            raise KediExecutionError("Cannot select another effective parent profile")
        active_stack = self.runtime._subagent_stack.get()
        if active_stack and active_stack[-1][1] != profile.name:
            raise KediExecutionError("Cannot impersonate another subagent parent")
        return await self._owner.start(subagent, task=task, parent_profile=profile)

    async def receive_messages(self) -> tuple[AgentInput, ...]:
        """Receive child updates without creating an implicit parent model call."""
        if not self._active or self._owner is None:
            raise KediExecutionError("Subagent scope is not active")
        return await self._owner.receive_messages()
