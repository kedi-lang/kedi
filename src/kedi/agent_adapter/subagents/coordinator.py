from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import AbstractContextManager, nullcontext
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

from ...errors import KediExecutionError, KediTraceFrame
from ...telemetry import TelemetrySpan, current_span
from ...telemetry import span as telemetry_span
from ..events import bind_agent_run, current_agent_run_identity
from ..invocation import InvocationUsage, observe_invocations
from ..telemetry import current_tool_call_name
from . import (
    DelegationResult,
    SubagentConversationContext,
    SubagentExecutionRequest,
    SubagentTurn,
    SupportsSubagentExecution,
    effective_final_schema,
    new_subagent_run_id,
    validate_structured_result,
    wrapped_final_schema,
)
from .attempts import bind_current_delegation_run
from .budget import (
    DelegationBudget,
    DelegationConcurrency,
    SubagentBudgetExceeded,
    SubagentConcurrencyExceeded,
)
from .limits import SubagentUsageLimits
from .persistence import StoredConversation, SubagentStateStore
from .runs import (
    SubagentCancelled,
    SubagentCompleted,
    SubagentEventLog,
    SubagentFailed,
    SubagentRunRegistry,
    SubagentRunStatus,
    SubagentStarted,
)
from .safety import SubagentSafetyCeiling

DEFAULT_MAX_SUBAGENT_DEPTH = 5
HARD_MAX_SUBAGENT_DEPTH = 5
DEFAULT_MAX_SUBAGENT_CONCURRENCY = 4
DEFAULT_SUBAGENT_TIMEOUT_SECONDS = 120.0
MAX_SUBAGENT_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_RETAINED_SUBAGENT_RUNS = 1024
MAX_SUBAGENT_CONVERSATION_TURNS = 8
_DELEGATION_TOOL_NAMES = frozenset({"delegate_task", "continue_subagent"})


def _subagent_run_span(
    attributes: Mapping[str, Any],
) -> AbstractContextManager[TelemetrySpan]:
    current = current_span()
    tool_name = current_tool_call_name()
    profile = attributes.get("kedi.subagent.profile")
    span_name = f"{profile} run" if isinstance(profile, str) and profile else "subagent run"
    if current.is_recording() and tool_name in _DELEGATION_TOOL_NAMES:
        current.update_name(span_name)
        current.set_attribute("kedi.operation.name", "run_agent")
        current.set_attribute("gen_ai.operation.name", "invoke_agent")
        for key, value in attributes.items():
            if isinstance(value, (bool, str, int, float)):
                current.set_attribute(key, value)
        return nullcontext(current)
    return telemetry_span(
        "agent",
        span_name,
        "run_agent",
        attributes={
            key: value
            for key, value in attributes.items()
            if isinstance(value, (bool, str, int, float))
        },
    )


@dataclass(frozen=True, slots=True)
class SubagentCoordinatorDependencies:
    profile_registry: Callable[[], Mapping[str, Any]]
    evaluate_profile: Callable[[Any], Any]
    resolve_adapter: Callable[[Any], Any]
    assert_profile_override_supported: Callable[[Any, Any], None]
    resolve_tools: Callable[[Any, Any], tuple[Any, ...]]
    instructions_for_profile: Callable[[Any, Any], str | None]
    trace_frame: Callable[[KediTraceFrame], AbstractContextManager[None]]
    active_stack: ContextVar[tuple[tuple[str, str], ...]]
    default_cwd: Callable[[], Path]


@dataclass(frozen=True, slots=True)
class _OwnedSubagentTask:
    """A child task together with the event loop that owns it."""

    task: asyncio.Task[DelegationResult]
    loop: asyncio.AbstractEventLoop


@dataclass(frozen=True, slots=True)
class SubagentHandle:
    run_id: str
    _coordinator: SubagentCoordinator

    @property
    def subagent(self) -> str:
        return self._coordinator.runs.get(self.run_id).profile_name

    @property
    def status(self) -> SubagentRunStatus:
        return self._coordinator.runs.get(self.run_id).status

    def as_tool_result(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "subagent": self.subagent,
            "status": self.status,
        }

    async def wait(self, *, timeout: float | None = None) -> DelegationResult:
        return await self._coordinator.wait(self.run_id, timeout=timeout)

    def cancel(self) -> bool:
        return self._coordinator.cancel(self.run_id)


class SubagentCoordinator:
    """Coordinate one isolated child run without owning runtime state."""

    def __init__(
        self,
        dependencies: SubagentCoordinatorDependencies,
        *,
        runs: SubagentRunRegistry | None = None,
        max_depth: int = DEFAULT_MAX_SUBAGENT_DEPTH,
        max_concurrency: int = DEFAULT_MAX_SUBAGENT_CONCURRENCY,
        timeout_seconds: float | None = DEFAULT_SUBAGENT_TIMEOUT_SECONDS,
        usage_limits: SubagentUsageLimits | None = None,
        max_retained_runs: int = DEFAULT_MAX_RETAINED_SUBAGENT_RUNS,
        state_path: str | Path | None = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("Subagent max depth must be at least 1")
        self._deps = dependencies
        self.runs = runs or SubagentRunRegistry()
        self.max_depth = min(max_depth, HARD_MAX_SUBAGENT_DEPTH)
        self.max_concurrency = DelegationConcurrency.for_limit(max_concurrency).limit
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("Subagent timeout must be greater than zero")
        self.timeout_seconds = (
            None if timeout_seconds is None else min(timeout_seconds, MAX_SUBAGENT_TIMEOUT_SECONDS)
        )
        self.usage_limits = (usage_limits or SubagentUsageLimits()).capped()
        if max_retained_runs < 1:
            raise ValueError("Subagent run retention must be at least 1")
        self.max_retained_runs = max_retained_runs
        self._coordination_lock = Lock()
        self._task_lock = Lock()
        self._state_lock = Lock()
        self._persistence_lock = Lock()
        self._active_runs = 0
        self._budget_stack: ContextVar[tuple[DelegationBudget, ...]] = ContextVar(
            f"kedi_subagent_budget_stack_{id(self)}",
            default=(),
        )
        self._concurrency: ContextVar[DelegationConcurrency | None] = ContextVar(
            f"kedi_subagent_concurrency_{id(self)}",
            default=None,
        )
        self._safety_ceiling: ContextVar[SubagentSafetyCeiling | None] = ContextVar(
            f"kedi_subagent_safety_ceiling_{id(self)}",
            default=None,
        )
        self.events = SubagentEventLog(max_events=max_retained_runs * 2)
        self._tasks: dict[str, _OwnedSubagentTask] = {}
        self._timeout_handles: dict[str, asyncio.TimerHandle] = {}
        self._cancel_requested: set[str] = set()
        self._results: dict[str, DelegationResult] = {}
        self._conversations: dict[str, StoredConversation] = {}
        self._state_store = SubagentStateStore(state_path) if state_path is not None else None
        self._restore_state()

    def new_concurrency(self) -> DelegationConcurrency:
        return DelegationConcurrency.for_limit(self.max_concurrency)

    @property
    def active_runs(self) -> int:
        with self._coordination_lock:
            return self._active_runs

    async def delegate(
        self,
        *,
        allowed_subagents: tuple[str, ...],
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: DelegationBudget,
        parent_concurrency: DelegationConcurrency,
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> DelegationResult:
        handle = self.start(
            allowed_subagents=allowed_subagents,
            subagent=subagent,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )
        return await self.wait(handle.run_id)

    def start(
        self,
        *,
        allowed_subagents: tuple[str, ...],
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: DelegationBudget,
        parent_concurrency: DelegationConcurrency,
        subagent_definitions: Mapping[str, Any] | None = None,
        previous_run_id: str | None = None,
        conversation_id: str | None = None,
        root_run_id_override: str | None = None,
    ) -> SubagentHandle:
        if subagent not in allowed_subagents:
            raise KediExecutionError(
                f"Subagent profile {subagent!r} is not allowed in the current profile"
            )
        definition = (
            subagent_definitions.get(subagent) if subagent_definitions is not None else None
        )
        if definition is None:
            definition = self._deps.profile_registry().get(subagent)
        if definition is None:
            raise KediExecutionError(f"Unknown subagent profile: {subagent!r}")
        if not task.strip():
            raise ValueError("Subagent task cannot be empty")

        stack = self._deps.active_stack.get()
        active_profiles = tuple(profile_name for _, profile_name in stack)
        budget_stack = self._budget_stack.get()
        parent_ceiling = self._safety_ceiling.get() or SubagentSafetyCeiling.from_parent(
            parent_profile,
            default_cwd=self._deps.default_cwd(),
        )
        if subagent in active_profiles:
            chain = " -> ".join((*active_profiles, subagent))
            raise KediExecutionError(f"Cyclic subagent invocation: {chain}")
        child_depth = len(stack) + 1
        if child_depth > self.max_depth:
            chain = " -> ".join(
                (
                    *(budget.owner_profile for budget in budget_stack),
                    parent_budget.owner_profile,
                    subagent,
                )
            )
            raise KediExecutionError(f"Subagent depth limit {self.max_depth} exceeded: {chain}")

        budgets = (*budget_stack, parent_budget)
        concurrency = self._concurrency.get() or parent_concurrency
        self._reserve_start(budgets, concurrency)
        run_id = new_subagent_run_id()
        parent_run_id = stack[-1][0] if stack else None
        root_run_id = root_run_id_override or (stack[0][0] if stack else run_id)
        conversation_id = conversation_id or run_id
        owner_profile_name = parent_profile.name
        try:
            self._activate_conversation(
                conversation_id=conversation_id,
                profile_name=subagent,
                run_id=run_id,
                previous_run_id=previous_run_id,
            )
            self.runs.create(
                run_id=run_id,
                parent_run_id=parent_run_id,
                root_run_id=root_run_id,
                previous_run_id=previous_run_id,
                conversation_id=conversation_id,
                owner_profile_name=owner_profile_name,
                profile_name=subagent,
                depth=child_depth,
            )
            bind_current_delegation_run(run_id)
            started = self.runs.transition(run_id, "running")
            self.events.append(SubagentStarted(started))
        except BaseException:
            self._deactivate_conversation(conversation_id, run_id)
            self._release_slot(concurrency)
            raise
        try:
            coroutine = self._execute_run(
                run_id=run_id,
                root_run_id=root_run_id,
                parent_run_id=parent_run_id,
                child_depth=child_depth,
                definition=definition,
                subagent=subagent,
                task=task,
                final_schema=final_schema,
                parent_ceiling=parent_ceiling,
                budgets=budgets,
                concurrency=concurrency,
                stack=stack,
                conversation_id=conversation_id,
            )
            task_context = copy_context()
            task_handle = _create_subagent_task(
                coroutine,
                name=f"kedi-subagent-{subagent}-{run_id}",
                context=task_context,
            )
            owner_loop = task_handle.get_loop()
        except BaseException as exc:
            failed = self.runs.transition(run_id, "failed", error=exc)
            self.events.append(SubagentFailed(failed))
            self._deactivate_conversation(conversation_id, run_id)
            self._persist_state()
            self._release_slot(concurrency)
            raise
        with self._task_lock:
            owned_task = _OwnedSubagentTask(
                task=task_handle,
                loop=owner_loop,
            )
            self._tasks[run_id] = owned_task
            if self.timeout_seconds is not None and parent_run_id is None:
                self._timeout_handles[run_id] = owner_loop.call_later(
                    self.timeout_seconds,
                    self._timeout_run,
                    run_id,
                    owned_task,
                    self.timeout_seconds,
                )
        task_handle.add_done_callback(
            lambda completed, completed_run_id=run_id: self._task_finished(
                completed_run_id,
                completed,
            )
        )
        self._persist_state()
        return SubagentHandle(run_id=run_id, _coordinator=self)

    async def continue_run(
        self,
        *,
        run_id: str,
        allowed_subagents: tuple[str, ...],
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: DelegationBudget,
        parent_concurrency: DelegationConcurrency,
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> DelegationResult:
        handle = self.start_continuation(
            run_id=run_id,
            allowed_subagents=allowed_subagents,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
        )
        return await self.wait(handle.run_id)

    def start_continuation(
        self,
        *,
        run_id: str,
        allowed_subagents: tuple[str, ...],
        task: str,
        final_schema: dict[str, Any] | None,
        parent_profile: Any,
        parent_budget: DelegationBudget,
        parent_concurrency: DelegationConcurrency,
        subagent_definitions: Mapping[str, Any] | None = None,
    ) -> SubagentHandle:
        previous = self._continuable_run(
            run_id,
            allowed_subagents=allowed_subagents,
            owner_profile_name=parent_profile.name,
        )
        return self.start(
            allowed_subagents=allowed_subagents,
            subagent=previous.profile_name,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
            subagent_definitions=subagent_definitions,
            previous_run_id=previous.run_id,
            conversation_id=previous.conversation_id,
            root_run_id_override=previous.root_run_id,
        )

    async def wait(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
    ) -> DelegationResult:
        owned_task = self._task_for(run_id, required=False)
        if owned_task is None:
            return self._terminal_result(run_id)
        awaited = self._bridge_task_wait(run_id, owned_task)
        if timeout is None:
            return await asyncio.shield(awaited)
        done, _ = await asyncio.wait((awaited,), timeout=timeout)
        if not done:
            raise TimeoutError
        return awaited.result()

    def cancel(self, run_id: str) -> bool:
        owned_task = self._task_for(run_id, required=False)
        if owned_task is None:
            self.status(run_id)
            return False
        with self._task_lock:
            current = self._tasks.get(run_id)
            if (
                current is not owned_task
                or owned_task.task.done()
                or run_id in self._cancel_requested
            ):
                return False
            self._cancel_requested.add(run_id)
        if owned_task.loop.is_closed():
            raise KediExecutionError(
                f"Subagent {run_id!r} cannot be cancelled because its owner event loop is closed"
            )
        if _running_loop() is owned_task.loop:
            owned_task.task.cancel()
        else:
            # ``call_soon_threadsafe(task.cancel)`` captures the caller's
            # ContextVar context. A foreign thread would then resume the child
            # coroutine with tokens created in a different context, so schedule
            # the callback with the child task's retained creation context.
            completed: Future[None] = Future()
            _schedule_owned_cancellation(owned_task, completed)
            completed.result()
        return True

    def status(self, run_id: str) -> SubagentRunStatus:
        try:
            return self.runs.get(run_id).status
        except KeyError:
            raise KediExecutionError(f"Unknown subagent run ID: {run_id!r}") from None

    def _task_for(
        self,
        run_id: str,
        *,
        required: bool = True,
    ) -> _OwnedSubagentTask | None:
        with self._task_lock:
            task = self._tasks.get(run_id)
        if task is not None or not required:
            return task
        raise KediExecutionError(f"Unknown subagent run ID: {run_id!r}")

    def _task_finished(
        self,
        run_id: str,
        task: asyncio.Task[DelegationResult],
    ) -> None:
        if not task.cancelled():
            task.exception()

        discarded: list[str] = []
        with self._task_lock:
            timeout_handle = self._timeout_handles.pop(run_id, None)
            if timeout_handle is not None:
                timeout_handle.cancel()
            if self._tasks.get(run_id) is None or self._tasks[run_id].task is not task:
                return
            for candidate_id, candidate in tuple(self._tasks.items()):
                if len(self._tasks) <= self.max_retained_runs:
                    break
                if candidate.task.done():
                    self._tasks.pop(candidate_id)
                    discarded.append(candidate_id)
        for candidate_id in discarded:
            conversation_id = self.runs.get(candidate_id).conversation_id
            self.runs.discard(candidate_id)
            with self._state_lock:
                self._results.pop(candidate_id, None)
                if not any(run.conversation_id == conversation_id for run in self.runs.snapshot()):
                    self._conversations.pop(conversation_id, None)
            self._persist_state()

    async def _wait_for_task(
        self,
        run_id: str,
        task: asyncio.Task[DelegationResult],
    ) -> DelegationResult:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            run = self.runs.get(run_id)
            if run.status != "timed_out":
                raise
            message = run.error_message or f"Subagent {run.profile_name!r} timed out"
            raise KediExecutionError(message) from TimeoutError()

    def _timeout_run(
        self,
        run_id: str,
        task: _OwnedSubagentTask | asyncio.Task[DelegationResult],
        timeout: float,
    ) -> None:
        owned_task = self._task_for(run_id, required=False)
        if owned_task is None or owned_task.task is not _unwrap_task(task):
            return
        if owned_task.task.done():
            return
        self._mark_timed_out(run_id, owned_task, timeout)

    def _mark_timed_out(
        self,
        run_id: str,
        task: _OwnedSubagentTask | asyncio.Task[DelegationResult],
        timeout: float,
    ) -> None:
        owned_task = self._task_for(run_id, required=False)
        if owned_task is None or owned_task.task is not _unwrap_task(task):
            return
        if _running_loop() is not owned_task.loop:
            if owned_task.loop.is_closed():
                return
            owned_task.loop.call_soon_threadsafe(
                self._mark_timed_out,
                run_id,
                owned_task,
                timeout,
            )
            return
        if owned_task.task.done() or self.runs.get(run_id).status != "running":
            return
        timeout_error = TimeoutError(
            f"Subagent {self.runs.get(run_id).profile_name!r} timed out after {timeout:g} seconds"
        )
        timed_out = self.runs.transition(run_id, "timed_out", error=timeout_error)
        self.events.append(SubagentFailed(timed_out))
        with self._task_lock:
            self._cancel_requested.add(run_id)
        _schedule_owned_cancellation(owned_task)

    def _bridge_task_wait(
        self,
        run_id: str,
        owned_task: _OwnedSubagentTask,
    ) -> asyncio.Future[DelegationResult] | asyncio.Task[DelegationResult]:
        if _running_loop() is owned_task.loop:
            return asyncio.create_task(self._wait_for_task(run_id, owned_task.task))
        if owned_task.loop.is_closed():
            raise KediExecutionError(
                f"Subagent {run_id!r} owner event loop is closed before completion"
            )
        bridge = asyncio.run_coroutine_threadsafe(
            self._wait_for_task(run_id, owned_task.task),
            owned_task.loop,
        )
        return asyncio.wrap_future(bridge)

    async def _execute_run(
        self,
        *,
        run_id: str,
        root_run_id: str,
        parent_run_id: str | None,
        child_depth: int,
        definition: Any,
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        parent_ceiling: SubagentSafetyCeiling,
        budgets: tuple[DelegationBudget, ...],
        concurrency: DelegationConcurrency,
        stack: tuple[tuple[str, str], ...],
        conversation_id: str,
    ) -> DelegationResult:
        active_agent_run = current_agent_run_identity()
        event_parent_run_id = parent_run_id or (
            active_agent_run.run_id if active_agent_run is not None else None
        )
        token = self._deps.active_stack.set((*stack, (run_id, subagent)))
        budget_token = self._budget_stack.set(budgets)
        concurrency_token = self._concurrency.set(concurrency)
        safety_token = None
        with _subagent_run_span(
            {
                "kedi.subagent.run_id": run_id,
                "kedi.subagent.root_run_id": root_run_id,
                "kedi.subagent.profile": subagent,
                "kedi.subagent.depth": child_depth,
                "kedi.subagent.structured_output": final_schema is not None,
            },
        ) as span:
            if parent_run_id is not None:
                span.set_attribute("kedi.subagent.parent_run_id", parent_run_id)
            try:
                child_profile = self._deps.evaluate_profile(definition)
                child_profile, child_ceiling = parent_ceiling.constrain_child(child_profile)
                final_schema = effective_final_schema(child_profile, final_schema)
                span.set_attribute(
                    "kedi.subagent.structured_output",
                    final_schema is not None,
                )
                child_adapter = self._deps.resolve_adapter(child_profile)
                if not isinstance(child_adapter, SupportsSubagentExecution):
                    raise KediExecutionError(
                        f"Adapter {getattr(child_adapter, 'shortname', type(child_adapter).__name__)!r} "
                        "does not implement Kedi child execution"
                    )
                model = child_profile.model or child_adapter.telemetry_model_name()
                self.runs.describe(
                    run_id,
                    adapter_shortname=child_adapter.shortname,
                    model=model,
                    structured_output=final_schema is not None,
                )
                span.set_attribute("kedi.subagent.adapter", child_adapter.shortname)
                if model is not None:
                    span.set_attribute("gen_ai.request.model", model)
                self._deps.assert_profile_override_supported(child_profile, child_adapter)
                child_tools = self._deps.resolve_tools(child_profile, child_adapter)
                instructions = self._deps.instructions_for_profile(child_profile, child_adapter)
                conversation = self._conversation_context(conversation_id)
                child_call = self._run_child(
                    run_id=run_id,
                    subagent=subagent,
                    task=task,
                    final_schema=final_schema,
                    adapter=child_adapter,
                    profile=child_profile,
                    tools=child_tools,
                    instructions=instructions,
                    approval_ceiling=parent_ceiling.approval_policies,
                    conversation=conversation,
                )
                safety_token = self._safety_ceiling.set(child_ceiling)
                with bind_agent_run(run_id, parent_run_id=event_parent_run_id):
                    with observe_invocations(
                        lambda usage: self._record_and_assert_usage(run_id, span, usage)
                    ):
                        result = await child_call
                self._assert_usage_limits(run_id)
                if self.runs.get(run_id).status != "running":
                    raise asyncio.CancelledError
                self._record_success(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    task=task,
                    result=result,
                    adapter_state=conversation.result_state,
                )
                completed = self.runs.transition(run_id, "completed")
                self.events.append(SubagentCompleted(completed))
                self._persist_state()
                span.set_attribute("kedi.subagent.status", "completed")
                return result
            except asyncio.CancelledError as exc:
                if self.runs.get(run_id).status == "running":
                    cancelled = self.runs.transition(run_id, "cancelled", error=exc)
                    self.events.append(SubagentCancelled(cancelled))
                    span.set_attribute("kedi.subagent.status", "cancelled")
                else:
                    span.set_attribute("kedi.subagent.status", "timed_out")
                self._deactivate_conversation(conversation_id, run_id)
                self._persist_state()
                raise
            except BaseException as exc:
                current = self.runs.get(run_id)
                if current.status == "running":
                    failed = self.runs.transition(run_id, "failed", error=exc)
                    self.events.append(SubagentFailed(failed))
                    span.set_attribute("kedi.subagent.status", "failed")
                else:
                    span.set_attribute("kedi.subagent.status", current.status)
                self._deactivate_conversation(conversation_id, run_id)
                self._persist_state()
                raise
            finally:
                run = self.runs.get(run_id)
                assert run.started_at is not None
                assert run.finished_at is not None
                span.set_attribute(
                    "kedi.subagent.duration_ms",
                    (run.finished_at - run.started_at) * 1000,
                )
                if safety_token is not None:
                    self._safety_ceiling.reset(safety_token)
                self._release_slot(concurrency)
                self._concurrency.reset(concurrency_token)
                self._budget_stack.reset(budget_token)
                self._deps.active_stack.reset(token)

    def _record_usage(
        self,
        run_id: str,
        span: TelemetrySpan,
        usage: InvocationUsage,
    ) -> None:
        run = self.runs.add_usage(
            run_id,
            requests=usage.requests,
            tool_calls=usage.tool_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )
        span.set_attribute("gen_ai.usage.input_tokens", run.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", run.output_tokens)
        span.set_attribute("kedi.subagent.requests", run.requests)
        span.set_attribute("kedi.subagent.tool_calls", run.tool_calls)

    def _record_and_assert_usage(
        self,
        run_id: str,
        span: TelemetrySpan,
        usage: InvocationUsage,
    ) -> None:
        self._record_usage(run_id, span, usage)
        self._assert_usage_limits(run_id)

    def _assert_usage_limits(self, run_id: str) -> None:
        run = self.runs.get(run_id)
        checks = (
            ("request", run.requests, self.usage_limits.request_limit),
            ("tool call", run.tool_calls, self.usage_limits.tool_calls_limit),
            ("input token", run.input_tokens, self.usage_limits.input_tokens_limit),
            ("output token", run.output_tokens, self.usage_limits.output_tokens_limit),
            ("total token", run.total_tokens, self.usage_limits.total_tokens_limit),
        )
        for label, actual, limit in checks:
            if limit is not None and actual > limit:
                raise KediExecutionError(
                    f"Subagent {label} limit {limit} exceeded: observed {actual}"
                )

    def _reserve_start(
        self,
        budgets: tuple[DelegationBudget, ...],
        concurrency: DelegationConcurrency,
    ) -> None:
        unique = tuple({id(budget): budget for budget in budgets}.values())
        with self._coordination_lock:
            exhausted = next(
                (budget for budget in unique if budget.started >= budget.limit),
                None,
            )
            if exhausted is not None:
                raise SubagentBudgetExceeded(
                    f"Subagent invocation limit {exhausted.limit} reached for profile "
                    f"{exhausted.owner_profile!r}"
                )
            if concurrency.active >= concurrency.limit or self._active_runs >= self.max_concurrency:
                raise SubagentConcurrencyExceeded(
                    f"Subagent concurrency limit {concurrency.limit} reached"
                )
            for budget in unique:
                budget.started += 1
            concurrency.active += 1
            self._active_runs += 1

    def _release_slot(self, concurrency: DelegationConcurrency) -> None:
        with self._coordination_lock:
            if concurrency.active < 1 or self._active_runs < 1:
                raise RuntimeError("Subagent concurrency slot underflow")
            concurrency.active -= 1
            self._active_runs -= 1

    async def _run_child(
        self,
        *,
        run_id: str,
        subagent: str,
        task: str,
        final_schema: dict[str, Any] | None,
        adapter: SupportsSubagentExecution,
        profile: Any,
        tools: tuple[Any, ...],
        instructions: str | None,
        approval_ceiling: tuple[Any, ...],
        conversation: SubagentConversationContext,
    ) -> DelegationResult:
        run_kwargs = dict(adapter.subagent_run_kwargs(self.usage_limits))
        if instructions is not None:
            run_kwargs["instructions"] = instructions

        request = SubagentExecutionRequest(
            profile=profile,
            tools=tools,
            instructions=instructions,
            approval_ceiling=approval_ceiling,
            usage_limits=self.usage_limits,
            conversation=conversation,
        )
        effective_task = task
        if conversation.turns and not bool(
            getattr(adapter, "supports_native_subagent_resume", False)
        ):
            effective_task = _task_with_verified_history(conversation.turns, task)
        with self._deps.trace_frame(KediTraceFrame(kind="subagent", label=f"delegate {subagent}")):
            with adapter.subagent_scope(request):
                if final_schema is None:
                    task_summary = await adapter.invoke(prompt=effective_task, **run_kwargs)
                    return DelegationResult(
                        run_id=run_id,
                        subagent=subagent,
                        task_summary=task_summary,
                    )

                schema = wrapped_final_schema(final_schema)
                produced = await adapter.produce_subagent(
                    template=effective_task,
                    output_schema=schema,
                    **run_kwargs,
                )
                result = validate_structured_result(produced, schema)
                return DelegationResult(
                    run_id=run_id,
                    subagent=subagent,
                    task_summary=result["task_summary"],
                    final_result=result["final_result"],
                )

    def _activate_conversation(
        self,
        *,
        conversation_id: str,
        profile_name: str,
        run_id: str,
        previous_run_id: str | None,
    ) -> None:
        with self._state_lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                if previous_run_id is not None:
                    raise KediExecutionError(f"Unknown subagent conversation: {conversation_id!r}")
                self._conversations[conversation_id] = StoredConversation(
                    conversation_id=conversation_id,
                    profile_name=profile_name,
                    active_run_id=run_id,
                    turns=[],
                    adapter_state={},
                )
                return
            if conversation.profile_name != profile_name:
                raise KediExecutionError("A subagent conversation cannot change child profiles")
            if conversation.active_run_id is not None:
                raise KediExecutionError(
                    f"Subagent conversation {conversation_id!r} already has an active run"
                )
            if not conversation.turns or conversation.turns[-1].run_id != previous_run_id:
                raise KediExecutionError("Only the latest completed subagent run can be continued")
            if len(conversation.turns) >= MAX_SUBAGENT_CONVERSATION_TURNS:
                raise KediExecutionError(
                    f"Subagent conversation turn limit {MAX_SUBAGENT_CONVERSATION_TURNS} reached"
                )
            conversation.active_run_id = run_id

    def _deactivate_conversation(self, conversation_id: str, run_id: str) -> None:
        with self._state_lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is not None and conversation.active_run_id == run_id:
                conversation.active_run_id = None

    def _conversation_context(self, conversation_id: str) -> SubagentConversationContext:
        with self._state_lock:
            conversation = self._conversations[conversation_id]
            return SubagentConversationContext(
                conversation_id=conversation_id,
                turns=tuple(conversation.turns),
                resume_state=dict(conversation.adapter_state),
            )

    def _record_success(
        self,
        *,
        conversation_id: str,
        run_id: str,
        task: str,
        result: DelegationResult,
        adapter_state: dict[str, Any],
    ) -> None:
        with self._state_lock:
            conversation = self._conversations[conversation_id]
            conversation.turns.append(
                SubagentTurn(
                    run_id=run_id,
                    task=task,
                    task_summary=result.task_summary,
                    final_result=result.final_result,
                )
            )
            conversation.active_run_id = None
            conversation.adapter_state = dict(adapter_state)
            self._results[run_id] = result

    def _continuable_run(
        self,
        run_id: str,
        *,
        allowed_subagents: tuple[str, ...],
        owner_profile_name: str,
    ) -> Any:
        try:
            run = self.runs.get(run_id)
        except KeyError:
            raise KediExecutionError(f"Unknown subagent run ID: {run_id!r}") from None
        if run.status != "completed":
            raise KediExecutionError("Only completed subagent runs can be continued")
        if (
            run.profile_name not in allowed_subagents
            or run.owner_profile_name != owner_profile_name
        ):
            raise KediExecutionError("The current profile does not own this subagent conversation")
        with self._state_lock:
            conversation = self._conversations.get(run.conversation_id)
            if conversation is None:
                raise KediExecutionError(f"Unknown subagent conversation: {run.conversation_id!r}")
            if conversation.active_run_id is not None:
                raise KediExecutionError("The subagent conversation is already running")
            if not conversation.turns or conversation.turns[-1].run_id != run_id:
                raise KediExecutionError("Only the latest completed subagent run can be continued")
        return run

    def _terminal_result(self, run_id: str) -> DelegationResult:
        try:
            run = self.runs.get(run_id)
        except KeyError:
            raise KediExecutionError(f"Unknown subagent run ID: {run_id!r}") from None
        if run.status == "completed":
            with self._state_lock:
                result = self._results.get(run_id)
            if result is None:
                raise KediExecutionError(f"Completed subagent run {run_id!r} has no result")
            return result
        if run.status in {"failed", "cancelled", "timed_out"}:
            message = run.error_message or f"Subagent {run.profile_name!r} {run.status}"
            raise KediExecutionError(message)
        raise KediExecutionError(f"Subagent run {run_id!r} is still {run.status}")

    def _restore_state(self) -> None:
        if self._state_store is None:
            return
        loaded = self._state_store.load()
        for run in loaded.runs:
            self.runs.restore(run)
        self._results.update(loaded.results)
        self._conversations.update(loaded.conversations)
        for conversation in self._conversations.values():
            conversation.active_run_id = None
        if any(run.error_type == "InterruptedError" for run in loaded.runs):
            self._persist_state()

    def _persist_state(self) -> None:
        if self._state_store is None:
            return
        with self._persistence_lock:
            with self._state_lock:
                results = dict(self._results)
                conversations = {
                    key: StoredConversation(
                        conversation_id=value.conversation_id,
                        profile_name=value.profile_name,
                        active_run_id=value.active_run_id,
                        turns=list(value.turns),
                        adapter_state=dict(value.adapter_state),
                    )
                    for key, value in self._conversations.items()
                }
            self._state_store.save(
                runs=self.runs.snapshot(),
                results=results,
                conversations=conversations,
            )


def _task_with_verified_history(turns: tuple[SubagentTurn, ...], task: str) -> str:
    history = [
        {
            "task": turn.task,
            "task_summary": turn.task_summary,
            "final_result": turn.final_result,
        }
        for turn in turns
    ]
    return (
        "Continue the same child task using this verified prior conversation state:\n"
        f"{json.dumps(history, ensure_ascii=False)}\n\n"
        f"New task:\n{task}"
    )


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _unwrap_task(
    owned_task: _OwnedSubagentTask | asyncio.Task[DelegationResult],
) -> asyncio.Task[DelegationResult]:
    if isinstance(owned_task, _OwnedSubagentTask):
        return owned_task.task
    return owned_task


def _create_subagent_task(
    coroutine: Any,
    *,
    name: str,
    context: Context,
) -> asyncio.Task[DelegationResult]:
    """Create a task in an explicitly retained ContextVar context."""

    # Creating inside the copied context works on every supported Python
    # version, including 3.10 where ``asyncio.create_task(context=...)`` is
    # unavailable. The task itself remains owned by the current event loop.
    return cast(
        asyncio.Task[DelegationResult],
        context.run(asyncio.create_task, coroutine, name=name),
    )


def _schedule_owned_cancellation(
    owned_task: _OwnedSubagentTask,
    completed: Future[None] | None = None,
) -> None:
    """Request cancellation in the child task's owner loop."""

    def cancel() -> None:
        try:
            owned_task.task.cancel()
        except BaseException as exc:
            if completed is not None and not completed.done():
                completed.set_exception(exc)
            return
        if completed is not None and not completed.done():
            completed.set_result(None)

    owned_task.loop.call_soon_threadsafe(cancel)
