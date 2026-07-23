from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from opentelemetry import trace

from ...errors import KediExecutionError, KediTraceFrame
from ..invocation import InvocationUsage, observe_invocations
from . import (
    DelegationResult,
    SubagentExecutionRequest,
    SupportsSubagentExecution,
    new_subagent_run_id,
    validate_structured_result,
    wrapped_final_schema,
)
from .attempts import bind_current_delegation_run
from .budget import DelegationBudget, DelegationConcurrency
from .limits import SubagentUsageLimits
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


@dataclass(frozen=True, slots=True)
class SubagentCoordinatorDependencies:
    profile_registry: Callable[[], Mapping[str, Any]]
    evaluate_profile: Callable[[Any], Any]
    resolve_adapter: Callable[[Any], Any]
    assert_profile_override_supported: Callable[[Any, Any], None]
    resolve_tools: Callable[[Any, Any], tuple[Any, ...]]
    instructions_for_profile: Callable[[Any], str | None]
    trace_frame: Callable[[KediTraceFrame], AbstractContextManager[None]]
    active_stack: ContextVar[tuple[tuple[str, str], ...]]
    default_cwd: Callable[[], Path]


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
        self._tracer = trace.get_tracer("kedi.subagents")
        self.events = SubagentEventLog(max_events=max_retained_runs * 2)
        self._tasks: dict[str, asyncio.Task[DelegationResult]] = {}
        self._timeout_handles: dict[str, asyncio.TimerHandle] = {}
        self._cancel_requested: set[str] = set()

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
    ) -> DelegationResult:
        handle = self.start(
            allowed_subagents=allowed_subagents,
            subagent=subagent,
            task=task,
            final_schema=final_schema,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
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
    ) -> SubagentHandle:
        if subagent not in allowed_subagents:
            raise KediExecutionError(
                f"Subagent profile {subagent!r} is not allowed in the current profile"
            )
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
        root_run_id = stack[0][0] if stack else run_id
        self.runs.create(
            run_id=run_id,
            parent_run_id=parent_run_id,
            root_run_id=root_run_id,
            profile_name=subagent,
            depth=child_depth,
        )
        bind_current_delegation_run(run_id)
        started = self.runs.transition(run_id, "running")
        self.events.append(SubagentStarted(started))
        try:
            task_handle = asyncio.create_task(
                self._execute_run(
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
                ),
                name=f"kedi-subagent-{subagent}-{run_id}",
            )
        except BaseException as exc:
            failed = self.runs.transition(run_id, "failed", error=exc)
            self.events.append(SubagentFailed(failed))
            self._release_slot(concurrency)
            raise
        with self._task_lock:
            self._tasks[run_id] = task_handle
            if self.timeout_seconds is not None and parent_run_id is None:
                self._timeout_handles[run_id] = asyncio.get_running_loop().call_later(
                    self.timeout_seconds,
                    self._timeout_run,
                    run_id,
                    task_handle,
                    self.timeout_seconds,
                )
        task_handle.add_done_callback(
            lambda completed, completed_run_id=run_id: self._task_finished(
                completed_run_id,
                completed,
            )
        )
        return SubagentHandle(run_id=run_id, _coordinator=self)

    async def wait(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
    ) -> DelegationResult:
        task = self._task_for(run_id)
        if timeout is None:
            return await self._wait_for_task(run_id, task)

        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            self.cancel(run_id)
            with suppress(asyncio.CancelledError):
                await task
            raise
        if done:
            return await self._wait_for_task(run_id, task)

        self._mark_timed_out(run_id, task, timeout)
        with suppress(asyncio.CancelledError):
            await task
        return await self._wait_for_task(run_id, task)

    def cancel(self, run_id: str) -> bool:
        task = self._task_for(run_id)
        if task.done() or run_id in self._cancel_requested:
            return False
        self._cancel_requested.add(run_id)
        task.cancel()
        return True

    def status(self, run_id: str) -> SubagentRunStatus:
        try:
            return self.runs.get(run_id).status
        except KeyError:
            raise KediExecutionError(f"Unknown subagent run ID: {run_id!r}") from None

    def _task_for(self, run_id: str) -> asyncio.Task[DelegationResult]:
        with self._task_lock:
            try:
                return self._tasks[run_id]
            except KeyError:
                raise KediExecutionError(f"Unknown subagent run ID: {run_id!r}") from None

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
            if self._tasks.get(run_id) is not task:
                return
            for candidate_id, candidate in tuple(self._tasks.items()):
                if len(self._tasks) <= self.max_retained_runs:
                    break
                if candidate.done():
                    self._tasks.pop(candidate_id)
                    discarded.append(candidate_id)
        for candidate_id in discarded:
            self.runs.discard(candidate_id)

    async def _wait_for_task(
        self,
        run_id: str,
        task: asyncio.Task[DelegationResult],
    ) -> DelegationResult:
        try:
            return await task
        except asyncio.CancelledError:
            run = self.runs.get(run_id)
            if run.status != "timed_out":
                raise
            message = run.error_message or f"Subagent {run.profile_name!r} timed out"
            raise KediExecutionError(message) from TimeoutError()

    def _timeout_run(
        self,
        run_id: str,
        task: asyncio.Task[DelegationResult],
        timeout: float,
    ) -> None:
        if task.done():
            return
        self._mark_timed_out(run_id, task, timeout)

    def _mark_timed_out(
        self,
        run_id: str,
        task: asyncio.Task[DelegationResult],
        timeout: float,
    ) -> None:
        if task.done() or self.runs.get(run_id).status != "running":
            return
        timeout_error = TimeoutError(
            f"Subagent {self.runs.get(run_id).profile_name!r} timed out after {timeout:g} seconds"
        )
        timed_out = self.runs.transition(run_id, "timed_out", error=timeout_error)
        self.events.append(SubagentFailed(timed_out))
        self._cancel_requested.add(run_id)
        task.cancel()

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
    ) -> DelegationResult:
        token = self._deps.active_stack.set((*stack, (run_id, subagent)))
        budget_token = self._budget_stack.set(budgets)
        concurrency_token = self._concurrency.set(concurrency)
        safety_token = None
        with self._tracer.start_as_current_span(
            "run subagent",
            attributes={
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
                instructions = self._deps.instructions_for_profile(child_profile)
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
                )
                safety_token = self._safety_ceiling.set(child_ceiling)
                with observe_invocations(
                    lambda usage: self._record_and_assert_usage(run_id, span, usage)
                ):
                    result = await child_call
                self._assert_usage_limits(run_id)
                if self.runs.get(run_id).status != "running":
                    raise asyncio.CancelledError
                completed = self.runs.transition(run_id, "completed")
                self.events.append(SubagentCompleted(completed))
                span.set_attribute("kedi.subagent.status", "completed")
                return result
            except asyncio.CancelledError as exc:
                if self.runs.get(run_id).status == "running":
                    cancelled = self.runs.transition(run_id, "cancelled", error=exc)
                    self.events.append(SubagentCancelled(cancelled))
                    span.set_attribute("kedi.subagent.status", "cancelled")
                else:
                    span.set_attribute("kedi.subagent.status", "timed_out")
                raise
            except BaseException as exc:
                current = self.runs.get(run_id)
                if current.status == "running":
                    failed = self.runs.transition(run_id, "failed", error=exc)
                    self.events.append(SubagentFailed(failed))
                    span.set_attribute("kedi.subagent.status", "failed")
                else:
                    span.set_attribute("kedi.subagent.status", current.status)
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
        span: trace.Span,
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
        span: trace.Span,
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
                raise KediExecutionError(
                    f"Subagent invocation limit {exhausted.limit} reached for profile "
                    f"{exhausted.owner_profile!r}"
                )
            if concurrency.active >= concurrency.limit or self._active_runs >= self.max_concurrency:
                raise KediExecutionError(f"Subagent concurrency limit {concurrency.limit} reached")
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
        )
        with self._deps.trace_frame(KediTraceFrame(kind="subagent", label=f"delegate {subagent}")):
            with adapter.subagent_scope(request):
                if final_schema is None:
                    task_summary = await adapter.invoke(prompt=task, **run_kwargs)
                    return DelegationResult(
                        run_id=run_id,
                        subagent=subagent,
                        task_summary=task_summary,
                    )

                schema = wrapped_final_schema(final_schema)
                produced = await adapter.produce_subagent(
                    template=task,
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
