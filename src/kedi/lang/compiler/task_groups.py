"""Completion-driven task groups with lexical, joined processing scopes."""

from __future__ import annotations

from concurrent.futures import Future
from collections import deque
from functools import partial
from threading import Lock
from typing import Any

from kedi.agent_adapter.subagents.groups import (
    ProcessingRegion,
    check_process_cancelled,
    register_process_child,
    register_process_future,
)
from kedi.agent_adapter.subagents.native import SubagentResult, SubagentTask
from kedi.agent_profile import AgentProfile
from kedi.core.runtime import KediRuntime
from kedi.core.scopes import ProcessingScope
from kedi.errors import KediExecutionError, KediTraceFrame
from kedi.lang.ast import ProcedureDef, TaskGroupArm, TaskGroupStmt
from kedi.run_budget import RunBudget, run_budget_scope

from .control_flow import active_scope
from .environment import BindingMetadata, ScopeFrame
from .evaluation import lookup_value


class _Completions:
    """One dispatcher, many completion callbacks, without blocking engine workers."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._queued: deque[tuple[int, bool, Future[Any]]] = deque()
        self._waiter: Future[tuple[int, bool, Future[Any]]] | None = None

    def put(self, index: int, processing: bool, future: Future[Any]) -> None:
        event = (index, processing, future)
        with self._lock:
            waiter, self._waiter = self._waiter, None
            if waiter is None:
                self._queued.append(event)
        if waiter is not None:
            waiter.set_result(event)

    def next(self) -> Future[tuple[int, bool, Future[Any]]]:
        waiter: Future[tuple[int, bool, Future[Any]]] = Future()
        with self._lock:
            if self._queued:
                waiter.set_result(self._queued.popleft())
            else:
                self._waiter = waiter
        return waiter


def execute_task_group(
    rt: KediRuntime, stmt: TaskGroupStmt, env: dict[str, Any], active: AgentProfile
) -> None:
    owner = rt._native_subagent_owner.get()
    if owner is None:
        raise RuntimeError("Task groups require an active Kedi invocation")
    jobs: list[SubagentTask] = []
    for arm in stmt.arms:
        job = lookup_value(rt, arm.wait.handle, env)
        if not isinstance(job, SubagentTask):
            raise TypeError(f"> await expects a subagent task, got {type(job).__name__}")
        if job._owner is not owner:
            raise KediExecutionError("Subagent task belongs to another invocation")
        owner._check_owner()
        jobs.append(job)

    region = ProcessingRegion(owner)
    for job in jobs:
        region.add_child(job)
    finished: Future[None] = Future()
    owner.register_group(finished, region.cancel)
    primary_error: BaseException | None = None
    events = _Completions()
    pending = 0

    try:
        with (
            region.activate(),
            rt.trace_frame(KediTraceFrame(kind="task_group", label="task group", span=stmt.loc)),
        ):
            # Register every waiter before admitting any processing work.
            for index, job in enumerate(jobs):
                register_process_child(job)
                future = owner.submit(job.wait)
                future.add_done_callback(partial(events.put, index, False))
                pending += 1
            profile = active.frozen_copy()
            while pending:
                index, processing, future = rt._engine.wait(events.next())
                pending -= 1
                try:
                    result = future.result()
                    if not processing and primary_error is None:
                        check_process_cancelled()
                        arm = stmt.arms[index]
                        if arm.body is not None:
                            process = rt._engine.schedule_control(
                                partial(_execute_process, rt, arm, env, profile, result)
                            )
                            register_process_future(process)
                            process.add_done_callback(partial(events.put, index, True))
                            pending += 1
                except BaseException as exc:
                    if primary_error is None:
                        primary_error = exc
                        region.cancel()
            if primary_error is not None:
                raise primary_error
    except BaseException as exc:
        primary_error = exc
        region.cancel()
        # Also join registered work if registration or scheduling itself failed.
        while pending:
            _, _, future = rt._engine.wait(events.next())
            pending -= 1
            try:
                future.result()
            except BaseException:
                pass
        raise
    finally:
        try:
            region.drain(primary_error)
        finally:
            finished.set_result(None)
            owner.release_group(finished)


def _execute_process(
    rt: KediRuntime,
    arm: TaskGroupArm,
    env: dict[str, Any],
    profile: AgentProfile,
    result: SubagentResult,
) -> None:
    from .statement_compiler import StatementExecutor

    owner = rt._native_subagent_owner.get()
    assert owner is not None
    region = ProcessingRegion(owner)
    frame = ScopeFrame(parent=env)
    if arm.wait.binder is not None:
        frame.declare(arm.wait.binder, result, BindingMetadata(source=arm.wait.loc))
    scope = ProcessingScope(dict(rt._procedures), [dict(types) for types in rt._type_env_stack])
    type_scope = dict(scope.types[-1]) if scope.types else {}
    scope.types.append(type_scope)
    token = rt._processing_scope.set(scope)
    primary_error: BaseException | None = None
    try:
        with region.activate(), active_scope(rt, frame), run_budget_scope(RunBudget()):
            with rt.trace_frame(
                KediTraceFrame(kind="process", label=arm.wait.handle, span=arm.process_loc)
            ):
                try:
                    # Keep declarations alive until deferred work has joined, not merely
                    # until the last source statement has scheduled its work.
                    StatementExecutor(rt)._execute_block(
                        arm.body or [],
                        ProcedureDef(name="<process>", params=[], body=arm.body or []),
                        frame,
                        type_scope,
                        {},
                        [],
                        profile,
                        None,
                        rt._profile_registry,
                    )
                except BaseException as exc:
                    primary_error = exc
                    region.cancel()
                    raise
                finally:
                    try:
                        rt._drain_outermost(primary_error)
                    except BaseException as exc:
                        primary_error = exc
                        region.cancel()
                        raise
                    finally:
                        region.drain(primary_error)
    finally:
        rt._processing_scope.reset(token)
