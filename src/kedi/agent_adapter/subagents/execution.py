from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from kedi.agent_adapter.events import bind_agent_run, current_agent_run_identity
from kedi.agent_adapter.invocation import observe_invocations
from kedi.agent_adapter.subagents import (
    DelegationResult,
    SubagentConversationContext,
    SubagentExecutionRequest,
    SubagentTurn,
    SupportsSubagentExecution,
    effective_final_schema,
    validate_structured_result,
    wrapped_final_schema,
)
from kedi.agent_adapter.subagents.budget import DelegationBudget, DelegationConcurrency
from kedi.agent_adapter.subagents.context import SubagentExecutionContext, _subagent_execution_scope
from kedi.agent_adapter.subagents.runs import SubagentCancelled, SubagentCompleted, SubagentFailed
from kedi.agent_adapter.subagents.safety import SubagentSafetyCeiling
from kedi.agent_adapter.telemetry import current_tool_call_name
from kedi.errors import KediExecutionError, KediTraceFrame
from kedi.telemetry import TelemetrySpan, current_span
from kedi.telemetry import span as telemetry_span

if TYPE_CHECKING:
    from kedi.agent_adapter.subagents.coordinator import SubagentCoordinator


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


async def _execute_run(
    self: SubagentCoordinator,
    *,
    run_id: str,
    root_run_id: str,
    parent_run_id: str | None,
    child_depth: int,
    definition: Any,
    subagent: str,
    task: str,
    final_schema: dict[str, Any] | None,
    parent_profile: Any,
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
            enforcement = getattr(parent_profile, "_hook_enforcement", None)
            if enforcement is not None and enforcement.active:
                child_hooks = getattr(child_profile, "hooks", None)
                child_profile = replace(
                    child_profile,
                    hooks=enforcement.compose_enforced(child_hooks),
                    _hook_enforcement=enforcement,
                )
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
            with bind_agent_run(
                run_id,
                parent_run_id=event_parent_run_id,
                agent_name=subagent,
            ):
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


async def _run_child(
    self: SubagentCoordinator,
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
    if conversation.turns and not bool(getattr(adapter, "supports_native_subagent_resume", False)):
        effective_task = _task_with_verified_history(conversation.turns, task)
    execution_context = SubagentExecutionContext(
        run_id=run_id,
        profile=profile,
        conversation_id=conversation.conversation_id,
    )
    with (
        self._deps.trace_frame(KediTraceFrame(kind="subagent", label=f"delegate {subagent}")),
        _subagent_execution_scope(execution_context),
        adapter.subagent_scope(request),
    ):
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


_DELEGATION_TOOL_NAMES = frozenset({"delegate_task", "continue_subagent"})
