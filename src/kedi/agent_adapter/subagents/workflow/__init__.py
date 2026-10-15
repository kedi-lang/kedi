from __future__ import annotations

import asyncio
import json
import keyword
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic_core import PydanticSerializationError, to_jsonable_python
from pydantic_monty import (
    CollectString,
    Monty,
    MontyCrashedError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
)

from kedi.agent_profile.capabilities import capability_subagent_policy

from ....telemetry import TelemetrySpan, current_span
from ....telemetry import span as telemetry_span
from ...telemetry import current_tool_call_name
from ...hooks.tool_wrapper import with_hook_lifecycle
from ...tools.specs import ToolSpec
from .. import validate_final_schema
from ..attempts import attempt_status_from_run, track_delegation_attempt
from ..budget import (
    DelegationBudget,
    DelegationConcurrency,
    SubagentBudgetExceeded,
)
from ..runs import safe_subagent_error_message
from .executor import WorkflowAbortError, WorkflowExecutor

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile
    from kedi.core import KediRuntime


MAX_WORKFLOW_PRINT_BYTES = 16 * 1024
MAX_WORKFLOW_RESULT_PREVIEW_CHARS = 300
MAX_WORKFLOW_SALVAGE_RESULTS = 20
WORKFLOW_MEMORY_LIMIT = 256 * 1024 * 1024
WORKFLOW_CPU_SECONDS = 5.0

_active_workflow: ContextVar[bool] = ContextVar("kedi_active_dynamic_workflow", default=False)


class WorkflowError(RuntimeError):
    """Base error for retryable dynamic workflow failures."""


class WorkflowTypeError(WorkflowError):
    pass


class WorkflowSyntaxError(WorkflowError):
    pass


class WorkflowRuntimeError(WorkflowError):
    pass


class WorkflowResourceError(WorkflowError):
    pass


@dataclass(frozen=True, slots=True)
class _CompletedCall:
    subagent: str
    task: str
    result: dict[str, Any]


@dataclass(slots=True)
class _WorkflowStats:
    active_calls: int = 0
    concurrency_peak: int = 0
    failed_calls: int = 0
    cancelled_calls: int = 0
    salvage_hits: int = 0
    deduplicated_calls: int = 0


def _dynamic_workflow_span(
    attributes: Mapping[str, Any],
) -> AbstractContextManager[TelemetrySpan]:
    current = current_span()
    tool_name = current_tool_call_name()
    profile = attributes.get("kedi.workflow.parent_profile")
    workflow_name = (
        f"{profile} workflow" if isinstance(profile, str) and profile else "dynamic workflow"
    )
    if current.is_recording() and tool_name == "run_workflow":
        current.update_name(workflow_name)
        current.set_attribute("kedi.operation.name", "run_workflow")
        current.set_attribute("gen_ai.operation.name", "execute_tool")
        for key, value in attributes.items():
            if isinstance(value, (bool, str, int, float)):
                current.set_attribute(key, value)
        return nullcontext(current)
    return telemetry_span(
        "agent",
        workflow_name,
        "run_workflow",
        attributes={
            key: value
            for key, value in attributes.items()
            if isinstance(value, (bool, str, int, float))
        },
    )


def dynamic_workflow_tool_spec(
    rt: KediRuntime, profile: AgentProfile, *, activation_profile: AgentProfile | None = None
) -> ToolSpec:
    from kedi.agent_profile.activation import require_subagent_activation

    activation_profile = profile if activation_profile is None else activation_profile
    allowed = tuple(profile.subagents)
    owned_policies = {
        name: capability_subagent_policy(profile, name, include_deferred=True) for name in allowed
    }
    definitions = _snapshot_subagent_definitions(rt, profile, allowed)
    native_owner = rt._native_subagent_owner.get()
    budget = (
        native_owner.budget_for(profile)
        if native_owner is not None
        else DelegationBudget.for_profile(profile_name=profile.name, max_agents=profile.max_agents)
    )
    concurrency = (
        native_owner.concurrency if native_owner is not None else rt._new_subagent_concurrency()
    )
    sandbox_to_profile = _sandbox_names(allowed)
    descriptions = {
        sandbox_name: _profile_description(rt, profile_name, definitions)
        for sandbox_name, profile_name in sandbox_to_profile.items()
    }
    salvage: dict[str, dict[str, Any]] = {}
    in_flight: dict[str, asyncio.Task[dict[str, Any]]] = {}

    async def run_workflow(*, code: str = "", **unsupported: Any) -> object:
        # Explicit-schema native tools can forward unvalidated keyword arguments.
        if unsupported:
            raise WorkflowTypeError(
                "run_workflow accepts only the code argument; unsupported arguments: "
                + ", ".join(sorted(unsupported))
                + ". Pass final_schema to a child call inside code, not to run_workflow."
            )
        if _active_workflow.get():
            raise WorkflowRuntimeError(
                "Dynamic workflows do not nest; return to the parent workflow first"
            )
        if not isinstance(code, str) or not code.strip():
            raise WorkflowTypeError("workflow code must be a non-empty string")

        completed_calls: list[_CompletedCall] = []
        stats = _WorkflowStats()

        async def dispatch(function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            subagent = sandbox_to_profile[function_name]
            require_subagent_activation(activation_profile, subagent)
            task, final_schema = _validate_child_arguments(function_name, arguments)
            if final_schema is not None:
                validate_final_schema(final_schema)
            salvage_key = _salvage_key(subagent, task, final_schema)
            cached = salvage.get(salvage_key)
            if cached is not None:
                stats.salvage_hits += 1
                return dict(cached)

            active_call = in_flight.get(salvage_key)
            if active_call is not None:
                stats.deduplicated_calls += 1
                return dict(await active_call)

            async def execute_child() -> dict[str, Any]:
                async def invoke(*, task: str, final_schema: dict[str, Any] | None):
                    return await _execute_dynamic_child(
                        rt=rt,
                        profile=profile,
                        allowed=allowed,
                        definitions=definitions,
                        budget=budget,
                        concurrency=concurrency,
                        subagent=subagent,
                        task=task,
                        final_schema=final_schema,
                    )

                def validate(arguments):
                    task, schema = _validate_child_arguments(function_name, arguments)
                    if schema is not None:
                        validate_final_schema(schema)
                    return {"task": task, "final_schema": schema}

                stats.active_calls += 1
                stats.concurrency_peak = max(stats.concurrency_peak, stats.active_calls)
                try:
                    policy = owned_policies[subagent]
                    if policy is None:
                        return await invoke(task=task, final_schema=final_schema)
                    from ...telemetry import adapter_shortname

                    tool = with_hook_lifecycle(
                        ToolSpec(
                            fn=invoke,
                            name=function_name,
                            risk="read_only",
                            capability_policy=policy,
                            argument_validator=validate,
                            metadata={"kind": "subagent", "subagent": subagent},
                        ),
                        adapter_shortname=adapter_shortname(rt.adapter),
                    )
                    return await tool.fn(task=task, final_schema=final_schema)
                except asyncio.CancelledError:
                    stats.cancelled_calls += 1
                    raise
                except BaseException:
                    stats.failed_calls += 1
                    raise
                finally:
                    stats.active_calls -= 1

            child_task = asyncio.create_task(execute_child())
            in_flight[salvage_key] = child_task
            try:
                snapshot = await child_task
            finally:
                del in_flight[salvage_key]
            if len(salvage) < MAX_WORKFLOW_SALVAGE_RESULTS:
                salvage[salvage_key] = snapshot
            completed_calls.append(_CompletedCall(subagent=subagent, task=task, result=snapshot))
            return dict(snapshot)

        workflow_id = uuid4().hex
        budget_before = budget.started
        started_at = perf_counter()
        span_attributes = {
            "kedi.workflow.id": workflow_id,
            "kedi.workflow.mode": "dynamic",
            "kedi.workflow.parent_profile": profile.name,
            "kedi.workflow.visible_subagents": len(sandbox_to_profile),
            "kedi.workflow.script_bytes": len(code.encode()),
            "kedi.workflow.budget_remaining_before": max(0, budget.limit - budget_before),
        }
        with _dynamic_workflow_span(span_attributes) as span:
            capture = CollectString(max_bytes=MAX_WORKFLOW_PRINT_BYTES)
            token = _active_workflow.set(True)
            result: object
            try:
                with Monty() as monty:
                    with monty.checkout(
                        limits={
                            "max_memory": WORKFLOW_MEMORY_LIMIT,
                            "max_feed_duration_secs": WORKFLOW_CPU_SECONDS,
                        },
                        type_check=True,
                        type_check_stubs=_type_check_stubs(descriptions),
                    ) as session:
                        state = session.feed_start(code, print_callback=capture)
                        complete = await WorkflowExecutor(
                            dispatch=dispatch,
                            valid_names=sandbox_to_profile,
                        ).run(state)
            except MontyTypingError as exc:
                if "error[invalid-syntax]" in exc.display():
                    raise WorkflowSyntaxError(
                        _workflow_error("Syntax error", exc, capture)
                    ) from exc
                raise WorkflowTypeError(_workflow_error("Type error", exc, capture)) from exc
            except MontySyntaxError as exc:
                raise WorkflowSyntaxError(_workflow_error("Syntax error", exc, capture)) from exc
            except MontyRuntimeError as exc:
                raise WorkflowRuntimeError(
                    _workflow_error("Runtime error", exc, capture)
                    + _completed_calls_text(completed_calls)
                ) from exc
            except MontyCrashedError as exc:
                raise WorkflowResourceError(
                    "Workflow sandbox exceeded its resource limits or stopped unexpectedly"
                    + _completed_calls_text(completed_calls)
                ) from exc
            except WorkflowAbortError as exc:
                result = {
                    "ok": False,
                    "error": "subagent_budget_exhausted",
                    "message": str(exc),
                    "completed_calls": len(completed_calls),
                }
            except BaseException as exc:
                if type(exc).__name__ != "PanicException":
                    raise
                raise WorkflowResourceError(
                    "Workflow sandbox stopped unexpectedly" + _completed_calls_text(completed_calls)
                ) from exc
            else:
                try:
                    result = _workflow_result(complete.output, capture.output)
                except PydanticSerializationError as exc:
                    raise WorkflowRuntimeError("Workflow result is not JSON serializable") from exc
            finally:
                _active_workflow.reset(token)
                span.set_attribute(
                    "kedi.workflow.duration_ms",
                    (perf_counter() - started_at) * 1000,
                )
                span.set_attribute(
                    "kedi.workflow.child_calls",
                    budget.started - budget_before,
                )
                span.set_attribute(
                    "kedi.workflow.completed_calls",
                    len(completed_calls),
                )
                span.set_attribute("kedi.workflow.failed_calls", stats.failed_calls)
                span.set_attribute("kedi.workflow.cancelled_calls", stats.cancelled_calls)
                span.set_attribute(
                    "kedi.workflow.concurrency_peak",
                    stats.concurrency_peak,
                )
                span.set_attribute("kedi.workflow.salvage_hits", stats.salvage_hits)
                span.set_attribute(
                    "kedi.workflow.deduplicated_calls",
                    stats.deduplicated_calls,
                )
                span.set_attribute(
                    "kedi.workflow.budget_remaining_after",
                    max(0, budget.limit - budget.started),
                )
            serialized_result = json.dumps(
                to_jsonable_python(result),
                ensure_ascii=True,
                separators=(",", ":"),
            )
            span.set_attribute("kedi.workflow.result_bytes", len(serialized_result.encode()))
            return result

    return ToolSpec(
        fn=run_workflow,
        name="run_workflow",
        description=_workflow_description(
            descriptions,
            remaining=max(0, budget.limit - budget.started),
        ),
        json_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Python orchestration code to execute in Monty.",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        return_schema={},
        metadata={
            "source": "kedi",
            "kind": "dynamic_workflow",
            "required_before_output": True,
            "code_arg_name": "code",
            "code_arg_language": "python",
            "kedi_workflow_catalog": descriptions,
            "kedi_workflow_remaining": max(0, budget.limit - budget.started),
        },
        risk="read_only",
        sequential=True,
    )


async def _execute_dynamic_child(
    *,
    rt: KediRuntime,
    profile: AgentProfile,
    allowed: tuple[str, ...],
    definitions: dict[str, Any],
    budget: DelegationBudget,
    concurrency: DelegationConcurrency,
    subagent: str,
    task: str,
    final_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    with track_delegation_attempt(
        requested_profile=subagent,
        task=task,
    ) as attempt:
        run_id: str | None = None
        try:
            handle = rt._start_subagent(
                allowed_subagents=allowed,
                subagent=subagent,
                task=task,
                final_schema=final_schema,
                parent_profile=profile,
                parent_budget=budget,
                parent_concurrency=concurrency,
                subagent_definitions=definitions,
            )
            run_id = handle.run_id
            result = await rt._wait_subagent(run_id)
            serialized = to_jsonable_python(result.as_tool_result())
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                if run_id is not None:
                    rt._cancel_subagent(run_id)
                    try:
                        await rt._wait_subagent(run_id)
                    except BaseException as cleanup_error:
                        if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
                            raise
                if attempt is not None:
                    attempt.fail(status="cancelled", error=exc)
                raise
            if attempt is not None:
                if run_id is None:
                    attempt.fail(status="rejected", error=exc)
                else:
                    status = rt._subagent_coordinator.runs.get(run_id).status
                    if status != "running":
                        attempt.fail(status=attempt_status_from_run(status), error=exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SubagentBudgetExceeded):
                raise WorkflowAbortError(exc.message) from exc
            message = safe_subagent_error_message(exc) or type(exc).__name__
            raise RuntimeError(f"subagent {subagent!r} failed: {message}") from exc
        if attempt is not None:
            attempt.complete()
        return serialized


def _sandbox_name(profile_name: str) -> str:
    if profile_name.isidentifier() and not keyword.iskeyword(profile_name):
        return profile_name
    return f"subagent_{profile_name}"


def _sandbox_names(profile_names: tuple[str, ...]) -> dict[str, str]:
    names: dict[str, str] = {}
    for profile_name in profile_names:
        base = _sandbox_name(profile_name)
        candidate = base
        suffix = 2
        while candidate in names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        names[candidate] = profile_name
    return names


def _profile_description(
    rt: KediRuntime,
    profile_name: str,
    definitions: dict[str, Any],
) -> str | None:
    definition = definitions.get(profile_name) or rt._profile_registry.get(profile_name)
    return definition.description if definition is not None else None


def _snapshot_subagent_definitions(
    rt: KediRuntime,
    profile: AgentProfile,
    allowed: tuple[str, ...],
) -> dict[str, Any]:
    registry = rt._profile_registry if isinstance(rt._profile_registry, dict) else {}
    definitions: dict[str, Any] = {}
    for profile_name in allowed:
        definition = profile.subagent_bindings.get(profile_name) or registry[profile_name]
        definitions[profile_name] = deepcopy(definition)
    return definitions


def _validate_child_arguments(
    function_name: str,
    arguments: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if "task" not in arguments:
        raise TypeError(f"{function_name}() missing required keyword argument: task")
    extra = sorted(set(arguments) - {"task", "final_schema"})
    if extra:
        raise TypeError(f"{function_name}() got unexpected keyword argument(s): {', '.join(extra)}")
    task = arguments["task"]
    if not isinstance(task, str) or not task.strip():
        raise TypeError(f"{function_name}() task must be a non-empty string")
    final_schema = arguments.get("final_schema")
    if final_schema is not None and not isinstance(final_schema, dict):
        raise TypeError(f"{function_name}() final_schema must be an object or null")
    return task, final_schema


def _salvage_key(
    subagent: str,
    task: str,
    final_schema: dict[str, Any] | None,
) -> str:
    payload = json.dumps(
        [subagent, task.strip(), final_schema],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode()).hexdigest()


def _type_check_stubs(descriptions: dict[str, str | None]) -> str:
    parts = [
        "from typing import Any, TypedDict",
        "class DelegationResult(TypedDict):\n"
        "    run_id: str\n"
        "    subagent: str\n"
        "    task_summary: str\n"
        "    final_result: Any",
    ]
    for name, description in descriptions.items():
        docstring = f"\n    {description!r}" if description else ""
        parts.append(
            f"async def {name}(*, task: str, final_schema: dict[str, Any] | None = None) "
            f"-> DelegationResult:{docstring}\n    raise NotImplementedError()"
        )
    return "\n\n".join(parts)


def _workflow_description(
    descriptions: dict[str, str | None],
    *,
    remaining: int,
) -> str:
    stubs = _type_check_stubs(descriptions)
    return f"""Write and run Python orchestration code that coordinates the available subagents.

Monty executes a restricted Python subset with no third-party packages, filesystem, environment,
network, clock, or process access. Import `asyncio` when using `asyncio.gather`. The subagent
functions below are already available as globals: do not import them. Await each function, pass a
self-contained non-empty `task` by keyword, and optionally pass a JSON Schema object as
`final_schema`. Results are dictionaries with `run_id`, `subagent`,
`task_summary`, and `final_result`; use dictionary indexing. Independent work should run concurrently
with `asyncio.gather`. The last expression is the workflow result; `print()` is only bounded debug
output. Build task text from prior results with f-strings, keeping every string literal closed on its
source line. After this tool returns, use its result to produce the parent response; never end the
parent turn with an empty response. This parent run has {remaining} subagent call(s) remaining.

Available subagents:
```python
{stubs}
```"""


def _workflow_error(label: str, exc: Any, capture: CollectString) -> str:
    message = f"{label} in workflow:\n{exc.display()}"
    printed = capture.output.rstrip("\n")
    if printed:
        return f"[stdout before error]\n{printed}\n[/stdout before error]\n{message}"
    return message


def _completed_calls_text(completed: list[_CompletedCall]) -> str:
    if not completed:
        return ""
    entries = completed[-MAX_WORKFLOW_SALVAGE_RESULTS:]
    lines = []
    for entry in entries:
        rendered = json.dumps(entry.result, ensure_ascii=True, separators=(",", ":"))
        if len(rendered) > MAX_WORKFLOW_RESULT_PREVIEW_CHARS:
            rendered = rendered[: MAX_WORKFLOW_RESULT_PREVIEW_CHARS - 16] + " ... [truncated]"
        lines.append(f"- {entry.subagent}(task={entry.task!r}) -> {rendered}")
    return "\n\nCompleted calls available to an identical retry:\n" + "\n".join(lines)


def _workflow_result(result: object, printed: str) -> object:
    normalized = to_jsonable_python(result)
    if not printed:
        return normalized if normalized is not None else {}
    output = printed.rstrip("\n")
    if normalized is None:
        return {"output": output}
    return {"output": output, "result": normalized}
