from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace
from typing import Any, Iterator, cast

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.adapters.pydantic import _validated_tool_function
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.events import (
    AgentRunStateEvent,
    AgentStreamEvent,
    agent_event_run,
    observe_agent_events,
)
from kedi.agent_adapter.invocation import (
    InvocationUsage,
    observe_invocations,
    report_invocation,
)
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.subagents import (
    MAX_FINAL_SCHEMA_BYTES,
    DelegationResult,
    SubagentExecutionRequest,
    SubagentTurn,
    effective_final_schema,
    subagent_tool_specs,
    validate_final_schema,
    validate_structured_result,
    wrapped_final_schema,
)
from kedi.agent_adapter.subagents import coordinator as coordinator_module
from kedi.agent_adapter.subagents import workflow as workflow_module
from kedi.agent_adapter.subagents.attempts import (
    DelegationFailurePolicy,
    UnresolvedDelegationError,
    delegation_scope,
    track_delegation_attempt,
)
from kedi.agent_adapter.subagents.budget import (
    MAX_SUBAGENT_CONCURRENCY,
    MAX_SUBAGENT_INVOCATIONS,
    DelegationBudget,
    DelegationConcurrency,
    SubagentBudgetExceeded,
)
from kedi.agent_adapter.subagents.coordinator import (
    MAX_SUBAGENT_CONVERSATION_TURNS,
    _OwnedSubagentTask,
    _schedule_owned_cancellation,
    _subagent_run_span,
    _unwrap_task,
)
from kedi.agent_adapter.subagents.limits import (
    MAX_SUBAGENT_REQUESTS,
    MAX_SUBAGENT_TOKENS,
    MAX_SUBAGENT_TOOL_CALLS,
    SubagentUsageLimits,
)
from kedi.agent_adapter.subagents.persistence import StoredConversation, SubagentStateStore
from kedi.agent_adapter.subagents.runs import (
    SubagentCancelled,
    SubagentCompleted,
    SubagentEventLog,
    SubagentFailed,
    SubagentRun,
    SubagentRunRegistry,
    SubagentStarted,
    safe_subagent_error_message,
)
from kedi.agent_adapter.subagents.safety import SubagentSafetyCeiling
from kedi.agent_adapter.subagents.workflow import (
    WorkflowResourceError,
    WorkflowRuntimeError,
    WorkflowSyntaxError,
    WorkflowTypeError,
    _completed_calls_text,
    _CompletedCall,
    _dynamic_workflow_span,
    _execute_dynamic_child,
    _profile_description,
    _sandbox_names,
    _validate_child_arguments,
    _workflow_error,
    _workflow_result,
)
from kedi.agent_adapter.subagents.workflow import executor as workflow_executor_module
from kedi.agent_adapter.subagents.workflow.executor import (
    WorkflowAbortError,
    WorkflowExecutor,
)
from kedi.agent_adapter.telemetry import tool_call_span
from kedi.agent_adapter.tool_override import (
    RequiredToolCallError,
    ToolSpec,
    required_tool_call_scope,
    validate_required_tool_calls,
    with_required_before_output,
)
from kedi.agent_profile import (
    AgentProfile,
    AgentSubagentUnsupported,
)
from kedi.agent_profile.compile import build_capture_env, profile_stmt_to_definition
from kedi.agent_profile.model import eval_profile_definition
from kedi.artifacts import artifact_instructions
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.lang import KediParseError, compile_program, parse_program
from kedi.lang.ast import ProfileDefStmt, ProfileOutputEntry, ProfileWorkflowEntry, TypeRef
from kedi.lsp import features
from tests.mock_adapter import MockAdapter
from tests.telemetry_helpers import installed_recording_backend


def test_subagent_run_reuses_delegation_tool_span() -> None:
    with installed_recording_backend() as backend:
        tool = ToolSpec(fn=lambda: None, name="delegate_task")
        with tool_call_span(tool, adapter_shortname="test", argument_count=0) as telemetry:
            assert telemetry.span is not None
            tool_span = telemetry.span
            with _subagent_run_span({"kedi.subagent.run_id": "run-1"}) as subagent_span:
                assert subagent_span is tool_span

    assert [recorded.name for recorded in backend.finished_spans] == ["subagent run"]
    assert backend.finished_spans[0].operation == "call_tool"
    assert backend.finished_spans[0].attributes["kedi.operation.name"] == "run_agent"
    assert backend.finished_spans[0].attributes["kedi.subagent.run_id"] == "run-1"


def test_subagent_run_keeps_unrelated_parent_span() -> None:
    with installed_recording_backend() as backend:
        with backend.start_span(
            scope="agent",
            name="agent kedi",
            operation="run_agent",
            kind="internal",
            level="lifecycle",
            attributes=None,
        ) as parent:
            with _subagent_run_span({"kedi.subagent.run_id": "run-1"}) as child:
                assert child is not parent
                assert child.parent is parent

    assert [recorded.name for recorded in backend.finished_spans] == [
        "subagent run",
        "agent kedi",
    ]


def test_dynamic_workflow_reuses_its_tool_span() -> None:
    with installed_recording_backend() as backend:
        tool = ToolSpec(fn=lambda: None, name="run_workflow")
        with tool_call_span(tool, adapter_shortname="test", argument_count=0) as telemetry:
            assert telemetry.span is not None
            tool_span = telemetry.span
            with _dynamic_workflow_span({"kedi.workflow.id": "workflow-1"}) as workflow_span:
                assert workflow_span is tool_span

    assert [recorded.name for recorded in backend.finished_spans] == ["dynamic workflow"]
    assert backend.finished_spans[0].attributes["kedi.operation.name"] == "run_workflow"
    assert backend.finished_spans[0].attributes["kedi.workflow.id"] == "workflow-1"


class RecordingPydanticAdapter(PydanticAdapter):
    def __init__(self) -> None:
        super().__init__(TestModel())
        self.invocations: list[tuple[str, str | None, tuple[str, ...], tuple[str, ...]]] = []

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        self.invocations.append(
            (
                prompt,
                kwargs.get("instructions"),
                tuple(tool.name for tool in self.scoped_tool_specs),
                self.active_profile.subagents,
            )
        )
        return f"child:{prompt}"

    async def produce(
        self,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        assert output_schema is None
        assert output_type is not None
        self.invocations.append(
            (
                template,
                kwargs.get("instructions"),
                tuple(tool.name for tool in self.scoped_tool_specs),
                self.active_profile.subagents,
            )
        )
        return {
            "task_summary": "Built the requested value.",
            "final_result": {"value": 7},
        }


class RecordingNeutralChildAdapter(MockAdapter):
    shortname = "neutral"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_profile_override=True,
        supports_subagents=True,
        supports_artifacts=True,
    )

    def __init__(self) -> None:
        super().__init__(default_model="neutral-model")
        self.child_requests: list[SubagentExecutionRequest] = []
        self.child_calls: list[tuple[str, dict[str, Any]]] = []

    def telemetry_model_name(self) -> str | None:
        return self.default_model()

    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Iterator[None]:
        self.child_requests.append(request)
        with self.tool_scope(request.tools), self.profile_override(request.profile):
            yield

    @staticmethod
    def subagent_run_kwargs(limits: SubagentUsageLimits) -> dict[str, Any]:
        return {"neutral_request_limit": limits.request_limit}

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        self.child_calls.append((prompt, kwargs))
        report_invocation(InvocationUsage(requests=1, total_tokens=3))
        return f"neutral:{prompt}"

    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.child_calls.append((template, {**kwargs, "output_schema": output_schema}))
        report_invocation(InvocationUsage(requests=1, total_tokens=5))
        return {
            "task_summary": "Neutral child completed the task.",
            "final_result": {"value": 11},
        }


def _profiles_source() -> str:
    return """
@lookup(query: str) -> str:
    = found-<query>

> profile: worker:
    ###
    Handles one isolated worker task.
    ###
    > adapter: pydantic
    > system: Work carefully.
    > use: lookup

> profile: manager:
    > adapter: pydantic
    > subagent: worker
"""


def _manager_tool() -> tuple[RecordingPydanticAdapter, Any, Any]:
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(_profiles_source()), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]
    return adapter, runtime, tool


def _manager_tools() -> tuple[RecordingPydanticAdapter, Any, dict[str, Any]]:
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(_profiles_source()), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}
    return adapter, runtime, tools


def test_subagent_foreign_loop_wait_does_not_cancel_owner_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, tools = _manager_tools()
    started = Event()
    release = Event()
    owner_loop = asyncio.new_event_loop()
    owner_ready = Event()

    def run_owner_loop() -> None:
        asyncio.set_event_loop(owner_loop)
        owner_ready.set()
        owner_loop.run_forever()
        owner_loop.close()

    owner_thread = Thread(target=run_owner_loop, daemon=True)

    async def block(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        started.set()
        await asyncio.to_thread(release.wait)
        return "completed by owner loop"

    monkeypatch.setattr(adapter, "invoke", block)
    owner_thread.start()
    assert owner_ready.wait(timeout=2)

    async def start_background() -> str:
        handle = await tools["delegate_task"].fn(
            subagent="worker",
            task="block until released",
            background=True,
        )
        return handle["run_id"]

    try:
        run_id = asyncio.run_coroutine_threadsafe(start_background(), owner_loop).result(timeout=2)
        assert started.wait(timeout=2)

        async def caller_timeout() -> None:
            with pytest.raises(TimeoutError):
                await runtime._subagent_coordinator.wait(run_id, timeout=0.01)

        asyncio.run(caller_timeout())
        assert runtime._subagent_coordinator.status(run_id) == "running"

        release.set()
        result = asyncio.run(runtime._subagent_coordinator.wait(run_id))
        assert result.task_summary == "completed by owner loop"
        assert runtime._subagent_coordinator.status(run_id) == "completed"
    finally:
        release.set()
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=2)


def test_subagent_foreign_thread_cancel_preserves_contextvars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, tools = _manager_tools()
    started = Event()
    owner_loop = asyncio.new_event_loop()
    owner_ready = Event()

    def run_owner_loop() -> None:
        asyncio.set_event_loop(owner_loop)
        owner_ready.set()
        owner_loop.run_forever()
        owner_loop.close()

    owner_thread = Thread(target=run_owner_loop, daemon=True)

    async def block(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(adapter, "invoke", block)
    owner_thread.start()
    assert owner_ready.wait(timeout=2)

    async def start_background() -> str:
        handle = await tools["delegate_task"].fn(
            subagent="worker",
            task="wait for cancellation",
            background=True,
        )
        return handle["run_id"]

    async def wait_for_cancellation(run_id: str) -> None:
        with pytest.raises(asyncio.CancelledError):
            await runtime._subagent_coordinator.wait(run_id)

    try:
        run_id = asyncio.run_coroutine_threadsafe(start_background(), owner_loop).result(timeout=2)
        assert started.wait(timeout=2)
        assert runtime._subagent_coordinator.cancel(run_id) is True
        asyncio.run_coroutine_threadsafe(wait_for_cancellation(run_id), owner_loop).result(
            timeout=2
        )
        assert runtime._subagent_coordinator.status(run_id) == "cancelled"
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=2)


def test_subagent_cross_loop_guard_paths_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator

    class PendingTask:
        def __init__(
            self, *, done: bool = False, cancel_error: BaseException | None = None
        ) -> None:
            self._done = done
            self._cancel_error = cancel_error
            self.cancelled = False

        def done(self) -> bool:
            return self._done

        def cancel(self) -> None:
            if self._cancel_error is not None:
                raise self._cancel_error
            self.cancelled = True

    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    closed_task = cast(Any, PendingTask())
    closed_owned = _OwnedSubagentTask(closed_task, closed_loop)
    coordinator._tasks["closed"] = closed_owned

    with pytest.raises(KediExecutionError, match="cannot be cancelled"):
        coordinator.cancel("closed")
    with pytest.raises(KediExecutionError, match="owner event loop is closed"):
        coordinator._bridge_task_wait("closed", closed_owned)

    done_task = cast(Any, PendingTask(done=True))
    done_owned = _OwnedSubagentTask(done_task, asyncio.new_event_loop())
    coordinator._tasks["done"] = done_owned
    coordinator._timeout_run("done", done_task, 1)
    done_owned.loop.close()

    scheduled: list[tuple[object, tuple[object, ...]]] = []

    class ForeignLoop:
        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, callback: object, *args: object, **_kwargs: object) -> None:
            scheduled.append((callback, args))

    foreign_task = cast(Any, PendingTask())
    foreign_owned = _OwnedSubagentTask(foreign_task, cast(Any, ForeignLoop()))
    coordinator._tasks["foreign"] = foreign_owned
    coordinator._mark_timed_out("foreign", foreign_task, 1)
    assert scheduled

    class ClosedForeignLoop(ForeignLoop):
        def is_closed(self) -> bool:
            return True

    closed_foreign_task = cast(Any, PendingTask())
    coordinator._tasks["closed-foreign"] = _OwnedSubagentTask(
        closed_foreign_task,
        cast(Any, ClosedForeignLoop()),
    )
    coordinator._mark_timed_out("closed-foreign", closed_foreign_task, 1)

    coordinator._tasks["terminal"] = _OwnedSubagentTask(
        cast(Any, PendingTask()),
        asyncio.new_event_loop(),
    )
    monkeypatch.setattr(
        coordinator_module, "_running_loop", lambda: coordinator._tasks["terminal"].loop
    )
    coordinator.runs = cast(
        Any, SimpleNamespace(get=lambda _run_id: SimpleNamespace(status="completed"))
    )
    coordinator._mark_timed_out("terminal", coordinator._tasks["terminal"].task, 1)
    coordinator._tasks["terminal"].loop.close()

    async def unwrap_raw_task() -> None:
        task = asyncio.create_task(asyncio.sleep(0))
        assert _unwrap_task(task) is task
        await task

    asyncio.run(unwrap_raw_task())

    async def cancellation_error() -> None:
        normal_task = cast(Any, PendingTask())
        normal_owned = _OwnedSubagentTask(normal_task, asyncio.get_running_loop())
        _schedule_owned_cancellation(normal_owned)
        await asyncio.sleep(0)
        assert normal_task.cancelled

        detached_error_task = cast(Any, PendingTask(cancel_error=RuntimeError("detached failure")))
        detached_error_owned = _OwnedSubagentTask(
            detached_error_task,
            asyncio.get_running_loop(),
        )
        _schedule_owned_cancellation(detached_error_owned)
        await asyncio.sleep(0)

        error_task = cast(Any, PendingTask(cancel_error=RuntimeError("cancel failed")))
        owned = _OwnedSubagentTask(error_task, asyncio.get_running_loop())
        completed: Future[None] = Future()
        _schedule_owned_cancellation(owned, completed)
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="cancel failed"):
            completed.result()

    asyncio.run(cancellation_error())


def test_subagent_profile_parses_and_compiles() -> None:
    source = _profiles_source().replace(
        "    > subagent: worker\n",
        "    > subagent: worker\n    > max_agents: 5\n",
    )
    program = parse_program(source)
    entry = program.profile_defs["manager"].subagent_members[0]
    assert entry.name == "worker"
    assert program.profile_defs["manager"].max_agents_members[0].value == 5

    runtime = compile_program(program, adapter=RecordingPydanticAdapter())
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    assert profile.subagents == ("worker",)
    assert profile.name == "manager"
    assert profile.max_agents == 5
    assert profile.workflow_mode is None


def test_omitted_workflow_mode_does_not_replace_an_explicit_strategy() -> None:
    dynamic = AgentProfile(workflow_mode="dynamic")
    assert dynamic.merge(AgentProfile(model="child-model")).workflow_mode == "dynamic"
    assert dynamic.merge(AgentProfile(workflow_mode="delegate")).workflow_mode == "delegate"


def test_subagent_workflow_profile_parses_and_compiles() -> None:
    source = _profiles_source().replace(
        "    > subagent: worker\n",
        "    > subagent: worker\n    > workflow: dynamic\n",
    )
    program = parse_program(source)
    entry = program.profile_defs["manager"].workflow_members[0]
    assert entry.mode == "dynamic"

    runtime = compile_program(program, adapter=RecordingPydanticAdapter())
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    assert profile.workflow_mode == "dynamic"


def test_subagent_profile_output_type_is_default_delegation_schema() -> None:
    source = _profiles_source().replace(
        "    > system: Work carefully.\n",
        "    > system: Work carefully.\n    > output: dict[str, int]\n",
    )
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["worker"]
    worker = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    assert worker.output_type == dict[str, int]

    manager_definition = runtime._profile_registry["manager"]
    manager = eval_profile_definition(
        runtime,
        manager_definition,
        build_capture_env(runtime),
    )
    tool = runtime._resolve_agent_tool_specs(manager, adapter)[0]
    result = asyncio.run(tool.fn(subagent="worker", task="Build a value"))

    assert result["task_summary"] == "Built the requested value."
    assert result["final_result"] == {"value": 7}


def test_explicit_delegation_schema_overrides_profile_output_type() -> None:
    explicit = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    profile = AgentProfile(name="worker", output_type=str)

    assert effective_final_schema(profile, explicit) is explicit


def test_profile_output_type_must_have_a_json_schema() -> None:
    profile = AgentProfile(name="broken", output_type=SimpleNamespace())

    with pytest.raises(
        KediExecutionError,
        match="profile 'broken' output type cannot be represented as JSON Schema",
    ):
        effective_final_schema(profile, None)


def test_profile_rejects_duplicate_output_types() -> None:
    source = "> profile: worker:\n    > output: str\n    > output: int\n"

    with pytest.raises(KediParseError, match="only one output type"):
        parse_program(source)

    with pytest.raises(KediExecutionError, match="only one output type"):
        profile_stmt_to_definition(
            ProfileDefStmt(
                name="worker",
                output_members=[
                    ProfileOutputEntry(type_annotation="str", type_expr=TypeRef("str")),
                    ProfileOutputEntry(type_annotation="int", type_expr=TypeRef("int")),
                ],
            )
        )


def test_explicit_delegate_workflow_preserves_default_tool_surface() -> None:
    default_adapter, default_runtime, default_tools = _manager_tools()
    explicit_source = _profiles_source().replace(
        "    > subagent: worker\n",
        "    > subagent: worker\n    > workflow: delegate\n",
    )
    explicit_adapter = RecordingPydanticAdapter()
    explicit_runtime = compile_program(parse_program(explicit_source), adapter=explicit_adapter)
    definition = explicit_runtime._profile_registry["manager"]
    profile = eval_profile_definition(
        explicit_runtime,
        definition,
        build_capture_env(explicit_runtime),
    )
    explicit_tools = explicit_runtime._resolve_agent_tool_specs(profile, explicit_adapter)

    assert tuple(tool.name for tool in explicit_tools) == tuple(
        tool.name for tool in default_tools.values()
    )
    assert default_adapter.invocations == []
    assert explicit_adapter.invocations == []
    assert default_runtime._subagent_coordinator.active_runs == 0
    assert explicit_runtime._subagent_coordinator.active_runs == 0


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "> profile: manager:\n    > workflow: automatic\n",
            "workflow mode must be 'delegate' or 'dynamic'",
        ),
        (
            "> profile: manager:\n    > workflow: delegate\n    > workflow: dynamic\n",
            "profile can define only one workflow mode",
        ),
        (
            "> profile: manager:\n    > workflow: dynamic\n",
            "dynamic workflow requires at least one direct subagent",
        ),
    ],
)
def test_subagent_workflow_profile_rejects_invalid_configuration(
    source: str,
    message: str,
) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


def test_workflow_profile_compilation_rechecks_ast_invariants() -> None:
    with pytest.raises(KediExecutionError, match="only one workflow mode"):
        profile_stmt_to_definition(
            ProfileDefStmt(
                name="manager",
                workflow_members=[
                    ProfileWorkflowEntry(mode="delegate"),
                    ProfileWorkflowEntry(mode="dynamic"),
                ],
            )
        )
    with pytest.raises(KediExecutionError, match="requires at least one direct subagent"):
        profile_stmt_to_definition(
            ProfileDefStmt(
                name="manager",
                workflow_members=[ProfileWorkflowEntry(mode="dynamic")],
            )
        )


def _dynamic_workflow_tool(
    *,
    max_agents: int = 5,
) -> tuple[RecordingPydanticAdapter, KediRuntime, ToolSpec]:
    source = _profiles_source().replace(
        "    > subagent: worker\n",
        f"    > subagent: worker\n    > workflow: dynamic\n    > max_agents: {max_agents}\n",
    )
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}
    assert "delegate_task" not in tools
    assert "continue_subagent" not in tools
    return adapter, runtime, tools["run_workflow"]


def test_dynamic_workflow_replaces_parent_delegation_tools() -> None:
    adapter, _runtime, tool = _dynamic_workflow_tool()

    assert tool.name == "run_workflow"
    assert tool.sequential is True
    assert (tool.metadata or {})["code_arg_language"] == "python"
    assert "worker" in (tool.description or "")
    assert "do not import them" in (tool.description or "")
    assert "never end the\nparent turn with an empty response" in (tool.description or "")
    assert (tool.metadata or {})["required_before_output"] is True
    assert adapter.invocations == []


def test_dynamic_workflow_rejects_parent_output_until_workflow_returns() -> None:
    parent_calls = 0

    async def respond(
        messages: list[object],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal parent_calls
        tool_names = {tool.name for tool in info.function_tools}
        if "run_workflow" not in tool_names:
            return ModelResponse(parts=[TextPart("verified child result")])

        parent_calls += 1
        if parent_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_workflow",
                        {"code": "await worker(task="},
                        tool_call_id="broken-workflow",
                    )
                ]
            )
        if parent_calls == 2:
            return ModelResponse(parts=[TextPart("fabricated parent result")])
        if parent_calls == 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_workflow",
                        {"code": "42"},
                        tool_call_id="valid-workflow",
                    )
                ]
            )
        retry_messages = " ".join(
            str(part)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        assert "did not complete successfully" in retry_messages
        return ModelResponse(parts=[TextPart("parent used verified workflow result")])

    source = (
        _profiles_source().replace(
            "    > subagent: worker\n",
            "    > subagent: worker\n    > workflow: dynamic\n",
        )
        + """
> use: manager
[answer] << Run the workflow and report its verified result.
= <answer>
"""
    )
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(respond), retries=2),
    )

    assert runtime.run_main() == "parent used verified workflow result"
    assert parent_calls == 4


def test_dynamic_workflow_retries_when_parent_never_calls_required_tool() -> None:
    parent_calls = 0

    async def respond(
        messages: list[object],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal parent_calls
        parent_calls += 1
        if parent_calls == 1:
            return ModelResponse(parts=[TextPart("fabricated parent result")])
        if parent_calls == 2:
            retry_messages = " ".join(
                str(part)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            assert "Call required tool run_workflow" in retry_messages
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_workflow",
                        {"code": "42"},
                        tool_call_id="valid-workflow",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("parent used verified workflow result")])

    source = (
        _profiles_source().replace(
            "    > subagent: worker\n",
            "    > subagent: worker\n    > workflow: dynamic\n",
        )
        + """
> use: manager
[answer] << Run the workflow and report its verified result.
= <answer>
"""
    )
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(respond), retries=1),
    )

    assert runtime.run_main() == "parent used verified workflow result"
    assert parent_calls == 3


class _RequiredWorkflowMockAdapter(MockAdapter):
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_profile_override=True,
        supports_subagents=True,
        supports_artifacts=True,
    )

    def __init__(self, *, workflow_code: str | None, ignore_tool_error: bool = False) -> None:
        super().__init__()
        self.workflow_code = workflow_code
        self.ignore_tool_error = ignore_tool_error

    def invoke_sync(self, *, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        workflow = next(tool for tool in self.scoped_tools if tool.name == "run_workflow")
        if self.workflow_code is None:
            return "parent adapter response"
        try:
            asyncio.run(workflow.fn(code=self.workflow_code))
        except WorkflowSyntaxError:
            if not self.ignore_tool_error:
                raise
        return "parent adapter response"


def _required_workflow_source() -> str:
    return """
> profile: worker:
    > system: Complete one isolated task.

> profile: manager:
    > subagent: worker
    > workflow: dynamic

> use: manager
[answer] << Run the workflow and report its result.
= <answer>
"""


def test_non_pydantic_adapter_rejects_output_after_required_tool_failure() -> None:
    runtime = compile_program(
        parse_program(_required_workflow_source()),
        adapter=_RequiredWorkflowMockAdapter(
            workflow_code="await worker(task=",
            ignore_tool_error=True,
        ),
    )

    with pytest.raises(
        KediExecutionError,
        match="Required tool run_workflow did not complete successfully",
    ):
        runtime.run_main()


def test_non_pydantic_adapter_rejects_output_without_required_tool_call() -> None:
    runtime = compile_program(
        parse_program(_required_workflow_source()),
        adapter=_RequiredWorkflowMockAdapter(workflow_code=None),
    )

    with pytest.raises(
        KediExecutionError,
        match="Required tool run_workflow was not called",
    ):
        runtime.run_main()


def test_non_pydantic_adapter_accepts_output_after_required_tool_returns() -> None:
    runtime = compile_program(
        parse_program(_required_workflow_source()),
        adapter=_RequiredWorkflowMockAdapter(workflow_code="42"),
    )

    assert runtime.run_main() == "parent adapter response"


def test_required_tool_wrapper_handles_sync_and_awaitable_results() -> None:
    def sync_tool() -> str:
        return "sync"

    sync_spec = with_required_before_output(
        ToolSpec(
            fn=sync_tool,
            name="required_sync",
            metadata={"required_before_output": True},
        )
    )
    assert with_required_before_output(sync_spec) is sync_spec
    assert sync_spec.fn() == "sync"
    with required_tool_call_scope((sync_spec,)):
        assert sync_spec.fn() == "sync"
        validate_required_tool_calls()

    def awaitable_tool() -> Any:
        async def result() -> str:
            return "awaited"

        return result()

    awaitable_spec = with_required_before_output(
        ToolSpec(
            fn=awaitable_tool,
            name="required_awaitable",
            metadata={"required_before_output": True},
        )
    )
    with required_tool_call_scope((awaitable_spec,)):
        assert asyncio.run(awaitable_spec.fn()) == "awaited"
        validate_required_tool_calls()


def test_nested_required_tool_scopes_are_isolated_by_default() -> None:
    spec = with_required_before_output(
        ToolSpec(
            fn=lambda: "done",
            name="required_nested",
            metadata={"required_before_output": True},
        )
    )

    with required_tool_call_scope((spec,)):
        with required_tool_call_scope((spec,)):
            assert spec.fn() == "done"
            validate_required_tool_calls()
        with pytest.raises(RequiredToolCallError, match="was not called"):
            validate_required_tool_calls()


def test_adapter_owned_required_tool_scope_can_be_inherited() -> None:
    spec = with_required_before_output(
        ToolSpec(
            fn=lambda: "done",
            name="required_inherited",
            metadata={"required_before_output": True},
        )
    )

    with required_tool_call_scope((spec,)):
        with required_tool_call_scope((spec,), inherit=True):
            assert spec.fn() == "done"
        validate_required_tool_calls()


def test_pydantic_maps_dynamic_workflow_errors_to_model_retry() -> None:
    from pydantic_ai import ModelRetry

    async def fail(*, code: str) -> None:
        raise WorkflowRuntimeError(f"bad workflow: {code}")

    wrapped = _validated_tool_function(
        ToolSpec(
            fn=fail,
            name="run_workflow",
            metadata={"kind": "dynamic_workflow"},
        )
    )
    with pytest.raises(ModelRetry, match="bad workflow"):
        asyncio.run(wrapped(code="broken()"))

    async def fail_normally(*, code: str) -> None:
        raise ValueError(code)

    normal = _validated_tool_function(
        ToolSpec(
            fn=fail_normally,
            name="run_workflow",
            metadata={"kind": "dynamic_workflow"},
        )
    )
    with pytest.raises(ValueError, match="broken"):
        asyncio.run(normal(code="broken"))


def test_dynamic_workflow_executes_sequential_parallel_and_structured_children() -> None:
    adapter, _runtime, tool = _dynamic_workflow_tool()

    async def scenario() -> None:
        sequential = await tool.fn(code="result = await worker(task='one')\nresult['task_summary']")
        assert sequential == "child:one"

        parallel = await tool.fn(
            code=(
                "import asyncio\n"
                "left, right = await asyncio.gather(\n"
                "    worker(task='left'), worker(task='right')\n"
                ")\n"
                "[left['task_summary'], right['task_summary']]"
            )
        )
        assert parallel == ["child:left", "child:right"]

        structured = await tool.fn(
            code=(
                "result = await worker(\n"
                "    task='structured',\n"
                "    final_schema={'type': 'object', 'properties': "
                "{'value': {'type': 'integer'}}, 'required': ['value']}\n"
                ")\n"
                "result['final_result']['value']"
            )
        )
        assert structured == 7

    asyncio.run(scenario())
    assert [call[0] for call in adapter.invocations] == [
        "one",
        "left",
        "right",
        "structured",
    ]


def test_dynamic_workflow_snapshots_local_child_definitions() -> None:
    adapter, runtime, tool = _dynamic_workflow_tool()
    runtime._profile_registry["worker"].members.clear()

    result = asyncio.run(
        tool.fn(code="result = await worker(task='snapshot')\nresult['task_summary']")
    )

    assert result == "child:snapshot"
    assert (adapter.invocations[-1][1] or "").startswith("Work carefully.")


def test_dynamic_workflow_validates_before_spending_budget_and_salvages_success() -> None:
    adapter, _runtime, tool = _dynamic_workflow_tool(max_agents=2)

    async def scenario() -> None:
        with pytest.raises(WorkflowSyntaxError):
            await tool.fn(code="await worker(task=")
        with pytest.raises(WorkflowTypeError):
            await tool.fn(code="unknown_name")
        assert adapter.invocations == []

        with pytest.raises(WorkflowRuntimeError, match="Completed calls"):
            await tool.fn(code="result = await worker(task='paid')\n1 // 0")
        assert len(adapter.invocations) == 1

        recovered = await tool.fn(code="result = await worker(task='paid')\nresult['task_summary']")
        assert recovered == "child:paid"
        assert len(adapter.invocations) == 1

        terminal = await tool.fn(
            code=("await worker(task='last')\nawait worker(task='over-budget')")
        )
        assert terminal == {
            "ok": False,
            "error": "subagent_budget_exhausted",
            "message": "Subagent invocation limit 2 reached for profile 'manager'",
            "completed_calls": 1,
        }
        assert len(adapter.invocations) == 2

    asyncio.run(scenario())


def test_dynamic_workflow_rejects_invalid_calls_and_nested_execution() -> None:
    _adapter, _runtime, tool = _dynamic_workflow_tool()

    async def scenario() -> None:
        with pytest.raises(WorkflowTypeError, match="positional"):
            await tool.fn(code="await worker('bad')")
        with pytest.raises(WorkflowTypeError, match="surprise"):
            await tool.fn(code="await worker(task='bad', surprise=True)")
        with pytest.raises(WorkflowTypeError, match="final_schema"):
            await tool.fn(code="await worker(task='bad', final_schema='object')")

        from kedi.agent_adapter.subagents.workflow import _active_workflow

        token = _active_workflow.set(True)
        try:
            with pytest.raises(WorkflowRuntimeError, match="do not nest"):
                await tool.fn(code="1")
        finally:
            _active_workflow.reset(token)

    asyncio.run(scenario())


def test_dynamic_workflow_deduplicates_concurrent_calls_and_recovers_child_failure() -> None:
    adapter, _runtime, tool = _dynamic_workflow_tool()
    original_invoke = adapter.invoke

    async def invoke(*, prompt: str, **kwargs: Any) -> str:
        if prompt == "bad":
            raise RuntimeError("child failed")
        await asyncio.sleep(0.01)
        return await original_invoke(prompt=prompt, **kwargs)

    adapter.invoke = invoke  # type: ignore[method-assign]

    async def scenario() -> None:
        duplicate = await tool.fn(
            code=(
                "import asyncio\n"
                "left, right = await asyncio.gather(\n"
                "    worker(task='same'), worker(task='same')\n"
                ")\n"
                "[left['task_summary'], right['task_summary']]"
            )
        )
        assert duplicate == ["child:same", "child:same"]
        assert [call[0] for call in adapter.invocations] == ["same"]

        recovered = await tool.fn(
            code=(
                "try:\n"
                "    await worker(task='bad')\n"
                "except RuntimeError:\n"
                "    result = await worker(task='good')\n"
                "result['task_summary']"
            )
        )
        assert recovered == "child:good"

    asyncio.run(scenario())


def test_dynamic_workflow_cancellation_unwinds_pending_child() -> None:
    adapter, runtime, tool = _dynamic_workflow_tool()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait_forever(*, prompt: str, **_kwargs: Any) -> str:
        assert prompt == "wait"
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    adapter.invoke = wait_forever  # type: ignore[method-assign]

    async def scenario() -> None:
        workflow = asyncio.create_task(tool.fn(code="await worker(task='wait')"))
        await asyncio.wait_for(started.wait(), timeout=1)
        workflow.cancel()
        with pytest.raises(asyncio.CancelledError):
            await workflow
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        assert runtime._subagent_coordinator.active_runs == 0

    asyncio.run(scenario())


def test_dynamic_workflow_exact_parallel_budget_and_cleanup() -> None:
    adapter, runtime, tool = _dynamic_workflow_tool(max_agents=2)
    original_invoke = adapter.invoke

    async def invoke(*, prompt: str, **kwargs: Any) -> str:
        await asyncio.sleep(0.01)
        return await original_invoke(prompt=prompt, **kwargs)

    adapter.invoke = invoke  # type: ignore[method-assign]

    async def scenario() -> None:
        result = await tool.fn(
            code=(
                "import asyncio\n"
                "results = await asyncio.gather(\n"
                "    worker(task='one'),\n"
                "    worker(task='two'),\n"
                "    worker(task='three'),\n"
                ")\n"
                "results"
            )
        )
        assert result == {
            "ok": False,
            "error": "subagent_budget_exhausted",
            "message": "Subagent invocation limit 2 reached for profile 'manager'",
            "completed_calls": 2,
        }
        assert sorted(call[0] for call in adapter.invocations) == ["one", "two"]
        assert runtime._subagent_coordinator.active_runs == 0

    asyncio.run(scenario())


def test_dynamic_workflow_context_isolated_between_parent_runs() -> None:
    left_adapter, left_runtime, left_tool = _dynamic_workflow_tool()
    right_adapter, right_runtime, right_tool = _dynamic_workflow_tool()

    async def scenario() -> None:
        left, right = await asyncio.gather(
            left_tool.fn(code="result = await worker(task='left')\nresult['task_summary']"),
            right_tool.fn(code="result = await worker(task='right')\nresult['task_summary']"),
        )
        assert left == "child:left"
        assert right == "child:right"
        assert left_runtime._subagent_coordinator.active_runs == 0
        assert right_runtime._subagent_coordinator.active_runs == 0

    asyncio.run(scenario())
    assert [call[0] for call in left_adapter.invocations] == ["left"]
    assert [call[0] for call in right_adapter.invocations] == ["right"]


def test_dynamic_workflow_salvage_cache_respects_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _runtime, tool = _dynamic_workflow_tool()
    monkeypatch.setattr(workflow_module, "MAX_WORKFLOW_SALVAGE_RESULTS", 0)

    async def scenario() -> None:
        code = "result = await worker(task='repeat')\nresult['task_summary']"
        assert await tool.fn(code=code) == "child:repeat"
        assert await tool.fn(code=code) == "child:repeat"

    asyncio.run(scenario())
    assert [call[0] for call in adapter.invocations] == ["repeat", "repeat"]


def test_dynamic_child_tracks_attempt_outcomes_and_interrupts() -> None:
    profile = AgentProfile(name="manager")
    budget = DelegationBudget(owner_profile="manager", limit=5)
    concurrency = DelegationConcurrency(limit=2)

    class Result:
        def as_tool_result(self) -> dict[str, object]:
            return {"task_summary": "done", "final_result": None}

    class Runtime:
        def __init__(
            self,
            *,
            start_error: BaseException | None = None,
            wait_errors: tuple[BaseException, ...] = (),
            status: str = "completed",
        ) -> None:
            self.start_error = start_error
            self.wait_errors = list(wait_errors)
            self.cancelled: list[str] = []
            self._subagent_coordinator = SimpleNamespace(
                runs=SimpleNamespace(get=lambda _run_id: SimpleNamespace(status=status))
            )

        def _start_subagent(self, **_kwargs: object) -> SimpleNamespace:
            if self.start_error is not None:
                raise self.start_error
            return SimpleNamespace(run_id="run-1")

        async def _wait_subagent(self, _run_id: str) -> Result:
            if self.wait_errors:
                raise self.wait_errors.pop(0)
            return Result()

        def _cancel_subagent(self, run_id: str) -> None:
            self.cancelled.append(run_id)

    async def execute(runtime: Runtime) -> dict[str, Any]:
        return await _execute_dynamic_child(
            rt=cast(Any, runtime),
            profile=profile,
            allowed=("worker",),
            definitions={},
            budget=budget,
            concurrency=concurrency,
            subagent="worker",
            task="inspect",
            final_schema=None,
        )

    async def scenario() -> None:
        with delegation_scope("recover") as ledger:
            assert (await execute(Runtime()))["task_summary"] == "done"
        assert ledger.snapshot()[0].status == "completed"

        with delegation_scope("recover") as ledger:
            with pytest.raises(RuntimeError, match="rejected"):
                await execute(Runtime(start_error=ValueError("rejected")))
        assert ledger.snapshot()[0].status == "rejected"

        with delegation_scope("recover") as ledger:
            with pytest.raises(RuntimeError, match="failed"):
                await execute(Runtime(wait_errors=(ValueError("failed"),), status="failed"))
        assert ledger.snapshot()[0].status == "failed"

        with delegation_scope("recover") as ledger:
            with pytest.raises(RuntimeError, match="interrupted"):
                await execute(
                    Runtime(
                        wait_errors=(ValueError("interrupted"),),
                        status="running",
                    )
                )
        assert ledger.snapshot()[0].status == "pending"

        with delegation_scope("recover") as ledger:
            with pytest.raises(asyncio.CancelledError):
                await execute(Runtime(start_error=asyncio.CancelledError()))
        assert ledger.snapshot()[0].status == "cancelled"

        interrupted = Runtime(
            wait_errors=(asyncio.CancelledError(), KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            await execute(interrupted)
        assert interrupted.cancelled == ["run-1"]

        with pytest.raises(KeyboardInterrupt):
            await execute(Runtime(start_error=KeyboardInterrupt()))

        exhausted = SubagentBudgetExceeded("limit reached")
        exhausted.message = "limit reached"
        with pytest.raises(WorkflowAbortError, match="limit reached"):
            await execute(Runtime(start_error=exhausted))

    asyncio.run(scenario())


def test_dynamic_workflow_helpers_cover_names_arguments_and_results() -> None:
    assert _sandbox_names(("class", "subagent_class")) == {
        "subagent_class": "class",
        "subagent_class_2": "subagent_class",
    }
    assert _validate_child_arguments("worker", {"task": "  inspect  "}) == (
        "  inspect  ",
        None,
    )
    with pytest.raises(TypeError, match="missing required"):
        _validate_child_arguments("worker", {})
    with pytest.raises(TypeError, match="unexpected keyword"):
        _validate_child_arguments("worker", {"task": "x", "other": True})
    with pytest.raises(TypeError, match="non-empty string"):
        _validate_child_arguments("worker", {"task": " "})
    with pytest.raises(TypeError, match="object or null"):
        _validate_child_arguments("worker", {"task": "x", "final_schema": []})

    assert _workflow_result(None, "") == {}
    assert _workflow_result(None, "debug\n") == {"output": "debug"}
    assert _workflow_result(7, "debug\n") == {"output": "debug", "result": 7}

    printed_error = _workflow_error(
        "Runtime error",
        SimpleNamespace(display=lambda: "boom"),
        SimpleNamespace(output="before\n"),
    )
    assert "[stdout before error]\nbefore" in printed_error

    completed = [
        _CompletedCall(
            subagent="worker",
            task="inspect",
            result={"task_summary": "x" * 500},
        )
    ]
    assert "[truncated]" in _completed_calls_text(completed)


def test_dynamic_workflow_direct_input_print_and_profile_fallback() -> None:
    _adapter, runtime, tool = _dynamic_workflow_tool()
    assert _profile_description(runtime, "worker", {}) == "Handles one isolated worker task."

    async def scenario() -> None:
        with pytest.raises(WorkflowTypeError, match="non-empty string"):
            await tool.fn(code="")
        assert await tool.fn(code="print('debug')") == {"output": "debug"}
        assert await tool.fn(code="print('debug')\n7") == {
            "output": "debug",
            "result": 7,
        }

    asyncio.run(scenario())


def test_dynamic_workflow_maps_sandbox_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    real_monty = workflow_module.Monty

    class FakeMontyError(Exception):
        def display(self) -> str:
            return str(self)

    class FakeSyntaxError(FakeMontyError):
        pass

    class FakeCrashedError(FakeMontyError):
        pass

    class RaisingMonty:
        def __init__(self, error: BaseException) -> None:
            self.error = error

        def __enter__(self) -> RaisingMonty:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def checkout(self, **_kwargs: object) -> None:
            raise self.error

    _adapter, _runtime, tool = _dynamic_workflow_tool()

    async def assert_failure(
        source_error: BaseException,
        expected_error: type[BaseException],
        message: str,
    ) -> None:
        monkeypatch.setattr(workflow_module, "Monty", lambda: RaisingMonty(source_error))
        with pytest.raises(expected_error, match=message):
            await tool.fn(code="1")

    async def scenario() -> None:
        monkeypatch.setattr(workflow_module, "MontySyntaxError", FakeSyntaxError)
        await assert_failure(FakeSyntaxError("bad syntax"), WorkflowSyntaxError, "bad syntax")

        monkeypatch.setattr(workflow_module, "MontyCrashedError", FakeCrashedError)
        await assert_failure(
            FakeCrashedError("crashed"),
            WorkflowResourceError,
            "resource limits",
        )

        panic_type = type("PanicException", (BaseException,), {})
        await assert_failure(
            panic_type("panic"),
            WorkflowResourceError,
            "stopped unexpectedly",
        )

        monkeypatch.setattr(
            workflow_module,
            "Monty",
            lambda: RaisingMonty(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            await tool.fn(code="1")

        monkeypatch.setattr(workflow_module, "Monty", real_monty)

        def fail_serialization(_result: object, _printed: str) -> object:
            from pydantic_core import PydanticSerializationError

            raise PydanticSerializationError("unsupported workflow result")

        monkeypatch.setattr(workflow_module, "_workflow_result", fail_serialization)
        with pytest.raises(WorkflowRuntimeError, match="not JSON serializable"):
            await tool.fn(code="1")

    asyncio.run(scenario())


def test_workflow_executor_snapshot_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class Complete:
        output = None

    class Lookup:
        def resume(self) -> Complete:
            return Complete()

    class Function:
        def __init__(
            self,
            *,
            name: str,
            os_function: bool = False,
            args: tuple[object, ...] = (),
        ) -> None:
            self.function_name = name
            self.is_os_function = os_function
            self.args = args
            self.kwargs = {"task": "x"}
            self.call_id = 1
            self.resumed: object = None

        def resume_auto(self) -> Complete:
            return Complete()

        def resume(self, value: object) -> Complete:
            self.resumed = value
            return Complete()

    class Future:
        pending_call_ids = (1,)

        def __init__(self) -> None:
            self.results: object = None

        def resume(self, results: object) -> Complete:
            self.results = results
            return Complete()

    monkeypatch.setattr(workflow_executor_module, "MontyComplete", Complete)
    monkeypatch.setattr(workflow_executor_module, "NameLookupSnapshot", Lookup)

    async def dispatch(_name: str, _arguments: dict[str, Any]) -> str:
        return "ok"

    async def scenario() -> None:
        executor = WorkflowExecutor(dispatch=dispatch, valid_names={"worker"})
        assert isinstance(await executor.run(Lookup()), Complete)

        assert isinstance(
            executor._dispatch(Function(name="print", os_function=True)),
            Complete,
        )

        unknown = Function(name="unknown")
        executor._dispatch(unknown)
        assert isinstance(cast(dict[str, Any], unknown.resumed)["exception"], NameError)

        positional = Function(name="worker", args=("x",))
        executor._dispatch(positional)
        assert isinstance(cast(dict[str, Any], positional.resumed)["exception"], TypeError)

        valid = Function(name="worker")
        executor._dispatch(valid)
        future = Future()
        await executor._resolve(cast(Any, future))
        settled = cast(dict[int, Any], future.results)[1]
        assert settled["return_value"] == "ok"

        async def pending() -> None:
            await asyncio.sleep(0)

        raw = pending()
        executor._pending[2] = raw
        assert isinstance(await executor.run(Complete()), Complete)

        async def fail() -> None:
            raise ValueError("child failure")

        failure_executor = WorkflowExecutor(dispatch=dispatch, valid_names={"worker"})
        failure_executor._pending[1] = asyncio.create_task(fail())
        failed_future = Future()
        await failure_executor._resolve(cast(Any, failed_future))
        settled_error = cast(dict[int, Any], failed_future.results)[1]
        sandbox_error = settled_error["exception"]
        assert isinstance(sandbox_error, RuntimeError)
        assert str(sandbox_error) == "child failure"
        assert sandbox_error.__traceback__ is None
        assert sandbox_error.__cause__ is None
        assert sandbox_error.__context__ is None

        async def abort() -> None:
            raise WorkflowAbortError("stop")

        abort_executor = WorkflowExecutor(dispatch=dispatch, valid_names={"worker"})
        abort_executor._pending[1] = asyncio.create_task(abort())
        with pytest.raises(WorkflowAbortError, match="stop"):
            await abort_executor._resolve(cast(Any, Future()))

        class Fatal(BaseException):
            pass

        async def fatal() -> None:
            raise Fatal()

        fatal_executor = WorkflowExecutor(dispatch=dispatch, valid_names={"worker"})
        fatal_executor._pending[1] = asyncio.create_task(fatal())
        with pytest.raises(Fatal):
            await fatal_executor._resolve(cast(Any, Future()))

    asyncio.run(scenario())


def test_delegation_budget_hard_cap() -> None:
    budget = DelegationBudget.for_profile(profile_name="manager", max_agents=100)
    assert budget.owner_profile == "manager"
    assert budget.limit == MAX_SUBAGENT_INVOCATIONS

    anonymous = DelegationBudget.for_profile(profile_name=None, max_agents=None)
    assert anonymous.owner_profile == "<anonymous>"
    assert anonymous.limit == MAX_SUBAGENT_INVOCATIONS

    for invalid in (-1, MAX_SUBAGENT_INVOCATIONS + 1, 1000):
        with pytest.raises(ValueError):
            DelegationBudget.for_profile(profile_name="manager", max_agents=invalid)

    disabled = DelegationBudget.for_profile(profile_name="manager", max_agents=0)
    assert disabled.limit == 0

    runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=RecordingPydanticAdapter(),
        subagent_max_concurrency=1000,
    )
    assert runtime._subagent_coordinator.max_concurrency == MAX_SUBAGENT_CONCURRENCY
    with pytest.raises(ValueError, match="at least 1"):
        compile_program(
            parse_program(_profiles_source()),
            adapter=RecordingPydanticAdapter(),
            subagent_max_concurrency=0,
        )


def test_subagent_tool_contract_exposes_direct_children_and_descriptions() -> None:
    source = """
> profile: hidden:
    ###
    Must not be directly routable.
    ###
    > adapter: pydantic

> profile: calculator:
    ###
    Performs arithmetic.
    ###
    > adapter: pydantic
    > subagent: hidden

> profile: summarizer:
    ###
    Condenses prose.
    ###
    > adapter: pydantic

> profile: manager:
    > adapter: pydantic
    > subagent: calculator
    > subagent: summarizer
"""
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    assert tool.json_schema is not None
    subagent_schema = tool.json_schema["properties"]["subagent"]
    assert subagent_schema["enum"] == ["calculator", "summarizer"]
    assert "hidden" not in subagent_schema["enum"]
    assert tool.description is not None
    assert "- calculator: Performs arithmetic." in tool.description
    assert "- summarizer: Condenses prose." in tool.description
    assert "Must not be directly routable" not in tool.description
    assert "does not inherit the parent conversation" in tool.description
    assert len(tool.description) < 700


def test_delegate_task_runs_text_child_in_isolated_profile() -> None:
    adapter, runtime, tool = _manager_tool()

    result = asyncio.run(tool.fn(subagent="worker", task="Inspect the parser", final_schema=None))

    assert result["subagent"] == "worker"
    assert len(result["run_id"]) == 32
    assert result["task_summary"] == "child:Inspect the parser"
    assert result["final_result"] is None
    assert adapter.invocations == [
        (
            "Inspect the parser",
            f"Work carefully.\n\n{artifact_instructions()}",
            (
                "lookup",
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            ),
            (),
        )
    ]
    assert runtime._subagent_stack.get() == ()
    run = runtime._subagent_coordinator.runs.get(result["run_id"])
    assert run.status == "completed"
    assert run.profile_name == "worker"
    assert run.parent_run_id is None
    assert run.root_run_id == run.run_id
    assert run.depth == 1
    assert run.started_at is not None
    assert run.finished_at is not None


def test_delegate_task_stream_events_link_child_to_active_agent_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, tools = _manager_tools()
    events: list[AgentStreamEvent] = []

    async def invoke(*, prompt: str, **kwargs: Any) -> str:
        del kwargs
        with agent_event_run("pydantic") as resolver:
            if resolver is not None:
                resolver.complete_run(final_text=f"child:{prompt}")
        return f"child:{prompt}"

    monkeypatch.setattr(adapter, "invoke", invoke)

    async def delegate() -> dict[str, Any]:
        with agent_event_run("manager") as resolver:
            result = await tools["delegate_task"].fn(
                subagent="worker",
                task="Inspect the parser",
            )
            if resolver is not None:
                resolver.complete_run(final_text="Manager complete.")
            return result

    with observe_agent_events(events.append):
        result = asyncio.run(delegate())

    started = [
        event
        for event in events
        if isinstance(event, AgentRunStateEvent) and event.state == "started"
    ]
    assert len(started) == 2
    manager, child = started
    assert child.run_id == result["run_id"]
    assert child.parent_run_id == manager.run_id


def test_delegate_task_runs_structured_child_and_validates_result() -> None:
    adapter, runtime, tool = _manager_tool()
    final_schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    result = asyncio.run(
        tool.fn(subagent="worker", task="Build a value", final_schema=final_schema)
    )

    assert result["task_summary"] == "Built the requested value."
    assert result["final_result"] == {"value": 7}
    assert adapter.invocations == [
        (
            "Build a value",
            f"Work carefully.\n\n{artifact_instructions()}",
            (
                "lookup",
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            ),
            (),
        )
    ]
    assert runtime._subagent_stack.get() == ()


def test_continue_subagent_uses_verified_history_and_new_run() -> None:
    _, runtime, tools = _manager_tools()
    adapter = RecordingNeutralChildAdapter()
    runtime._resolve_adapter = lambda _profile: adapter  # type: ignore[method-assign]

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        first = await tools["delegate_task"].fn(
            subagent="worker",
            task="Inspect the parser",
        )
        second = await tools["continue_subagent"].fn(
            run_id=first["run_id"],
            task="Now identify one edge case",
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert second["run_id"] != first["run_id"]
    first_run = runtime._subagent_coordinator.runs.get(first["run_id"])
    second_run = runtime._subagent_coordinator.runs.get(second["run_id"])
    assert second_run.previous_run_id == first_run.run_id
    assert second_run.conversation_id == first_run.conversation_id
    assert "verified prior conversation state" in adapter.child_calls[1][0]
    assert "neutral:Inspect the parser" in adapter.child_calls[1][0]
    with pytest.raises(KediExecutionError, match="latest completed"):
        asyncio.run(
            tools["continue_subagent"].fn(
                run_id=first["run_id"],
                task="Branch from stale state",
            )
        )


def test_continue_subagent_supports_background_lifecycle() -> None:
    _, runtime, tools = _manager_tools()

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        first = await tools["delegate_task"].fn(
            subagent="worker",
            task="Inspect the parser",
        )
        handle = await tools["continue_subagent"].fn(
            run_id=first["run_id"],
            task="Continue in background",
            background=True,
        )
        result = await tools["wait_subagent"].fn(run_id=handle["run_id"])
        return handle, result

    handle, result = asyncio.run(scenario())

    assert handle["status"] == "running"
    assert result["task_summary"] == "child:Continue in background"
    assert runtime._subagent_coordinator.status(handle["run_id"]) == "completed"


def test_continue_subagent_rejects_background_for_blocking_adapter() -> None:
    _adapter, runtime, _tools = _manager_tools()
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tools = {
        tool.name: tool for tool in subagent_tool_specs(runtime, profile, supports_background=False)
    }
    first = asyncio.run(
        tools["delegate_task"].fn(
            subagent="worker",
            task="Inspect the parser",
        )
    )

    with pytest.raises(ValueError, match="does not support background subagents"):
        asyncio.run(
            tools["continue_subagent"].fn(
                run_id=first["run_id"],
                task="Continue in background",
                background=True,
            )
        )


def test_pydantic_subagent_continuation_uses_native_message_history() -> None:
    calls: list[list[object]] = []

    async def respond(messages: list[object], _info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        return ModelResponse(parts=[TextPart(f"turn-{len(calls)}")])

    adapter = PydanticAdapter(FunctionModel(respond))
    runtime = compile_program(parse_program(_profiles_source()), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}

    async def scenario() -> None:
        first = await tools["delegate_task"].fn(subagent="worker", task="First task")
        await tools["continue_subagent"].fn(
            run_id=first["run_id"],
            task="Follow-up task",
        )

    asyncio.run(scenario())

    assert len(calls) == 2
    assert len(calls[1]) > len(calls[0])
    assert any(
        isinstance(part, TextPart) and part.content == "turn-1"
        for message in calls[1]
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


def test_subagent_state_restores_terminal_results_and_continuation(tmp_path: Path) -> None:
    state_path = tmp_path / "subagents.json"
    first_adapter = RecordingPydanticAdapter()
    first_runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=first_adapter,
        subagent_state_path=state_path,
    )
    definition = first_runtime._profile_registry["manager"]
    profile = eval_profile_definition(first_runtime, definition, build_capture_env(first_runtime))
    first_tools = {
        tool.name: tool for tool in first_runtime._resolve_agent_tool_specs(profile, first_adapter)
    }
    first = asyncio.run(
        first_tools["delegate_task"].fn(subagent="worker", task="Persist this result")
    )
    assert state_path.exists()
    assert state_path.stat().st_mode & 0o777 == 0o600

    second_adapter = RecordingPydanticAdapter()
    second_runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=second_adapter,
        subagent_state_path=state_path,
    )
    restored = asyncio.run(second_runtime._wait_subagent(first["run_id"]))
    assert restored.task_summary == "child:Persist this result"
    definition = second_runtime._profile_registry["manager"]
    profile = eval_profile_definition(
        second_runtime,
        definition,
        build_capture_env(second_runtime),
    )
    second_tools = {
        tool.name: tool
        for tool in second_runtime._resolve_agent_tool_specs(profile, second_adapter)
    }
    continued = asyncio.run(
        second_tools["continue_subagent"].fn(
            run_id=first["run_id"],
            task="Continue after restart",
        )
    )
    assert continued["task_summary"] == "child:Continue after restart"


def test_subagent_state_marks_nonterminal_runs_interrupted(tmp_path: Path) -> None:
    state_path = tmp_path / "subagents.json"
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=adapter,
        subagent_state_path=state_path,
    )
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}
    result = asyncio.run(tools["delegate_task"].fn(subagent="worker", task="Initial"))

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["runs"][0]["status"] = "running"
    payload["runs"][0]["finished_at"] = None
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = compile_program(
        parse_program(_profiles_source()),
        adapter=RecordingPydanticAdapter(),
        subagent_state_path=state_path,
    )
    with pytest.raises(KediExecutionError, match="interrupted by a process restart"):
        asyncio.run(restored._wait_subagent(result["run_id"]))
    run = restored._subagent_coordinator.runs.get(result["run_id"])
    assert run.status == "failed"
    assert run.error_type == "InterruptedError"


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not-json", "Invalid subagent state file"),
        ('{"version": 999}', "expected version 1"),
        ('{"version": 1, "runs": {}}', "runs must be an array"),
    ],
)
def test_subagent_state_rejects_invalid_data(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    state_path = tmp_path / "subagents.json"
    state_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        compile_program(
            parse_program(_profiles_source()),
            adapter=RecordingPydanticAdapter(),
            subagent_state_path=state_path,
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("results", [], "results must be an object"),
        ("conversations", [], "conversations must be an object"),
        ("runs", ["invalid"], "run records must be objects"),
    ],
)
def test_subagent_state_rejects_invalid_collection_shapes(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    state_path = tmp_path / "subagents.json"
    payload: dict[str, object] = {
        "version": 1,
        "runs": [],
        "results": {},
        "conversations": {},
    }
    payload[field] = value
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        SubagentStateStore(state_path).load()


def test_subagent_state_rejects_unknown_run_status(tmp_path: Path) -> None:
    state_path = tmp_path / "subagents.json"
    run = SubagentRun(
        run_id="run",
        parent_run_id=None,
        root_run_id="run",
        previous_run_id=None,
        conversation_id="run",
        owner_profile_name="manager",
        profile_name="worker",
        depth=1,
        status="completed",
        created_at=1.0,
    )
    payload = {
        "version": 1,
        "runs": [{**asdict(run), "status": "unknown"}],
        "results": {},
        "conversations": {},
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown subagent status"):
        SubagentStateStore(state_path).load()


def test_subagent_state_rejects_non_serializable_adapter_state(tmp_path: Path) -> None:
    store = SubagentStateStore(tmp_path / "subagents.json")
    conversation = StoredConversation(
        conversation_id="conversation",
        profile_name="worker",
        active_run_id=None,
        turns=[],
        adapter_state={"invalid": object()},
    )

    with pytest.raises(ValueError, match="must be JSON serializable"):
        store.save(runs=(), results={}, conversations={"conversation": conversation})


def test_subagent_state_removes_temporary_file_after_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SubagentStateStore(tmp_path / "subagents.json")

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("kedi.agent_adapter.subagents.persistence.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.save(runs=(), results={}, conversations={})
    assert tuple(tmp_path.iterdir()) == ()


def test_subagent_state_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(_profiles_source()), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    asyncio.run(tool.fn(subagent="worker", task="Do not persist"))

    assert tuple(tmp_path.iterdir()) == ()


def test_subagent_run_registry_rejects_duplicate_restore() -> None:
    registry = SubagentRunRegistry()
    run = registry.create(
        run_id="duplicate",
        parent_run_id=None,
        root_run_id="duplicate",
        profile_name="worker",
        depth=1,
    )

    with pytest.raises(ValueError, match="Duplicate subagent run ID"):
        registry.restore(run)


def test_concurrent_subagent_state_writes_cannot_restore_an_older_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(_profiles_source()), adapter=adapter)
    coordinator = runtime._subagent_coordinator

    class BlockingStateStore:
        def __init__(self) -> None:
            self.first_entered = Event()
            self.second_entered = Event()
            self.release_first = Event()
            self._calls_lock = Lock()
            self._calls = 0
            self.saved_result_ids: list[frozenset[str]] = []

        def save(
            self,
            *,
            runs: tuple[Any, ...],
            results: dict[str, DelegationResult],
            conversations: dict[str, Any],
        ) -> None:
            del runs, conversations
            with self._calls_lock:
                self._calls += 1
                call = self._calls
            if call == 1:
                self.first_entered.set()
                assert self.release_first.wait(timeout=1)
            else:
                self.second_entered.set()
            self.saved_result_ids.append(frozenset(results))

    store = BlockingStateStore()
    monkeypatch.setattr(coordinator, "_state_store", store)
    coordinator._results["older"] = DelegationResult(
        run_id="older",
        subagent="worker",
        task_summary="older",
    )
    first = Thread(target=coordinator._persist_state)
    first.start()
    assert store.first_entered.wait(timeout=1)

    coordinator._results["newer"] = DelegationResult(
        run_id="newer",
        subagent="worker",
        task_summary="newer",
    )
    second = Thread(target=coordinator._persist_state)
    second.start()
    assert not store.second_entered.wait(timeout=0.05)

    store.release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert store.saved_result_ids == [
        frozenset({"older"}),
        frozenset({"older", "newer"}),
    ]


def test_subagent_continuation_is_limited_to_owning_profile() -> None:
    source = (
        _profiles_source()
        + """

> profile: other_manager:
    > adapter: pydantic
    > subagent: worker
"""
    )
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)

    def tools_for(profile_name: str) -> dict[str, Any]:
        definition = runtime._profile_registry[profile_name]
        profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
        return {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}

    first = asyncio.run(
        tools_for("manager")["delegate_task"].fn(subagent="worker", task="Owned task")
    )
    with pytest.raises(KediExecutionError, match="does not own"):
        asyncio.run(
            tools_for("other_manager")["continue_subagent"].fn(
                run_id=first["run_id"],
                task="Cross-owner continuation",
            )
        )


def test_pydantic_native_message_history_survives_restart(tmp_path: Path) -> None:
    calls: list[list[object]] = []

    async def respond(messages: list[object], _info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        return ModelResponse(parts=[TextPart(f"turn-{len(calls)}")])

    state_path = tmp_path / "subagents.json"
    first_adapter = PydanticAdapter(FunctionModel(respond))
    first_runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=first_adapter,
        subagent_state_path=state_path,
    )
    definition = first_runtime._profile_registry["manager"]
    profile = eval_profile_definition(first_runtime, definition, build_capture_env(first_runtime))
    first_tools = {
        tool.name: tool for tool in first_runtime._resolve_agent_tool_specs(profile, first_adapter)
    }
    first = asyncio.run(
        first_tools["delegate_task"].fn(subagent="worker", task="First persisted task")
    )

    second_adapter = PydanticAdapter(FunctionModel(respond))
    second_runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=second_adapter,
        subagent_state_path=state_path,
    )
    definition = second_runtime._profile_registry["manager"]
    profile = eval_profile_definition(
        second_runtime,
        definition,
        build_capture_env(second_runtime),
    )
    second_tools = {
        tool.name: tool
        for tool in second_runtime._resolve_agent_tool_specs(profile, second_adapter)
    }
    asyncio.run(
        second_tools["continue_subagent"].fn(
            run_id=first["run_id"],
            task="Continue persisted task",
        )
    )

    assert len(calls) == 2
    assert any(
        isinstance(part, TextPart) and part.content == "turn-1"
        for message in calls[1]
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


def test_delegate_task_uses_adapter_neutral_child_execution_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runtime, tool = _manager_tool()
    adapter = RecordingNeutralChildAdapter()
    monkeypatch.setattr(runtime, "_resolve_adapter", lambda _profile: adapter)

    raw = asyncio.run(tool.fn(subagent="worker", task="Inspect the parser"))
    structured = asyncio.run(
        tool.fn(
            subagent="worker",
            task="Build a value",
            final_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    )

    assert raw["task_summary"] == "neutral:Inspect the parser"
    assert structured["task_summary"] == "Neutral child completed the task."
    assert structured["final_result"] == {"value": 11}
    assert len(adapter.child_requests) == 2
    assert all(request.profile.name == "worker" for request in adapter.child_requests)
    assert all(
        tuple(tool.name for tool in request.tools)
        == (
            "lookup",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
        for request in adapter.child_requests
    )
    assert all(
        request.instructions == f"Work carefully.\n\n{artifact_instructions()}"
        for request in adapter.child_requests
    )
    assert adapter.child_calls[0][1]["neutral_request_limit"] is not None
    assert (
        adapter.child_calls[0][1]["instructions"] == f"Work carefully.\n\n{artifact_instructions()}"
    )
    structured_schema = adapter.child_calls[1][1]["output_schema"]
    assert structured_schema["required"] == ["task_summary", "final_result"]
    runs = runtime._subagent_coordinator.runs.snapshot()
    assert [run.adapter_shortname for run in runs] == ["neutral", "neutral"]
    assert [run.total_tokens for run in runs] == [3, 5]


def test_adapter_neutral_child_execution_enforces_reported_usage_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RecordingNeutralChildAdapter()

    async def over_limit(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        report_invocation(InvocationUsage(requests=2))
        return "completed"

    monkeypatch.setattr(adapter, "invoke", over_limit)
    runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=adapter,
        subagent_usage_limits=SubagentUsageLimits(request_limit=1),
    )
    monkeypatch.setattr(runtime, "_resolve_adapter", lambda _profile: adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    with pytest.raises(KediExecutionError, match="request limit 1 exceeded"):
        asyncio.run(tool.fn(subagent="worker", task="exceed limit"))

    run = runtime._subagent_coordinator.runs.snapshot()[-1]
    assert run.requests == 2
    assert run.status == "failed"


def test_pydantic_subagent_execution_requires_an_object_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RecordingPydanticAdapter()

    async def invalid_result(**_kwargs: Any) -> str:
        return "not an object"

    monkeypatch.setattr(adapter, "produce", invalid_result)
    with pytest.raises(TypeError, match="must be an object"):
        asyncio.run(
            adapter.produce_subagent(
                template="delegate",
                output_schema={"type": "object"},
            )
        )


def test_delegate_task_without_child_system_has_default_artifact_contract() -> None:
    source = """
> profile: worker:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
"""
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    asyncio.run(tool.fn(subagent="worker", task="Inspect the parser"))

    assert adapter.invocations == [
        (
            "Inspect the parser",
            artifact_instructions(),
            (
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            ),
            (),
        )
    ]


def test_delegate_task_omits_artifact_contract_for_unsupported_child_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = """
> profile: worker:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
"""
    parent_adapter = RecordingPydanticAdapter()
    child_adapter = RecordingNeutralChildAdapter()
    child_adapter.capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_profile_override=True,
        supports_subagents=True,
    )
    runtime = compile_program(parse_program(source), adapter=parent_adapter)
    monkeypatch.setattr(runtime, "_resolve_adapter", lambda _profile: child_adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, parent_adapter)[0]

    asyncio.run(tool.fn(subagent="worker", task="Inspect the parser"))

    assert child_adapter.child_requests[0].instructions is None
    assert child_adapter.child_requests[0].tools == ()
    assert "instructions" not in child_adapter.child_calls[0][1]


def test_pydantic_agent_delegates_and_resumes_parent_run() -> None:
    calls: list[tuple[tuple[str, ...], tuple[tuple[str, object], ...]]] = []

    async def respond(
        messages: list[object],
        info: AgentInfo,
    ) -> ModelResponse:
        tool_names = tuple(tool.name for tool in info.function_tools)
        tool_returns = tuple(
            (part.tool_name, part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        )
        calls.append((tool_names, tool_returns))
        if "delegate_task" not in tool_names:
            return ModelResponse(parts=[TextPart("child completed")])
        if tool_returns:
            return ModelResponse(parts=[TextPart("parent received child result")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "delegate_task",
                    {"subagent": "worker", "task": "Inspect the parser"},
                    tool_call_id="delegate-1",
                )
            ]
        )

    source = """
> profile: worker:
    > adapter: pydantic
    > system: Work carefully.

> profile: manager:
    > adapter: pydantic
    > subagent: worker

> use: manager
[answer] << Delegate this task and report the result.
= <answer>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )

    assert runtime.run_main() == "parent received child result"
    lifecycle_tools = (
        "delegate_task",
        "continue_subagent",
        "wait_subagent",
        "cancel_subagent",
        "subagent_status",
    )
    artifact_tools = (
        "read_artifact",
        "search_artifacts",
        "release_artifact",
        "run_artifact_code",
    )
    assert calls[0] == (lifecycle_tools + artifact_tools, ())
    assert calls[1] == (artifact_tools, ())
    assert calls[2][0] == lifecycle_tools + artifact_tools
    tool_name, tool_result = calls[2][1][0]
    assert tool_name == "delegate_task"
    assert isinstance(tool_result, dict)
    assert tool_result["subagent"] == "worker"
    assert tool_result["task_summary"] == "child completed"
    assert tool_result["final_result"] is None


def test_background_subagent_lifecycle_is_observable_and_repeatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _runtime, tools = _manager_tools()

    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed(*, prompt: str, **kwargs: Any) -> str:
            del kwargs
            started.set()
            await release.wait()
            return f"completed:{prompt}"

        monkeypatch.setattr(adapter, "invoke", delayed)
        with delegation_scope("fail_closed"):
            handle = await tools["delegate_task"].fn(
                subagent="worker",
                task="parallel work",
                background=True,
            )
            await started.wait()
            assert tools["subagent_status"].fn(run_id=handle["run_id"]) == {
                "run_id": handle["run_id"],
                "subagent": "worker",
                "status": "running",
            }
            release.set()
            first = await tools["wait_subagent"].fn(run_id=handle["run_id"])
            second = await tools["wait_subagent"].fn(run_id=handle["run_id"])
            assert first == second
            assert first["task_summary"] == "completed:parallel work"
            assert first["final_result"] is None

    asyncio.run(scenario())


def test_background_subagent_must_be_observed_before_parent_finishes() -> None:
    _adapter, _runtime, tools = _manager_tools()

    async def scenario() -> None:
        with pytest.raises(UnresolvedDelegationError, match="without a successful recovery"):
            with delegation_scope("fail_closed"):
                handle = await tools["delegate_task"].fn(
                    subagent="worker",
                    task="unobserved work",
                    background=True,
                )
                await asyncio.sleep(0)
                assert handle["run_id"]

    asyncio.run(scenario())


def test_background_subagent_cancel_and_timeout_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _runtime, tools = _manager_tools()

    async def scenario() -> None:
        started = asyncio.Event()

        async def blocked(*, prompt: str, **kwargs: Any) -> str:
            del prompt, kwargs
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        monkeypatch.setattr(adapter, "invoke", blocked)
        with delegation_scope("recover") as cancelled:
            handle = await tools["delegate_task"].fn(
                subagent="worker",
                task="cancel work",
                background=True,
            )
            await started.wait()
            response = tools["cancel_subagent"].fn(run_id=handle["run_id"])
            assert response["cancel_requested"] is True
            assert tools["cancel_subagent"].fn(run_id=handle["run_id"])["cancel_requested"] is False
            with pytest.raises(asyncio.CancelledError):
                await tools["wait_subagent"].fn(run_id=handle["run_id"])
        assert cancelled.snapshot()[0].status == "cancelled"

        started.clear()
        with delegation_scope("recover") as timed_out:
            handle = await tools["delegate_task"].fn(
                subagent="worker",
                task="timeout work",
                background=True,
            )
            await started.wait()
            with pytest.raises(TimeoutError):
                await tools["wait_subagent"].fn(run_id=handle["run_id"], timeout=0.001)
            assert tools["cancel_subagent"].fn(run_id=handle["run_id"])["cancel_requested"] is True
            with pytest.raises(asyncio.CancelledError):
                await tools["wait_subagent"].fn(run_id=handle["run_id"])
        assert timed_out.snapshot()[0].status == "cancelled"

    asyncio.run(scenario())


def test_background_subagent_runtime_timeout_starts_with_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=adapter,
        subagent_timeout_seconds=0.005,
    )
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}

    async def scenario() -> None:
        async def blocked(*, prompt: str, **kwargs: Any) -> str:
            del prompt, kwargs
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        monkeypatch.setattr(adapter, "invoke", blocked)
        with delegation_scope("recover") as timed_out:
            handle = await tools["delegate_task"].fn(
                subagent="worker",
                task="automatic timeout",
                background=True,
            )
            await asyncio.sleep(0.02)
            assert tools["subagent_status"].fn(run_id=handle["run_id"])["status"] == "timed_out"
            with pytest.raises(KediExecutionError, match="timed out after 0.005 seconds"):
                await tools["wait_subagent"].fn(run_id=handle["run_id"])
        assert timed_out.snapshot()[0].status == "timed_out"

    asyncio.run(scenario())


def test_background_subagent_lifecycle_rejects_unknown_run_ids() -> None:
    _adapter, _runtime, tools = _manager_tools()

    with pytest.raises(KediExecutionError, match="Unknown subagent run ID"):
        tools["subagent_status"].fn(run_id="missing")
    with pytest.raises(KediExecutionError, match="Unknown subagent run ID"):
        tools["cancel_subagent"].fn(run_id="missing")
    with pytest.raises(KediExecutionError, match="Unknown subagent run ID"):
        asyncio.run(tools["wait_subagent"].fn(run_id="missing"))


def test_subagent_concurrency_limit_is_shared_across_root_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=adapter,
        subagent_max_concurrency=1,
    )
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    first_tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}
    second_tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}

    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked(*, prompt: str, **kwargs: Any) -> str:
            del prompt, kwargs
            started.set()
            await release.wait()
            return "done"

        monkeypatch.setattr(adapter, "invoke", blocked)
        first = await first_tools["delegate_task"].fn(
            subagent="worker",
            task="first root",
            background=True,
        )
        await started.wait()
        assert runtime._subagent_coordinator.active_runs == 1
        with pytest.raises(KediExecutionError, match="concurrency limit 1"):
            await second_tools["delegate_task"].fn(
                subagent="worker",
                task="second root",
                background=True,
            )
        release.set()
        await first_tools["wait_subagent"].fn(run_id=first["run_id"])
        assert runtime._subagent_coordinator.active_runs == 0

    asyncio.run(scenario())


def test_delegate_task_rejects_invalid_requests() -> None:
    _, runtime, tool = _manager_tool()

    with pytest.raises(KediExecutionError, match="not allowed"):
        asyncio.run(tool.fn(subagent="other", task="x"))
    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(tool.fn(subagent="worker", task=" "))

    runtime._profile_registry.pop("worker")
    with pytest.raises(KediExecutionError, match="Unknown subagent"):
        asyncio.run(tool.fn(subagent="worker", task="x"))


def test_max_agents_counts_repeated_failed_and_rejected_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _profiles_source().replace(
        "    > subagent: worker\n",
        "    > subagent: worker\n    > max_agents: 2\n",
    )
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    asyncio.run(tool.fn(subagent="worker", task="first"))
    asyncio.run(tool.fn(subagent="worker", task="second"))
    with pytest.raises(KediExecutionError, match="limit 2 reached.*manager"):
        asyncio.run(tool.fn(subagent="worker", task="third"))
    assert len(runtime._subagent_coordinator.runs.snapshot()) == 2

    failing_source = source.replace("> max_agents: 2", "> max_agents: 1")
    failing_adapter = RecordingPydanticAdapter()
    failing_runtime = compile_program(parse_program(failing_source), adapter=failing_adapter)
    failing_definition = failing_runtime._profile_registry["manager"]
    failing_profile = eval_profile_definition(
        failing_runtime,
        failing_definition,
        build_capture_env(failing_runtime),
    )
    failing_tool = failing_runtime._resolve_agent_tool_specs(
        failing_profile,
        failing_adapter,
    )[0]

    async def fail(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        raise RuntimeError("failed after start")

    monkeypatch.setattr(failing_adapter, "invoke", fail)
    with pytest.raises(KediExecutionError, match="failed after start"):
        asyncio.run(failing_tool.fn(subagent="worker", task="fails"))
    with pytest.raises(KediExecutionError, match="limit 1 reached"):
        asyncio.run(failing_tool.fn(subagent="worker", task="cannot retry"))
    assert len(failing_runtime._subagent_coordinator.runs.snapshot()) == 1

    rejected_adapter = RecordingPydanticAdapter()
    rejected_runtime = compile_program(parse_program(failing_source), adapter=rejected_adapter)
    rejected_definition = rejected_runtime._profile_registry["manager"]
    rejected_profile = eval_profile_definition(
        rejected_runtime,
        rejected_definition,
        build_capture_env(rejected_runtime),
    )
    rejected_tool = rejected_runtime._resolve_agent_tool_specs(
        rejected_profile,
        rejected_adapter,
    )[0]
    with pytest.raises(KediExecutionError, match="not allowed"):
        asyncio.run(rejected_tool.fn(subagent="other", task="rejected"))
    result = asyncio.run(rejected_tool.fn(subagent="worker", task="accepted"))
    assert result["task_summary"] == "child:accepted"


def test_max_agents_reservation_is_atomic_for_parallel_calls() -> None:
    source = _profiles_source().replace(
        "    > subagent: worker\n",
        "    > subagent: worker\n    > max_agents: 1\n",
    )
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    async def race() -> list[object]:
        return await asyncio.gather(
            tool.fn(subagent="worker", task="first"),
            tool.fn(subagent="worker", task="second"),
            return_exceptions=True,
        )

    results = asyncio.run(race())
    assert sum(isinstance(result, dict) for result in results) == 1
    errors = [result for result in results if isinstance(result, KediExecutionError)]
    assert len(errors) == 1
    assert "limit 1 reached" in str(errors[0])
    assert len(runtime._subagent_coordinator.runs.snapshot()) == 1


def test_concurrency_limit_fails_fast_and_releases_success_and_cancel_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=adapter,
        subagent_max_concurrency=1,
    )
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def block(*, prompt: str, **kwargs: Any) -> str:
            del kwargs
            started.set()
            await release.wait()
            return f"done:{prompt}"

        monkeypatch.setattr(adapter, "invoke", block)
        first = asyncio.create_task(tool.fn(subagent="worker", task="first"))
        await started.wait()
        with pytest.raises(KediExecutionError, match="concurrency limit 1 reached"):
            await tool.fn(subagent="worker", task="second")
        release.set()
        assert (await first)["task_summary"] == "done:first"

        monkeypatch.setattr(adapter, "invoke", RecordingPydanticAdapter.invoke.__get__(adapter))
        assert (await tool.fn(subagent="worker", task="after-success"))[
            "task_summary"
        ] == "child:after-success"

        started = asyncio.Event()
        release = asyncio.Event()
        monkeypatch.setattr(adapter, "invoke", block)
        cancelled = asyncio.create_task(tool.fn(subagent="worker", task="cancel"))
        await started.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        active_run = next(
            run for run in runtime._subagent_coordinator.runs.snapshot() if run.status == "running"
        )
        assert runtime._subagent_coordinator.cancel(active_run.run_id) is True
        await asyncio.sleep(0)

        monkeypatch.setattr(adapter, "invoke", RecordingPydanticAdapter.invoke.__get__(adapter))
        assert (await tool.fn(subagent="worker", task="after-cancel"))[
            "task_summary"
        ] == "child:after-cancel"

    asyncio.run(scenario())


def test_parallel_subagent_results_keep_profile_tools_schema_and_run_identity_isolated() -> None:
    source = """
@alpha_tool(value: str) -> str:
    = alpha-<value>

@beta_tool(value: str) -> str:
    = beta-<value>

> profile: alpha:
    > adapter: pydantic
    > use: alpha_tool

> profile: beta:
    > adapter: pydantic
    > use: beta_tool

> profile: manager:
    > adapter: pydantic
    > subagent: alpha
    > subagent: beta
    > max_agents: 4
"""
    completed: list[str] = []
    observed: dict[str, tuple[str | None, tuple[str, ...]]] = {}

    class ParallelAdapter(RecordingPydanticAdapter):
        async def produce(
            self,
            template: str,
            output_schema: dict[str, Any] | None = None,
            output_type: type[Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            del output_schema, output_type, kwargs
            profile = self.active_profile
            observed[template] = (
                profile.name,
                tuple(tool.name for tool in self.scoped_tool_specs),
            )
            delays = {"slow-int": 0.04, "fast-text": 0.005, "beta-object": 0.02}
            await asyncio.sleep(delays.get(template, 0.01))
            if template == "beta-failure":
                raise RuntimeError("isolated beta failure")
            completed.append(template)
            values: dict[str, Any] = {
                "slow-int": 11,
                "fast-text": "fast",
                "beta-object": {"ok": True},
            }
            return {"task_summary": template, "final_result": values[template]}

    adapter = ParallelAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    async def scenario() -> list[object]:
        return await asyncio.gather(
            tool.fn(
                subagent="alpha",
                task="slow-int",
                final_schema={"type": "integer"},
            ),
            tool.fn(
                subagent="alpha",
                task="fast-text",
                final_schema={"type": "string"},
            ),
            tool.fn(
                subagent="beta",
                task="beta-object",
                final_schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            ),
            tool.fn(
                subagent="beta",
                task="beta-failure",
                final_schema={"type": "string"},
            ),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    successful = [result for result in results if isinstance(result, dict)]
    failure = next(result for result in results if isinstance(result, RuntimeError))
    runs = runtime._subagent_coordinator.runs.snapshot()

    assert [result["final_result"] for result in successful] == [
        11,
        "fast",
        {"ok": True},
    ]
    assert "isolated beta failure" in str(failure)
    assert completed == ["fast-text", "beta-object", "slow-int"]
    assert observed == {
        "slow-int": (
            "alpha",
            (
                "alpha_tool",
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            ),
        ),
        "fast-text": (
            "alpha",
            (
                "alpha_tool",
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            ),
        ),
        "beta-object": (
            "beta",
            (
                "beta_tool",
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            ),
        ),
        "beta-failure": (
            "beta",
            (
                "beta_tool",
                "read_artifact",
                "search_artifacts",
                "release_artifact",
                "run_artifact_code",
            ),
        ),
    }
    assert len({result["run_id"] for result in successful}) == 3
    assert [run.profile_name for run in runs] == ["alpha", "alpha", "beta", "beta"]
    assert [run.status for run in runs].count("completed") == 3
    assert [run.status for run in runs].count("failed") == 1


def test_nested_delegation_consumes_parent_and_child_budgets() -> None:
    source = """
> profile: leaf:
    > adapter: pydantic

> profile: worker:
    > adapter: pydantic
    > subagent: leaf
    > max_agents: 1

> profile: manager:
    > adapter: pydantic
    > subagent: worker
    > max_agents: 2
"""

    class NestedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "outer":
                delegate = next(
                    tool for tool in self.scoped_tool_specs if tool.name == "delegate_task"
                )
                child = await delegate.fn(subagent="leaf", task="inner")
                return f"outer:{child['task_summary']}"
            return await super().invoke(prompt=prompt, **kwargs)

    adapter = NestedAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    result = asyncio.run(tool.fn(subagent="worker", task="outer"))
    assert result["task_summary"] == "outer:child:inner"
    runs = runtime._subagent_coordinator.runs.snapshot()
    assert [run.profile_name for run in runs] == ["worker", "leaf"]
    assert runs[1].parent_run_id == runs[0].run_id
    assert runs[1].root_run_id == runs[0].run_id
    assert runs[1].depth == 2
    with pytest.raises(KediExecutionError, match="limit 2 reached.*manager"):
        asyncio.run(tool.fn(subagent="worker", task="another"))


def test_runtime_depth_limit_uses_invocation_stack_and_rejects_before_start() -> None:
    source = """
> profile: leaf:
    > adapter: pydantic

> profile: verifier:
    > adapter: pydantic
    > subagent: leaf

> profile: planner:
    > adapter: pydantic
    > subagent: verifier

> profile: manager:
    > adapter: pydantic
    > subagent: planner
"""

    class DepthAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            next_child = {"planner": "verifier", "verifier": "leaf"}.get(prompt)
            if next_child is not None:
                delegate = next(
                    tool for tool in self.scoped_tool_specs if tool.name == "delegate_task"
                )
                child = await delegate.fn(subagent=next_child, task=next_child)
                return f"{prompt}:{child['task_summary']}"
            return await super().invoke(prompt=prompt, **kwargs)

    def run(max_depth: int) -> tuple[KediRuntime, Any]:
        adapter = DepthAdapter()
        runtime = compile_program(
            parse_program(source),
            adapter=adapter,
            subagent_max_depth=max_depth,
        )
        definition = runtime._profile_registry["manager"]
        profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
        tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]
        return runtime, tool

    boundary_runtime, boundary_tool = run(3)
    result = asyncio.run(boundary_tool.fn(subagent="planner", task="planner"))
    assert result["task_summary"] == "planner:verifier:child:leaf"
    assert [run.depth for run in boundary_runtime._subagent_coordinator.runs.snapshot()] == [
        1,
        2,
        3,
    ]

    limited_runtime, limited_tool = run(2)
    with pytest.raises(
        KediExecutionError,
        match=r"depth limit 2 exceeded: manager -> planner -> verifier -> leaf",
    ):
        asyncio.run(limited_tool.fn(subagent="planner", task="planner"))
    runs = limited_runtime._subagent_coordinator.runs.snapshot()
    assert [run.profile_name for run in runs] == ["planner", "verifier"]
    assert [run.status for run in runs] == ["failed", "failed"]

    assert (
        compile_program(
            parse_program(source),
            adapter=DepthAdapter(),
            subagent_max_depth=50,
        )._subagent_coordinator.max_depth
        == 5
    )
    with pytest.raises(ValueError, match="at least 1"):
        compile_program(
            parse_program(source),
            adapter=DepthAdapter(),
            subagent_max_depth=0,
        )


def test_timeout_and_parent_cancellation_propagate_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=adapter,
        subagent_max_concurrency=1,
        subagent_timeout_seconds=0.01,
    )
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    async def block(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        await asyncio.sleep(60)
        return "unreachable"

    monkeypatch.setattr(adapter, "invoke", block)
    with pytest.raises(KediExecutionError, match="timed out after 0.01 seconds"):
        asyncio.run(tool.fn(subagent="worker", task="timeout"))
    assert runtime._subagent_coordinator.runs.snapshot()[-1].status == "timed_out"

    monkeypatch.setattr(adapter, "invoke", RecordingPydanticAdapter.invoke.__get__(adapter))
    result = asyncio.run(tool.fn(subagent="worker", task="after-timeout"))
    assert result["task_summary"] == "child:after-timeout"

    assert (
        compile_program(
            parse_program(_profiles_source()),
            adapter=RecordingPydanticAdapter(),
            subagent_timeout_seconds=1000,
        )._subagent_coordinator.timeout_seconds
        == 600
    )
    with pytest.raises(ValueError, match="greater than zero"):
        compile_program(
            parse_program(_profiles_source()),
            adapter=RecordingPydanticAdapter(),
            subagent_timeout_seconds=0,
        )


def test_timeout_cancels_active_nested_subtree() -> None:
    source = """
> profile: leaf:
    > adapter: pydantic
> profile: worker:
    > adapter: pydantic
    > subagent: leaf
> profile: manager:
    > adapter: pydantic
    > subagent: worker
"""

    class NestedTimeoutAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "outer":
                delegate = next(
                    tool for tool in self.scoped_tool_specs if tool.name == "delegate_task"
                )
                await delegate.fn(subagent="leaf", task="block")
                return "unreachable"
            await asyncio.sleep(60)
            return "unreachable"

    adapter = NestedTimeoutAdapter()
    runtime = compile_program(
        parse_program(source),
        adapter=adapter,
        subagent_timeout_seconds=0.02,
    )
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    with pytest.raises(KediExecutionError, match="worker.*timed out"):
        asyncio.run(tool.fn(subagent="worker", task="outer"))
    runs = runtime._subagent_coordinator.runs.snapshot()
    assert [run.status for run in runs] == ["timed_out", "cancelled"]


def test_delegate_task_rejects_adapter_without_child_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runtime, tool = _manager_tool()
    monkeypatch.setattr(runtime, "_resolve_adapter", lambda _profile: MockAdapter())

    with pytest.raises(KediExecutionError, match="does not implement Kedi child execution"):
        asyncio.run(tool.fn(subagent="worker", task="Inspect the parser"))


def test_subagent_graph_rejects_unknown_and_cycles() -> None:
    with pytest.raises(KediExecutionError, match="Unknown subagent profile 'missing'"):
        compile_program(
            parse_program("> profile: parent:\n    > subagent: missing\n"),
            adapter=RecordingPydanticAdapter(),
        )

    source = """
> profile: first:
    > subagent: second
> profile: second:
    > subagent: first
"""
    with pytest.raises(
        KediExecutionError,
        match="first -> second -> first",
    ):
        compile_program(parse_program(source), adapter=RecordingPydanticAdapter())


def test_non_pydantic_adapter_rejects_subagent_tools() -> None:
    source = """
> profile: child:
    > model: test
> profile: parent:
    > subagent: child
> use: parent
>> Produce [value].
"""
    runtime = compile_program(parse_program(source), adapter=MockAdapter())
    with pytest.raises(AgentSubagentUnsupported):
        runtime.run_main()


def test_runtime_cycle_guard_restores_stack() -> None:
    _, runtime, tool = _manager_tool()
    token = runtime._subagent_stack.set((("parent-run", "worker"),))
    try:
        with pytest.raises(KediExecutionError, match="Cyclic subagent invocation"):
            asyncio.run(tool.fn(subagent="worker", task="loop"))
    finally:
        runtime._subagent_stack.reset(token)


def test_subagent_run_registry_parent_links_failures_and_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, tool = _manager_tool()
    token = runtime._subagent_stack.set((("root-run", "root"),))

    async def fail(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        raise RuntimeError("child failed")

    monkeypatch.setattr(adapter, "invoke", fail)
    try:
        with pytest.raises(KediExecutionError, match="child failed"):
            asyncio.run(tool.fn(subagent="worker", task="fail"))
    finally:
        runtime._subagent_stack.reset(token)

    run = runtime._subagent_coordinator.runs.snapshot()[-1]
    assert run.status == "failed"
    assert run.parent_run_id == "root-run"
    assert run.root_run_id == "root-run"
    assert run.depth == 2
    assert run.finished_at is not None

    registry = SubagentRunRegistry()
    registry.create(
        run_id="run",
        parent_run_id=None,
        root_run_id="run",
        profile_name="worker",
        depth=1,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        registry.create(
            run_id="run",
            parent_run_id=None,
            root_run_id="run",
            profile_name="worker",
            depth=1,
        )
    with pytest.raises(RuntimeError, match="pending -> completed"):
        registry.transition("run", "completed")
    with pytest.raises(ValueError, match="may record an error"):
        registry.transition("run", "running", error=RuntimeError("not terminal"))

    cancelled_adapter, cancelled_runtime, cancelled_tool = _manager_tool()

    async def cancel(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(cancelled_adapter, "invoke", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled_tool.fn(subagent="worker", task="cancel"))
    assert cancelled_runtime._subagent_coordinator.runs.snapshot()[-1].status == "cancelled"


def test_subagent_failure_metadata_is_sanitized() -> None:
    error = KediExecutionError("API_KEY=very-secret\nBearer also-secret child exploded")
    message = safe_subagent_error_message(error)
    assert message is not None
    assert "very-secret" not in message
    assert "also-secret" not in message
    assert "API_KEY=<redacted>" in message
    assert "Bearer <redacted>" in message


@pytest.mark.parametrize(
    ("failure_policy", "fails_closed"),
    [("fail_closed", True), ("recover", False)],
)
def test_pydantic_parent_failure_policy_controls_unresolved_delegation(
    failure_policy: DelegationFailurePolicy,
    fails_closed: bool,
) -> None:
    parent_calls = 0

    async def respond(
        messages: list[object],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal parent_calls
        if "delegate_task" not in {tool.name for tool in info.function_tools}:
            raise RuntimeError("API_KEY=very-secret child exploded")
        parent_calls += 1
        if parent_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {"subagent": "worker", "task": "fail"},
                        tool_call_id="delegate-failure",
                    )
                ]
            )
        retry_messages = " ".join(
            str(part)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        assert "very-secret" not in retry_messages
        assert "API_KEY=<redacted>" in retry_messages
        return ModelResponse(parts=[TextPart("parent recovered")])

    source = """
> profile: worker:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
> use: manager
[answer] << Delegate once, then recover if the child fails.
= <answer>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(
            FunctionModel(respond),
            subagent_failure_policy=failure_policy,
        ),
    )

    if fails_closed:
        with pytest.raises(
            KediExecutionError,
            match="Subagent delegation failed without a successful recovery",
        ):
            runtime.run_main()
    else:
        assert runtime.run_main() == "parent recovered"
    run = runtime._subagent_coordinator.runs.snapshot()[0]
    assert run.status == "failed"
    assert run.error_type == "KediExecutionError"
    assert run.error_message is not None
    assert "very-secret" not in run.error_message


def test_pydantic_parent_can_recover_with_successful_replacement_delegation() -> None:
    parent_calls = 0
    child_calls = 0

    async def respond(
        _messages: list[object],
        info: AgentInfo,
    ) -> ModelResponse:
        nonlocal child_calls, parent_calls
        if "delegate_task" not in {tool.name for tool in info.function_tools}:
            child_calls += 1
            if child_calls == 1:
                raise RuntimeError("first child attempt failed")
            return ModelResponse(parts=[TextPart("verified child result")])

        parent_calls += 1
        if parent_calls <= 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "delegate_task",
                        {"subagent": "worker", "task": "perform the task"},
                        tool_call_id=f"delegate-{parent_calls}",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("parent used verified recovery")])

    source = """
> profile: worker:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
    > max_agents: 2
> use: manager
[answer] << Delegate the task and retry it once if the first child fails.
= <answer>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(FunctionModel(respond)),
    )

    assert runtime.run_main() == "parent used verified recovery"
    assert [run.status for run in runtime._subagent_coordinator.runs.snapshot()] == [
        "failed",
        "completed",
    ]


def test_delegation_ledger_requires_sequential_same_task_recovery() -> None:
    with delegation_scope("fail_closed") as recovered:
        with track_delegation_attempt(
            requested_profile="worker",
            task="inspect parser",
        ) as failed:
            assert failed is not None
            failed.fail(status="rejected", error=RuntimeError("unavailable"))
        with track_delegation_attempt(
            requested_profile="backup",
            task="inspect parser",
        ) as replacement:
            assert replacement is not None
            replacement.complete()

    attempts = recovered.snapshot()
    assert attempts[0].replacement_attempt_id == attempts[1].attempt_id
    assert recovered.unresolved() == ()

    with pytest.raises(UnresolvedDelegationError) as unrelated:
        with delegation_scope("fail_closed"):
            with track_delegation_attempt(
                requested_profile="worker",
                task="inspect parser",
            ) as failed:
                assert failed is not None
                failed.fail(status="failed", error=RuntimeError("provider failed"))
            with track_delegation_attempt(
                requested_profile="backup",
                task="write documentation",
            ) as different_task:
                assert different_task is not None
                different_task.complete()
    assert unrelated.value.attempts[0].requested_profile == "worker"

    with pytest.raises(UnresolvedDelegationError):
        with delegation_scope("fail_closed"):
            with track_delegation_attempt(
                requested_profile="worker",
                task="inspect parser",
            ) as failed:
                with track_delegation_attempt(
                    requested_profile="backup",
                    task="inspect parser",
                ) as parallel_success:
                    assert failed is not None
                    assert parallel_success is not None
                    failed.fail(status="failed", error=RuntimeError("provider failed"))
                    parallel_success.complete()


def test_delegation_attempt_guards_reject_invalid_state_changes() -> None:
    with pytest.raises(ValueError, match="Unknown subagent failure policy"):
        with delegation_scope("invalid"):  # type: ignore[arg-type]
            raise AssertionError("unreachable")

    with delegation_scope("recover") as ledger:
        with track_delegation_attempt(
            requested_profile="worker",
            task="inspect parser",
        ) as attempt:
            assert attempt is not None
            attempt.complete()
            with pytest.raises(RuntimeError, match="already completed"):
                attempt.complete()

        pending = ledger.begin(requested_profile="worker", task="invalid status")
        with pytest.raises(ValueError, match="Invalid delegation failure status"):
            ledger.fail(
                pending.attempt_id,
                status="completed",
                error=RuntimeError("invalid"),
            )

    with track_delegation_attempt(
        requested_profile="worker",
        task="outside invocation",
    ) as inactive:
        assert inactive is None


def test_pydantic_adapter_rejects_unknown_subagent_failure_policy() -> None:
    with pytest.raises(ValueError, match="Unknown subagent failure policy"):
        PydanticAdapter(
            TestModel(),
            subagent_failure_policy="invalid",  # type: ignore[arg-type]
        )


def test_delegation_ledger_tracks_runtime_terminal_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, tool = _manager_tool()

    with delegation_scope("recover") as rejected:
        with pytest.raises(KediExecutionError, match="not allowed"):
            asyncio.run(tool.fn(subagent="other", task="inspect parser"))
    assert rejected.snapshot()[0].status == "rejected"
    assert rejected.snapshot()[0].run_id is None

    async def fail(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        raise RuntimeError("provider failed")

    monkeypatch.setattr(adapter, "invoke", fail)
    with delegation_scope("recover") as failed:
        with pytest.raises(KediExecutionError, match="provider failed"):
            asyncio.run(tool.fn(subagent="worker", task="inspect parser"))
    failed_attempt = failed.snapshot()[0]
    assert failed_attempt.status == "failed"
    assert failed_attempt.run_id is not None

    async def invalid_structured_result(
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del template, output_schema, output_type, kwargs
        return {"task_summary": "invalid", "final_result": "not-an-integer"}

    monkeypatch.setattr(adapter, "produce", invalid_structured_result)
    with delegation_scope("recover") as invalid:
        with pytest.raises(KediExecutionError):
            asyncio.run(
                tool.fn(
                    subagent="worker",
                    task="build integer",
                    final_schema={"type": "integer"},
                )
            )
    assert invalid.snapshot()[0].status == "failed"

    timeout_adapter = RecordingPydanticAdapter()
    timeout_runtime = compile_program(
        parse_program(_profiles_source()),
        adapter=timeout_adapter,
        subagent_timeout_seconds=0.001,
    )
    timeout_definition = timeout_runtime._profile_registry["manager"]
    timeout_profile = eval_profile_definition(
        timeout_runtime,
        timeout_definition,
        build_capture_env(timeout_runtime),
    )
    timeout_tool = timeout_runtime._resolve_agent_tool_specs(
        timeout_profile,
        timeout_adapter,
    )[0]

    async def wait_forever(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(timeout_adapter, "invoke", wait_forever)
    with delegation_scope("recover") as timed_out:
        with pytest.raises(KediExecutionError, match="timed out"):
            asyncio.run(timeout_tool.fn(subagent="worker", task="wait"))
    assert timed_out.snapshot()[0].status == "timed_out"

    cancel_adapter, _cancel_runtime, cancel_tool = _manager_tool()
    cancel_definition = _cancel_runtime._profile_registry["manager"]
    cancel_profile = eval_profile_definition(
        _cancel_runtime,
        cancel_definition,
        build_capture_env(_cancel_runtime),
    )
    cancel_tools = {
        tool.name: tool
        for tool in _cancel_runtime._resolve_agent_tool_specs(cancel_profile, cancel_adapter)
    }

    async def cancel_scenario() -> Any:
        started = asyncio.Event()

        async def block(*, prompt: str, **kwargs: Any) -> str:
            del prompt, kwargs
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        monkeypatch.setattr(cancel_adapter, "invoke", block)
        with delegation_scope("recover") as cancelled:
            pending = asyncio.create_task(cancel_tool.fn(subagent="worker", task="cancel"))
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            active_run = next(
                run
                for run in _cancel_runtime._subagent_coordinator.runs.snapshot()
                if run.status == "running"
            )
            assert _cancel_runtime._subagent_coordinator.cancel(active_run.run_id) is True
            with pytest.raises(asyncio.CancelledError):
                await cancel_tools["wait_subagent"].fn(run_id=active_run.run_id)
        return cancelled

    cancelled = asyncio.run(cancel_scenario())
    assert cancelled.snapshot()[0].status == "cancelled"


class _ApprovalCall:
    tool_name = "write_file"
    tool_call_id = "call-1"

    def __init__(self, arguments: dict[str, Any]) -> None:
        self._arguments = arguments

    def args_as_dict(self) -> dict[str, Any]:
        return dict(self._arguments)


class _ApprovalRequests:
    def __init__(self, arguments: dict[str, Any]) -> None:
        self.approvals = [_ApprovalCall(arguments)]

    def build_results(self, *, approvals: dict[str, Any]) -> dict[str, Any]:
        return approvals


def _native_approval_result(
    *,
    parent: ApprovalPolicy | None,
    child: ApprovalPolicy | None,
    arguments: dict[str, Any] | None = None,
) -> Any:
    adapter = PydanticAdapter(TestModel())
    tool = ToolSpec(
        fn=lambda **kwargs: kwargs,
        name="write_file",
        risk="mutating",
        risk_resolver=lambda values: (
            "sensitive" if Path(str(values["path"])).name == ".env" else "mutating"
        ),
    )
    profile = AgentProfile(approval=child)
    with (
        adapter.tool_scope((tool,)),
        adapter.profile_override(profile),
        adapter.approval_ceiling((parent,)),
    ):
        results = asyncio.run(
            adapter._handle_deferred_tool_calls(
                None,
                _ApprovalRequests(arguments or {"path": "notes.txt"}),
            )
        )
    return results["call-1"]


def test_parent_approval_ceiling_cannot_be_widened_by_child() -> None:
    denied_by_parent = _native_approval_result(
        parent=ApprovalPolicy.deny(),
        child=ApprovalPolicy.allow(),
    )
    denied_by_child = _native_approval_result(
        parent=ApprovalPolicy.allow(),
        child=ApprovalPolicy.deny(),
    )

    assert denied_by_parent.kind == "tool-denied"
    assert "static deny policy" in denied_by_parent.message
    assert denied_by_child.kind == "tool-denied"
    assert "static deny policy" in denied_by_child.message
    inherited_allow = _native_approval_result(
        parent=ApprovalPolicy.allow(),
        child=None,
    )
    assert inherited_allow.kind == "tool-approved"


def test_parent_approval_edit_is_rechecked_by_child_policy() -> None:
    seen: list[tuple[str, str]] = []

    def redirect(_request: Any) -> ApprovalDecision:
        return ApprovalDecision.edit({"path": ".env"})

    def child_policy(request: Any) -> ApprovalDecision:
        seen.append((request.arguments["path"], request.risk))
        return (
            ApprovalDecision.deny(reason="secret path")
            if request.risk == "sensitive"
            else ApprovalDecision.allow()
        )

    result = _native_approval_result(
        parent=ApprovalPolicy.dynamic(redirect),
        child=ApprovalPolicy.dynamic(child_policy),
    )

    assert result.kind == "tool-denied"
    assert result.message == "Tool 'write_file' was denied: secret path"
    assert seen == [(".env", "sensitive")]


def test_subagent_safety_ceiling_restricts_cwd_sandbox_and_nested_profiles(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = AgentProfile(
        approval=ApprovalPolicy.allow(),
        settings={"cwd": str(workspace), "sandbox": "workspace-write"},
    )
    ceiling = SubagentSafetyCeiling.from_parent(parent, default_cwd=tmp_path)

    child, child_ceiling = ceiling.constrain_child(
        AgentProfile(
            approval=ApprovalPolicy.deny(),
            settings={"cwd": "nested", "sandbox": "read-only"},
        )
    )
    grandchild, grandchild_ceiling = child_ceiling.constrain_child(AgentProfile.empty())

    assert child.settings["cwd"] == str(workspace / "nested")
    assert child.settings["sandbox"] == "read-only"
    assert grandchild.settings["sandbox"] == "read-only"
    assert grandchild_ceiling.approval_policies == (
        ApprovalPolicy.allow(),
        ApprovalPolicy.deny(),
        None,
    )

    with pytest.raises(KediExecutionError, match="escapes parent cwd boundary"):
        ceiling.constrain_child(AgentProfile(settings={"cwd": str(tmp_path)}))
    with pytest.raises(KediExecutionError, match="broader than parent sandbox"):
        child_ceiling.constrain_child(AgentProfile(settings={"sandbox": "danger-full-access"}))
    with pytest.raises(TypeError, match="cwd must be str or Path"):
        ceiling.constrain_child(AgentProfile(settings={"cwd": 42}))
    with pytest.raises(KediExecutionError, match="Subagent sandbox must be"):
        ceiling.constrain_child(AgentProfile(settings={"sandbox": "unconfined"}))


def test_subagent_internal_guards_are_covered() -> None:
    async def fail() -> None:
        raise KediExecutionError("ordinary tool failure")

    validated = _validated_tool_function(ToolSpec(fn=fail, name="ordinary"))
    with pytest.raises(KediExecutionError, match="ordinary tool failure"):
        asyncio.run(validated())

    _, runtime, _ = _manager_tool()
    concurrency = DelegationConcurrency.for_limit(1)
    with pytest.raises(RuntimeError, match="slot underflow"):
        runtime._subagent_coordinator._release_slot(concurrency)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        (SubagentUsageLimits(request_limit=1), "request_limit"),
        (SubagentUsageLimits(request_limit=5, tool_calls_limit=1), "tool_calls_limit"),
    ],
)
def test_subagent_pydantic_usage_limits_stop_child_loops_and_cleanup(
    limits: SubagentUsageLimits,
    message: str,
) -> None:
    async def repeat_tool(_messages: list[object], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("loop", {}, tool_call_id="repeat-tool")])

    source = """
@loop() -> str:
    = looped

> profile: worker:
    > adapter: pydantic
    > approval: allow
    > use: loop

> profile: manager:
    > adapter: pydantic
    > approval: allow
    > subagent: worker
"""
    adapter = PydanticAdapter(FunctionModel(repeat_tool))
    runtime = compile_program(
        parse_program(source),
        adapter=adapter,
        subagent_usage_limits=limits,
    )
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    for _ in range(2):
        with pytest.raises(KediExecutionError, match=message):
            asyncio.run(tool.fn(subagent="worker", task="keep calling"))

    assert [run.status for run in runtime._subagent_coordinator.runs.snapshot()] == [
        "failed",
        "failed",
    ]
    assert all(run.requests >= 1 for run in runtime._subagent_coordinator.runs.snapshot())
    assert all(
        run.adapter_shortname == "pydantic" for run in runtime._subagent_coordinator.runs.snapshot()
    )
    assert runtime._subagent_coordinator._concurrency.get() is None


def test_subagent_records_native_pydantic_usage_and_invocation_metadata() -> None:
    source = """
> profile: worker:
    > adapter: pydantic

> profile: manager:
    > adapter: pydantic
    > subagent: worker
"""
    adapter = PydanticAdapter(TestModel(custom_output_text="child output", call_tools=[]))
    runtime = compile_program(parse_program(source), adapter=adapter)
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]

    result = asyncio.run(tool.fn(subagent="worker", task="reply"))
    run = runtime._subagent_coordinator.runs.get(result["run_id"])

    assert result["task_summary"] == "child output"
    assert run.adapter_shortname == "pydantic"
    assert run.model == "test"
    assert run.structured_output is False
    assert run.requests == 1
    assert run.tool_calls == 0
    assert run.input_tokens > 0
    assert run.output_tokens > 0
    assert run.total_tokens == run.input_tokens + run.output_tokens


def test_pydantic_invocation_observer_accepts_existing_usage_and_model_variants() -> None:
    from pydantic_ai.usage import RunUsage

    adapter = PydanticAdapter(TestModel(custom_output_text="done"))
    usage = RunUsage()
    observed: list[InvocationUsage] = []

    async def scenario() -> None:
        with observe_invocations(observed.append):
            result = await adapter._run_agent(adapter, "reply", {"usage": usage})
        assert result.output == "done"

    asyncio.run(scenario())
    assert observed[0].requests == 1
    report_invocation(InvocationUsage())

    named = PydanticAdapter("test")
    assert named.telemetry_model_name() == "test"
    named._agent_model_arg = None
    assert named.telemetry_model_name() is None


def test_subagent_allows_an_adapter_without_a_telemetry_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, tool = _manager_tool()
    monkeypatch.setattr(adapter, "telemetry_model_name", lambda: None)

    result = asyncio.run(tool.fn(subagent="worker", task="reply"))
    run = runtime._subagent_coordinator.runs.get(result["run_id"])

    assert run.status == "completed"
    assert run.model is None


def test_subagent_start_cleans_up_when_task_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runtime, _ = _manager_tool()
    coordinator = runtime._subagent_coordinator
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    concurrency = coordinator.new_concurrency()

    def fail_create_task(
        coroutine: Coroutine[Any, Any, Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        del name
        coroutine.close()
        raise RuntimeError("task creation failed")

    monkeypatch.setattr(asyncio, "create_task", fail_create_task)
    with pytest.raises(RuntimeError, match="task creation failed"):
        coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="fail",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=DelegationBudget.for_profile(
                profile_name="manager",
                max_agents=1,
            ),
            parent_concurrency=concurrency,
        )

    assert concurrency.active == 0
    assert coordinator.runs.snapshot()[-1].status == "failed"
    assert isinstance(coordinator.events.snapshot()[-1], SubagentFailed)


def test_subagent_start_falls_back_to_global_definition_after_private_lookup_miss() -> None:
    _, runtime, _ = _manager_tool()
    coordinator = runtime._subagent_coordinator
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))

    async def scenario() -> None:
        handle = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="reply",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=DelegationBudget.for_profile(profile_name="manager", max_agents=1),
            parent_concurrency=coordinator.new_concurrency(),
            subagent_definitions={},
        )
        result = await coordinator.wait(handle.run_id)
        assert result.task_summary == "child:reply"

    asyncio.run(scenario())


def test_subagent_start_uses_private_definition_without_global_lookup() -> None:
    _, runtime, _ = _manager_tool()
    coordinator = runtime._subagent_coordinator
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))

    async def scenario() -> None:
        handle = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="reply",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=DelegationBudget.for_profile(profile_name="manager", max_agents=1),
            parent_concurrency=coordinator.new_concurrency(),
            subagent_definitions={"worker": runtime._profile_registry["worker"]},
        )
        result = await coordinator.wait(handle.run_id)
        assert result.task_summary == "child:reply"

    asyncio.run(scenario())


def test_subagent_start_rejects_unknown_global_definition_after_private_lookup_miss() -> None:
    _, runtime, _ = _manager_tool()
    coordinator = runtime._subagent_coordinator
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))

    with pytest.raises(KediExecutionError, match="Unknown subagent profile: 'missing'"):
        coordinator.start(
            allowed_subagents=("missing",),
            subagent="missing",
            task="reply",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=DelegationBudget.for_profile(profile_name="manager", max_agents=1),
            parent_concurrency=coordinator.new_concurrency(),
            subagent_definitions={},
        )


def test_subagent_handle_wait_cancel_timeout_and_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, _ = _manager_tool()
    coordinator = runtime._subagent_coordinator
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    parent_budget = DelegationBudget.for_profile(profile_name="manager", max_agents=5)
    parent_concurrency = coordinator.new_concurrency()

    async def scenario() -> None:
        first = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="first",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        result = await first.wait()
        assert first.status == "completed"
        assert await first.wait() == result
        assert first.cancel() is False

        started = asyncio.Event()

        async def block(*, prompt: str, **kwargs: Any) -> str:
            del prompt, kwargs
            started.set()
            await asyncio.sleep(60)
            return "unreachable"

        monkeypatch.setattr(adapter, "invoke", block)
        cancelled = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="cancel",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        await started.wait()
        assert cancelled.cancel() is True
        assert cancelled.cancel() is False
        with pytest.raises(asyncio.CancelledError):
            await cancelled.wait()
        assert cancelled.status == "cancelled"

        started.clear()
        timed_out = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="timeout",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        await started.wait()
        with pytest.raises(TimeoutError):
            await timed_out.wait(timeout=0.01)
        assert timed_out.status == "running"
        assert timed_out.cancel() is True
        with pytest.raises(asyncio.CancelledError):
            await timed_out.wait()
        assert timed_out.status == "cancelled"

    asyncio.run(scenario())
    events = coordinator.events.snapshot()
    assert [type(event) for event in events] == [
        SubagentStarted,
        SubagentCompleted,
        SubagentStarted,
        SubagentCancelled,
        SubagentStarted,
        SubagentCancelled,
    ]
    assert [event.run.run_id for event in events[::2]] == [
        event.run.run_id for event in events[1::2]
    ]


def test_subagent_coordinator_bounds_terminal_run_retention() -> None:
    _adapter, runtime, _ = _manager_tool()
    coordinator = runtime._subagent_coordinator
    coordinator.max_retained_runs = 2
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    parent_budget = DelegationBudget.for_profile(profile_name="manager", max_agents=3)
    parent_concurrency = coordinator.new_concurrency()

    async def scenario() -> str:
        first_run_id = ""
        for index in range(3):
            handle = coordinator.start(
                allowed_subagents=("worker",),
                subagent="worker",
                task=f"task-{index}",
                final_schema=None,
                parent_profile=parent_profile,
                parent_budget=parent_budget,
                parent_concurrency=parent_concurrency,
            )
            first_run_id = first_run_id or handle.run_id
            await handle.wait()
            await asyncio.sleep(0)
        return first_run_id

    first_run_id = asyncio.run(scenario())
    assert len(coordinator.runs.snapshot()) == 2
    with pytest.raises(KediExecutionError, match="Unknown subagent run ID"):
        coordinator.status(first_run_id)


def test_subagent_retention_keeps_conversation_for_latest_continuation() -> None:
    _adapter, runtime, tools = _manager_tools()
    coordinator = runtime._subagent_coordinator
    coordinator.max_retained_runs = 1

    async def scenario() -> tuple[str, str]:
        first = await tools["delegate_task"].fn(
            subagent="worker",
            task="first turn",
        )
        second = await tools["continue_subagent"].fn(
            run_id=first["run_id"],
            task="second turn",
        )
        await asyncio.sleep(0)
        return first["run_id"], second["run_id"]

    first_run_id, second_run_id = asyncio.run(scenario())

    with pytest.raises(KediExecutionError, match="Unknown subagent run ID"):
        coordinator.status(first_run_id)
    second = coordinator.runs.get(second_run_id)
    assert second.conversation_id in coordinator._conversations


def test_subagent_lifecycle_rejects_invalid_configuration_and_background_mode() -> None:
    _adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    blocking_delegate = subagent_tool_specs(
        runtime,
        profile,
        supports_background=False,
    )[0]

    with pytest.raises(ValueError, match="does not support background subagents"):
        asyncio.run(
            blocking_delegate.fn(
                subagent="worker",
                task="background work",
                background=True,
            )
        )
    with pytest.raises(ValueError, match="run retention must be at least 1"):
        type(coordinator)(coordinator._deps, max_retained_runs=0)
    with pytest.raises(ValueError, match="event retention must be at least 1"):
        SubagentEventLog(max_events=0)


def test_subagent_start_releases_slot_when_lifecycle_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator
    definition = runtime._profile_registry["manager"]
    profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    budget = DelegationBudget.for_profile(profile_name="manager", max_agents=1)
    concurrency = coordinator.new_concurrency()

    def fail_activation(**_kwargs: Any) -> None:
        raise RuntimeError("activation failed")

    monkeypatch.setattr(coordinator, "_activate_conversation", fail_activation)

    with pytest.raises(RuntimeError, match="activation failed"):
        coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="fail during setup",
            final_schema=None,
            parent_profile=profile,
            parent_budget=budget,
            parent_concurrency=concurrency,
        )
    assert coordinator.active_runs == 0
    assert concurrency.active == 0


def test_subagent_conversation_and_terminal_state_guards() -> None:
    _adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator

    with pytest.raises(KediExecutionError, match="Unknown subagent conversation"):
        coordinator._activate_conversation(
            conversation_id="missing",
            profile_name="worker",
            run_id="next",
            previous_run_id="previous",
        )

    coordinator._conversations["profile-change"] = StoredConversation(
        conversation_id="profile-change",
        profile_name="worker",
        active_run_id=None,
        turns=[],
        adapter_state={},
    )
    with pytest.raises(KediExecutionError, match="cannot change child profiles"):
        coordinator._activate_conversation(
            conversation_id="profile-change",
            profile_name="other",
            run_id="next",
            previous_run_id=None,
        )

    coordinator._conversations["active"] = StoredConversation(
        conversation_id="active",
        profile_name="worker",
        active_run_id="running",
        turns=[],
        adapter_state={},
    )
    with pytest.raises(KediExecutionError, match="already has an active run"):
        coordinator._activate_conversation(
            conversation_id="active",
            profile_name="worker",
            run_id="next",
            previous_run_id=None,
        )

    coordinator._conversations["stale"] = StoredConversation(
        conversation_id="stale",
        profile_name="worker",
        active_run_id=None,
        turns=[],
        adapter_state={},
    )
    with pytest.raises(KediExecutionError, match="latest completed"):
        coordinator._activate_conversation(
            conversation_id="stale",
            profile_name="worker",
            run_id="next",
            previous_run_id="previous",
        )

    turns = [
        SubagentTurn(
            run_id=f"run-{index}",
            task="task",
            task_summary="summary",
            final_result=None,
        )
        for index in range(MAX_SUBAGENT_CONVERSATION_TURNS)
    ]
    coordinator._conversations["full"] = StoredConversation(
        conversation_id="full",
        profile_name="worker",
        active_run_id=None,
        turns=turns,
        adapter_state={},
    )
    with pytest.raises(KediExecutionError, match="turn limit"):
        coordinator._activate_conversation(
            conversation_id="full",
            profile_name="worker",
            run_id="next",
            previous_run_id=turns[-1].run_id,
        )

    coordinator._deactivate_conversation("missing", "run")
    coordinator._deactivate_conversation("active", "different")
    assert coordinator._conversations["active"].active_run_id == "running"
    coordinator._deactivate_conversation("active", "running")
    assert coordinator._conversations["active"].active_run_id is None


def test_subagent_coordinator_rejects_invalid_restored_lifecycle_state() -> None:
    _adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator
    registry = coordinator.runs

    with pytest.raises(KediExecutionError, match="Unknown subagent run ID"):
        coordinator._task_for("missing")
    with pytest.raises(KediExecutionError, match="Unknown subagent run ID"):
        coordinator._continuable_run(
            "missing",
            allowed_subagents=("worker",),
            owner_profile_name="manager",
        )

    pending = registry.create(
        run_id="pending",
        parent_run_id=None,
        root_run_id="pending",
        profile_name="worker",
        depth=1,
        owner_profile_name="manager",
    )
    assert coordinator.cancel(pending.run_id) is False
    with pytest.raises(KediExecutionError, match="Only completed"):
        coordinator._continuable_run(
            pending.run_id,
            allowed_subagents=("worker",),
            owner_profile_name="manager",
        )
    with pytest.raises(KediExecutionError, match="still pending"):
        coordinator._terminal_result(pending.run_id)

    completed = registry.create(
        run_id="completed",
        parent_run_id=None,
        root_run_id="completed",
        profile_name="worker",
        depth=1,
        owner_profile_name="manager",
    )
    registry.transition(completed.run_id, "running")
    registry.transition(completed.run_id, "completed")
    with pytest.raises(KediExecutionError, match="Unknown subagent conversation"):
        coordinator._continuable_run(
            completed.run_id,
            allowed_subagents=("worker",),
            owner_profile_name="manager",
        )
    coordinator._conversations[completed.conversation_id] = StoredConversation(
        conversation_id=completed.conversation_id,
        profile_name="worker",
        active_run_id="active",
        turns=[
            SubagentTurn(
                run_id=completed.run_id,
                task="task",
                task_summary="summary",
                final_result=None,
            )
        ],
        adapter_state={},
    )
    with pytest.raises(KediExecutionError, match="already running"):
        coordinator._continuable_run(
            completed.run_id,
            allowed_subagents=("worker",),
            owner_profile_name="manager",
        )
    with pytest.raises(KediExecutionError, match="has no result"):
        coordinator._terminal_result(completed.run_id)


def test_subagent_wait_cancellation_and_completed_timeout_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator
    coordinator.timeout_seconds = None
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    parent_budget = DelegationBudget.for_profile(profile_name="manager", max_agents=3)
    parent_concurrency = coordinator.new_concurrency()

    async def scenario() -> None:
        completed = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="complete",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        assert (await completed.wait(timeout=1)).task_summary == "child:complete"

        started = asyncio.Event()

        async def block(*, prompt: str, **kwargs: Any) -> str:
            del prompt, kwargs
            started.set()
            await asyncio.sleep(60)
            return "unreachable"

        monkeypatch.setattr(adapter, "invoke", block)
        cancelled = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="cancel waiter",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        waiter = asyncio.create_task(cancelled.wait(timeout=10))
        await started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert cancelled.status == "running"
        assert cancelled.cancel() is True
        with pytest.raises(asyncio.CancelledError):
            await cancelled.wait()
        assert cancelled.status == "cancelled"

    asyncio.run(scenario())


def test_subagent_task_cleanup_handles_races_and_active_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator
    coordinator.timeout_seconds = None
    coordinator.max_retained_runs = 1
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    parent_budget = DelegationBudget.for_profile(profile_name="manager", max_agents=3)
    parent_concurrency = coordinator.new_concurrency()

    async def scenario() -> None:
        releases = {
            "first": asyncio.Event(),
            "second": asyncio.Event(),
        }

        async def block(*, prompt: str, **kwargs: Any) -> str:
            del kwargs
            await releases[prompt].wait()
            return prompt

        monkeypatch.setattr(adapter, "invoke", block)
        first = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="first",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        second = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="second",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        await asyncio.sleep(0)
        releases["second"].set()
        await second.wait()
        await asyncio.sleep(0)
        assert first.run_id in coordinator._tasks

        async def unrelated_result() -> Any:
            return None

        unrelated = asyncio.create_task(unrelated_result())
        await unrelated
        coordinator._task_finished(first.run_id, unrelated)
        coordinator._timeout_run(second.run_id, unrelated, 1)
        coordinator._mark_timed_out(second.run_id, unrelated, 1)

        releases["first"].set()
        await first.wait()

    asyncio.run(scenario())


def test_subagent_terminal_state_races_preserve_existing_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator
    coordinator.timeout_seconds = None
    definition = runtime._profile_registry["manager"]
    parent_profile = eval_profile_definition(runtime, definition, build_capture_env(runtime))
    parent_budget = DelegationBudget.for_profile(profile_name="manager", max_agents=2)
    parent_concurrency = coordinator.new_concurrency()

    def running_id() -> str:
        return next(run.run_id for run in coordinator.runs.snapshot() if run.status == "running")

    async def finish_after_timeout(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        run_id = running_id()
        coordinator.runs.transition(run_id, "timed_out", error=TimeoutError("external timeout"))
        return "late result"

    monkeypatch.setattr(adapter, "invoke", finish_after_timeout)

    async def timed_out_scenario() -> None:
        handle = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="late",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        with pytest.raises(KediExecutionError, match="external timeout"):
            await handle.wait()

    asyncio.run(timed_out_scenario())

    async def fail_after_terminal_state(*, prompt: str, **kwargs: Any) -> str:
        del prompt, kwargs
        run_id = running_id()
        coordinator.runs.transition(run_id, "failed", error=RuntimeError("first failure"))
        raise RuntimeError("late failure")

    monkeypatch.setattr(adapter, "invoke", fail_after_terminal_state)

    async def failed_scenario() -> None:
        handle = coordinator.start(
            allowed_subagents=("worker",),
            subagent="worker",
            task="fail late",
            final_schema=None,
            parent_profile=parent_profile,
            parent_budget=parent_budget,
            parent_concurrency=parent_concurrency,
        )
        with pytest.raises(RuntimeError, match="late failure"):
            await handle.wait()

    asyncio.run(failed_scenario())


def test_subagent_usage_limits_validate_cap_and_build_pydantic_limits() -> None:
    limits = SubagentUsageLimits(
        request_limit=1000,
        tool_calls_limit=1000,
        input_tokens_limit=2_000_000,
        output_tokens_limit=2_000_000,
        total_tokens_limit=2_000_000,
    ).capped()
    native = limits.to_pydantic()

    assert limits.request_limit == MAX_SUBAGENT_REQUESTS
    assert limits.tool_calls_limit == MAX_SUBAGENT_TOOL_CALLS
    assert limits.input_tokens_limit == MAX_SUBAGENT_TOKENS
    assert limits.output_tokens_limit == MAX_SUBAGENT_TOKENS
    assert limits.total_tokens_limit == MAX_SUBAGENT_TOKENS
    assert native.request_limit == MAX_SUBAGENT_REQUESTS
    assert native.count_tokens_before_request is True

    for field in (
        "request_limit",
        "tool_calls_limit",
        "input_tokens_limit",
        "output_tokens_limit",
        "total_tokens_limit",
    ):
        with pytest.raises(ValueError, match=f"{field} must be at least 1"):
            SubagentUsageLimits(**{field: 0})


def test_final_schema_guards_and_result_validation() -> None:
    schema = {"type": "integer"}
    wrapped = wrapped_final_schema(schema)
    assert wrapped["properties"]["final_result"] == schema

    with pytest.raises(ValueError, match=r"does not support \$ref"):
        validate_final_schema({"$ref": "https://example.com/schema"})
    with pytest.raises(ValueError, match="nesting depth"):
        nested: dict[str, Any] = {}
        current = nested
        for _ in range(18):
            child: dict[str, Any] = {}
            current["child"] = child
            current = child
        validate_final_schema(nested)
    with pytest.raises(ValueError, match="size limit"):
        validate_final_schema({"description": "x" * MAX_FINAL_SCHEMA_BYTES})
    with pytest.raises(ValueError, match="JSON serializable"):
        validate_final_schema({"default": object()})
    with pytest.raises(Exception):
        validate_final_schema({"type": "definitely-not-json-schema"})

    with pytest.raises(TypeError, match="must be an object"):
        validate_structured_result([], wrapped)
    with pytest.raises(Exception):
        validate_structured_result(
            {"task_summary": "done", "final_result": "not-an-int"},
            wrapped,
        )


def test_imported_profile_brings_private_subagent_dependencies(tmp_path: Path) -> None:
    (tmp_path / "profiles.kedi").write_text(
        """
> profile: leaf:
    ###
    Resolves one focused leaf task.
    ###
    > adapter: pydantic
> profile: worker:
    ###
    Coordinates leaf-level analysis.
    ###
    > adapter: pydantic
    > subagent: leaf
> profile: manager:
    > adapter: pydantic
    > subagent: worker
    > subagent: worker
> export:
    manager
""",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: profiles\n= ok\n", encoding="utf-8")
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(
        parse_program(main.read_text(encoding="utf-8"), source_path=str(main)),
        adapter=adapter,
    )

    assert set(runtime._profile_registry) == {"manager"}
    manager = runtime._profile_registry["manager"]
    worker = manager.subagent_bindings["worker"]
    leaf = worker.subagent_bindings["leaf"]
    assert worker.name == "worker"
    assert leaf.name == "leaf"
    profile = eval_profile_definition(runtime, manager, build_capture_env(runtime))
    delegate_tool = runtime._resolve_agent_tool_specs(profile, adapter)[0]
    assert delegate_tool.description is not None
    assert "- worker: Coordinates leaf-level analysis." in delegate_tool.description
    assert "Resolves one focused leaf task." not in delegate_tool.description


def test_subagent_lsp_diagnostics_definition_and_tokens() -> None:
    source = """
> profile: child:
    > model: test
> profile: parent:
    > subagent: child
    > subagent: child
    > max_agents: 5
"""
    assert features.compute_diagnostics(source, "/tmp/subagents.kedi") == []

    definition = features.compute_definition(
        source,
        "file:///tmp/subagents.kedi",
        "/tmp/subagents.kedi",
        features.lsp.Position(line=5, character=17),
    )
    assert definition is not None

    tokens = features.compute_semantic_tokens_full(source, "/tmp/subagents.kedi")
    assert tokens.data
    hover = features.compute_hover(
        source,
        "/tmp/subagents.kedi",
        features.lsp.Position(line=6, character=8),
    )
    assert hover is not None
    assert "Subagent budget" in hover.contents.value
    assert "5" in hover.contents.value

    unknown = source.replace("subagent: child", "subagent: missing")
    diagnostics = features.compute_diagnostics(unknown, "/tmp/subagents.kedi")
    assert any(item.code == "K-SUBAGENT-UNKNOWN" for item in diagnostics)

    cycle = """
> profile: a:
    > subagent: b
> profile: b:
    > subagent: a
"""
    diagnostics = features.compute_diagnostics(cycle, "/tmp/subagents.kedi")
    assert any(item.code == "K-SUBAGENT-CYCLE" for item in diagnostics)

    unsupported = source.replace(
        "> profile: parent:",
        "> profile: parent:\n    > agent: acp",
    )
    diagnostics = features.compute_diagnostics(unsupported, "/tmp/subagents.kedi")
    assert any(item.code == "K-ADAPTER-SUBAGENTS" for item in diagnostics)

    inherited_unsupported = source + "\n> agent: acp\n" + "> use: parent\n"
    diagnostics = features.compute_diagnostics(
        inherited_unsupported,
        "/tmp/subagents.kedi",
    )
    assert any(item.code == "K-ADAPTER-SUBAGENTS" for item in diagnostics)


def test_subagent_coordinator_close_is_fail_closed_and_cleans_owned_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _adapter, runtime, _tools = _manager_tools()
    coordinator = runtime._subagent_coordinator

    async def exercise() -> None:
        task = asyncio.create_task(asyncio.sleep(60))
        coordinator._tasks["active-close"] = _OwnedSubagentTask(
            cast("asyncio.Task[DelegationResult]", task),
            asyncio.get_running_loop(),
        )
        timeout = asyncio.get_running_loop().call_later(60, lambda: None)
        coordinator._timeout_handles["active-close"] = timeout

        def reject_cancel(_run_id: str) -> bool:
            raise KediExecutionError("synthetic cancellation failure")

        async def skip_wait(_run_id: str, _owned: _OwnedSubagentTask) -> None:
            return None

        monkeypatch.setattr(coordinator, "cancel", reject_cancel)
        monkeypatch.setattr(coordinator, "_bridge_task_wait", skip_wait)
        await coordinator.aclose()

        assert timeout.cancelled()
        with pytest.raises(RuntimeError, match="closed"):
            coordinator.start(
                allowed_subagents=(),
                subagent="worker",
                task="task",
                final_schema=None,
                parent_profile=AgentProfile.empty(),
                parent_budget=cast(Any, None),
                parent_concurrency=cast(Any, None),
            )
        with pytest.raises(RuntimeError, match="closed"):
            coordinator._reserve_start(
                (DelegationBudget.for_profile(profile_name="parent", max_agents=1),),
                DelegationConcurrency.for_limit(1),
            )

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
