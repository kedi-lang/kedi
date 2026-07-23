from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.invocation import (
    InvocationUsage,
    observe_invocations,
    report_invocation,
)
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.subagents import (
    MAX_FINAL_SCHEMA_BYTES,
    SubagentExecutionRequest,
    subagent_tool_specs,
    validate_final_schema,
    validate_structured_result,
    wrapped_final_schema,
)
from kedi.agent_adapter.tool_override import ToolSpec
from kedi.agent_profile import (
    AgentProfile,
    AgentSubagentUnsupported,
)
from kedi.core import KediRuntime
from kedi.agent_profile.compile import build_capture_env
from kedi.agent_profile.model import eval_profile_definition
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.compiler.program_compiler import ProgramCompiler
from kedi.lsp import features
from kedi.agent_adapter.subagents.budget import (
    MAX_SUBAGENT_CONCURRENCY,
    MAX_SUBAGENT_INVOCATIONS,
    DelegationBudget,
    DelegationConcurrency,
)
from kedi.agent_adapter.subagents.attempts import (
    DelegationFailurePolicy,
    UnresolvedDelegationError,
    delegation_scope,
    track_delegation_attempt,
)
from kedi.agent_adapter.subagents.runs import SubagentRunRegistry
from kedi.agent_adapter.subagents.runs import (
    SubagentCancelled,
    SubagentCompleted,
    SubagentEventLog,
    SubagentFailed,
    SubagentStarted,
    safe_subagent_error_message,
)
from kedi.agent_adapter.subagents.safety import SubagentSafetyCeiling
from kedi.agent_adapter.subagents.limits import (
    MAX_SUBAGENT_REQUESTS,
    MAX_SUBAGENT_TOKENS,
    MAX_SUBAGENT_TOOL_CALLS,
    SubagentUsageLimits,
)
from tests.mock_adapter import MockAdapter


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


def test_delegation_budget_hard_cap() -> None:
    budget = DelegationBudget.for_profile(profile_name="manager", max_agents=1000)
    assert budget.owner_profile == "manager"
    assert budget.limit == MAX_SUBAGENT_INVOCATIONS

    anonymous = DelegationBudget.for_profile(profile_name=None, max_agents=None)
    assert anonymous.owner_profile == "<anonymous>"
    assert anonymous.limit == MAX_SUBAGENT_INVOCATIONS

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
    assert adapter.invocations == [("Inspect the parser", "Work carefully.", ("lookup",), ())]
    assert runtime._subagent_stack.get() == ()
    run = runtime._subagent_coordinator.runs.get(result["run_id"])
    assert run.status == "completed"
    assert run.profile_name == "worker"
    assert run.parent_run_id is None
    assert run.root_run_id == run.run_id
    assert run.depth == 1
    assert run.started_at is not None
    assert run.finished_at is not None


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
    assert adapter.invocations == [("Build a value", "Work carefully.", ("lookup",), ())]
    assert runtime._subagent_stack.get() == ()


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
        tuple(tool.name for tool in request.tools) == ("lookup",)
        for request in adapter.child_requests
    )
    assert all(request.instructions == "Work carefully." for request in adapter.child_requests)
    assert adapter.child_calls[0][1]["neutral_request_limit"] is not None
    assert adapter.child_calls[0][1]["instructions"] == "Work carefully."
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


def test_delegate_task_without_child_system_has_no_instructions() -> None:
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

    assert adapter.invocations == [("Inspect the parser", None, (), ())]


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
        "wait_subagent",
        "cancel_subagent",
        "subagent_status",
    )
    assert calls[0] == (lifecycle_tools, ())
    assert calls[1] == ((), ())
    assert calls[2][0] == lifecycle_tools
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
            with pytest.raises(KediExecutionError, match="timed out"):
                await tools["wait_subagent"].fn(run_id=handle["run_id"], timeout=0.001)
        assert timed_out.snapshot()[0].status == "timed_out"

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
        "slow-int": ("alpha", ("alpha_tool",)),
        "fast-text": ("alpha", ("alpha_tool",)),
        "beta-object": ("beta", ("beta_tool",)),
        "beta-failure": ("beta", ("beta_tool",)),
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
        if not info.function_tools:
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
        if not info.function_tools:
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
    assert result.message.endswith(": secret path")
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
    adapter = PydanticAdapter(TestModel(custom_output_text="child output"))
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
        with pytest.raises(KediExecutionError, match="timed out after 0.01 seconds"):
            await timed_out.wait(timeout=0.01)
        assert timed_out.status == "timed_out"

    asyncio.run(scenario())
    events = coordinator.events.snapshot()
    assert [type(event) for event in events] == [
        SubagentStarted,
        SubagentCompleted,
        SubagentStarted,
        SubagentCancelled,
        SubagentStarted,
        SubagentFailed,
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
    > adapter: pydantic
> profile: worker:
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

    assert {"manager", "worker", "leaf"} <= runtime._profile_registry.keys()

    compiler = ProgramCompiler(adapter=None)
    imported_runtime = KediRuntime()
    compiler._merge_profile_dependencies(
        AgentProfile(subagents=("missing",)),
        imported_runtime,
        seen=set(),
    )
    assert compiler._profile_subagent_names(AgentProfile(subagents=("leaf",))) == ("leaf",)


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
