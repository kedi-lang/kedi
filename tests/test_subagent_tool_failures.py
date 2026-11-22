"""Terminal child failures are feedback, not requests to correct tool arguments."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import NodeCancelledError
from pydantic_ai import ToolFailed, UsageLimits
from pydantic_ai.exceptions import RunCancelled, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.subagents.attempts import (
    UnresolvedDelegationError,
    delegation_scope,
)
from kedi.agent_adapter.subagents.tool_schemas import schema_validated
from kedi.agent_adapter.tools.retry import ToolRetryPolicy, with_body_retry
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile.compile import build_capture_env
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope
from tests.test_langchain_branch_parity import ScriptedModel


def scripted_adapter(backend: str, batches: list[list[dict[str, Any]]]):
    seen = []
    if backend == "pydantic":

        async def respond(messages, info):
            seen.append(list(messages))
            calls = batches.pop(0) if batches else []
            return ModelResponse(
                parts=[ToolCallPart(c["name"], c["args"], tool_call_id=c["id"]) for c in calls]
                or [TextPart("done")]
            )

        return PydanticAdapter(FunctionModel(respond)), seen
    model = ScriptedModel(
        responses=[AIMessage(content="", tool_calls=calls) for calls in batches]
        + [AIMessage(content="done")]
    )
    return LangChainAdapter(model), model.seen


def call(index: int, *, valid: bool = True) -> dict[str, Any]:
    return {
        "name": "delegate_task",
        "args": {"task": f"work-{index}"} if valid else {"parameters": {}},
        "id": f"child-{index}",
    }


def failure_spec(fn) -> ToolSpec:
    return schema_validated(
        [
            ToolSpec(
                fn=fn,
                name="delegate_task",
                description="Delegate work",
                metadata={"kind": "subagent"},
                risk="read_only",
                artifact_result="never",
                json_schema={
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                    "additionalProperties": False,
                },
            )
        ]
    )[0]


def failed_results(messages, backend):
    if backend == "langchain":
        return [m for m in messages if isinstance(m, ToolMessage) and m.status == "error"]
    return [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.outcome == "failed"
    ]


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("parallel", [False, True])
def test_four_terminal_failures_do_not_exhaust_corrections(backend, parallel):
    executed = []

    async def fail(*, task: str):
        executed.append(task)
        await asyncio.sleep(0)
        raise UsageLimitExceeded("API_KEY=secret child request_limit exhausted")

    calls = [call(i) for i in range(4)]
    adapter, seen = scripted_adapter(backend, [calls] if parallel else [[c] for c in calls])
    with adapter.tool_scope([failure_spec(fail)]):
        assert adapter.invoke_sync(prompt="Delegate work") == "done"
    assert sorted(executed) == [f"work-{i}" for i in range(4)]
    failures = failed_results(seen[-1], backend)
    assert len(failures) == 4
    assert "secret" not in str(failures)
    assert "API_KEY=<redacted>" in str(failures)


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("parallel", [False, True])
def test_malformed_subagent_arguments_still_exhaust_corrections(backend, parallel):
    async def body(*, task: str):
        pytest.fail("Malformed arguments must not execute a tool body")

    calls = [call(i, valid=False) for i in range(4)]
    # Pydantic AI counts parallel corrections by model round, not by physical call.
    adapter, _ = scripted_adapter(backend, [calls] * 4 if parallel else [[c] for c in calls])
    with (
        adapter.tool_scope([failure_spec(body)]),
        pytest.raises(UnexpectedModelBehavior, match="max retries"),
    ):
        adapter.invoke_sync(prompt="Delegate work")


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("failure", ["budget", "cancel"])
def test_control_failures_propagate_from_subagent_tools(backend, failure):
    error = {
        "budget": RunBudgetExceeded("requests", 1, 1, 1),
        "cancel": asyncio.CancelledError(),
    }[failure]

    async def fail(*, task: str):
        raise error

    adapter, _ = scripted_adapter(backend, [[call(0)]])
    expected = NodeCancelledError if backend == "langchain" and failure == "cancel" else type(error)
    with adapter.tool_scope([failure_spec(fail)]), pytest.raises(expected):
        adapter.invoke_sync(prompt="Delegate work")


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("budget_kind", ["request", "tool", "native_tool"])
def test_terminal_failures_remain_bounded_by_run_limits(backend, budget_kind):
    async def fail(*, task: str):
        raise RuntimeError("child failed")

    calls = [call(i) for i in range(5)]
    adapter, _ = scripted_adapter(
        backend, [calls] if budget_kind == "native_tool" else [[c] for c in calls]
    )
    budget = (
        RunBudget(request_limit=2) if budget_kind == "request" else RunBudget(tool_attempt_limit=2)
    )
    managed_body = with_body_retry(fail, ToolRetryPolicy(), isolate_arguments=False)
    with adapter.tool_scope([failure_spec(managed_body)]):
        if budget_kind == "native_tool":
            with pytest.raises(UsageLimitExceeded, match="tool_calls_limit"):
                adapter.invoke_sync(
                    prompt="Delegate work", usage_limits=UsageLimits(tool_calls_limit=2)
                )
        else:
            with run_budget_scope(budget), pytest.raises(RunBudgetExceeded):
                adapter.invoke_sync(prompt="Delegate work")


SOURCE = """
> profile: worker:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
"""


def runtime_tools(child):
    runtime = compile_program(parse_program(SOURCE), adapter=child)
    profile = eval_profile_definition(
        runtime, runtime._profile_registry["manager"], build_capture_env(runtime)
    )
    tools = {t.name: t for t in runtime._resolve_agent_tool_specs(profile, child)}
    return runtime, tools


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
async def test_failed_wait_is_repeatable_and_direct_wait_still_raises(backend):
    async def fail(messages, info):
        raise RuntimeError("API_KEY=secret child failed")

    runtime, tools = runtime_tools(PydanticAdapter(FunctionModel(fail)))
    coordinator = runtime._subagent_coordinator
    try:
        with pytest.raises(UnresolvedDelegationError):
            with delegation_scope("fail_closed") as ledger:
                handle = await tools["delegate_task"].fn(
                    subagent="worker", task="work", background=True
                )
                results = [
                    await tools["wait_subagent"].fn(run_id=handle["run_id"]) for _ in range(4)
                ]
                assert all(result == results[0] for result in results)
                assert results[0]["status"] == "failed"
                assert "API_KEY=<redacted>" in results[0]["error"]
                assert "secret" not in str(results)
                assert ledger.snapshot()[0].status == "failed"
                assert ledger.unresolved()
                Draft202012Validator(tools["wait_subagent"].return_schema).validate(results[0])
                calls = [
                    {"name": "wait_subagent", "args": {"run_id": handle["run_id"]}, "id": f"w-{i}"}
                    for i in range(4)
                ]
                parent, _ = scripted_adapter(backend, [[c] for c in calls])
                with parent.tool_scope([tools["wait_subagent"]]):
                    assert await parent.invoke(prompt="Observe the failed child") == "done"
                with pytest.raises(KediExecutionError, match="child failed"):
                    await coordinator.wait(handle["run_id"])
                with pytest.raises(KediExecutionError, match="child failed"):
                    await runtime._wait_subagent(handle["run_id"])
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_actual_child_failures_preserve_fail_closed_and_replacement(backend, recover, parallel):
    child_calls = 0

    async def child_response(messages, info):
        nonlocal child_calls
        child_calls += 1
        if child_calls <= 4:
            raise RuntimeError("child failed")
        return ModelResponse(parts=[TextPart("verified replacement")])

    calls = [
        {
            "name": "delegate_task",
            "args": {"subagent": "worker", "task": "same task"},
            "id": f"c-{i}",
        }
        for i in range(5 if recover else 4)
    ]
    batches = [calls[:4]] + ([[calls[4]]] if recover else []) if parallel else [[c] for c in calls]
    parent, _ = scripted_adapter(backend, batches)
    child = PydanticAdapter(FunctionModel(child_response))
    source = (
        SOURCE.replace(
            "> profile: manager:\n    > adapter: pydantic",
            f"> profile: manager:\n    > adapter: {backend}",
        )
        + "\n> use: manager\n[answer] << Delegate the task.\n= <answer>\n"
    )
    runtime = compile_program(parse_program(source), adapter=parent)
    coordinator = runtime._subagent_coordinator
    coordinator._deps = replace(coordinator._deps, resolve_adapter=lambda profile: child)
    try:
        if recover:
            assert runtime.run_main() == "done"
        else:
            with pytest.raises(KediExecutionError, match="without a successful recovery"):
                runtime.run_main()
        assert [r.status for r in coordinator.runs.snapshot()] == ["failed"] * 4 + (
            ["completed"] if recover else []
        )
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_pydantic_execution_uses_official_terminal_failure(asynchronous):
    from kedi.agent_adapter.adapters.pydantic.tools import _validated_tool_function

    def fail(*, task: str):
        raise RuntimeError("child failed")

    async def afail(*, task: str):
        fail(task=task)

    wrapped = _validated_tool_function(failure_spec(afail if asynchronous else fail))
    with pytest.raises(ToolFailed, match="child failed"):
        if asynchronous:
            asyncio.run(wrapped(task="work"))
        else:
            wrapped(task="work")


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
def test_native_cancellation_is_not_reclassified_by_tool_wrapper(backend):
    from kedi.agent_adapter.adapters.langchain.tools import _langchain_async_tool_function
    from kedi.agent_adapter.adapters.pydantic.tools import _validated_tool_function

    async def fail(*, task: str):
        raise RunCancelled("cancelled")

    spec = failure_spec(fail)
    wrapped = (
        _validated_tool_function(spec)
        if backend == "pydantic"
        else _langchain_async_tool_function(spec)
    )
    with pytest.raises(RunCancelled):
        asyncio.run(wrapped(task="work"))


@pytest.mark.asyncio
async def test_langchain_terminal_feedback_does_not_increment_corrections():
    from kedi.agent_adapter.adapters.langchain.runtime import (
        LangChainRunMiddleware,
        langchain_run_scope,
    )
    from kedi.agent_adapter.adapters.langchain.tools import _langchain_tool_from_spec

    async def fail(*, task: str):
        raise RuntimeError("child failed")

    native = _langchain_tool_from_spec(failure_spec(fail))
    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)

    async def handler(request):
        return await native.ainvoke({**request.tool_call, "type": "tool_call"})

    with langchain_run_scope(None) as state:
        for i in range(4):
            result = await middleware.awrap_tool_call(
                SimpleNamespace(tool_call=call(i), tool=native), handler
            )
            assert result.status == "error"
            assert "child failed" in result.content
        assert state.corrections == {}
        assert state.native_usage.tool_calls == 4
        assert state.pending_tools == 0
        assert len(state.completed_tools) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RunBudgetExceeded("request", 1, 1, 1),
        KediExecutionError("child cancelled", original=RunCancelled("cancelled")),
    ],
)
async def test_wait_preserves_fatal_failure_on_every_observation(error):
    async def fail(messages, info):
        raise error

    runtime, tools = runtime_tools(PydanticAdapter(FunctionModel(fail)))
    try:
        with delegation_scope("recover") as ledger:
            handle = await tools["delegate_task"].fn(
                subagent="worker", task="work", background=True
            )
            for _ in range(4):
                with pytest.raises(type(error)):
                    await tools["wait_subagent"].fn(run_id=handle["run_id"])
            assert ledger.snapshot()[0].status == "failed"
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_subagent_capability_activation_errors_remain_bounded_corrections(backend, wrapped):
    from kedi.agent_adapter.tools.errors import CapabilityActivationError, terminal_tool_failure

    original = CapabilityActivationError("Activate worker before delegating")
    error = KediExecutionError("inactive child", original=original) if wrapped else original

    async def fail(*, task: str):
        raise error

    spec = failure_spec(fail)
    assert terminal_tool_failure(spec, error) is None
    adapter, seen = scripted_adapter(backend, [[call(i)] for i in range(4)])
    with adapter.tool_scope([spec]), pytest.raises(UnexpectedModelBehavior, match="max retries"):
        adapter.invoke_sync(prompt="Delegate work")
    assert len(seen) == 4
