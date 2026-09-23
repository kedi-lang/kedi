from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Barrier
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

import kedi
from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_adapter.adapters.langchain.middleware import _langchain_approval_middleware
from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.hooks import (
    AdapterHookSnapshot,
    HookDispatcher,
    HookDispatcherSnapshot,
    PostToolResultFailureEvent,
    PostToolUseEvent,
    bind_hook_dispatcher,
    with_hook_lifecycle,
)
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.agent_adapter.tools.execution import with_artifact_result
from kedi.agent_adapter.tools.records import ToolCallRecord, ToolCallTracker
from kedi.agent_adapter.tools.retry import ToolRetryPolicy, with_body_retry
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.lang import parse_program
from kedi.lang import compile_program
from kedi.run_budget import (
    RunBudget,
    RunBudgetExceeded,
    admit_run_operation,
    run_budget_scope,
)


class _BudgetChatModel(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "budget-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _BudgetChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        response = (
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {"x": "ok"}, "id": "one"}])
            if self.calls == 1
            else AIMessage(content="done")
        )
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_run_budget_validates_and_charges_ancestors_atomically() -> None:
    for invalid in (-1, True, 1.0):
        with pytest.raises(ValueError):
            RunBudget(request_limit=invalid)  # type: ignore[arg-type]

    with run_budget_scope(RunBudget(request_limit=2, tool_attempt_limit=1)) as root:
        assert root is not None
        with run_budget_scope(RunBudget(request_limit=1)) as child:
            assert child is not None
            admit_run_operation("request")
            with pytest.raises(RunBudgetExceeded, match="request"):
                admit_run_operation("request")
            assert root.spent("request") == 1
            assert child.spent("request") == 1
        admit_run_operation("request")
        with pytest.raises(RunBudgetExceeded, match="request"):
            admit_run_operation("request")
        assert root.spent("request") == 2


def test_run_budget_last_slot_is_atomic_for_siblings() -> None:
    barrier = Barrier(2)
    with run_budget_scope(RunBudget(request_limit=1)) as root:
        assert root is not None

        def attempt() -> bool:
            barrier.wait()
            try:
                root.admit("request")
            except RunBudgetExceeded:
                return False
            return True

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _index: attempt(), range(2)))
        assert sorted(outcomes) == [False, True]
        assert root.spent("request") == 1


def test_retry_consumes_tool_attempts_but_does_not_retry_budget_denial() -> None:
    calls = 0

    def body() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("body failed")

    wrapped = with_body_retry(body, ToolRetryPolicy(retries=3), isolate_arguments=False)
    with run_budget_scope(RunBudget(tool_attempt_limit=2)) as ledger:
        assert ledger is not None
        with pytest.raises(RunBudgetExceeded):
            wrapped()
        assert calls == 2
        assert ledger.spent("tool_attempt") == 2


def test_tool_record_preserves_body_output_when_post_hook_fails() -> None:
    observed: list[PostToolResultFailureEvent] = []

    def handler(event: Any) -> None:
        if isinstance(event, PostToolUseEvent):
            raise RuntimeError("post hook failed")
        if isinstance(event, PostToolResultFailureEvent):
            observed.append(event)

    def body() -> None:
        return None

    tool = with_hook_lifecycle(python_callable_tool_spec("body", body), adapter_shortname="fake")
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use=(handler,), post_tool_result_failure=(handler,)),
        )
    )
    with bind_hook_dispatcher(dispatcher):
        with pytest.raises(RuntimeError, match="post hook failed") as raised:
            tool.fn()
    record = raised.value.tool_call
    assert isinstance(record, ToolCallRecord)
    assert (record.execution_state, record.result_state, record.failure_stage) == (
        "returned",
        "failed",
        "post_hook",
    )
    assert record.output_available and record.output is None
    assert observed[0].record == record
    assert "output=" not in repr(record)


def test_artifact_failure_preserves_body_result_without_retry(monkeypatch: Any) -> None:
    import kedi.agent_adapter.tools.execution as execution

    calls = 0
    observed: list[PostToolResultFailureEvent] = []

    def body() -> str:
        nonlocal calls
        calls += 1
        return "native result"

    def reject_artifact(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("artifact admission failed")

    monkeypatch.setattr(execution, "admit_artifact_result", reject_artifact)
    tool = python_callable_tool_spec(
        "body", with_body_retry(body, ToolRetryPolicy(retries=2), isolate_arguments=False)
    )
    tool = with_hook_lifecycle(
        with_artifact_result(tool, manager=object(), policy=object()),
        adapter_shortname="fake",
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_result_failure=(observed.append,)),
        )
    )
    with bind_hook_dispatcher(dispatcher):
        with pytest.raises(RuntimeError, match="artifact admission failed") as raised:
            tool.fn()
    record = raised.value.tool_call
    assert isinstance(record, ToolCallRecord)
    assert (record.execution_state, record.result_state, record.failure_stage) == (
        "returned",
        "failed",
        "artifact",
    )
    assert record.output_available and record.output == "native result"
    assert calls == 1
    assert len(observed) == 1


def test_budget_directive_parses_top_level_procedure_and_profile() -> None:
    program = parse_program(
        "> budget:\n"
        "  request_limit: 2\n"
        "@worker() -> str:\n"
        "  > budget:\n"
        "    tool_attempt_limit: 3\n"
        '  = `"done"`\n'
        "> profile: child:\n"
        "  > budget:\n"
        "    request_limit: 1\n"
    )
    assert program.ordered_toplevel[0].request_limit == 2
    assert program.procedures[0].body[0].tool_attempt_limit == 3
    assert program.profile_defs["child"].budget_members[0].request_limit == 1


def test_async_retry_records_separate_attempts() -> None:
    from kedi.agent_adapter.tools.records import track_tool_call

    attempts = 0

    async def body() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("try again")
        return "ready"

    wrapped = with_body_retry(body, ToolRetryPolicy(retries=1), isolate_arguments=False)

    async def exercise() -> tuple[ToolCallRecord, ...]:
        with track_tool_call("call-1") as tracker:
            assert await wrapped() == "ready"
            return tracker.records

    records = asyncio.run(exercise())
    assert [(item.attempt, item.execution_state) for item in records] == [
        (1, "raised"),
        (2, "returned"),
    ]


def test_sync_callable_returning_awaitable_remains_body_until_awaited() -> None:
    from kedi.agent_adapter.tools.records import track_tool_call

    calls = 0

    def body() -> Any:
        async def finish() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("not yet")
            return "done"

        return finish()

    wrapped = with_body_retry(body, ToolRetryPolicy(retries=1), isolate_arguments=False)

    async def exercise() -> tuple[ToolCallRecord, ...]:
        with run_budget_scope(RunBudget(tool_attempt_limit=2)):
            with track_tool_call("sync-awaitable") as tracker:
                assert await wrapped() == "done"
                return tracker.records

    records = asyncio.run(exercise())
    assert [(item.attempt, item.execution_state) for item in records] == [
        (1, "raised"),
        (2, "returned"),
    ]
    assert records[-1].output == "done"


@pytest.mark.parametrize("adapter_kind", ["pydantic", "langchain"])
def test_native_tool_loop_charges_each_generation_and_managed_body(
    adapter_kind: str, monkeypatch: Any
) -> None:
    model_calls = 0
    handed_off: list[ToolCallRecord] = []
    original_handed_off = ToolCallTracker.handed_off

    def capture_handoff(tracker: ToolCallTracker) -> ToolCallRecord | None:
        record = original_handed_off(tracker)
        if record is not None:
            handed_off.append(record)
        return record

    monkeypatch.setattr(ToolCallTracker, "handed_off", capture_handoff)

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return (
            ModelResponse(parts=[ToolCallPart("echo", {"x": "ok"})])
            if model_calls == 1
            else ModelResponse(parts=[TextPart("done")])
        )

    @kedi.tool(risk="read_only")
    def echo(x: str) -> str:
        return x

    chat_model = _BudgetChatModel()
    adapter = (
        PydanticAdapter(FunctionModel(respond))
        if adapter_kind == "pydantic"
        else LangChainAdapter(chat_model)
    )
    with run_budget_scope(RunBudget(request_limit=1, tool_attempt_limit=1)) as ledger:
        assert ledger is not None
        with adapter.tool_scope((python_callable_tool_spec("echo", echo),)):
            with pytest.raises(RunBudgetExceeded, match="request"):
                if adapter_kind == "pydantic":
                    adapter.run_sync("go")
                else:
                    adapter.invoke_sync(prompt="go")
        assert ledger.spent("request") == 1
        assert ledger.spent("tool_attempt") == 1
    assert (model_calls if adapter_kind == "pydantic" else chat_model.calls) == 1
    assert any(
        record.execution_state == "returned"
        and record.result_state == "handed_off"
        and record.output == "ok"
        for record in handed_off
    )


@pytest.mark.parametrize("adapter_kind", ["pydantic", "langchain"])
def test_zero_request_budget_prevents_first_model_call(adapter_kind: str) -> None:
    model_calls = 0

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("done")])

    chat_model = _BudgetChatModel()
    adapter = (
        PydanticAdapter(FunctionModel(respond))
        if adapter_kind == "pydantic"
        else LangChainAdapter(chat_model)
    )
    with run_budget_scope(RunBudget(request_limit=0)):
        with pytest.raises(RunBudgetExceeded, match="request"):
            if adapter_kind == "pydantic":
                adapter.run_sync("go")
            else:
                adapter.invoke_sync(prompt="go")
    assert model_calls == chat_model.calls == 0


@pytest.mark.parametrize("adapter_kind", ["pydantic", "langchain"])
def test_native_post_result_failure_retains_body_evidence_without_retry(adapter_kind: str) -> None:
    body_calls = 0
    observed: list[PostToolResultFailureEvent] = []

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("echo", {"x": "ok"})])

    @kedi.tool(risk="read_only", retries=2)
    def echo(x: str) -> str:
        nonlocal body_calls
        body_calls += 1
        return f"{x}:{body_calls}"

    def hook(event: Any) -> None:
        if isinstance(event, PostToolUseEvent):
            raise RuntimeError("observer failed")
        if isinstance(event, PostToolResultFailureEvent):
            observed.append(event)

    adapter = (
        PydanticAdapter(FunctionModel(respond), hook_handler=hook)
        if adapter_kind == "pydantic"
        else LangChainAdapter(_BudgetChatModel(), hook_handler=hook)
    )
    with adapter.tool_scope((python_callable_tool_spec("echo", echo),)):
        with pytest.raises(RuntimeError, match="observer failed") as raised:
            if adapter_kind == "pydantic":
                adapter.run_sync("go")
            else:
                adapter.invoke_sync(prompt="go")
    record = raised.value.tool_call
    assert isinstance(record, ToolCallRecord)
    assert (record.execution_state, record.result_state, record.failure_stage) == (
        "returned",
        "failed",
        "post_hook",
    )
    assert record.output_available and record.output == "ok:1"
    assert body_calls == 1
    assert len(observed) == 1 and observed[0].tool_call == record


def test_branch_budget_does_not_escape_lexical_scope() -> None:
    model_calls = 0

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("ok")])

    program = parse_program(
        "> if: `True`:\n  > budget:\n    request_limit: 0\n[answer] << Say ok.\n= `answer`\n"
    )
    runtime = compile_program(program, adapter=PydanticAdapter(FunctionModel(respond)))
    assert runtime.run_main() == "ok"
    assert model_calls == 1


def test_python_root_budget_starts_fresh_for_each_runtime_invocation() -> None:
    model_calls = 0

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("ok")])

    program = parse_program("[answer] << Say ok.\n= `answer`\n")
    runtime = compile_program(
        program,
        adapter=PydanticAdapter(FunctionModel(respond)),
        run_budget=RunBudget(request_limit=1),
    )
    assert runtime.run_main() == "ok"
    assert runtime.run_main() == "ok"
    assert model_calls == 2


@pytest.mark.parametrize(
    "source",
    [
        "> profile: limited:\n"
        "  > budget:\n"
        "    request_limit: 0\n"
        "> use: limited\n"
        "[answer] << Say ok.\n"
        "= `answer`\n",
        "> profile: limited:\n"
        "  > budget:\n"
        "    request_limit: 0\n"
        "@worker() -> str:\n"
        "  > use: limited\n"
        "  [answer] << Say ok.\n"
        "  = `answer`\n"
        "= `worker()`\n",
    ],
)
def test_used_profile_budget_is_enforced_before_model_call(source: str) -> None:
    model_calls = 0

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("ok")])

    runtime = compile_program(
        parse_program(source), adapter=PydanticAdapter(FunctionModel(respond))
    )
    with pytest.raises(RunBudgetExceeded, match="request"):
        runtime.run_main()
    assert model_calls == 0


def test_native_pydantic_toolset_is_rejected_before_generation_under_attempt_budget() -> None:
    model_calls = 0

    async def respond(_messages: Any, _info: Any) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("ok")])

    adapter = PydanticAdapter(FunctionModel(respond), toolsets=[FunctionToolset()])
    with run_budget_scope(RunBudget(tool_attempt_limit=1)):
        with pytest.raises(Exception, match="cannot enforce tool_attempt run budget"):
            adapter.run_sync("go")
    assert model_calls == 0


def test_langchain_result_error_after_body_return_is_not_model_feedback() -> None:
    calls = 0
    observed: list[PostToolResultFailureEvent] = []

    def body() -> str:
        nonlocal calls
        calls += 1
        return "native output"

    wrapped = with_body_retry(body, ToolRetryPolicy(retries=2), isolate_arguments=False)
    tool = python_callable_tool_spec("echo", wrapped)
    middleware = _langchain_approval_middleware(
        profile=AgentProfile.empty(), tools=(tool,), approval_enabled=False
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_result_failure=(observed.append,)),
        )
    )

    @dataclass(frozen=True)
    class Request:
        tool_call: dict[str, Any]
        tool: Any = None

        def override(self, **changes: Any) -> Request:
            return replace(self, **changes)

    async def handler(_request: Request) -> ToolMessage:
        wrapped()
        return ToolMessage(
            content="serialization failed",
            name="echo",
            tool_call_id="call-1",
            status="error",
        )

    async def exercise() -> None:
        with bind_hook_dispatcher(dispatcher):
            with pytest.raises(RuntimeError, match="serialization failed") as raised:
                await middleware.awrap_tool_call(
                    Request({"name": "echo", "args": {}, "id": "call-1"}), handler
                )
        record = raised.value.tool_call
        assert isinstance(record, ToolCallRecord)
        assert record.execution_state == "returned"
        assert record.result_state == "failed"
        assert record.failure_stage == "serialization"
        assert record.output_available and record.output == "native output"

    asyncio.run(exercise())
    assert calls == 1
    assert len(observed) == 1
