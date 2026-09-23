from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.wrapper import WrapperModel

import kedi
from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.adapters.pydantic.budget import budget_model
from kedi.agent_adapter.codemode.errors import CodeModeResultError
from kedi.agent_adapter.hooks import PostToolResultFailureEvent
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.agent_profile import CodeModeSettings
from kedi.errors import KediExecutionError
from kedi.run_budget import (
    RunBudget,
    RunBudgetExceeded,
    assert_no_hidden_model_retries,
    run_budget_scope,
)


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("reject_response", [False, True])
def test_fallback_admits_each_actual_attempt_without_mutating_models(
    limit: int, reject_response: bool
) -> None:
    calls: list[str] = []

    def primary(messages: Any, info: Any) -> ModelResponse:
        calls.append("primary")
        if reject_response:
            return ModelResponse(parts=[TextPart("rejected")])
        raise RuntimeError("temporary failure")

    def secondary(messages: Any, info: Any) -> ModelResponse:
        calls.append("secondary")
        return ModelResponse(parts=[TextPart("accepted")])

    def reject(response: ModelResponse) -> bool:
        return response.parts == [TextPart("rejected")]

    original = FallbackModel(
        FunctionModel(primary), FunctionModel(secondary), fallback_on=[Exception, reject]
    )
    wrapped = WrapperModel(original)

    async def run() -> None:
        with run_budget_scope(RunBudget(request_limit=limit)) as ledger:
            managed = budget_model(wrapped)
            assert managed is not wrapped and wrapped.wrapped is original
            if limit == 1:
                with pytest.raises(RunBudgetExceeded):
                    await managed.request([], None, ModelRequestParameters())
            else:
                result = await managed.request([], None, ModelRequestParameters())
                assert result.parts == [TextPart("accepted")]
            assert ledger is not None and ledger.spent("request") == limit
        assert calls == ["primary", "secondary"][:limit]
        assert original.models[0].__class__ is FunctionModel

    asyncio.run(run())


def test_hidden_sdk_retries_rejected_only_for_finite_request_budget() -> None:
    model = SimpleNamespace(provider=SimpleNamespace(client=SimpleNamespace(max_retries=2)))
    assert_no_hidden_model_retries(model)
    with run_budget_scope(RunBudget(tool_attempt_limit=0)):
        assert_no_hidden_model_retries(model)
    with run_budget_scope(RunBudget(request_limit=1)):
        with pytest.raises(KediExecutionError, match="max_retries=0"):
            assert_no_hidden_model_retries(model)
        model.provider.client.max_retries = 0
        assert_no_hidden_model_retries(model)
        with pytest.raises(KediExecutionError, match="opaque retry policy"):
            assert_no_hidden_model_retries(
                SimpleNamespace(provider=SimpleNamespace(client=object()))
            )


def test_streamed_model_generation_is_admitted_and_budget_wrapper_is_idempotent() -> None:
    from pydantic_ai.models.test import TestModel

    async def run() -> None:
        base = TestModel()
        assert budget_model(base) is base
        with run_budget_scope(RunBudget(request_limit=1)) as ledger:
            model = budget_model(base)
            assert budget_model(model) is model
            async with model.request_stream([], None, ModelRequestParameters()) as response:
                assert response is not None
            assert ledger is not None and ledger.spent("request") == 1
            with pytest.raises(RunBudgetExceeded):
                async with model.request_stream([], None, ModelRequestParameters()):
                    pytest.fail("second generation must not start")

    asyncio.run(run())


class _CodeModel(BaseChatModel):
    calls: int = 0
    code: str = "await lookup()"

    @property
    def _llm_type(self) -> str:
        return "codemode-regression"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _CodeModel:
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls += 1
        response = (
            AIMessage(
                content="",
                tool_calls=[{"name": "execute_code", "args": {"code": self.code}, "id": "outer"}],
            )
            if self.calls == 1
            else AIMessage(content="done")
        )
        return ChatResult(generations=[ChatGeneration(message=response)])


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("mode", ["success", "size_failure", "script_failure", "exhausted"])
def test_codemode_retry_budget_and_delivery(kind: str, mode: str, monkeypatch: Any) -> None:
    bodies = 0
    turns = 0
    handoffs = []
    failures: list[PostToolResultFailureEvent] = []
    code = "await lookup()" + ("\n1 / 0" if mode == "script_failure" else "")

    @kedi.tool(risk="read_only", retries=1, retry_on=(TimeoutError,))
    def lookup() -> str:
        nonlocal bodies
        bodies += 1
        if bodies == 1:
            raise TimeoutError("transient failure")
        return "complete result"

    def hook(event: Any) -> None:
        if isinstance(event, PostToolResultFailureEvent):
            failures.append(event)

    async def respond(messages: Any, info: Any) -> ModelResponse:
        nonlocal turns
        turns += 1
        return (
            ModelResponse(
                parts=[ToolCallPart("execute_code", {"code": code}, tool_call_id="outer")]
            )
            if turns == 1
            else ModelResponse(parts=[TextPart("done")])
        )

    from kedi.agent_adapter.tools.records import ToolResultHandoff

    original = ToolResultHandoff.finish

    def capture(self: ToolResultHandoff, *, accepted: bool) -> None:
        original(self, accepted=accepted)
        if self.tracker is not None:
            handoffs.append(self.tracker.records)

    monkeypatch.setattr(ToolResultHandoff, "finish", capture)
    options = CodeModeSettings(
        preload_tools=("lookup",), max_tool_result_bytes=8 if mode == "size_failure" else 1024
    )
    chat = _CodeModel(code=code)
    adapter = (
        PydanticAdapter(FunctionModel(respond), codemode=options, hook_handler=hook)
        if kind == "pydantic"
        else LangChainAdapter(chat, codemode=options, hook_handler=hook)
    )
    with run_budget_scope(
        RunBudget(request_limit=3, tool_attempt_limit=1 if mode == "exhausted" else 2)
    ) as ledger:
        with adapter.tool_scope((python_callable_tool_spec("lookup", lookup),)):

            def invoke() -> Any:
                return (
                    adapter.run_sync("go")
                    if kind == "pydantic"
                    else adapter.invoke_sync(prompt="go")
                )

            if mode in {"size_failure", "exhausted"}:
                with pytest.raises(
                    CodeModeResultError if mode == "size_failure" else RunBudgetExceeded
                ) as raised:
                    invoke()
                if mode == "size_failure":
                    record = raised.value.tool_call
                    assert record.output == "complete result"
                    assert record.failure_stage == "serialization"
                    assert record.result_state == "failed"
                    assert record.destination == "sandbox"
                    assert len(failures) == 1 and failures[0].record == record
                assert (turns if kind == "pydantic" else chat.calls) == 1
            else:
                invoke()
        assert ledger is not None and ledger.spent("tool_attempt") == bodies
    assert bodies == (1 if mode == "exhausted" else 2)
    if mode != "exhausted":
        assert handoffs[0][0].attempt == 1 and handoffs[0][0].execution_state == "raised"
        assert handoffs[0][1].attempt == 2 and handoffs[0][1].execution_state == "returned"
        assert handoffs[0][0].tool_call_id == handoffs[0][1].tool_call_id
        assert (
            handoffs[0][1].result_state
            == {"success": "handed_off", "size_failure": "failed", "script_failure": "unknown"}[
                mode
            ]
        )
