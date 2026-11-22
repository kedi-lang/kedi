from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.execution import RunDeadlineExceeded, use_run_deadline
from kedi.agent_adapter.invocation import observe_invocations
from kedi.agent_adapter.request_retry import (
    request_retry_limit,
    retry_delay,
    wait_for_request_retry,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.directives import eval_settings_directive
from kedi.core import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.parser import KediParseError
from kedi.lsp.features.diagnostics import compute_diagnostics
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope
from tests.test_langchain_branch_parity import ScriptedModel

Reply = str | list[tuple[str, dict[str, Any]]]


@pytest.fixture(autouse=True)
def skip_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def sleep(delay: float) -> None:
        assert delay >= 0

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", sleep)


def scripted_adapter(
    kind: str,
    respond: Callable[[Sequence[Any], set[str]], Reply],
    **kwargs: Any,
) -> Any:
    sequence = 0

    def response(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal sequence
        sequence += 1
        return respond(messages, names)

    if kind == "pydantic":

        def native(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            value = response(messages, {t.name for t in info.function_tools})
            return ModelResponse(
                parts=[TextPart(value)]
                if isinstance(value, str)
                else [
                    ToolCallPart(name, args, tool_call_id=f"call-{sequence}-{index}")
                    for index, (name, args) in enumerate(value)
                ]
            )

        return PydanticAdapter(FunctionModel(native), **kwargs)

    class ChatModel(ScriptedModel):
        names: set[str] = set()

        def bind_tools(self, tools: Any, **settings: Any) -> ChatModel:
            self.names = {
                tool.name if hasattr(tool, "name") else tool["function"]["name"] for tool in tools
            }
            return self

        async def _agenerate(self, messages: list[BaseMessage], **settings: Any) -> ChatResult:
            value = response(messages, self.names)
            message = (
                AIMessage(content=value)
                if isinstance(value, str)
                else AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": f"call-{sequence}-{index}"}
                        for index, (name, args) in enumerate(value)
                    ],
                )
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

    return LangChainAdapter(ChatModel(responses=[]), **kwargs)


def tool_returns(messages: Sequence[Any]) -> list[Any]:
    return [message for message in messages if isinstance(message, ToolMessage)] + [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_transient_retry_does_not_replay_a_completed_tool(kind: str) -> None:
    requests = 0
    effects = []

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal requests
        requests += 1
        if requests == 1:
            return [("record", {})]
        assert len(tool_returns(messages)) == 1
        if requests == 2:
            raise ModelHTTPError(429, "test", {"message": "temporarily throttled"})
        return "done"

    def record() -> str:
        effects.append("written")
        return "recorded"

    adapter = scripted_adapter(kind, respond)
    spec = ToolSpec(
        fn=record,
        name="record",
        description="Record one event.",
        json_schema={"type": "object", "properties": {}},
        risk="read_only",
        artifact_result="never",
    )
    with adapter.tool_scope([spec]):
        assert adapter.invoke_sync(prompt="Record an event and report completion.") == "done"
    assert requests == 3
    assert effects == ["written"]


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_transient_child_retry_does_not_consume_another_agent_start(kind: str) -> None:
    children = 0

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal children
        if "delegate_task" in names:
            if not tool_returns(messages):
                return [("delegate_task", {"subagent": "worker", "task": "Verify the record."})]
            return "done"
        children += 1
        if children == 1:
            raise ModelHTTPError(503, "test", {"message": "temporarily unavailable"})
        return "verified"

    adapter = scripted_adapter(kind, respond)
    runtime = compile_program(
        parse_program(
            "> profile: worker:\n  > instructions: Verify the supplied record.\n"
            "> profile: coordinator:\n  > subagent: worker\n  > max_agents: 1\n"
            "> use: coordinator\n[reply] << Delegate verification to worker.\n= `reply`\n"
        ),
        adapter=adapter,
    )
    try:
        assert runtime.run_main() == "done"
        runs = runtime._subagent_coordinator.runs.snapshot()
        assert len(runs) == 1
        assert runs[0].status == "completed"
    finally:
        asyncio.run(runtime.aclose())
    assert children == 2


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_request_retries_are_finite_and_preserve_the_provider_error(kind: str, status: int) -> None:
    calls = 0
    error = ModelHTTPError(status, "test", {"message": "unavailable"})

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal calls
        calls += 1
        raise error

    adapter = scripted_adapter(kind, respond)
    with pytest.raises(ModelHTTPError) as raised:
        adapter.invoke_sync(prompt="Report the result.")
    assert raised.value is error
    assert calls == 3


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_nontransient_provider_errors_are_not_retried(kind: str, status: int) -> None:
    calls = 0
    error = ModelHTTPError(status, "test", {})

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ModelHTTPError) as raised:
        scripted_adapter(kind, respond).invoke_sync(prompt="Report the result.")
    assert raised.value is error
    assert calls == 1


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("profile_override", [True, False])
def test_zero_request_retries_can_disable_recovery(kind: str, profile_override: bool) -> None:
    calls = 0

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal calls
        calls += 1
        raise ModelHTTPError(429, "test", {})

    adapter = scripted_adapter(kind, respond, request_retries=2 if profile_override else 0)
    profile = AgentProfile(settings={"request_retries": 0} if profile_override else {})
    with adapter.profile_override(profile), pytest.raises(ModelHTTPError):
        adapter.invoke_sync(prompt="Report the result.")
    assert calls == 1


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_transient_attempts_respect_native_request_limits(kind: str) -> None:
    calls = 0

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal calls
        calls += 1
        raise ModelHTTPError(503, "test", {})

    adapter = scripted_adapter(kind, respond, default_usage_limits=UsageLimits(request_limit=1))
    with pytest.raises(UsageLimitExceeded, match="request_limit"):
        adapter.invoke_sync(prompt="Report the result.")
    assert calls == 1


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_transient_attempts_respect_strict_run_request_budgets(kind: str) -> None:
    calls = 0

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal calls
        calls += 1
        raise ModelHTTPError(503, "test", {})

    adapter = scripted_adapter(kind, respond)
    with run_budget_scope(RunBudget(request_limit=1)) as ledger:
        with pytest.raises(RunBudgetExceeded):
            adapter.invoke_sync(prompt="Report the result.")
        assert ledger is not None
        assert ledger.spent("request") == 1
    assert calls == 1


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_invocation_usage_counts_failed_http_attempts(kind: str) -> None:
    calls = 0

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelHTTPError(429, "test", {})
        return "done"

    usages = []
    with observe_invocations(usages.append):
        assert scripted_adapter(kind, respond).invoke_sync(prompt="Report the result.") == "done"
    assert usages[-1].requests == calls == 2


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_cancellation_and_deadline_interrupt_retry_backoff(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = False

    async def sleep(delay: float) -> None:
        nonlocal entered
        entered = True
        await asyncio.sleep(60)

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", sleep)

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        raise ModelHTTPError(429, "test", {})

    import time

    adapter = scripted_adapter(kind, respond)
    with use_run_deadline(time.monotonic() + 1), pytest.raises(RunDeadlineExceeded):
        adapter.invoke_sync(prompt="Report the result.")
    assert entered


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("value", [True, -1, 0.5, "2", None])
def test_request_retry_setting_rejects_invalid_values(kind: str, value: Any) -> None:
    with pytest.raises(ValueError, match="request_retries"):
        scripted_adapter(kind, lambda messages, names: "done", request_retries=value)
    with pytest.raises(ValueError, match="request_retries"):
        request_retry_limit({"request_retries": value})


def provider_error_with_retry_after(value: str) -> ModelHTTPError:
    error = ModelHTTPError(429, "test", {})
    response = httpx.Response(
        429, headers={"Retry-After": value}, request=httpx.Request("GET", "https://example.test")
    )
    error.__cause__ = httpx.HTTPStatusError(
        "throttled", request=response.request, response=response
    )
    return error


@pytest.mark.asyncio
async def test_retry_after_seconds_are_honored_without_retrying_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", sleep)
    assert await wait_for_request_retry(provider_error_with_retry_after("7"), attempt=1, retries=2)
    assert delays == [7]
    assert not await wait_for_request_retry(
        provider_error_with_retry_after("31"), attempt=1, retries=2
    )
    assert delays == [7]


@pytest.mark.parametrize("header", ["invalid", "nan", "inf", "1e999", "Thu, 01 Jan 1970 00:00:00"])
def test_invalid_retry_after_uses_bounded_backoff(header: str) -> None:
    assert retry_delay(provider_error_with_retry_after(header), 1) == 0.5


def test_retry_after_http_date_and_negative_seconds() -> None:
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=15), usegmt=True)
    assert 13 <= retry_delay(provider_error_with_retry_after(future), 1) <= 15
    assert retry_delay(provider_error_with_retry_after("-1"), 1) == 0
    assert retry_delay(ModelHTTPError(503, "test", {}), 100) == 30
    assert request_retry_limit({}) == 2


@pytest.mark.asyncio
async def test_nonprovider_errors_are_not_retried() -> None:
    assert not await wait_for_request_retry(ValueError("invalid"), attempt=1, retries=2)
    assert not await wait_for_request_retry(SimpleNamespace(status_code=True), attempt=1, retries=2)


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_long_retry_after_preserves_original_error_without_an_early_retry(kind: str) -> None:
    calls = 0
    error = provider_error_with_retry_after("31")

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ModelHTTPError) as raised:
        scripted_adapter(kind, respond).invoke_sync(prompt="Report the result.")
    assert raised.value is error
    assert calls == 1


@pytest.mark.asyncio
async def test_pydantic_retry_capability_supports_a_context_without_usage_limits() -> None:
    from pydantic_ai import RetryModelRequest, RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    from kedi.agent_adapter.adapters.pydantic.request_retry import RequestRetryCapability

    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    request_context = SimpleNamespace(attempt=1)
    with pytest.raises(RetryModelRequest):
        await RequestRetryCapability(2).on_model_request_error(
            ctx, request_context=request_context, error=ModelHTTPError(429, "test", {})
        )
    assert ctx.usage.requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
async def test_real_sdk_http_429_retries_preserve_history_and_tool_effects(kind: str) -> None:
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    requests = []
    effects = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        index = len(requests)
        if index == 2:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error": {"message": "Temporarily throttled", "type": "rate_limit_error"}},
            )
        message = {"role": "assistant", "content": "done"}
        if index == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "write-once",
                        "type": "function",
                        "function": {"name": "record", "arguments": "{}"},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": f"chat-{index}",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if index == 1 else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )

    def record() -> str:
        effects.append("written")
        return "recorded"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if kind == "pydantic":
            sdk = AsyncOpenAI(
                api_key="test",
                base_url="https://example.test/v1",
                max_retries=0,
                http_client=client,
            )
            adapter = PydanticAdapter(
                OpenAIChatModel("test", provider=OpenAIProvider(openai_client=sdk))
            )
        else:
            adapter = LangChainAdapter(
                ChatOpenAI(
                    model="test",
                    api_key="test",
                    base_url="https://example.test/v1",
                    max_retries=0,
                    http_async_client=client,
                )
            )
        spec = ToolSpec(
            fn=record,
            name="record",
            description="Record an event.",
            json_schema={"type": "object", "properties": {}},
            risk="read_only",
            artifact_result="never",
        )
        usages = []
        with adapter.tool_scope([spec]), observe_invocations(usages.append):
            assert await adapter.invoke(prompt="Record once and report completion.") == "done"
    assert effects == ["written"]
    assert len(requests) == 3
    assert requests[1]["messages"] == requests[2]["messages"]
    assert usages[-1].requests == 3
    assert usages[-1].input_tokens == 20
    assert usages[-1].output_tokens == 6


@pytest.mark.asyncio
async def test_pydantic_stream_failure_after_output_is_not_replayed() -> None:
    attempts = 0

    async def stream(messages: list[ModelMessage], info: AgentInfo):
        nonlocal attempts
        attempts += 1
        yield "partial"
        yield " output"
        raise ModelHTTPError(503, "test", {})

    adapter = PydanticAdapter(FunctionModel(stream_function=stream))
    with pytest.raises(ModelHTTPError):
        async with adapter.run_stream("Report progress.") as result:
            async for _text in result.stream_text(debounce_by=None):
                pass
    assert attempts == 1


@pytest.mark.parametrize("value", ["-1", "true", "false", '"2"', "none", "1.5"])
def test_request_retry_setting_rejects_invalid_literals_in_parser_and_editor(value: str) -> None:
    from lsprotocol.types import DiagnosticSeverity

    source = f"> settings:\n  request_retries: {value}\n"
    with pytest.raises(KediParseError):
        parse_program(source)
    errors = [
        diagnostic
        for diagnostic in compute_diagnostics(source, None)
        if diagnostic.severity == DiagnosticSeverity.Error
    ]
    assert errors
    assert errors[0].range.start.line == 1


@pytest.mark.parametrize("value", [0, 2, 5])
def test_request_retry_setting_accepts_static_and_computed_integers(value: int) -> None:
    for literal in (str(value), "`limit`"):
        statement = parse_program(f"> settings:\n  request_retries: {literal}\n").ordered_toplevel[
            0
        ]
        profile = eval_settings_directive(KediRuntime(), statement, {"limit": value})
        assert profile.settings["request_retries"] == value


@pytest.mark.parametrize("value", [True, False, -1, "2", None])
def test_request_retry_setting_validates_computed_values(value: Any) -> None:
    statement = parse_program("> settings:\n  request_retries: `limit`\n").ordered_toplevel[0]
    with pytest.raises(ValueError, match="request_retries"):
        eval_settings_directive(KediRuntime(), statement, {"limit": value})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("second_retries", [0, 1])
async def test_concurrent_request_retries_isolate_profiles_and_usage(
    kind: str, second_retries: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: dict[str, int] = {}
    both_started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def respond(messages: Sequence[Any], names: set[str]) -> Reply:
        if kind == "pydantic":
            prompt = next(
                part.content
                for message in reversed(messages)
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart)
            )
        else:
            prompt = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        assert isinstance(prompt, str)
        attempts[prompt] = attempts.get(prompt, 0) + 1
        if len(attempts) == 2:
            loop.call_soon_threadsafe(both_started.set)
        if attempts[prompt] == 1:
            raise ModelHTTPError(503, "test", {})
        return prompt + " completed"

    async def wait_for_both(delay: float) -> None:
        await asyncio.wait_for(both_started.wait(), timeout=2)

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", wait_for_both)
    adapter = scripted_adapter(kind, respond)
    usages = []

    async def invoke(prompt: str, retries: int) -> str:
        with adapter.profile_override(AgentProfile(settings={"request_retries": retries})):
            return await adapter.invoke(prompt=prompt)

    with observe_invocations(usages.append):
        first, second = await asyncio.gather(
            invoke("alpha", 1), invoke("beta", second_retries), return_exceptions=True
        )
    assert first == "alpha completed"
    if second_retries:
        assert second == "beta completed"
    else:
        assert isinstance(second, ModelHTTPError)
    assert attempts == {"alpha": 2, "beta": 1 + second_retries}
    assert sorted(usage.requests for usage in usages) == sorted([2, 1 + second_retries])
    assert adapter.active_profile.settings.get("request_retries") is None
