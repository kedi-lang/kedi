from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import ModelHTTPError

from kedi.agent_adapter.adapters import LangChainAdapter
from kedi.agent_adapter.adapters.langchain.fallback import (
    LangChainModelFallbackMiddleware,
    _model_key,
    _ProgressHandler,
    _StreamProgress,
    prepare_model_history,
    response_started,
    retain_fallback_continuation_reset,
)
from kedi.agent_adapter.execution import RunDeadlineExceeded
from kedi.agent_adapter.history_processing import HistoryProcessingError
from kedi.agent_adapter.invocation import observe_invocations
from kedi.agent_profile import AgentProfile
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope
from tests.test_langchain_branch_parity import ScriptedModel, tool_spec


class Candidate(ScriptedModel):
    outcomes: list[Any] = Field(default_factory=list)
    ledger: Any = None
    emit_before_error: bool = False

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.ledger.append((self.name, list(messages)))
        self.seen.append(list(messages))
        outcome = self.outcomes.pop(0)
        if self.emit_before_error and run_manager is not None:
            await run_manager.on_llm_new_token("partial")
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            outcome = outcome(messages)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
        return ChatResult(generations=[ChatGeneration(message=outcome)])


def candidate(name, outcomes, ledger, **kwargs):
    return Candidate(name=name, responses=[], outcomes=outcomes, ledger=ledger, **kwargs)


def http_error(status=503):
    return ModelHTTPError(status, "test-provider", {})


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def sleep(delay):
        assert delay >= 0

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", sleep)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, 2])
async def test_each_fallback_attempt_requires_admission(limit):
    calls = []
    primary = candidate("primary", [http_error()], calls)
    backup = candidate("backup", [AIMessage(content="done")], calls)
    adapter = LangChainAdapter(primary, fallback_models=[backup], request_retries=0)
    records = []
    with (
        run_budget_scope(RunBudget(request_limit=limit)) as ledger,
        observe_invocations(records.append),
    ):
        if limit < 2:
            with pytest.raises(RunBudgetExceeded):
                await adapter.invoke(prompt="Finish")
        else:
            assert await adapter.invoke(prompt="Finish") == "done"
    assert [name for name, _ in calls] == ["primary", "backup"][:limit]
    assert ledger.spent("request") == limit
    assert (records[-1].requests if records else 0) == limit


@pytest.mark.asyncio
async def test_candidate_retries_finish_before_ordered_fallback():
    calls = []
    primary = candidate("primary", [http_error(), http_error()], calls)
    backup = candidate("backup", [http_error(429), http_error(429)], calls)
    final = candidate("final", [http_error(408), AIMessage(content="done")], calls)
    adapter = LangChainAdapter(primary, fallback_models=[backup, final], request_retries=1)
    records = []
    with observe_invocations(records.append):
        assert await adapter.invoke(prompt="Finish") == "done"
    assert [name for name, _ in calls] == [
        "primary",
        "primary",
        "backup",
        "backup",
        "final",
        "final",
    ]
    assert records[-1].requests == 6


@pytest.mark.asyncio
async def test_primary_success_skips_alternatives():
    calls = []
    primary = candidate("primary", [AIMessage(content="done")], calls)
    backup = candidate("backup", [], calls)
    assert (
        await LangChainAdapter(primary, fallback_models=[backup]).invoke(prompt="Finish") == "done"
    )
    assert [name for name, _ in calls] == ["primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        http_error(400),
        http_error(401),
        http_error(403),
        ValueError("invalid"),
        RunDeadlineExceeded("elapsed"),
    ],
)
async def test_terminal_errors_do_not_select_an_alternative(error):
    calls = []
    adapter = LangChainAdapter(
        candidate("primary", [error], calls), fallback_models=[candidate("backup", [], calls)]
    )
    with pytest.raises(type(error)):
        await adapter.invoke(prompt="Finish")
    assert [name for name, _ in calls] == ["primary"]


@pytest.mark.asyncio
async def test_caller_cancellation_does_not_select_an_alternative():
    calls = []
    started = asyncio.Event()

    async def pending(messages):
        started.set()
        await asyncio.Event().wait()

    adapter = LangChainAdapter(
        candidate("primary", [pending], calls), fallback_models=[candidate("backup", [], calls)]
    )
    task = asyncio.create_task(adapter.invoke(prompt="Wait"))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [name for name, _ in calls] == ["primary"]
    assert response_started() is False


@pytest.mark.asyncio
async def test_ordered_exhaustion_keeps_last_error_and_attempt_context():
    calls = []
    error = http_error(429)
    adapter = LangChainAdapter(
        candidate("primary", [http_error()], calls),
        fallback_models=[candidate("backup", [error], calls)],
        request_retries=0,
    )
    with pytest.raises(ModelHTTPError) as raised:
        await adapter.invoke(prompt="Finish")
    assert raised.value is error
    assert any("primary -> backup" in note for note in error.__notes__)


@pytest.mark.asyncio
async def test_completed_business_tool_is_not_replayed_after_fallback():
    calls, effects = [], []

    def effect(step: int) -> str:
        effects.append(step)
        return "recorded"

    primary = candidate(
        "primary",
        [
            AIMessage(
                content="", tool_calls=[{"name": "terminal", "args": {"step": 7}, "id": "call-1"}]
            ),
            http_error(),
        ],
        calls,
    )
    backup = candidate("backup", [AIMessage(content="done")], calls)
    adapter = LangChainAdapter(primary, fallback_models=[backup], request_retries=0)
    with adapter.tool_scope([tool_spec(effect)]):
        assert await adapter.invoke(prompt="Record once") == "done"
    assert effects == [7]
    assert [name for name, _ in calls] == ["primary", "primary", "backup"]
    assert any(
        isinstance(message, ToolMessage) and message.content == "recorded"
        for message in calls[-1][1]
    )


@pytest.mark.asyncio
async def test_fallback_preserves_typed_output_contract():
    class Report(BaseModel):
        city: str

    calls = []
    backup = candidate(
        "backup",
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "Report", "args": {"city": "Istanbul"}, "id": "report-1"}],
            )
        ],
        calls,
    )
    adapter = LangChainAdapter(
        candidate("primary", [http_error()], calls), fallback_models=[backup], request_retries=0
    )
    result = await adapter.produce(template="The city is [city].", output_type=Report)
    assert isinstance(result, Report)
    assert result.city == "Istanbul"


@pytest.mark.asyncio
async def test_partial_stream_failure_is_not_retried_or_fallen_back():
    calls = []
    primary = candidate("primary", [http_error()], calls, emit_before_error=True)
    adapter = LangChainAdapter(primary, fallback_models=[candidate("backup", [], calls)])
    with pytest.raises(ModelHTTPError):
        await adapter.invoke(prompt="Finish")
    assert [name for name, _ in calls] == ["primary"]
    assert response_started() is False


@pytest.mark.parametrize("value", ["model", b"model", 4, [None], [""], [" "]])
def test_constructor_rejects_malformed_alternatives(value):
    with pytest.raises(TypeError):
        LangChainAdapter(candidate("primary", [], []), fallback_models=value)


@pytest.mark.asyncio
async def test_profiles_replace_inherit_and_clear_constructor_fallback(monkeypatch):
    calls = []
    primary = candidate("primary", [http_error(), http_error(), http_error()], calls)
    backup = candidate("backup", [AIMessage(content="constructor")], calls)
    other = candidate("other", [AIMessage(content="profile")], calls)
    adapter = LangChainAdapter(primary, fallback_models=[backup], request_retries=0)
    monkeypatch.setattr(adapter, "_resolve_model", lambda name: other)
    with adapter.profile_override(AgentProfile()):
        assert await adapter.invoke(prompt="Finish") == "constructor"
    with adapter.profile_override(AgentProfile(fallback_models=("other",))):
        assert await adapter.invoke(prompt="Finish") == "profile"
    with adapter.profile_override(AgentProfile(fallback_models=())):
        with pytest.raises(ModelHTTPError):
            await adapter.invoke(prompt="Finish")
    assert [name for name, _ in calls] == ["primary", "backup", "primary", "other", "primary"]


@pytest.mark.asyncio
async def test_concurrent_invocations_have_independent_fallback_state():
    calls = []

    async def success(messages):
        await asyncio.sleep(0)
        return AIMessage(content=messages[-1].content)

    primary = candidate("primary", [http_error(), http_error()], calls)
    backup = candidate("backup", [success, success], calls)
    adapter = LangChainAdapter(primary, fallback_models=[backup], request_retries=0)
    results = await asyncio.gather(adapter.invoke(prompt="first"), adapter.invoke(prompt="second"))
    assert results == ["first", "second"]
    assert [name for name, _ in calls].count("primary") == 2
    assert [name for name, _ in calls].count("backup") == 2
    assert adapter._model is primary


def test_model_switch_preserves_local_history_without_mutating_original():
    model = candidate("backup", [], [], model_kwargs={"temperature": 0.4})
    previous = AIMessage(
        content="previous", response_metadata={"id": "old", "kedi_fallback_model": "other-model"}
    )
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="first"), previous, HumanMessage(content="next")],
        model_settings={"prompt_cache_key": "old-cache"},
    )
    prepared = prepare_model_history(request)
    assert prepared.model.model_kwargs == {"temperature": 0.4}
    assert "prompt_cache_key" not in prepared.model_settings
    assert model.model_kwargs == {"temperature": 0.4}
    assert previous.response_metadata["id"] == "old"
    assert prepared.messages[1].content == "previous"


@pytest.mark.parametrize("reference", ["previous_response_id", "conversation_id", "conversation"])
@pytest.mark.parametrize("with_local_response", [False, True])
@pytest.mark.parametrize("location", ["model_kwargs", "settings"])
def test_server_only_continuation_cannot_silently_lose_context(
    reference, with_local_response, location
):
    model = candidate(
        "backup", [], [], model_kwargs={reference: "opaque"} if location == "model_kwargs" else {}
    )
    messages = [HumanMessage(content="next")]
    if with_local_response:
        messages.insert(0, AIMessage(content="A recent response, not the full history"))
    request = ModelRequest(
        model=model,
        messages=messages,
        model_settings={reference: "opaque"} if location == "settings" else {},
    )
    with pytest.raises(HistoryProcessingError, match="server-only"):
        prepare_model_history(request, switching=True)


def test_signed_reasoning_cannot_be_replayed_to_another_model():
    model = candidate("backup", [], [])
    previous = AIMessage(
        content=[{"type": "thinking", "thinking": "private", "signature": "signed"}],
        response_metadata={"kedi_fallback_model": "other"},
    )
    request = ModelRequest(model=model, messages=[previous], model_settings={})
    with pytest.raises(HistoryProcessingError):
        prepare_model_history(request)


@pytest.mark.asyncio
async def test_middleware_retains_native_message_and_usage_on_success():
    calls = []
    primary = candidate("primary", [], calls)
    backup = candidate("backup", [], calls)
    middleware = LangChainModelFallbackMiddleware(backup)
    request = ModelRequest(
        model=primary, messages=[HumanMessage(content="hello")], model_settings={}
    )
    message = AIMessage(
        content="done", usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
    )

    async def handler(request):
        return ModelResponse(result=[message])

    result = await middleware.awrap_model_call(request, handler)
    assert result.result[0].usage_metadata == message.usage_metadata
    assert result.result[0].response_metadata["kedi_fallback_model"] == _model_key(primary)


@pytest.mark.parametrize("location", ["model_kwargs", "settings"])
@pytest.mark.asyncio
async def test_fallback_rejects_partial_local_history_with_explicit_server_context(location):
    primary = candidate(
        "primary",
        [],
        [],
        model_kwargs={"conversation_id": "server-only-prefix"}
        if location == "model_kwargs"
        else {},
    )
    backup = candidate("backup", [], [])
    request = ModelRequest(
        model=primary,
        messages=[AIMessage(content="Only the latest answer"), HumanMessage(content="Continue")],
        model_settings={"conversation_id": "server-only-prefix"} if location == "settings" else {},
    )
    seen = []

    async def handler(request):
        seen.append(request.model.name)
        raise http_error()

    with pytest.raises(HistoryProcessingError, match="server-only"):
        await LangChainModelFallbackMiddleware(backup).awrap_model_call(request, handler)
    assert seen == ["primary"]


def test_automatic_continuation_and_model_cache_are_disabled_on_a_portable_copy():
    from langchain_core.messages import SystemMessage

    original = candidate("backup", [], []).model_copy(
        update={"use_previous_response_id": True, "prompt_cache_key": "stale"}
    )
    request = ModelRequest(
        model=original,
        messages=[HumanMessage(content="Local context")],
        system_message=SystemMessage(content="Keep the contract"),
        model_settings={},
    )
    prepared = prepare_model_history(request, switching=True)
    assert prepared.model.use_previous_response_id is False
    assert prepared.model.prompt_cache_key is None
    assert original.use_previous_response_id is True
    assert original.prompt_cache_key == "stale"
    assert prepared.system_message.content == "Keep the contract"


def test_model_without_provider_settings_keeps_its_identity_when_replaying():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    model = FakeListChatModel(responses=["done"])
    request = ModelRequest(
        model=model, messages=[HumanMessage(content="Full history")], model_settings={}
    )
    prepared = prepare_model_history(request, switching=True)
    assert prepared.model is model
    assert prepared.messages[0].content == "Full history"


def test_portable_multimodal_history_preserves_content_and_removes_nested_cache_hints():
    content = [
        "A caption",
        {"type": "cache_point"},
        {"cachePoint": {"type": "default"}},
        {"type": "text", "text": "A diagram", "id": "provider-only", "cache_control": {}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "audio", "source": [{"data": "AAAA", "cache_point": {}}]},
    ]
    message = HumanMessage(
        content=content, additional_kwargs={"kedi_asset": "asset-1", "vendor": 1}
    )
    prepared = prepare_model_history(
        ModelRequest(model=candidate("backup", [], []), messages=[message], model_settings={}),
        switching=True,
    )
    assert prepared.messages[0].content == [
        "A caption",
        {"type": "text", "text": "A diagram"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "audio", "source": [{"data": "AAAA"}]},
    ]
    assert prepared.messages[0].additional_kwargs == {"kedi_asset": "asset-1"}
    assert message.content == content


@pytest.mark.parametrize("has_portable_call", [False, True])
def test_provider_tool_blocks_require_normalized_tool_calls(has_portable_call):
    message = AIMessage(
        content=[{"type": "tool_use", "name": "lookup", "id": "call-1", "input": {}}],
        tool_calls=[{"name": "lookup", "id": "call-1", "args": {}}] if has_portable_call else [],
    )
    request = ModelRequest(model=candidate("backup", [], []), messages=[message], model_settings={})
    if not has_portable_call:
        with pytest.raises(HistoryProcessingError, match="portable tool calls"):
            prepare_model_history(request, switching=True)
    else:
        prepared = prepare_model_history(request, switching=True)
        assert prepared.messages[0].tool_calls == message.tool_calls
        assert prepared.messages[0].content == []


@pytest.mark.parametrize("kind", ["reasoning", "computer_use", "unknown-provider-block"])
def test_nonportable_content_blocks_fail_instead_of_disappearing(kind):
    request = ModelRequest(
        model=candidate("backup", [], []),
        messages=[AIMessage(content=[{"type": kind, "text": "Do not discard"}])],
        model_settings={},
    )
    with pytest.raises(HistoryProcessingError, match="provider-owned content block"):
        prepare_model_history(request, switching=True)


@pytest.mark.parametrize("key", ["reasoning", "reasoning_content", "thinking", "refusal"])
def test_provider_reasoning_and_refusals_are_not_silently_discarded(key):
    request = ModelRequest(
        model=candidate("backup", [], []),
        messages=[AIMessage(content="", additional_kwargs={key: "private"})],
        model_settings={},
    )
    with pytest.raises(HistoryProcessingError, match="reasoning/refusal"):
        prepare_model_history(request, switching=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event,started",
    [("content-block-start", True), ("content-block-delta", True), ("usage", False)],
)
async def test_stream_events_only_mark_visible_content(event, started):
    from uuid import uuid4

    state = _StreamProgress()
    callback = _ProgressHandler(state)
    await callback.on_llm_new_token("", run_id=uuid4())
    assert state.started is False
    await callback.on_stream_event({"type": event})
    assert state.started is started


@pytest.mark.asyncio
async def test_tool_call_stream_chunks_prevent_replay_without_text_tokens():
    from uuid import uuid4

    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    state = _StreamProgress()
    await _ProgressHandler(state).on_llm_new_token(
        "",
        run_id=uuid4(),
        chunk=ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": "lookup", "args": "", "id": "call-1", "index": 0}],
            )
        ),
    )
    assert state.started is True


@pytest.mark.asyncio
async def test_callback_manager_and_continuation_reset_survive_candidate_switch():
    from langchain_core.callbacks import AsyncCallbackManager

    primary = candidate("primary", [], [], callbacks=AsyncCallbackManager([]))
    backup = candidate("backup", [], [])
    request = ModelRequest(model=primary, messages=[], model_settings={})
    seen = []

    async def handler(request):
        seen.append(request.model.name)
        if len(seen) == 1:
            assert retain_fallback_continuation_reset(True) is True
            raise http_error()
        assert retain_fallback_continuation_reset(False) is True
        return ModelResponse(
            result=[AIMessage(content="done"), ToolMessage(content="ok", tool_call_id="earlier")]
        )

    assert retain_fallback_continuation_reset(False) is False
    result = await LangChainModelFallbackMiddleware(backup).awrap_model_call(request, handler)
    assert result.result[0].content == "done"
    assert result.result[1].content == "ok"
    assert primary.callbacks.handlers == []
    assert seen == ["primary", "backup"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile,tools,response_format,content,error",
    [
        ({"tool_calling": False}, [object()], None, "hello", "required tools"),
        (
            {"tool_calling": False, "structured_output": False},
            [],
            object(),
            "hello",
            "structured output",
        ),
        (
            {"image_inputs": False},
            [],
            None,
            [{"type": "image_url", "image_url": "image"}],
            "required media",
        ),
        ({"audio_inputs": False}, [], None, [{"type": "audio", "data": "AAAA"}], "required media"),
        ({}, [], None, ["text", {"type": "text", "text": "hello"}], None),
    ],
)
async def test_declared_candidate_contract_is_checked_before_io(
    profile, tools, response_format, content, error
):
    backup = candidate("backup", [], [], profile=profile)
    request = ModelRequest(
        model=candidate("primary", [], []),
        messages=[HumanMessage(content=content)],
        model_settings={},
        tools=tools,
        response_format=response_format,
    )
    calls = []

    async def handler(request):
        calls.append(request.model.name)
        return ModelResponse(result=[AIMessage(content="done")])

    middleware = LangChainModelFallbackMiddleware(backup)
    if error:
        with pytest.raises(ValueError, match=error):
            await middleware.awrap_model_call(request, handler)
        assert calls == []
    else:
        await middleware.awrap_model_call(request, handler)
        assert calls == ["primary"]


@pytest.mark.asyncio
async def test_candidate_without_native_tool_binding_is_rejected():
    from langchain_core.language_models import BaseChatModel

    class NoTools(Candidate):
        bind_tools = BaseChatModel.bind_tools

    request = ModelRequest(
        model=candidate("primary", [], []), messages=[], model_settings={}, tools=[object()]
    )

    async def handler(request):
        pytest.fail("Unsupported tool contract must fail before I/O")

    with pytest.raises(ValueError, match="cannot bind"):
        await LangChainModelFallbackMiddleware(
            NoTools(responses=[], outcomes=[], ledger=[])
        ).awrap_model_call(request, handler)
