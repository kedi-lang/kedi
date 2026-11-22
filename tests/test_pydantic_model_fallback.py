"""Deterministic model fallback and per-provider admission contracts."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_profile import AgentProfile
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope


@pytest.fixture(autouse=True)
def skip_retry_sleep(monkeypatch):
    async def sleep(delay):
        pass

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", sleep)


def response(text="done"):
    return ModelResponse(
        parts=[TextPart(text)], usage=RequestUsage(input_tokens=11, output_tokens=3)
    )


def model(name: str, outcomes: list[Any], calls: list[str], **kwargs):
    def respond(messages, info):
        calls.append(name)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return FunctionModel(respond, model_name=name, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1, 2])
async def test_native_capability_selected_backup_obeys_request_admission(limit):
    calls = []
    primary = model("primary", [ModelHTTPError(503, "primary", {})], calls)
    backup = model("backup", [response()], calls)
    adapter = PydanticAdapter(primary, request_retries=0)
    usage = RunUsage()
    with run_budget_scope(RunBudget(request_limit=limit)) as ledger:
        if limit < 2:
            with pytest.raises(RunBudgetExceeded):
                await adapter.invoke(
                    prompt="Finish", capabilities=[ModelFallback(backup)], usage=usage
                )
        else:
            assert (
                await adapter.invoke(
                    prompt="Finish", capabilities=[ModelFallback(backup)], usage=usage
                )
                == "done"
            )
    assert calls == ["primary", "backup"][:limit]
    assert ledger.spent("request") == usage.requests == limit


@pytest.mark.asyncio
async def test_ordered_fallback_exhausts_retries_for_each_candidate():
    calls = []
    primary = model("primary", [ModelHTTPError(503, "primary", {}) for _ in range(2)], calls)
    backup = model("backup", [ModelHTTPError(429, "backup", {}) for _ in range(2)], calls)
    final = model("final", [ModelHTTPError(408, "final", {}), response()], calls)
    adapter = PydanticAdapter(primary, fallback_models=[backup, final], request_retries=1)
    usage = RunUsage()
    assert await adapter.invoke(prompt="Finish", usage=usage) == "done"
    assert calls == ["primary", "primary", "backup", "backup", "final", "final"]
    assert usage.requests == 6
    assert (usage.input_tokens, usage.output_tokens) == (11, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_configured_fallback_and_retries_share_budget(limit):
    calls = []
    primary = model("primary", [ModelHTTPError(503, "primary", {}) for _ in range(2)], calls)
    backup = model("backup", [response()], calls)
    adapter = PydanticAdapter(primary, fallback_models=[backup], request_retries=1)
    usage = RunUsage()
    with run_budget_scope(RunBudget(request_limit=limit)) as ledger:
        if limit < 3:
            with pytest.raises(RunBudgetExceeded):
                await adapter.invoke(prompt="Finish", usage=usage)
        else:
            assert await adapter.invoke(prompt="Finish", usage=usage) == "done"
    assert calls == ["primary", "primary", "backup"][:limit]
    assert ledger.spent("request") == usage.requests == limit


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_non_transient_provider_errors_do_not_select_fallback(status):
    calls = []
    error = ModelHTTPError(status, "primary", {})
    adapter = PydanticAdapter(
        model("primary", [error], calls),
        fallback_models=[model("backup", [response()], calls)],
    )
    with pytest.raises(ModelHTTPError) as raised:
        await adapter.invoke(prompt="Finish")
    assert raised.value is error
    assert calls == ["primary"]


@pytest.mark.asyncio
async def test_ordered_exhaustion_reports_native_attempts():
    calls = []
    adapter = PydanticAdapter(
        model("primary", [ModelHTTPError(503, "primary", {})], calls),
        fallback_models=[model("backup", [ModelHTTPError(503, "backup", {})], calls)],
        request_retries=0,
    )
    usage = RunUsage()
    with pytest.raises(FallbackExceptionGroup) as raised:
        await adapter.invoke(prompt="Finish", usage=usage)
    assert calls == ["primary", "backup"]
    assert [attempt.model_name for attempt in raised.value.attempts] == calls
    assert usage.requests == 2


@pytest.mark.asyncio
async def test_primary_success_does_not_touch_backups():
    calls = []
    adapter = PydanticAdapter(
        model("primary", [response()], calls),
        fallback_models=[model("backup", [], calls)],
    )
    assert await adapter.invoke(prompt="Finish") == "done"
    assert calls == ["primary"]


@pytest.mark.parametrize("value", ["test:backup", b"backup", [False], [None], 1])
def test_constructor_rejects_malformed_fallback_models(value):
    with pytest.raises(TypeError, match="fallback_models"):
        PydanticAdapter(fallback_models=value)


@pytest.mark.parametrize("value", [[""], ["  "]])
def test_constructor_rejects_empty_model_ids(value):
    with pytest.raises(ValueError, match="non-empty"):
        PydanticAdapter(fallback_models=value)


@pytest.mark.asyncio
async def test_profile_replaces_inherits_and_clears_constructor_alternatives():
    from kedi.agent_adapter.model_resolution import pydantic_model_resolution

    calls, resolved = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {}) for _ in range(4)], calls)
    default = model("default", [response("default"), response("default")], calls)
    alternate = model("alternate", [response("alternate")], calls)

    def resolve(name):
        resolved.append(name)
        return alternate

    adapter = PydanticAdapter(primary, fallback_models=[default], request_retries=0)
    with adapter.profile_override(AgentProfile(fallback_models=None)):
        assert await adapter.invoke(prompt="Finish") == "default"
    with (
        pydantic_model_resolution(resolve),
        adapter.profile_override(AgentProfile(fallback_models=("test:alternate",))),
    ):
        assert await adapter.invoke(prompt="Finish") == "alternate"
    with adapter.profile_override(AgentProfile(fallback_models=())), pytest.raises(ModelHTTPError):
        await adapter.invoke(prompt="Finish")
    assert await adapter.invoke(prompt="Finish") == "default"
    assert resolved == ["test:alternate"]
    assert calls == ["primary", "default", "primary", "alternate", "primary", "primary", "default"]
    assert adapter.fallback_models == (default,)


@pytest.mark.asyncio
async def test_constructor_string_models_use_existing_factory():
    from kedi.agent_adapter.model_resolution import pydantic_model_resolution

    calls, resolved = [], []
    backup = model("backup", [response()], calls)

    def resolve(name):
        resolved.append(name)
        return backup

    adapter = PydanticAdapter(
        model("primary", [ModelHTTPError(503, "primary", {})], calls),
        fallback_models=["test/backup"],
        request_retries=0,
    )
    with pydantic_model_resolution(resolve):
        assert await adapter.invoke(prompt="Finish") == "done"
    assert resolved == ["test:backup"]


@pytest.mark.asyncio
async def test_fallback_does_not_supply_missing_primary(monkeypatch):
    monkeypatch.delenv("KEDI_MODEL", raising=False)
    calls = []
    adapter = PydanticAdapter(fallback_models=[model("backup", [response()], calls)])
    with pytest.raises(Exception, match="model"):
        await adapter.invoke(prompt="Finish")
    assert calls == []


@pytest.mark.asyncio
async def test_successful_tool_result_and_approval_are_not_replayed():
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    from kedi.agent_adapter.approvals.policy import ApprovalDecision, ApprovalPolicy
    from kedi.agent_adapter.tools.specs import ToolSpec

    calls, effects, approvals = [], [], []

    def record():
        effects.append("written")
        return "recorded"

    def approve(request):
        approvals.append(request.tool_name)
        return ApprovalDecision.allow()

    first = ModelResponse(parts=[ToolCallPart("record", {}, tool_call_id="once")])
    primary = model("primary", [first, ModelHTTPError(503, "primary", {})], calls)

    def backup(messages, info):
        calls.append("backup")
        results = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(results) == 1 and results[0].content == "recorded"
        return response()

    adapter = PydanticAdapter(primary, fallback_models=[FunctionModel(backup)], request_retries=0)
    spec = ToolSpec(
        fn=record,
        name="record",
        description="Record an event",
        json_schema={"type": "object", "properties": {}},
        artifact_result="never",
    )
    with (
        adapter.tool_scope([spec]),
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))),
    ):
        assert await adapter.invoke(prompt="Record once") == "done"
    assert calls == ["primary", "primary", "backup"]
    assert effects == ["written"]
    assert approvals == ["record"]


@pytest.mark.asyncio
async def test_fallback_keeps_native_typed_output_contract():
    from pydantic import BaseModel
    from pydantic_ai.messages import ToolCallPart

    class Answer(BaseModel):
        count: int

    calls = []

    def backup(messages, info):
        calls.append("backup")
        output = info.output_tools[0]
        assert "count" in output.parameters_json_schema["properties"]
        return ModelResponse(parts=[ToolCallPart(output.name, {"count": 7})])

    adapter = PydanticAdapter(
        model("primary", [ModelHTTPError(503, "primary", {})], calls),
        fallback_models=[FunctionModel(backup)],
        request_retries=0,
    )
    assert await adapter.produce("Count", output_type=Answer) == Answer(count=7)
    assert calls == ["primary", "backup"]


@pytest.mark.asyncio
@pytest.mark.parametrize("after_output", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_stream_opening_only_fallback_obeys_budget(after_output, limit):
    calls, output = [], []
    usage = RunUsage()

    async def primary(messages, info):
        calls.append("primary")
        if after_output:
            yield "partial"
            yield " output"
        raise ModelHTTPError(503, "primary", {})

    async def backup(messages, info):
        calls.append("backup")
        yield "done"

    adapter = PydanticAdapter(
        FunctionModel(stream_function=primary),
        fallback_models=[FunctionModel(stream_function=backup)],
        request_retries=1,
    )

    async def consume():
        async with adapter.run_stream("Finish", usage=usage) as result:
            async for text in result.stream_text(debounce_by=None):
                output.append(text)

    with run_budget_scope(RunBudget(request_limit=limit)) as ledger:
        if after_output:
            with pytest.raises(ModelHTTPError):
                await consume()
        elif limit < 3:
            with pytest.raises(RunBudgetExceeded):
                await consume()
        else:
            await consume()
    expected = ["primary"] if after_output else ["primary", "primary", "backup"][:limit]
    assert calls == expected
    assert usage.requests == ledger.spent("request") == len(expected)
    if not after_output and limit == 3:
        assert output[-1] == "done"


@pytest.mark.asyncio
async def test_concurrent_runs_keep_models_and_budgets_isolated():
    calls = []
    entered = asyncio.Event()

    async def fail(messages, info):
        calls.append("primary")
        if calls.count("primary") == 2:
            entered.set()
        await asyncio.wait_for(entered.wait(), 2)
        raise ModelHTTPError(503, "primary", {})

    adapter = PydanticAdapter(
        FunctionModel(fail),
        fallback_models=[model("backup", [response(), response()], calls)],
        request_retries=0,
    )

    async def invoke(limit):
        usage = RunUsage()
        with run_budget_scope(RunBudget(request_limit=limit)) as ledger:
            try:
                result = await adapter.invoke(prompt="Finish", usage=usage)
            except RunBudgetExceeded:
                result = "blocked"
        return result, usage.requests, ledger.spent("request")

    assert await asyncio.gather(invoke(1), invoke(2)) == [("blocked", 1, 1), ("done", 2, 2)]
    assert calls == ["primary", "primary", "backup"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError(), ValueError("not a provider error")])
async def test_control_and_non_provider_errors_propagate(error):
    calls = []
    adapter = PydanticAdapter(
        model("primary", [error], calls), fallback_models=[model("backup", [], calls)]
    )
    with pytest.raises(type(error)) as raised:
        await adapter.invoke(prompt="Finish")
    assert raised.value is error
    assert calls == ["primary"]


@pytest.mark.asyncio
async def test_native_selected_nested_composite_is_guarded():
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.wrapper import WrapperModel

    calls = []
    primary = model("primary", [ModelHTTPError(503, "primary", {})], calls)
    nested = WrapperModel(
        FallbackModel(
            model("nested-first", [ModelHTTPError(503, "nested-first", {})], calls),
            model("nested-second", [response()], calls),
        )
    )
    usage = RunUsage()
    adapter = PydanticAdapter(primary, request_retries=0)
    with run_budget_scope(RunBudget(request_limit=2)) as ledger, pytest.raises(RunBudgetExceeded):
        await adapter.invoke(prompt="Finish", capabilities=[ModelFallback(nested)], usage=usage)
    assert calls == ["primary", "nested-first"]
    assert usage.requests == ledger.spent("request") == 2


@pytest.mark.asyncio
async def test_native_agent_admission_does_not_depend_on_request_retry():
    from pydantic_ai import Agent

    from kedi.agent_adapter.adapters.pydantic.budget import ModelAdmissionCapability, budget_model

    calls = []
    with run_budget_scope(RunBudget(request_limit=1)) as ledger:
        agent = Agent(
            budget_model(model("primary", [ModelHTTPError(503, "primary", {})], calls)),
            capabilities=[
                ModelFallback(model("backup", [response()], calls)),
                ModelAdmissionCapability(),
            ],
        )
        with pytest.raises(RunBudgetExceeded):
            await agent.run("Finish")
    assert calls == ["primary"]
    assert ledger.spent("request") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,value",
    [
        ("openai_previous_response_id", "resp_private"),
        ("openai_conversation_id", "conv_private"),
        ("google_cached_content", "cached/private"),
    ],
)
async def test_explicit_server_owned_state_is_not_sent_to_fallback(key, value):
    calls = []
    adapter = PydanticAdapter(
        model("primary", [ModelHTTPError(503, "primary", {})], calls),
        fallback_models=[model("backup", [], calls)],
        request_retries=0,
    )
    with pytest.raises(ValueError, match="server-owned state"):
        await adapter.invoke(prompt="Finish", model_settings={key: value})
    assert calls == ["primary"]


@pytest.mark.asyncio
async def test_automatic_continuation_resets_without_mutating_portable_history():
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    calls, settings = [], []
    previous = ModelResponse(
        parts=[TextPart("earlier answer")],
        model_name="primary",
        provider_name="function",
        provider_response_id="resp_private",
        provider_details={"conversation_id": "conv_private"},
    )

    def backup(messages, info):
        calls.append("backup")
        settings.append(info.model_settings)
        earlier = next(message for message in messages if isinstance(message, ModelResponse))
        assert earlier.text == "earlier answer"
        assert earlier.provider_response_id is None
        assert earlier.provider_details is None
        return response()

    adapter = PydanticAdapter(
        model("primary", [ModelHTTPError(503, "primary", {})], calls),
        fallback_models=[FunctionModel(backup)],
        request_retries=0,
    )
    assert (
        await adapter.invoke(
            prompt="Finish",
            message_history=[ModelRequest(parts=[UserPromptPart("Earlier")]), previous],
            model_settings={"openai_previous_response_id": "auto"},
        )
        == "done"
    )
    assert settings[0]["openai_previous_response_id"] is None
    assert previous.provider_response_id == "resp_private"
    assert previous.provider_details == {"conversation_id": "conv_private"}


@pytest.mark.asyncio
async def test_portable_settings_and_candidate_defaults_are_preserved():
    calls, received = [], []

    def backup(messages, info):
        calls.append("backup")
        received.append(info.model_settings)
        return response()

    alternate = FunctionModel(backup, settings={"temperature": 0.4, "max_tokens": 50})
    primary = model(
        "primary",
        [ModelHTTPError(503, "primary", {})],
        calls,
        settings={"temperature": 0.8, "top_p": 0.1},
    )
    adapter = PydanticAdapter(primary, fallback_models=[alternate], request_retries=0)
    assert await adapter.invoke(prompt="Finish", model_settings={"temperature": 0.2}) == "done"
    assert received == [{"temperature": 0.2, "max_tokens": 50}]
    assert primary.settings == {"temperature": 0.8, "top_p": 0.1}
    assert alternate.settings == {"temperature": 0.4, "max_tokens": 50}


@pytest.mark.asyncio
async def test_long_retry_after_skips_early_retry_but_allows_an_alternative():
    import httpx

    calls = []
    error = ModelHTTPError(429, "primary", {})
    request = httpx.Request("POST", "https://example.test")
    error.__cause__ = httpx.HTTPStatusError(
        "throttled",
        request=request,
        response=httpx.Response(429, headers={"Retry-After": "120"}, request=request),
    )
    adapter = PydanticAdapter(
        model("primary", [error], calls),
        fallback_models=[model("backup", [response()], calls)],
        request_retries=2,
    )
    assert await adapter.invoke(prompt="Finish") == "done"
    assert calls == ["primary", "backup"]


@pytest.mark.asyncio
async def test_deadline_cancellation_does_not_select_an_alternative():
    calls = []
    started = asyncio.Event()

    async def primary(messages, info):
        calls.append("primary")
        started.set()
        await asyncio.Event().wait()

    adapter = PydanticAdapter(FunctionModel(primary), fallback_models=[model("backup", [], calls)])
    async with asyncio.timeout(2):
        task = asyncio.create_task(adapter.invoke(prompt="Finish"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert calls == ["primary"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings,error",
    [
        ({"function_vendor_option": "private"}, "provider settings"),
        ({"function_vendor_option": None, "temperature": 0.2}, None),
    ],
)
async def test_cross_provider_fallback_keeps_portable_settings_only(settings, error):
    class OtherProvider(FunctionModel):
        @property
        def system(self):
            return "other"

    calls = []

    def respond(messages, info):
        calls.append("backup")
        assert info.model_settings["temperature"] == 0.2
        return response()

    adapter = PydanticAdapter(
        model("primary", [ModelHTTPError(503, "primary", {})], calls),
        fallback_models=[OtherProvider(respond)],
        request_retries=0,
    )
    if error:
        with pytest.raises(ValueError, match=error):
            await adapter.invoke(prompt="Finish", model_settings=settings)
        assert calls == ["primary"]
    else:
        assert await adapter.invoke(prompt="Finish", model_settings=settings) == "done"
        assert calls == ["primary", "backup"]


@pytest.mark.asyncio
async def test_fallback_refuses_suspended_response_even_with_local_text():
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.tools import RunContext

    from kedi.agent_adapter.adapters.pydantic.fallback import RequestFallbackCapability

    calls = []
    primary = model("primary", [], calls)
    backup = model("backup", [], calls)
    capability = RequestFallbackCapability([backup])
    context = RunContext(deps=None, model=primary, usage=RunUsage(), prompt="Continue")
    request = ModelRequestContext(
        model=primary,
        messages=[ModelResponse(parts=[TextPart("Partial answer")], state="suspended")],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    await capability.before_model_request(context, request)
    request.model = backup
    with pytest.raises(ValueError, match="suspended"):
        await capability.prepare_model_request(context, request)
    assert calls == []
