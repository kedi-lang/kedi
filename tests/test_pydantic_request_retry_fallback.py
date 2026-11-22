"""Count provider attempts hidden inside native Pydantic fallback requests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from pydantic_ai.capabilities import AbstractCapability, ModelFallback
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError, RetryModelRequest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.adapters.pydantic.budget import budget_model
from kedi.agent_adapter.adapters.pydantic.request_retry import RequestRetryCapability
from kedi.agent_adapter.invocation import observe_invocations
from kedi.run_budget import RunBudget, RunBudgetExceeded, run_budget_scope


@pytest.fixture(autouse=True)
def skip_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def sleep(delay: float) -> None:
        pass

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", sleep)


def response(text: str = "done") -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(text)], usage=RequestUsage(input_tokens=11, output_tokens=3)
    )


def model(name: str, outcomes: list[Any], calls: list[str]) -> FunctionModel:
    def respond(messages, info):
        calls.append(name)
        value = outcomes.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    return FunctionModel(respond, model_name=name)


@pytest.mark.asyncio
@pytest.mark.parametrize("native_capability", [False, True])
async def test_fallback_success_counts_failed_primary_without_inventing_tokens(native_capability):
    calls, usages = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {})], calls)
    backup = model("backup", [response()], calls)
    adapter = PydanticAdapter(primary if native_capability else FallbackModel(primary, backup))
    capabilities = [ModelFallback(backup)] if native_capability else []
    with observe_invocations(usages.append):
        assert await adapter.invoke(prompt="Finish", capabilities=capabilities) == "done"
    assert calls == ["primary", "backup"]
    assert usages[-1].requests == 2
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (11, 3)


@pytest.mark.asyncio
async def test_exhausted_fallback_counts_each_failed_provider_once():
    calls, usages = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {})], calls)
    backup = model("backup", [ModelHTTPError(503, "backup", {})], calls)
    adapter = PydanticAdapter(FallbackModel(primary, backup))
    with observe_invocations(usages.append), pytest.raises(FallbackExceptionGroup):
        await adapter.invoke(prompt="Finish")
    assert calls == ["primary", "backup"]
    assert usages[-1].requests == 2
    assert usages[-1].input_tokens == usages[-1].output_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 1])
async def test_budget_refusal_preserves_usage_of_only_admitted_fallback_attempts(limit):
    calls, usages = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {})], calls)
    backup = model("backup", [response()], calls)
    adapter = PydanticAdapter(FallbackModel(primary, backup))
    with (
        run_budget_scope(RunBudget(request_limit=limit)) as ledger,
        observe_invocations(usages.append),
        pytest.raises(RunBudgetExceeded),
    ):
        await adapter.invoke(prompt="Finish")
    assert ledger.spent("request") == limit
    assert calls == (["primary"] if limit else [])
    assert usages[-1].requests == limit
    assert usages[-1].input_tokens == usages[-1].output_tokens == 0


@pytest.mark.asyncio
async def test_rejected_fallback_response_retains_only_measured_tokens():
    calls, usages = [], []
    primary = model("primary", [response("rejected")], calls)
    backup = model("backup", [response()], calls)

    def reject(value: ModelResponse) -> bool:
        return value.text == "rejected"

    adapter = PydanticAdapter(FallbackModel(primary, backup, fallback_on=[reject]))
    with observe_invocations(usages.append):
        assert await adapter.invoke(prompt="Finish") == "done"
    assert calls == ["primary", "backup"]
    assert usages[-1].requests == 2
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (22, 6)


@pytest.mark.asyncio
async def test_direct_retry_followed_by_fallback_does_not_count_http_error_twice():
    calls, usages = [], []
    primary = model(
        "primary",
        [ModelHTTPError(429, "primary", {}), ModelHTTPError(503, "primary", {})],
        calls,
    )
    backup = model("backup", [response()], calls)
    adapter = PydanticAdapter(
        FallbackModel(primary, backup, fallback_on=lambda error: error.status_code == 503)
    )
    with observe_invocations(usages.append):
        assert await adapter.invoke(prompt="Finish") == "done"
    assert calls == ["primary", "primary", "backup"]
    assert usages[-1].requests == 3
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (11, 3)


@pytest.mark.asyncio
async def test_response_hook_retry_does_not_recount_previous_fallback_attempts():
    calls, usages = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {}), response()], calls)
    backup = model("backup", [response()], calls)

    class RetryOnce(AbstractCapability):
        async def after_model_request(self, ctx, *, request_context, response):
            if request_context.attempt == 1:
                raise RetryModelRequest()
            return response

    adapter = PydanticAdapter(FallbackModel(primary, backup))
    with observe_invocations(usages.append):
        assert await adapter.invoke(prompt="Finish", capabilities=[RetryOnce()]) == "done"
    assert calls == ["primary", "backup", "primary"]
    assert usages[-1].requests == 3
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (22, 6)


@pytest.mark.asyncio
async def test_nested_exhausted_fallback_is_not_an_extra_provider_attempt():
    calls, usages = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {})], calls)
    secondary = model("secondary", [ModelHTTPError(503, "secondary", {})], calls)
    backup = model("backup", [response()], calls)
    chain = FallbackModel(
        FallbackModel(primary, secondary), backup, fallback_on=(FallbackExceptionGroup,)
    )
    with observe_invocations(usages.append):
        assert await PydanticAdapter(chain).invoke(prompt="Finish") == "done"
    assert calls == ["primary", "secondary", "backup"]
    assert usages[-1].requests == 3
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (11, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 3])
async def test_retry_after_exhausted_group_deduplicates_cumulative_attempts(budget):
    calls, usages = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {})] * 2, calls)
    backup = model("backup", [ModelHTTPError(503, "backup", {}), response()], calls)

    class RetryGroupOnce(AbstractCapability):
        async def on_model_request_error(self, ctx, *, request_context, error):
            if isinstance(error, FallbackExceptionGroup) and request_context.attempt == 1:
                raise RetryModelRequest()
            raise error

    adapter = PydanticAdapter(FallbackModel(primary, backup))
    with (
        run_budget_scope(RunBudget(request_limit=budget)),
        observe_invocations(usages.append),
    ):
        if budget is None:
            assert await adapter.invoke(prompt="Finish", capabilities=[RetryGroupOnce()]) == "done"
        else:
            with pytest.raises(RunBudgetExceeded):
                await adapter.invoke(prompt="Finish", capabilities=[RetryGroupOnce()])
    assert calls == ["primary", "backup", "primary"] + (["backup"] if budget is None else [])
    assert usages[-1].requests == len(calls)
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (
        (11, 3) if budget is None else (0, 0)
    )


@pytest.mark.asyncio
async def test_terminal_http_error_counts_itself_and_carried_primary_once():
    calls, usages = [], []
    primary = model("primary", [ModelHTTPError(503, "primary", {})], calls)
    error = ModelHTTPError(400, "backup", {})
    backup = model("backup", [error], calls)
    chain = FallbackModel(primary, backup, fallback_on=lambda error: error.status_code == 503)
    with observe_invocations(usages.append), pytest.raises(ModelHTTPError) as raised:
        await PydanticAdapter(chain).invoke(prompt="Finish")
    assert raised.value is error
    assert calls == ["primary", "backup"]
    assert usages[-1].requests == 2
    assert usages[-1].input_tokens == usages[-1].output_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 2])
@pytest.mark.parametrize("initial_composite", [False, True])
async def test_identical_http_messages_do_not_merge_distinct_direct_and_internal_attempts(
    budget, initial_composite
):
    calls, usages = [], []
    errors = [ModelHTTPError(503, "primary", {}) for _ in range(2)]
    assert errors[0] is not errors[1] and str(errors[0]) == str(errors[1])
    internal_error = errors[1]
    primary = model("primary", errors, calls)
    backup = model("backup", [response()], calls)
    fallback = FallbackModel(primary, backup, fallback_on=lambda error: error is internal_error)

    class SwitchToFallback(AbstractCapability):
        async def prepare_model_request(self, ctx, request_context):
            if request_context.attempt > 1:
                return replace(request_context, model=budget_model(fallback))
            return request_context

    adapter = PydanticAdapter(fallback if initial_composite else primary)
    capabilities = [] if initial_composite else [SwitchToFallback()]
    with run_budget_scope(RunBudget(request_limit=budget)), observe_invocations(usages.append):
        if budget is None:
            assert await adapter.invoke(prompt="Finish", capabilities=capabilities) == "done"
        else:
            with pytest.raises(RunBudgetExceeded):
                await adapter.invoke(prompt="Finish", capabilities=capabilities)
    assert calls == ["primary", "primary"] + (["backup"] if budget is None else [])
    assert usages[-1].requests == len(calls)
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (
        (11, 3) if budget is None else (0, 0)
    )


@pytest.mark.asyncio
async def test_native_fallback_exhaustion_counts_identically_named_provider_failures():
    calls, usages = [], []
    primary = model("same-model", [ModelHTTPError(503, "same-model", {})], calls)
    backup = model("same-model", [ModelHTTPError(503, "same-model", {})], calls)
    with observe_invocations(usages.append), pytest.raises(FallbackExceptionGroup):
        await PydanticAdapter(primary).invoke(prompt="Finish", capabilities=[ModelFallback(backup)])
    assert calls == ["same-model", "same-model"]
    assert usages[-1].requests == 2
    assert usages[-1].input_tokens == usages[-1].output_tokens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [None, 3])
async def test_leaf_named_like_a_composite_is_still_a_provider_attempt(budget):
    calls, usages = [], []
    first = model("nested-first", [ModelHTTPError(503, "nested-first", {})], calls)
    second = model("nested-second", [ModelHTTPError(503, "nested-second", {})], calls)
    nested = FallbackModel(first, second)
    leaf = model(nested.model_name, [ModelHTTPError(503, "leaf", {})], calls)
    final = model("final", [response()], calls)
    chain = FallbackModel(leaf, nested, final, fallback_on=(ModelHTTPError, FallbackExceptionGroup))
    with run_budget_scope(RunBudget(request_limit=budget)), observe_invocations(usages.append):
        if budget is None:
            assert await PydanticAdapter(chain).invoke(prompt="Finish") == "done"
        else:
            with pytest.raises(RunBudgetExceeded):
                await PydanticAdapter(chain).invoke(prompt="Finish")
    assert calls == [nested.model_name, "nested-first", "nested-second"] + (
        ["final"] if budget is None else []
    )
    assert usages[-1].requests == len(calls)


@pytest.mark.asyncio
async def test_native_fallback_discovers_dynamically_selected_composite():
    calls, usages = [], []
    initial = model("initial", [ModelHTTPError(503, "initial", {})], calls)
    first = model("nested-first", [ModelHTTPError(503, "nested-first", {})], calls)
    second = model("nested-second", [ModelHTTPError(503, "nested-second", {})], calls)
    nested = FallbackModel(first, second)
    final = model("final", [response()], calls)
    native = ModelFallback(nested, final, fallback_on=(ModelHTTPError, FallbackExceptionGroup))
    with observe_invocations(usages.append):
        assert (
            await PydanticAdapter(initial).invoke(prompt="Finish", capabilities=[native]) == "done"
        )
    assert calls == ["initial", "nested-first", "nested-second", "final"]
    assert usages[-1].requests == 4
    assert (usages[-1].input_tokens, usages[-1].output_tokens) == (11, 3)


@pytest.mark.asyncio
async def test_accounting_capability_is_fresh_per_run_and_resets_between_steps():
    calls, usages = [], []
    shared = RequestRetryCapability(2)
    primary = model(
        "primary",
        [
            ModelHTTPError(503, "primary", {}),
            response(),
            ModelHTTPError(503, "primary", {}),
            response(),
        ],
        calls,
    )
    backup = model("backup", [response(), response()], calls)
    adapter = PydanticAdapter(FallbackModel(primary, backup))

    class AnotherStep(AbstractCapability):
        async def after_model_request(self, ctx, *, request_context, response):
            if ctx.run_step == 1:
                from pydantic_ai import ModelRetry

                raise ModelRetry("One more step")
            return response

    with observe_invocations(usages.append):
        for _ in range(2):
            assert (
                await adapter.invoke(prompt="Finish", capabilities=[shared, AnotherStep()])
                == "done"
            )
    assert [usage.requests for usage in usages] == [3, 3]
    assert shared._attempts == {}


@pytest.mark.asyncio
async def test_shared_capability_keeps_concurrent_native_fallback_accounting_isolated():
    shared = RequestRetryCapability(2)
    calls, usages = [], []
    both_entered = asyncio.Event()

    async def fail(messages, info):
        calls.append("primary")
        if len(calls) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), 2)
        raise ModelHTTPError(503, "primary", {})

    primary = FunctionModel(fail, model_name="primary")
    backup = model("backup", [response(), response()], calls)
    adapter = PydanticAdapter(primary)
    with observe_invocations(usages.append):
        results = await asyncio.gather(
            *(
                adapter.invoke(prompt="Finish", capabilities=[shared, ModelFallback(backup)])
                for _ in range(2)
            )
        )
    assert results == ["done", "done"]
    assert [usage.requests for usage in usages] == [2, 2]
    assert shared._attempts == {}
    assert shared._pending_http == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_output", [False, True])
@pytest.mark.parametrize("native_capability", [False, True])
async def test_stream_fallback_preserves_native_boundary_and_counts_once(
    after_output, native_capability
):
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

    first = FunctionModel(stream_function=primary, model_name="primary")
    second = FunctionModel(stream_function=backup, model_name="backup")
    adapter = PydanticAdapter(first if native_capability else FallbackModel(first, second))
    capabilities = [ModelFallback(second)] if native_capability else []

    async def consume():
        async with adapter.run_stream("Finish", usage=usage, capabilities=capabilities) as result:
            async for text in result.stream_text(debounce_by=None):
                output.append(text)

    if after_output:
        with pytest.raises(ModelHTTPError):
            await consume()
    else:
        await consume()
    assert calls == (["primary"] if after_output else ["primary", "backup"])
    assert usage.requests == len(calls)
    if not after_output:
        assert output[-1] == "done"


@pytest.mark.asyncio
async def test_interrupted_stream_does_not_recount_its_direct_opening_retry():
    calls = []
    usage = RunUsage()

    async def stream(messages, info):
        calls.append("provider")
        if len(calls) > 1:
            yield "partial"
            yield " output"
        raise ModelHTTPError(503, "provider", {})

    adapter = PydanticAdapter(FunctionModel(stream_function=stream, model_name="provider"))
    with pytest.raises(ModelHTTPError):
        async with adapter.run_stream("Finish", usage=usage) as result:
            async for _ in result.stream_text(debounce_by=None):
                pass
    assert calls == ["provider", "provider"]
    assert usage.requests == 2
