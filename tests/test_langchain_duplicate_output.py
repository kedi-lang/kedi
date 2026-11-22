from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from jsonschema.validators import validator_for
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import MultipleStructuredOutputsError, ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

from kedi.agent_adapter.adapters.langchain.adapter import LangChainAdapter, _accept_typed_output
from kedi.agent_adapter.adapters.langchain.duplicate_output import (
    LangChainDuplicateOutputMiddleware,
    RejectedDuplicateOutput,
)
from kedi.agent_adapter.adapters.langchain.request_retry import RequestRetryMiddleware
from kedi.agent_adapter.adapters.langchain.runtime import (
    LangChainOutputCorrectionMiddleware,
    LangChainRunMiddleware,
    langchain_run_scope,
)
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
from kedi.agent_adapter.subagents import (
    SubagentExecutionRequest,
    SubagentOutputState,
    accept_subagent_output,
    wrapped_final_schema,
)
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_profile import AgentProfile
from tests.test_langchain_branch_parity import ScriptedModel, tool_spec


class Result(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    answer: Any


class IntegerResult(Result):
    answer: int


class DateResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    when: date


def _reply(
    values: list[dict[str, Any]],
    *,
    name: str = "Result",
    business: bool = False,
    finish_reason: str = "tool_calls",
    call_prefix: str = "out",
) -> AIMessage:
    calls: list[Any] = [
        {"name": name, "args": value, "id": f"{call_prefix}-{index}", "type": "tool_call"}
        for index, value in enumerate(values)
    ]
    if business:
        calls.append({"name": "effect", "args": {"value": 7}, "id": "effect-1"})
    return AIMessage(
        id=f"provider-{call_prefix}",
        content="Checked the evidence.",
        tool_calls=calls,
        additional_kwargs={"provider_data": {"preserve": True}},
        response_metadata={"finish_reason": finish_reason, "provider_request_id": "provider-1"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def _handled(message: AIMessage, *, name: str = "Result") -> ModelResponse[Any]:
    return ModelResponse(
        result=[
            message,
            *[
                ToolMessage(
                    id=f"ack-{index}",
                    content="Error: multiple structured responses. Please fix your mistakes.",
                    name=name,
                    tool_call_id=call["id"],
                    artifact={"preserve": index},
                    response_metadata={"synthetic": True},
                )
                for index, call in enumerate(message.tool_calls)
                if call["name"] == name
            ],
        ]
    )


async def _normalize(
    response: ModelResponse[Any],
    middleware: LangChainDuplicateOutputMiddleware | None = None,
    *,
    tools: list[Any] | None = None,
) -> ModelResponse[Any]:
    calls = 0

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal calls
        calls += 1
        return response

    schema = Result.model_json_schema()
    request = ModelRequest(
        model=ScriptedModel(responses=[]),
        messages=[],
        tools=tools or [],
        response_format=ToolStrategy(schema),
    )
    result = await (
        middleware or LangChainDuplicateOutputMiddleware(schema=schema, output_model=Result)
    ).awrap_model_call(request, handler)
    assert calls == 1
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [2, 3])
async def test_equivalent_final_calls_preserve_messages_usage_metadata_and_pairing(
    count: int,
) -> None:
    message = _reply([{"answer": "delivered"}] * count)
    original = _handled(message)
    snapshot = [result.model_dump() for result in original.result]

    response = await _normalize(original)

    assert response.structured_response == {"answer": "delivered"}
    assert response.result[0] is message
    assert message.usage_metadata == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    assert [result.model_dump() for result in original.result] == snapshot
    for before, after in zip(original.result[1:], response.result[1:], strict=True):
        assert isinstance(before, ToolMessage) and isinstance(after, ToolMessage)
        assert after.status == "success"
        assert after.content == "Equivalent structured responses accepted."
        assert after.model_dump(exclude={"content", "status"}) == before.model_dump(
            exclude={"content", "status"}
        )
    assert {call["id"] for call in message.tool_calls} == {
        result.tool_call_id for result in response.result if isinstance(result, ToolMessage)
    }


@pytest.mark.asyncio
async def test_nested_mapping_key_order_is_irrelevant() -> None:
    values = [
        {"answer": {"a": 1, "b": [{"x": "first", "y": "second"}]}},
        {"answer": {"b": [{"y": "second", "x": "first"}], "a": 1}},
    ]
    response = await _normalize(_handled(_reply(values)))
    assert response.structured_response == values[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values",
    [
        [{"answer": 1}, {"answer": 2}],
        [{"answer": True}, {"answer": 1}],
        [{"answer": [1, 2]}, {"answer": [2, 1]}],
        [{"answer": {"a": 1}}, {"answer": {"a": 1, "extra": 2}}],
        [{"answer": "done"}, {"answer": "done "}],
    ],
    ids=["conflict", "bool-int", "array-order", "extra-field", "text"],
)
async def test_conflicting_values_keep_native_correction(values: list[dict[str, Any]]) -> None:
    original = _handled(_reply(values))
    response = await _normalize(original)
    assert response.structured_response is None
    assert response.result[0] is original.result[0]
    for before, after in zip(original.result[1:], response.result[1:], strict=True):
        assert isinstance(after, ToolMessage)
        assert after.status == "error"
        assert after.model_dump(exclude={"status"}) == before.model_dump(exclude={"status"})


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [{"wrong": "field"}, {"answer": True}, {"answer": 1.5}])
async def test_identical_invalid_typed_candidates_are_not_accepted(value: dict[str, Any]) -> None:
    middleware = LangChainDuplicateOutputMiddleware(
        schema=IntegerResult.model_json_schema(), output_model=IntegerResult
    )
    response = await _normalize(
        _handled(_reply([value, value], name="IntegerResult"), name="IntegerResult"), middleware
    )
    assert response.structured_response is None
    assert all(
        isinstance(result, ToolMessage) and result.status == "error"
        for result in response.result[1:]
    )


@pytest.mark.asyncio
async def test_every_equivalent_candidate_gets_native_validation() -> None:
    validated = []

    class CheckedResult(Result):
        @field_validator("answer")
        @classmethod
        def check_answer(cls, value: Any) -> Any:
            validated.append(value)
            return value

    middleware = LangChainDuplicateOutputMiddleware(
        schema=CheckedResult.model_json_schema(), output_model=CheckedResult
    )
    response = await _normalize(
        _handled(_reply([{"answer": 1}] * 3, name="CheckedResult"), name="CheckedResult"),
        middleware,
    )
    assert response.structured_response == {"answer": 1}
    assert validated == [1, 1, 1]


@pytest.mark.asyncio
async def test_typed_native_validator_failure_is_not_hidden_by_schema_validity() -> None:
    class CheckedResult(IntegerResult):
        @field_validator("answer")
        @classmethod
        def check_answer(cls, value: int) -> int:
            if value == 13:
                raise ValueError("Not an accepted result")
            return value

    middleware = LangChainDuplicateOutputMiddleware(
        schema=CheckedResult.model_json_schema(), output_model=CheckedResult
    )
    response = await _normalize(
        _handled(_reply([{"answer": 13}] * 2, name="CheckedResult"), name="CheckedResult"),
        middleware,
    )
    assert response.structured_response is None


@pytest.mark.asyncio
async def test_unexpected_native_validator_errors_propagate() -> None:
    class BrokenResult(Result):
        @field_validator("answer")
        @classmethod
        def check_answer(cls, value: Any) -> Any:
            raise TypeError("Application validator bug")

    middleware = LangChainDuplicateOutputMiddleware(
        schema=BrokenResult.model_json_schema(), output_model=BrokenResult
    )
    with pytest.raises(TypeError, match="Application validator bug"):
        await _normalize(
            _handled(_reply([{"answer": 1}] * 2, name="BrokenResult"), name="BrokenResult"),
            middleware,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_values_are_not_recovered(value: float) -> None:
    response = await _normalize(_handled(_reply([{"answer": value}] * 2)))
    assert response.structured_response is None


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[None, "b"], ["", "b"], ["a", "a"], ["a", 1]])
async def test_malformed_or_duplicate_call_ids_are_untouched(ids: list[Any]) -> None:
    original = _handled(_reply([{"answer": 1}] * 2))
    message = original.result[0]
    assert isinstance(message, AIMessage)
    message.tool_calls = [
        {**call, "id": call_id} for call, call_id in zip(message.tool_calls, ids, strict=True)
    ]
    assert await _normalize(original) is original


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong-name", "extra"])
async def test_ambiguous_acknowledgements_are_untouched(problem: str) -> None:
    original = _handled(_reply([{"answer": 1}] * 2))
    if problem == "missing":
        original.result.pop()
    elif problem == "duplicate":
        original.result.append(original.result[-1])
    elif problem == "wrong-name":
        original.result[-1] = original.result[-1].model_copy(update={"name": "other"})
    else:
        original.result.append(ToolMessage(content="unrelated", tool_call_id="unrelated"))
    assert await _normalize(original) is original


@pytest.mark.asyncio
async def test_invalid_calls_are_preserved_and_prevent_recovery() -> None:
    original = _handled(_reply([{"answer": 1}] * 2))
    message = original.result[0]
    assert isinstance(message, AIMessage)
    message.invalid_tool_calls = [
        {
            "name": "broken",
            "args": "{",
            "id": "bad",
            "error": "invalid JSON",
            "type": "invalid_tool_call",
        }
    ]
    response = await _normalize(original)
    assert response.structured_response is None
    assert response.result[0] is message
    assert message.invalid_tool_calls[0]["args"] == "{"


@pytest.mark.asyncio
async def test_business_tool_name_collision_disables_recovery() -> None:
    def result(answer: int) -> int:
        return answer

    tool = StructuredTool.from_function(result, name="Result", description="Business tool")
    original = _handled(_reply([{"answer": 1}] * 2))
    for exposed in (tool, {"type": "function", "function": {"name": "Result"}}):
        assert await _normalize(original, tools=[exposed]) is original


@pytest.mark.asyncio
async def test_native_output_single_calls_and_untitled_schemas_are_untouched() -> None:
    native = ModelResponse(
        result=[AIMessage(content='{"answer": 1}')], structured_response={"answer": 1}
    )
    assert await _normalize(native) is native
    single = _handled(_reply([{"answer": 1}]))
    assert await _normalize(single) is single
    duplicate = _handled(_reply([{"answer": 1}] * 2))
    untitled = LangChainDuplicateOutputMiddleware(schema={"type": "object"})
    assert await _normalize(duplicate, untitled) is duplicate


@pytest.mark.asyncio
@pytest.mark.parametrize("ai_count", [0, 2], ids=["no-ai", "multiple-ai"])
async def test_responses_without_exactly_one_ai_message_are_untouched(ai_count: int) -> None:
    original = _handled(_reply([{"answer": 1}] * 2))
    if ai_count == 0:
        original.result.pop(0)
    else:
        original.result.insert(0, AIMessage(content="Earlier response.", id="earlier"))
    messages = original.result
    snapshot = [message.model_dump() for message in messages]

    response = await _normalize(original)

    assert response is original
    assert response.result is messages
    assert response.structured_response is None
    assert [message.model_dump() for message in messages] == snapshot


@pytest.mark.asyncio
async def test_strict_json_date_validation_is_preserved() -> None:
    middleware = LangChainDuplicateOutputMiddleware(
        schema=DateResult.model_json_schema(), output_model=DateResult
    )
    response = await _normalize(
        _handled(_reply([{"when": "2026-11-22"}] * 2, name="DateResult"), name="DateResult"),
        middleware,
    )
    assert DateResult.model_validate_json(json.dumps(response.structured_response)).when == date(
        2026, 11, 22
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("different_summary", [False, True])
async def test_subagent_candidates_never_commit_real_output_state(different_summary: bool) -> None:
    schema = wrapped_final_schema(DateResult.model_json_schema())
    output_state = SubagentOutputState(TypeAdapter(DateResult))
    previous = output_state.typed_output
    values = [
        {"task_summary": "Checked.", "final_result": {"when": "2026-11-22"}},
        {
            "task_summary": "Different." if different_summary else "Checked.",
            "final_result": {"when": "2026-11-22"},
        },
    ]
    middleware = LangChainDuplicateOutputMiddleware(schema=schema, output_state=output_state)
    response = await _normalize(
        _handled(_reply(values, name="KediSubagentResult"), name="KediSubagentResult"), middleware
    )
    assert output_state.typed_output is previous
    if different_summary:
        assert response.structured_response is None
    else:
        assert response.structured_response == values[0]
        accept_subagent_output(
            response.structured_response, validator_for(schema)(schema), output_state
        )
        assert output_state.typed_output.when == date(2026, 11, 22)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_invalid_subagent_json_schema_and_native_candidates_are_not_accepted(
    native: bool,
) -> None:
    class NativeResult(IntegerResult):
        @field_validator("answer")
        @classmethod
        def check_answer(cls, value: int) -> int:
            if value == 13:
                raise ValueError("Not an accepted result")
            return value

    schema = wrapped_final_schema(NativeResult.model_json_schema())
    state = SubagentOutputState(TypeAdapter(NativeResult))
    previous = state.typed_output
    value = (
        {"task_summary": "Checked.", "final_result": {"answer": 13}}
        if native
        else {"task_summary": "Missing final result."}
    )
    middleware = LangChainDuplicateOutputMiddleware(schema=schema, output_state=state)
    response = await _normalize(
        _handled(_reply([value] * 2, name="KediSubagentResult"), name="KediSubagentResult"),
        middleware,
    )
    assert response.structured_response is None
    assert state.typed_output is previous


@pytest.mark.asyncio
async def test_schema_only_subagent_acceptance_retains_raw_values() -> None:
    schema = wrapped_final_schema(IntegerResult.model_json_schema())
    value = {"task_summary": "Checked.", "final_result": {"answer": 7}}
    middleware = LangChainDuplicateOutputMiddleware(schema=schema)
    response = await _normalize(
        _handled(_reply([value] * 2, name="KediSubagentResult"), name="KediSubagentResult"),
        middleware,
    )
    assert response.structured_response == value
    assert isinstance(response.structured_response, dict)


@pytest.mark.asyncio
async def test_only_normal_outer_validation_commits_subagent_output_state() -> None:
    schema = wrapped_final_schema(DateResult.model_json_schema())
    state = SubagentOutputState(TypeAdapter(DateResult))
    previous = state.typed_output
    validator = validator_for(schema)(schema)
    value = {"task_summary": "Checked.", "final_result": {"when": "2026-11-22"}}
    model = ScriptedModel(responses=[_reply([value] * 3, name="KediSubagentResult")])
    accepted = []

    def accept(output: Any) -> Any:
        assert state.typed_output is previous
        accepted.append(output)
        return accept_subagent_output(output, validator, state)

    agent = create_agent(
        model=model,
        response_format=schema,
        middleware=[
            LangChainOutputCorrectionMiddleware(retries=0, accept=accept),
            LangChainDuplicateOutputMiddleware(schema=schema, output_state=state),
            RequestRetryMiddleware(retries=2),
            LangChainRunMiddleware(archive=None, tool_retries=3),
        ],
    )
    with langchain_run_scope(UsageLimits(request_limit=1)):
        result = await agent.ainvoke({"messages": [("user", "Finish")]})
    assert result["structured_response"] == value
    assert state.typed_output.when == date(2026, 11, 22)
    assert accepted == [value]
    assert len(model.seen) == 1


def test_typed_and_subagent_modes_cannot_be_combined() -> None:
    with pytest.raises(ValueError, match="not both"):
        LangChainDuplicateOutputMiddleware(
            schema=Result.model_json_schema(),
            output_model=Result,
            output_state=SubagentOutputState(TypeAdapter(Result)),
        )


def _pipeline(model: type[BaseModel] = Result, *, output_retries: int = 1) -> list[Any]:
    return [
        LangChainOutputCorrectionMiddleware(
            retries=output_retries, accept=lambda value: _accept_typed_output(model, value)
        ),
        LangChainDuplicateOutputMiddleware(schema=model.model_json_schema(), output_model=model),
        RequestRetryMiddleware(retries=2),
        LangChainRunMiddleware(archive=None, tool_retries=3),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("output_retries", [0, 1])
async def test_identical_final_output_succeeds_with_request_limit_one(output_retries: int) -> None:
    model = ScriptedModel(responses=[_reply([{"answer": 1}] * 3)])
    agent = create_agent(
        model=model,
        response_format=Result.model_json_schema(),
        middleware=_pipeline(output_retries=output_retries),
    )
    with langchain_run_scope(UsageLimits(request_limit=1)) as run:
        result = await agent.ainvoke({"messages": [("user", "Finish")]})
        assert result["structured_response"] == Result(answer=1)
        assert len(model.seen) == run.native_usage.requests == 1
        assert run.native_usage.input_tokens == 10
        assert run.native_usage.output_tokens == 5
        assert run.native_usage.tool_calls == 0
        assert not run.corrections


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["conflict", "invalid", "valid-invalid", "mixed-equal", "mixed-conflict"]
)
async def test_native_correction_preserves_business_effects_pairing_and_accounting(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    effects = []

    def effect(value: int) -> str:
        """Record the supplied value."""
        effects.append(value)
        return "recorded"

    async def unexpected_request_retry(*args: Any, **kwargs: Any) -> bool:
        pytest.fail("Structured feedback must not enter transport retry handling")

    monkeypatch.setattr(
        "kedi.agent_adapter.adapters.langchain.request_retry.wait_for_request_retry",
        unexpected_request_retry,
    )
    values = [{"answer": 1}, {"answer": 2}]
    if case in {"invalid", "valid-invalid"}:
        values = [{"answer": True}] * 2 if case == "invalid" else [{"answer": 1}, {"answer": True}]
    elif case == "mixed-equal":
        values = [{"answer": 1}] * 2
    mixed = case.startswith("mixed")
    model = ScriptedModel(
        responses=[
            _reply(values, name="IntegerResult", business=mixed),
            _reply([{"answer": 9}], name="IntegerResult", call_prefix="final"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[effect],
        response_format=IntegerResult.model_json_schema(),
        middleware=_pipeline(IntegerResult),
    )
    with langchain_run_scope(UsageLimits(request_limit=2)) as run:
        result = await agent.ainvoke({"messages": [("user", "Finish")]})
        assert result["structured_response"] == IntegerResult(answer=9)
        assert len(model.seen) == run.native_usage.requests == 2
        assert run.native_usage.input_tokens == 20
        assert run.native_usage.output_tokens == 10
        assert run.native_usage.tool_calls == int(mixed)
        assert not run.corrections
    assert effects == ([7] if mixed else [])
    corrected_request = model.seen[1]
    called = {
        call["id"]
        for message in corrected_request
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    answered = {
        message.tool_call_id for message in corrected_request if isinstance(message, ToolMessage)
    }
    assert called == answered
    rejected = [
        message
        for message in corrected_request
        if isinstance(message, ToolMessage) and message.name == "IntegerResult"
    ]
    assert len(rejected) == 2
    assert all(
        message.status == "error" and "multiple structured" in message.content
        for message in rejected
    )
    assert len([message for message in result["messages"] if isinstance(message, AIMessage)]) == 2


@pytest.mark.asyncio
async def test_mixed_business_argument_validation_is_preserved() -> None:
    effects = []

    def effect(value: int) -> str:
        """Record an integer value."""
        effects.append(value)
        return "recorded"

    initial = _reply([{"answer": 1}] * 2, business=True)
    initial.tool_calls[-1]["args"] = {"value": "not-an-integer"}
    model = ScriptedModel(responses=[initial, _reply([{"answer": 9}], call_prefix="final")])
    agent = create_agent(
        model=model,
        tools=[effect],
        response_format=Result.model_json_schema(),
        middleware=_pipeline(),
    )
    with langchain_run_scope(UsageLimits(request_limit=2)):
        result = await agent.ainvoke({"messages": [("user", "Finish")]})
    assert result["structured_response"] == Result(answer=9)
    assert not effects
    tool_errors = [
        message
        for message in model.seen[1]
        if isinstance(message, ToolMessage) and message.name == "effect"
    ]
    assert len(tool_errors) == 1
    assert tool_errors[0].status == "error"
    assert "value" in tool_errors[0].content


@pytest.mark.asyncio
async def test_conflicting_output_cannot_bypass_request_limit_one() -> None:
    model = ScriptedModel(responses=[_reply([{"answer": 1}, {"answer": 2}])])
    agent = create_agent(
        model=model, response_format=Result.model_json_schema(), middleware=_pipeline()
    )
    with langchain_run_scope(UsageLimits(request_limit=1)) as run:
        with pytest.raises(UsageLimitExceeded):
            await agent.ainvoke({"messages": [("user", "Finish")]})
        assert len(model.seen) == run.native_usage.requests == 1


@pytest.mark.asyncio
async def test_truncated_duplicates_remain_terminal_failures() -> None:
    model = ScriptedModel(responses=[_reply([{"answer": 1}] * 2, finish_reason="length")])
    agent = create_agent(
        model=model, response_format=Result.model_json_schema(), middleware=_pipeline()
    )
    with langchain_run_scope(UsageLimits(request_limit=1)):
        with pytest.raises(UnexpectedModelBehavior, match="incomplete"):
            await agent.ainvoke({"messages": [("user", "Finish")]})
    assert len(model.seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_type", "wire", "name", "expected"),
    [
        (IntegerResult, {"answer": 7}, "IntegerResult", IntegerResult(answer=7)),
        (DateResult, {"when": "2026-11-22"}, "DateResult", DateResult(when=date(2026, 11, 22))),
        (int, {"response": 7}, "KediTypedOutput", 7),
    ],
    ids=["typed", "strict-json-date", "scalar-wrapper"],
)
async def test_adapter_registration_recovers_typed_output_with_one_request(
    output_type: type[Any], wire: dict[str, Any], name: str, expected: Any
) -> None:
    model = ScriptedModel(responses=[_reply([wire] * 3, name=name)])
    adapter = LangChainAdapter(model, output_retries=0)
    usage: list[InvocationUsage] = []
    with observe_invocations(usage.append):
        result = await adapter.produce(
            "Finish", output_type=output_type, usage_limits=UsageLimits(request_limit=1)
        )
    assert result == expected
    assert type(result) is output_type
    assert len(model.seen) == 1
    assert sum(item.requests for item in usage) == 1
    assert sum(item.input_tokens for item in usage) == 10
    assert sum(item.output_tokens for item in usage) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True], ids=["schema-only", "native-state"])
@pytest.mark.parametrize("conflicting_summary", [False, True], ids=["equal", "conflict"])
async def test_adapter_registration_preserves_subagent_validation_and_summary(
    typed: bool, conflicting_summary: bool
) -> None:
    value = {"task_summary": "Checked.", "final_result": {"when": "2026-11-22"}}
    values = [value, {**value, "task_summary": "Different." if conflicting_summary else "Checked."}]
    replies = [_reply(values, name="KediSubagentResult")]
    if conflicting_summary:
        replies.append(_reply([value], name="KediSubagentResult", call_prefix="final"))
    model = ScriptedModel(responses=replies)
    state = SubagentOutputState(TypeAdapter(DateResult)) if typed else None
    request = SubagentExecutionRequest(
        profile=AgentProfile(name="worker", output_type=DateResult if typed else None),
        tools=(),
        instructions=None,
        approval_ceiling=(),
        usage_limits=SubagentUsageLimits(request_limit=2 if conflicting_summary else 1),
        output_state=state,
    )
    adapter = LangChainAdapter(model)
    usage: list[InvocationUsage] = []
    with adapter.subagent_scope(request), observe_invocations(usage.append):
        result = await adapter.produce_subagent(
            template="Finish",
            output_schema=wrapped_final_schema(DateResult.model_json_schema()),
            **adapter.subagent_run_kwargs(request.usage_limits),
        )
    assert result == value
    if state is not None:
        assert isinstance(state.typed_output, DateResult)
        assert state.typed_output.when == date(2026, 11, 22)
    expected_requests = 2 if conflicting_summary else 1
    assert len(model.seen) == expected_requests
    assert sum(item.requests for item in usage) == expected_requests
    assert sum(item.input_tokens for item in usage) == 10 * expected_requests
    assert sum(item.output_tokens for item in usage) == 5 * expected_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["conflict", "invalid", "mixed-equal", "mixed-conflict"])
async def test_adapter_registration_keeps_corrections_and_business_effects(case: str) -> None:
    effects = []

    def terminal(step: int) -> str:
        effects.append(step)
        return "recorded"

    values = [{"answer": 1}, {"answer": 2}]
    if case == "invalid":
        values = [{"answer": True}] * 2
    elif case == "mixed-equal":
        values = [{"answer": 1}] * 2
    initial = _reply(values, name="IntegerResult")
    mixed = case.startswith("mixed")
    if mixed:
        initial.tool_calls.append(
            {"name": "terminal", "args": {"step": 7}, "id": "effect-1", "type": "tool_call"}
        )
    model = ScriptedModel(
        responses=[initial, _reply([{"answer": 9}], name="IntegerResult", call_prefix="final")]
    )
    adapter = LangChainAdapter(model)
    usage: list[InvocationUsage] = []
    with adapter.tool_scope([tool_spec(terminal)]), observe_invocations(usage.append):
        result = await adapter.produce(
            "Finish", output_type=IntegerResult, usage_limits=UsageLimits(request_limit=2)
        )
    assert result == IntegerResult(answer=9)
    assert effects == ([7] if mixed else [])
    assert len(model.seen) == 2
    assert sum(item.requests for item in usage) == 2
    assert sum(item.input_tokens for item in usage) == 20
    assert sum(item.output_tokens for item in usage) == 10
    called = {
        call["id"]
        for message in model.seen[1]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    answered = {
        message.tool_call_id for message in model.seen[1] if isinstance(message, ToolMessage)
    }
    assert called == answered


@pytest.mark.asyncio
@pytest.mark.parametrize("output_retries", [0, 1, 2])
@pytest.mark.parametrize("case", ["conflict", "invalid", "valid-invalid"])
async def test_output_only_duplicate_rejections_use_existing_output_retry_budget(
    case: str, output_retries: int
) -> None:
    values = [{"answer": 1}, {"answer": 2}]
    if case == "invalid":
        values = [{"answer": True}] * 2
    elif case == "valid-invalid":
        values = [{"answer": 1}, {"answer": True}]
    failed = [
        _reply(values, name="IntegerResult", call_prefix=f"attempt-{index}")
        for index in range(output_retries + 1)
    ]
    spare = _reply([{"answer": 9}], name="IntegerResult", call_prefix="unused")
    model = ScriptedModel(responses=[*failed, spare])
    adapter = LangChainAdapter(model, output_retries=output_retries)
    usage: list[InvocationUsage] = []
    with observe_invocations(usage.append):
        with pytest.raises(MultipleStructuredOutputsError) as raised:
            await adapter.produce(
                "Finish",
                output_type=IntegerResult,
                usage_limits=UsageLimits(request_limit=output_retries + 1),
            )
    assert raised.value.ai_message.id == failed[-1].id
    assert raised.value.ai_message.tool_calls == failed[-1].tool_calls
    assert raised.value.tool_names == ["IntegerResult", "IntegerResult"]
    assert len(model.seen) == output_retries + 1
    assert model.responses == [spare]
    assert sum(item.requests for item in usage) == output_retries + 1
    assert sum(item.input_tokens for item in usage) == 10 * (output_retries + 1)
    assert sum(item.output_tokens for item in usage) == 5 * (output_retries + 1)
    for index, request in enumerate(model.seen[1:], start=1):
        called = {
            call["id"]
            for message in request
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        }
        acknowledgements = [message for message in request if isinstance(message, ToolMessage)]
        assert len(acknowledgements) == index * 2
        assert called == {message.tool_call_id for message in acknowledgements}
        assert all(message.status == "error" for message in acknowledgements)


@pytest.mark.asyncio
async def test_exhaustion_raises_the_contract_error_without_dropping_acknowledgements() -> None:
    original = _handled(_reply([{"answer": 1}, {"answer": 2}]))
    rejection = await _normalize(original)
    assert isinstance(rejection, RejectedDuplicateOutput)
    assert rejection.error.ai_message is original.result[0]
    snapshot = [message.model_dump() for message in rejection.result]
    requests = 0

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        nonlocal requests
        requests += 1
        return rejection

    request = ModelRequest(model=ScriptedModel(responses=[]), messages=[])
    correction = LangChainOutputCorrectionMiddleware(retries=0)
    with pytest.raises(MultipleStructuredOutputsError) as raised:
        await correction.awrap_model_call(request, handler)
    assert raised.value is rejection.error
    assert requests == 1
    assert [message.model_dump() for message in rejection.result] == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_first", [False, True])
async def test_duplicate_and_single_output_validation_share_one_attempt_counter(
    duplicate_first: bool,
) -> None:
    duplicate = _reply([{"answer": 1}, {"answer": 2}], name="IntegerResult")
    invalid_single = _reply([{"answer": True}], name="IntegerResult", call_prefix="invalid")
    spare = _reply([{"answer": 9}], name="IntegerResult", call_prefix="unused")
    rejected = [duplicate, invalid_single] if duplicate_first else [invalid_single, duplicate]
    model = ScriptedModel(responses=[*rejected, spare])
    adapter = LangChainAdapter(model, output_retries=1)
    error = ValidationError if duplicate_first else MultipleStructuredOutputsError
    with pytest.raises(error):
        await adapter.produce(
            "Finish", output_type=IntegerResult, usage_limits=UsageLimits(request_limit=2)
        )
    assert len(model.seen) == 2
    assert model.responses == [spare]


@pytest.mark.asyncio
@pytest.mark.parametrize("output_retries", [0, 1])
async def test_invalid_duplicate_subagent_exhaustion_preserves_real_output_state(
    output_retries: int,
) -> None:
    schema = wrapped_final_schema(DateResult.model_json_schema())
    state = SubagentOutputState(TypeAdapter(DateResult))
    previous = state.typed_output
    invalid = {"task_summary": "Missing final result."}
    model = ScriptedModel(
        responses=[
            _reply([invalid] * 2, name="KediSubagentResult", call_prefix=f"attempt-{index}")
            for index in range(output_retries + 1)
        ]
    )
    request = SubagentExecutionRequest(
        profile=AgentProfile(name="worker", output_type=DateResult),
        tools=(),
        instructions=None,
        approval_ceiling=(),
        usage_limits=SubagentUsageLimits(request_limit=output_retries + 1),
        output_state=state,
    )
    adapter = LangChainAdapter(model, output_retries=output_retries)
    with adapter.subagent_scope(request), pytest.raises(MultipleStructuredOutputsError):
        await adapter.produce_subagent(
            template="Finish",
            output_schema=schema,
            **adapter.subagent_run_kwargs(request.usage_limits),
        )
    assert state.typed_output is previous
    assert len(model.seen) == output_retries + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("conflicting", [False, True])
async def test_mixed_duplicate_turn_keeps_business_effect_with_zero_output_retries(
    conflicting: bool,
) -> None:
    effects = []

    def terminal(step: int) -> str:
        effects.append(step)
        return "recorded"

    values = [{"answer": 1}, {"answer": 2 if conflicting else 1}]
    initial = _reply(values, name="IntegerResult")
    initial.tool_calls.append(
        {"name": "terminal", "args": {"step": 7}, "id": "effect-1", "type": "tool_call"}
    )
    model = ScriptedModel(
        responses=[initial, _reply([{"answer": 9}], name="IntegerResult", call_prefix="final")]
    )
    adapter = LangChainAdapter(model, output_retries=0)
    with adapter.tool_scope([tool_spec(terminal)]):
        result = await adapter.produce(
            "Finish", output_type=IntegerResult, usage_limits=UsageLimits(request_limit=2)
        )
    assert result == IntegerResult(answer=9)
    assert effects == [7]
    assert len(model.seen) == 2
