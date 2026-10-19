from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Literal

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, TypeAdapter, model_validator
from pydantic_ai import UsageLimits
from pydantic_ai.capabilities import AbstractCapability, OutputContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.subagents.runs import SubagentCompleted, SubagentFailed
from kedi.agent_adapter.subagents import wrapped_final_schema
from kedi.agent_adapter.subagents import DelegationResult, SubagentOutputState
from kedi.agent_adapter.subagents.native import SubagentTask
from kedi.lang import compile_program, parse_program

FINAL_SCHEMA = {
    "type": "object",
    "properties": {"investigation": {"type": "string", "enum": ["checked"]}},
    "required": ["investigation"],
    "additionalProperties": False,
}
GOOD = {"task_summary": "Read evidence", "final_result": {"investigation": "checked"}}
MISNESTED = {"final_result": {"investigation": "checked", "task_summary": "Read evidence"}}
AT_LIMIT = {**GOOD, "task_summary": "a" * 1000}
OVER_LIMIT = {**GOOD, "task_summary": "a" * 1001}
SOURCE = """
> profile: analyst:
  > instructions: Return the requested field.
> profile: coordinator:
  > subagent: analyst
  > max_agents: 2
> use: coordinator
> task [job]: analyst:
  >> The investigation status is [investigation: str].
> await [result]: job
= `result.output.investigation`
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", [GOOD, MISNESTED, AT_LIMIT, OVER_LIMIT])
async def test_native_task_repairs_invalid_responses_output(initial: dict[str, Any]) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(requests)}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "offline-model",
                "output": [
                    {
                        "type": "function_call",
                        "id": f"fc_{len(requests)}",
                        "call_id": f"call_{len(requests)}",
                        "name": "final_result",
                        "arguments": json.dumps(initial if len(requests) == 1 else GOOD),
                        "status": "completed",
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http,
        AsyncOpenAI(api_key="offline-fixture", http_client=http, max_retries=0) as client,
    ):
        model = OpenAIResponsesModel("offline-model", provider=OpenAIProvider(openai_client=client))
        with use_async_owner_loop():
            runtime = compile_program(
                parse_program(SOURCE),
                adapter=PydanticAdapter(model),
                subagent_max_depth=1,
                subagent_max_concurrency=2,
            )
            try:
                assert await asyncio.to_thread(runtime.run_main) == "checked"
            finally:
                await runtime.aclose()

    assert len(requests) == (1 if initial in (GOOD, AT_LIMIT) else 2)
    output_tool = next(tool for tool in requests[0]["tools"] if tool["name"] == "final_result")
    assert output_tool["strict"] is True
    assert "maxLength" not in output_tool["parameters"]["properties"]["task_summary"]
    assert wrapped_final_schema(FINAL_SCHEMA)["properties"]["task_summary"]["maxLength"] == 1000
    if initial in (MISNESTED, OVER_LIMIT):
        feedback = [
            item for item in requests[1]["input"] if item.get("type") == "function_call_output"
        ]
        assert any("JSON Schema" in item["output"] for item in feedback)
        if initial == MISNESTED:
            assert any("required property" in item["output"] for item in feedback)
            assert any("unexpected" in item["output"] for item in feedback)
        else:
            assert any("length 1001 exceeds maxLength 1000" in item["output"] for item in feedback)


@pytest.mark.asyncio
async def test_native_task_repairs_pydantic_business_validator_before_completion() -> None:
    validations = 0

    class Stock(BaseModel):
        available: int
        requested: int
        sufficient: bool

        @model_validator(mode="after")
        def consistent(self) -> Stock:
            nonlocal validations
            validations += 1
            if self.sufficient != (self.available >= self.requested):
                raise ValueError("Availability decision contradicts the quantities")
            return self

    calls = 0
    feedback: list[RetryPromptPart] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        feedback.extend(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "task_summary": "Checked inventory",
                        "final_result": {
                            "stock": {
                                "available": 20,
                                "requested": 18,
                                "sufficient": calls == 2,
                            }
                        },
                    },
                )
            ]
        )

    source = """
> profile: worker:
  > instructions: Check the inventory.
> profile: coordinator:
  > subagent: worker
> use: coordinator
> task [job]: worker:
  >> The stock status is [stock: Stock].
> await [result]: job
= `result.output.stock.sufficient`
"""
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(source),
            adapter=PydanticAdapter(FunctionModel(respond), retries={"output": 1}),
            runtime_globals={"Stock": Stock},
        )
        try:
            assert await asyncio.to_thread(runtime.run_main) is True
            assert runtime._globals_env["result"].output.stock.available == 20
            assert runtime._globals_env["result"].output.stock.sufficient is True
        finally:
            await runtime.aclose()

    assert calls == 2
    assert validations == 2
    assert len(feedback) == 1
    assert "Availability decision contradicts the quantities" in str(feedback[0].content)


@pytest.mark.asyncio
async def test_invalid_native_validator_never_commits_completed_child(tmp_path: Any) -> None:
    class Stock(BaseModel):
        available: int
        requested: int
        sufficient: bool

        @model_validator(mode="after")
        def consistent(self) -> Stock:
            if self.sufficient != (self.available >= self.requested):
                raise ValueError("Availability decision contradicts the quantities")
            return self

    attempts = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "task_summary": "Checked",
                        "final_result": {
                            "stock": {
                                "available": 20,
                                "requested": 18,
                                "sufficient": False,
                            }
                        },
                    },
                )
            ]
        )

    source = """
> profile: worker:
  > instructions: Check stock.
> profile: coordinator:
  > subagent: worker
> use: coordinator
> task [job]: worker:
  >> The stock status is [stock: Stock].
> await [result]: job
= `result.output.stock.sufficient`
"""
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(source),
            adapter=PydanticAdapter(FunctionModel(respond), retries={"output": 1}),
            runtime_globals={"Stock": Stock},
            subagent_state_path=tmp_path / "children.json",
        )
        try:
            with pytest.raises(Exception, match="Exceeded maximum output retries"):
                await asyncio.to_thread(runtime.run_main)
            events = runtime._subagent_coordinator.events.snapshot()
            assert not any(isinstance(event, SubagentCompleted) for event in events)
            assert sum(isinstance(event, SubagentFailed) for event in events) == 1
        finally:
            await runtime.aclose()
    assert attempts == 2
    persisted = json.loads((tmp_path / "children.json").read_text())
    assert persisted["results"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [True, False])
async def test_native_validator_transformation_is_used_once_and_persisted(
    tmp_path: Any, capture: bool
) -> None:
    validations = 0

    class Label(BaseModel):
        value: str

        @model_validator(mode="after")
        def normalize(self) -> Label:
            nonlocal validations
            validations += 1
            self.value = self.value.upper()
            return self

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        value = {"label": {"value": "ready"}} if capture else {"value": "ready"}
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "task_summary": "Checked",
                        "final_result": value,
                    },
                )
            ]
        )

    output_directive = "" if capture else "  > output: Label\n"
    task = "The label is [label: Label]." if capture else "Check the label."
    access = "result.output.label.value" if capture else "result.output.value"
    source = (
        "> profile: worker:\n"
        "  > instructions: Check a label.\n"
        + output_directive
        + "> profile: coordinator:\n  > subagent: worker\n"
        + "> use: coordinator\n"
        + f"> task [job]: worker:\n  >> {task}\n"
        + "> await [result]: job\n"
        + f"= `{access}`\n"
    )
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(source),
            adapter=PydanticAdapter(FunctionModel(respond)),
            runtime_globals={"Label": Label},
            subagent_state_path=tmp_path / "children.json",
        )
        try:
            assert await asyncio.to_thread(runtime.run_main) == "READY"
        finally:
            await runtime.aclose()

    assert validations == 1
    persisted = json.loads((tmp_path / "children.json").read_text())
    result = next(iter(persisted["results"].values()))
    field = result["final_result"]["label"] if capture else result["final_result"]
    assert field["value"] == "READY"


@pytest.mark.asyncio
async def test_repair_feedback_bounds_many_schema_errors() -> None:
    required = {f"field_{index}": {"type": "integer"} for index in range(12)}
    schema = wrapped_final_schema(
        {
            "type": "object",
            "properties": required,
            "required": list(required),
            "additionalProperties": False,
        }
    )
    feedback: list[RetryPromptPart] = []
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        feedback.extend(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        value = {} if requests == 1 else {name: 1 for name in required}
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "task_summary": "Checked",
                        "final_result": value,
                    },
                )
            ]
        )

    adapter = PydanticAdapter(FunctionModel(respond), retries={"output": 1})
    result = await adapter.produce_subagent(template="Check fields", output_schema=schema)
    assert result["final_result"] == {name: 1 for name in required}
    assert requests == 2
    assert len(feedback) == 1
    assert str(feedback[0].content).count("required property") == 8
    assert len(str(feedback[0].content)) <= 4000


@pytest.mark.asyncio
async def test_validator_normalization_cannot_commit_schema_invalid_value() -> None:
    class Gate(BaseModel):
        value: Literal["ready"]

        @model_validator(mode="after")
        def alter(self) -> Gate:
            object.__setattr__(self, "value", "changed")
            return self

    attempts = 0
    feedback: list[RetryPromptPart] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        feedback.extend(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "task_summary": "Checked",
                        "final_result": {"gate": {"value": "ready"}},
                    },
                )
            ]
        )

    source = """
> profile: worker:
  > instructions: Check a gate.
> profile: coordinator:
  > subagent: worker
> use: coordinator
> task [job]: worker:
  >> The gate is [gate: Gate].
> await [result]: job
= `result.output.gate.value`
"""
    with use_async_owner_loop():
        runtime = compile_program(
            parse_program(source),
            adapter=PydanticAdapter(FunctionModel(respond), retries={"output": 1}),
            runtime_globals={"Gate": Gate},
        )
        try:
            with pytest.raises(Exception, match="Exceeded maximum output retries"):
                await asyncio.to_thread(runtime.run_main)
            assert not any(
                isinstance(event, SubagentCompleted)
                for event in runtime._subagent_coordinator.events.snapshot()
            )
        finally:
            await runtime.aclose()
    assert attempts == 2
    assert any("serialization violates its JSON Schema" in str(part.content) for part in feedback)


@pytest.mark.asyncio
async def test_native_wait_rejects_unvalidated_success_from_an_adapter() -> None:
    async def wait(run_id: str) -> DelegationResult:
        return DelegationResult(
            run_id=run_id, subagent="worker", task_summary="Done", final_result="unchecked"
        )

    owner = SimpleNamespace(
        _check_owner=lambda: None,
        _bind_execution_loop=lambda: None,
        runtime=SimpleNamespace(_wait_subagent=wait, _subagent_status=lambda run_id: "running"),
        _observed=set(),
    )
    task = SubagentTask(owner, "child", SubagentOutputState(TypeAdapter(str)))
    with pytest.raises(RuntimeError, match="without native output validation"):
        await task.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        MISNESTED,
        {"task_summary": "Read evidence"},
        {**GOOD, "extra": True},
        {**GOOD, "task_summary": "x" * 1001},
        {**GOOD, "final_result": {"investigation": "unsupported"}},
        {**GOOD, "final_result": {"investigation": 42}},
    ],
)
async def test_subagent_repairs_full_schema_and_preserves_capabilities(
    invalid: dict[str, Any],
) -> None:
    attempts = 0
    observed: list[Any] = []
    feedback: list[RetryPromptPart] = []

    class Observer(AbstractCapability[Any]):
        async def after_output_process(
            self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
        ) -> Any:
            observed.append(output)
            return output

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        feedback.extend(
            part
            for message in messages
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        )
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, invalid if attempts == 1 else GOOD)]
        )

    adapter = PydanticAdapter(FunctionModel(respond), retries={"output": 1})
    capabilities = [Observer()]
    result = await adapter.produce_subagent(
        template="Review evidence",
        output_schema=wrapped_final_schema(FINAL_SCHEMA),
        capabilities=capabilities,
    )
    assert result == GOOD
    assert attempts == 2
    assert len(capabilities) == 1
    assert observed == [GOOD]
    assert len(feedback) == 1
    assert "JSON Schema" in str(feedback[0].content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retries", "limits", "error", "expected_requests"),
    [
        (0, UsageLimits(request_limit=10), UnexpectedModelBehavior, 1),
        (2, UsageLimits(request_limit=10), UnexpectedModelBehavior, 3),
        (5, UsageLimits(request_limit=1), UsageLimitExceeded, 1),
        (5, UsageLimits(total_tokens_limit=1), UsageLimitExceeded, 1),
    ],
)
async def test_subagent_output_repair_obeys_retry_and_usage_limits(
    retries: int, limits: UsageLimits, error: type[Exception], expected_requests: int
) -> None:
    attempts = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, MISNESTED)],
            usage=RequestUsage(input_tokens=1, output_tokens=1),
        )

    adapter = PydanticAdapter(FunctionModel(respond))
    with pytest.raises(error):
        await adapter.produce_subagent(
            template="Review evidence",
            output_schema=wrapped_final_schema(FINAL_SCHEMA),
            retries={"output": retries},
            usage_limits=limits,
        )
    assert attempts == expected_requests


@pytest.mark.asyncio
async def test_concurrent_child_repairs_do_not_leak_schemas_or_restart_siblings() -> None:
    attempts = {"integer": 0, "string": 0}
    arrived = 0
    both_started = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal arrived
        kind = info.output_tools[0].parameters_json_schema["properties"]["final_result"]["type"]
        attempts[kind] += 1
        if attempts[kind] == 1:
            arrived += 1
            if arrived == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=5)
        value = 7 if kind == "integer" and attempts[kind] > 1 else "checked"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name, {"task_summary": "Done", "final_result": value}
                )
            ]
        )

    adapter = PydanticAdapter(FunctionModel(respond))
    results = await asyncio.gather(
        *(
            adapter.produce_subagent(
                template=f"Return a {kind}",
                output_schema=wrapped_final_schema({"type": kind}),
            )
            for kind in ("integer", "string")
        )
    )
    assert [result["final_result"] for result in results] == [7, "checked"]
    assert attempts == {"integer": 2, "string": 1}

    class OrdinaryOutput(BaseModel):
        value: str

    def ordinary_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"value": "unrelated"})]
        )

    with adapter.override(model=FunctionModel(ordinary_response)):
        output = await adapter.produce(template="Return a value", output_type=OrdinaryOutput)
    assert output.value == "unrelated"
