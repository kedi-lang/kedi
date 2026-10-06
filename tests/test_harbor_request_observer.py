from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage

from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.request import (
    ConversationCoordinates,
    RawPromptBuilder,
    RunRequestAssembler,
    TextOutput,
    lower_request_sync,
)
from kedi.agent_profile import AgentProfile
from kedi.integrations.harbor.request_observer import PydanticRequestEvidenceCapability


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _openai_response() -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-test",
        "output": [
            {
                "type": "message",
                "id": "msg_test",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "unchanged result",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": None,
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 19,
            "input_tokens_details": {"cached_tokens": 7},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 22,
        },
    }


def test_request_observer_correlates_semantic_and_native_requests(tmp_path: Path) -> None:
    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart("unchanged result")],
            usage=RequestUsage(
                input_tokens=19,
                cache_read_tokens=7,
                output_tokens=3,
                cost=Decimal("0.0125"),
            ),
        )

    capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path,
        trial_id="trial-1",
        manifest_digest="manifest-1",
    )
    adapter = PydanticAdapter(FunctionModel(respond), capabilities=[capability])
    request = RunRequestAssembler().assemble(
        prompt=RawPromptBuilder(content="inspect the repository").build(),
        instructions=(),
        conversation=ConversationCoordinates(
            session_id="session-secret",
            lane_identity="lane-1",
            cache_epoch=2,
            cache_key="cache-secret",
            history_count=4,
        ),
        tool_specs=(),
        output=TextOutput(),
        run_id="run-1",
        turn_sequence=3,
        call_kind="invoke",
    )

    assert lower_request_sync(adapter, request) == "unchanged result"

    events = _events(capability.event_path)
    assert [event["event"] for event in events] == [
        "model_request_started",
        "model_request_completed",
    ]
    started = events[0]
    assert started["correlation"]["kedi_run_id"] == "run-1"
    assert started["correlation"]["kedi_turn_sequence"] == 3
    assert started["agent_request"]["lane_identity"] == "lane-1"
    assert started["request"]["cache"]["kedi_cache_epoch"] == 2
    assert "session-secret" not in capability.event_path.read_text(encoding="utf-8")
    assert "cache-secret" not in capability.event_path.read_text(encoding="utf-8")
    assert events[1]["response"]["usage"]["cache_read_tokens"] == 7
    assert events[1]["response"]["usage"]["cost_usd"] == 0.0125
    assert capability.summary().usable is True


def test_profile_effort_preserves_constructor_request_capabilities(tmp_path: Path) -> None:
    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.thinking == "high"
        return ModelResponse(parts=[TextPart("observed")])

    capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path,
        trial_id="profile-effort",
    )
    adapter = PydanticAdapter(
        FunctionModel(respond, profile=ModelProfile(supports_thinking=True)),
        capabilities=[capability],
    )
    request = RunRequestAssembler().assemble(
        prompt=RawPromptBuilder(content="observe this request").build(),
        instructions=(),
        conversation=None,
        tool_specs=(),
        output=TextOutput(),
        run_id="profile-effort-run",
        turn_sequence=0,
        call_kind="invoke",
    )

    with adapter.profile_override(AgentProfile(effort="high")):
        assert lower_request_sync(adapter, request) == "observed"

    assert capability.summary().usable is True


def test_request_observer_does_not_change_openai_transport_payload(tmp_path: Path) -> None:
    async def exercise(
        capability: PydanticRequestEvidenceCapability | None,
    ) -> tuple[str, list[bytes]]:
        bodies: list[bytes] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            bodies.append(bytes(request.content))
            return httpx2.Response(200, json=_openai_response())

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://private.example.test/v1",
            http_client=client,
        )
        model = OpenAIResponsesModel("gpt-test", provider=provider)
        capabilities = [] if capability is None else [capability]
        agent = Agent(
            model,
            instructions="stable instructions",
            capabilities=capabilities,
        )
        try:
            results = [
                await agent.run(
                    "stable prompt",
                    model_settings=cast(
                        Any,
                        {
                            "openai_prompt_cache_key": "stable-cache-key",
                            "openai_store": False,
                        },
                    ),
                )
                for _ in range(2)
            ]
        finally:
            await client.aclose()
        return results[-1].output, bodies

    capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path,
        trial_id="transport-identity",
    )
    without_observer, baseline_bodies = asyncio.run(exercise(None))
    with_observer, observed_bodies = asyncio.run(exercise(capability))

    assert with_observer == without_observer == "unchanged result"
    assert observed_bodies == baseline_bodies
    assert observed_bodies[0] == observed_bodies[1]
    assert json.loads(observed_bodies[0])["prompt_cache_key"] == "stable-cache-key"
    evidence = capability.event_path.read_text(encoding="utf-8")
    assert "private.example.test" not in evidence
    completed = _events(capability.event_path)[-1]
    assert completed["response"]["provider_endpoint"].startswith("custom:")
    starts = [
        event
        for event in _events(capability.event_path)
        if event["event"] == "model_request_started"
    ]
    assert len({event["request"]["signature_digest"] for event in starts}) == 1
    assert all(event["request"]["transport"]["observed"] is False for event in starts)


def test_request_observer_records_normalized_usage_when_provider_omits_it(
    tmp_path: Path,
) -> None:
    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path,
        trial_id="missing-usage",
    )
    result = Agent(FunctionModel(respond), capabilities=[capability]).run_sync("hello")
    assert result.output == "done"

    completed = _events(capability.event_path)[-1]
    recorded = completed["response"]["usage"]
    assert recorded["input_tokens"] == result.usage.input_tokens
    assert recorded["output_tokens"] == result.usage.output_tokens
    assert recorded["total_tokens"] == result.usage.total_tokens
    assert recorded["provenance"] == "pydantic_ai_normalized_response_usage"


def test_request_observer_correlates_structured_output_retry(tmp_path: Path) -> None:
    class StructuredResult(BaseModel):
        value: int

    calls = 0

    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        arguments: dict[str, Any] = {"value": "invalid"} if calls == 1 else {"value": 7}
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, arguments, tool_call_id=f"out-{calls}")]
        )

    capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path,
        trial_id="structured-retry",
    )
    agent = Agent(
        FunctionModel(respond),
        output_type=StructuredResult,
        retries=1,
        capabilities=[capability],
    )

    assert agent.run_sync("produce a value").output == StructuredResult(value=7)
    events = _events(capability.event_path)
    assert [event["event"] for event in events] == [
        "model_request_started",
        "model_request_completed",
        "model_request_started",
        "model_request_completed",
    ]
    second = events[2]["request"]["history"]
    assert second["common_prefix_messages"] == 1
    assert second["added_messages"] == 2


def test_request_observer_correlates_parallel_tool_completion(tmp_path: Path) -> None:
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("delayed", {"value": "a"}, tool_call_id="call-a"),
                    ToolCallPart("immediate", {"value": "b"}, tool_call_id="call-b"),
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    async def delayed(value: str) -> str:
        await asyncio.sleep(0.01)
        return value

    async def immediate(value: str) -> str:
        await asyncio.sleep(0)
        return value

    capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path,
        trial_id="parallel-tools",
    )
    agent = Agent(
        FunctionModel(respond),
        tools=cast(Any, [delayed, immediate]),
        capabilities=[capability],
    )

    assert agent.run_sync("call both tools").output == "done"
    started = [
        event
        for event in _events(capability.event_path)
        if event["event"] == "model_request_started"
    ]
    assert len(started) == 2
    assert started[1]["request"]["history"]["common_prefix_messages"] == 1
    assert started[1]["request"]["history"]["added_messages"] == 2


def test_request_observer_detects_tool_turn_prefix_and_same_length_mutation(
    tmp_path: Path,
) -> None:
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("echo", {"value": "x"}, tool_call_id="call-1")]
            )
        return ModelResponse(parts=[TextPart("done")])

    def echo(value: str) -> str:
        return value

    capability = PydanticRequestEvidenceCapability(records_dir=tmp_path, trial_id="tool-turn")
    agent = Agent(
        FunctionModel(respond),
        tools=cast(Any, [echo]),
        capabilities=cast(Any, [capability]),
    )

    assert agent.run_sync("hello").output == "done"
    started = [
        event for event in _events(capability.event_path) if event["event"].endswith("started")
    ]
    assert len(started) == 2
    assert started[1]["request"]["history"]["common_prefix_messages"] == 1
    assert started[1]["request"]["history"]["added_messages"] > 0

    message = ModelRequest(parts=[UserPromptPart("alpha")])
    before = capability._message_fingerprint(message)  # noqa: SLF001
    part = message.parts[0]
    assert isinstance(part, UserPromptPart)
    part.content = "bravo"
    after = capability._message_fingerprint(message)  # noqa: SLF001
    assert before.digest != after.digest

    same_content = ModelRequest(parts=[UserPromptPart("bravo")])
    same = capability._message_fingerprint(same_content)  # noqa: SLF001
    assert same.digest == after.digest

    semantic_timestamp = ModelRequest(
        parts=[
            ToolReturnPart(
                "echo", {"timestamp": "semantic-value", "value": "bravo"}, tool_call_id="semantic"
            )
        ]
    )
    changed_semantic_timestamp = ModelRequest(
        parts=[
            ToolReturnPart(
                "echo", {"timestamp": "changed-value", "value": "bravo"}, tool_call_id="semantic"
            )
        ]
    )
    assert (
        capability._message_fingerprint(semantic_timestamp).digest  # noqa: SLF001
        != capability._message_fingerprint(changed_semantic_timestamp).digest  # noqa: SLF001
    )


def test_request_observer_preserves_errors_cancellation_and_capture_failures(
    tmp_path: Path,
) -> None:
    async def fail(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        raise RuntimeError("provider failed with a sensitive payload")

    failed_capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path / "failed",
        trial_id="failed",
    )
    failed_agent = Agent(FunctionModel(fail), capabilities=[failed_capability])
    with pytest.raises(RuntimeError, match="provider failed"):
        failed_agent.run_sync("hello")
    failed_events = _events(failed_capability.event_path)
    assert failed_events[-1]["event"] == "model_request_failed"
    assert failed_events[-1]["error"] == {"cancelled": False, "type": "RuntimeError"}
    assert "sensitive payload" not in failed_capability.event_path.read_text(encoding="utf-8")

    async def cancel(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        raise asyncio.CancelledError

    cancelled_capability = PydanticRequestEvidenceCapability(
        records_dir=tmp_path / "cancelled",
        trial_id="cancelled",
    )
    cancelled_agent = Agent(FunctionModel(cancel), capabilities=[cancelled_capability])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled_agent.run("hello"))
    assert _events(cancelled_capability.event_path)[-1]["error"]["cancelled"] is True

    invalid_records_dir = tmp_path / "not-a-directory"
    invalid_records_dir.write_text("occupied", encoding="utf-8")
    capture_failure = PydanticRequestEvidenceCapability(
        records_dir=invalid_records_dir,
        trial_id="capture-failure",
    )

    async def succeed(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("still succeeds")])

    result = Agent(FunctionModel(succeed), capabilities=[capture_failure]).run_sync("hello")
    assert result.output == "still succeeds"
    summary = capture_failure.summary()
    assert summary.observer_failures == 2
    assert summary.usable is False
