from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool

from kedi.agent_adapter import HistoryProcessingError, HistoryProcessorContext, PydanticAdapter
from kedi.agent_adapter.adapters.pydantic.history_processing import (
    PydanticHistoryMessageCodec,
    PydanticHistoryProcessingCapability,
)
from kedi.agent_adapter.compaction import (
    HistoryArchiveSettings,
    PydanticArtifactHistoryCapability,
)
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.history_processing import HistoryProcessorRunner
from kedi.agent_adapter.tools.scope import artifact_admission_scope
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.artifacts import ArtifactManager, ArtifactPolicy


def _serialized_history() -> list[dict[str, Any]]:
    return ModelMessagesTypeAdapter.dump_python(
        [
            ModelRequest(parts=[UserPromptPart(content="obsolete")]),
            ModelResponse(parts=[TextPart(content="obsolete answer")]),
        ],
        mode="json",
    )


def _pydantic_native_history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content="obsolete")]),
        ModelResponse(parts=[TextPart(content="obsolete answer")]),
        ModelRequest(parts=[UserPromptPart(content="current task")]),
    ]


def test_pydantic_processor_persists_into_tool_loop_and_resets_continuation_once() -> None:
    seen_messages: list[list[ModelMessage]] = []
    seen_settings: list[dict[str, Any]] = []
    request_indices: list[int] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_messages.append(list(messages))
        seen_settings.append(dict(info.model_settings or {}))
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart("read_file", {}, tool_call_id="call-current")])

    def processor(ctx: HistoryProcessorContext[ModelMessage]) -> Sequence[ModelMessage]:
        request_indices.append(ctx.request_index)
        if ctx.request_index == 1:
            return ctx.messages[-1:]
        return ctx.messages

    adapter = PydanticAdapter(
        FunctionModel(respond),
        history_processor=processor,
        model_settings={
            "openai_previous_response_id": "auto",
            "openai_prompt_cache_key": "old-key",
        },
        tools=[Tool(lambda: "file contents", name="read_file")],
    )
    conversation = AdapterConversationContext(
        session_id="session",
        cache_epoch=0,
        cache_key="old-key",
        resume_state={"pydantic_messages": _serialized_history()},
        lane_identity="pydantic",
    )

    with adapter.conversation_scope(conversation):
        result = adapter.run_sync("current task")

    assert result.output == "done"
    assert request_indices == [1, 2]
    assert all("obsolete" not in str(messages) for messages in seen_messages)
    assert seen_settings[0]["openai_previous_response_id"] is None
    assert seen_settings[1]["openai_previous_response_id"] == "auto"
    assert seen_settings[0]["openai_prompt_cache_key"].endswith("_e1")
    assert seen_settings[1]["openai_prompt_cache_key"].endswith("_e1")
    assert conversation.cache_epoch == 1
    assert conversation.cache_prefix_changed
    serialized = conversation.result_state["pydantic_messages"]
    assert "obsolete" not in str(serialized)


def test_pydantic_equivalent_copy_preserves_cache_and_continuation() -> None:
    settings_seen: list[dict[str, Any]] = []

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings_seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart(content="done")])

    adapter = PydanticAdapter(
        FunctionModel(respond),
        history_processor=lambda ctx: list(ctx.messages),
        model_settings={
            "openai_previous_response_id": "auto",
            "openai_prompt_cache_key": "stable-key",
        },
    )
    conversation = AdapterConversationContext(
        session_id="session",
        cache_epoch=7,
        cache_key="stable-key",
        lane_identity="pydantic",
    )
    with adapter.conversation_scope(conversation):
        assert adapter.run_sync("current task").output == "done"

    assert settings_seen == [
        {
            "openai_previous_response_id": "auto",
            "openai_prompt_cache_key": "stable-key",
        }
    ]
    assert conversation.cache_epoch == 7
    assert not conversation.cache_prefix_changed


def test_pydantic_rejects_ambiguous_native_process_history() -> None:
    capability = ProcessHistory(lambda _ctx, messages: messages)
    with pytest.raises(HistoryProcessingError, match="another Pydantic ProcessHistory"):
        PydanticAdapter(
            FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("ok")])),
            history_processor=lambda ctx: ctx.messages,
            capabilities=[capability],
        )


def test_pydantic_rejects_invalid_per_adapter_capability_container() -> None:
    with pytest.raises(TypeError, match="capabilities must be a list or tuple"):
        PydanticAdapter(
            FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("ok")])),
            history_processor=lambda ctx: ctx.messages,
            capabilities=cast("Any", object()),
        )

    PydanticAdapter(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("ok")])),
        history_processor=lambda ctx: ctx.messages,
        capabilities=[PydanticArtifactHistoryCapability()],
    )


def test_pydantic_capability_handles_explicit_and_unknown_model_ids_without_cache_key() -> None:
    model_ids: list[str | None] = []

    def processor(ctx: HistoryProcessorContext[ModelMessage]) -> Sequence[ModelMessage]:
        model_ids.append(ctx.model_id)
        return ctx.messages[-1:]

    runner = HistoryProcessorRunner(processor, PydanticHistoryMessageCodec())
    capability = PydanticHistoryProcessingCapability(runner)

    async def run(model_id: str | None, model: Any) -> ModelRequestContext:
        request = ModelRequestContext(
            model=model,
            model_id=model_id,
            messages=_pydantic_native_history(),
            model_settings=cast(
                "ModelSettings",
                {
                    "openai_previous_response_id": "auto",
                    "openai_conversation_id": "stale-conversation",
                },
            ),
            model_request_parameters=ModelRequestParameters(),
        )
        with runner.run_scope(None):
            return await capability.before_model_request(
                cast("Any", SimpleNamespace(messages=list(request.messages))), request
            )

    first = asyncio.run(run("explicit-model", object()))
    second = asyncio.run(run(None, object()))

    assert model_ids == ["explicit-model", None]
    assert first.model_settings == {
        "openai_previous_response_id": None,
        "openai_conversation_id": None,
    }
    assert second.model_settings == first.model_settings
    assert capability.get_serialization_name() is None


def test_pydantic_processor_failure_stops_before_model_and_preserves_conversation() -> None:
    model_calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart(content="unexpected")])

    def processor(_ctx: HistoryProcessorContext[ModelMessage]) -> Sequence[ModelMessage]:
        raise ValueError("redaction policy failed")

    adapter = PydanticAdapter(FunctionModel(respond), history_processor=processor)
    conversation = AdapterConversationContext(
        session_id="failure",
        cache_epoch=5,
        cache_key="cache-e5",
        resume_state={"pydantic_messages": _serialized_history()},
        lane_identity="pydantic",
    )

    with (
        adapter.conversation_scope(conversation),
        pytest.raises(HistoryProcessingError, match="redaction policy failed"),
    ):
        adapter.run_sync("current task")

    assert model_calls == 0
    assert conversation.cache_epoch == 5
    assert not conversation.cache_prefix_changed
    assert conversation.result_state == {}


def test_pydantic_archive_runs_before_user_processor(tmp_path: Path) -> None:
    saw_checkpoint: list[bool] = []
    step = 0

    def read_file() -> str:
        nonlocal step
        step += 1
        return "x" * 900

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if step >= 12:
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(
            parts=[ToolCallPart("read_file", {}, tool_call_id=f"current-call-{step}")]
        )

    def processor(ctx: HistoryProcessorContext[ModelMessage]) -> Sequence[ModelMessage]:
        saw_checkpoint.append(any(entry.kind == "checkpoint" for entry in ctx.entries))
        return ctx.messages

    adapter = PydanticAdapter(
        FunctionModel(respond),
        history_archive=HistoryArchiveSettings(
            threshold_tokens=2_000,
            preserve_recent_tokens=500,
            minimum_reduction_tokens=500,
        ),
        history_processor=processor,
    )
    tool = ToolSpec(
        fn=read_file,
        name="read_file",
        description="Read synthetic evidence",
        json_schema={"type": "object", "properties": {}},
        risk="read_only",
        artifact_result="never",
    )
    manager = ArtifactManager(session_id="archive-before-processor")
    policy = ArtifactPolicy.patch(
        {
            "enabled": True,
            "store": "file",
            "path": tmp_path,
            "threshold_bytes": 100_000,
        }
    )
    try:
        with (
            artifact_admission_scope(manager=manager, policy=policy),
            adapter.tool_scope([tool]),
        ):
            result = adapter.run_sync("current task")
    finally:
        manager.close()

    assert result.output == "done"
    assert saw_checkpoint[-1]


@pytest.mark.parametrize("native_entry", [True, False])
def test_pydantic_nested_native_run_has_independent_history_counter(native_entry: bool) -> None:
    indices: list[int] = []

    def processor(ctx: HistoryProcessorContext[ModelMessage]) -> Sequence[ModelMessage]:
        indices.append(ctx.request_index)
        return ctx.messages

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        parts = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ]
        if any(isinstance(part, UserPromptPart) and part.content == "child" for part in parts):
            return ModelResponse(parts=[TextPart("child done")])
        returned = [part for part in parts if isinstance(part, ToolReturnPart)]
        if returned:
            assert returned[-1].content == "child done"
            return ModelResponse(parts=[TextPart("parent done")])
        return ModelResponse(parts=[ToolCallPart("delegate", {}, tool_call_id="call-child")])

    adapter = PydanticAdapter(FunctionModel(respond), history_processor=processor)

    async def delegate() -> str:
        return (await adapter.run("child")).output

    tool = ToolSpec(
        fn=delegate,
        name="delegate",
        description="Delegate a child request",
        json_schema={"type": "object", "properties": {}},
        risk="read_only",
        artifact_result="never",
    )
    with adapter.tool_scope([tool]):
        result = (
            adapter.run_sync("parent").output
            if native_entry
            else adapter.invoke_sync(prompt="parent")
        )

    assert result == "parent done"
    assert indices == [1, 1, 2]


def test_streaming_retains_new_messages_boundary_after_old_history_removal() -> None:
    seen: list[list[ModelMessage]] = []

    async def stream(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        seen.append(messages)
        yield "current "
        yield "answer"

    adapter = PydanticAdapter(
        FunctionModel(stream_function=stream),
        history_processor=lambda ctx: ctx.messages[-1:],
    )
    old = _pydantic_native_history()[:2]

    async def scenario() -> None:
        async with adapter.run_stream("current task", message_history=old) as result:
            assert await result.get_output() == "current answer"
            assert len(result.new_messages()) == 2
            assert result.new_messages() == result.all_messages()

    asyncio.run(scenario())
    assert len(seen) == 1
    assert "obsolete" not in str(seen[0])
    assert "obsolete" in str(old)


def test_explicit_response_id_stays_disabled_after_first_rewrite() -> None:
    runner = HistoryProcessorRunner(lambda ctx: ctx.messages[-1:], PydanticHistoryMessageCodec())
    capability = PydanticHistoryProcessingCapability(runner)

    async def scenario() -> None:
        with runner.run_scope(None):
            for messages in (_pydantic_native_history(), _pydantic_native_history()[-1:]):
                request = ModelRequestContext(
                    model=cast("Any", object()),
                    messages=messages,
                    model_settings=cast(
                        "ModelSettings", {"openai_previous_response_id": "resp_old"}
                    ),
                    model_request_parameters=ModelRequestParameters(),
                )
                updated = await capability.before_model_request(
                    cast("Any", SimpleNamespace(messages=list(request.messages))), request
                )
                assert updated.model_settings is not None
                assert updated.model_settings.get("openai_previous_response_id") is None

    asyncio.run(scenario())
