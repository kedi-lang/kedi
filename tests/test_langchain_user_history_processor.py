from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    message_to_dict,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from kedi.agent_adapter import HistoryProcessingError, HistoryProcessorContext, LangChainAdapter
from kedi.agent_adapter.adapters.langchain.runtime import (
    LangChainRunMiddleware,
    _without_stale_continuation,
)
from kedi.agent_adapter.compaction import HistoryArchiveSettings
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.tools.scope import artifact_admission_scope
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.artifacts import ArtifactManager, ArtifactPolicy


class HistoryScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    previous_response_flags: list[bool] = Field(default_factory=list)
    settings_seen: list[dict[str, Any]] = Field(default_factory=list)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    use_previous_response_id: bool = True
    model_name: str | None = None
    conversation: object | None = None
    conversation_id: str | None = None
    stop_on_checkpoint: bool = False
    steps: int = 0

    @property
    def _llm_type(self) -> str:
        return "history-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        self.seen.append(list(messages))
        self.previous_response_flags.append(self.use_previous_response_id)
        self.settings_seen.append(dict(kwargs))
        if self.stop_on_checkpoint:
            if any(
                message.additional_kwargs.get("kedi_compaction_checkpoint") for message in messages
            ):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
            self.steps += 1
            if self.steps > 12:
                raise AssertionError("history archive did not produce a checkpoint")
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "read_file",
                                    "args": {},
                                    "id": f"current-call-{self.steps}",
                                }
                            ],
                        )
                    )
                ]
            )
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def _tool_spec(result: str = "file contents") -> ToolSpec:
    return ToolSpec(
        fn=lambda: result,
        name="read_file",
        description="Read a file",
        json_schema={"type": "object", "properties": {}},
        risk="read_only",
        artifact_result="never",
    )


def test_langchain_processor_overwrites_graph_and_resets_continuation_once() -> None:
    request_indices: list[int] = []
    model = HistoryScriptedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {}, "id": "call-current"}],
            ),
            AIMessage(content="done"),
        ],
        model_kwargs={"prompt_cache_key": "old-key"},
    )

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        request_indices.append(ctx.request_index)
        if ctx.request_index == 1:
            return ctx.messages[-1:]
        return ctx.messages

    adapter = LangChainAdapter(model, history_processor=processor)
    conversation = AdapterConversationContext(
        session_id="session",
        cache_epoch=0,
        cache_key="old-key",
        resume_state={
            "langchain_messages": [
                message_to_dict(HumanMessage(content="obsolete")),
                message_to_dict(AIMessage(content="obsolete answer")),
            ]
        },
        lane_identity="langchain",
    )

    with adapter.conversation_scope(conversation), adapter.tool_scope([_tool_spec()]):
        assert adapter.invoke_sync(prompt="current task") == "done"

    assert request_indices == [1, 2]
    assert all("obsolete" not in str(messages) for messages in model.seen)
    assert model.previous_response_flags == [False, True]
    assert conversation.cache_epoch == 1
    assert conversation.cache_prefix_changed
    assert "obsolete" not in str(conversation.result_state["langchain_messages"])


def test_langchain_equivalent_copy_preserves_continuation() -> None:
    model_ids: list[str | None] = []
    model = HistoryScriptedModel(
        responses=[AIMessage(content="done")],
        model_kwargs={"prompt_cache_key": "stable-key"},
        model_name="history-model",
    )

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        model_ids.append(ctx.model_id)
        return list(ctx.messages)

    adapter = LangChainAdapter(model, history_processor=processor)
    conversation = AdapterConversationContext(
        session_id="session",
        cache_epoch=3,
        cache_key="stable-key",
        lane_identity="langchain",
    )

    with adapter.conversation_scope(conversation):
        assert adapter.invoke_sync(prompt="current task") == "done"

    assert model.previous_response_flags == [True]
    assert model_ids == ["history-model"]
    assert conversation.cache_epoch == 3
    assert not conversation.cache_prefix_changed


def test_langchain_middleware_uses_staged_cache_key_for_every_request_after_rewrite() -> None:
    model = HistoryScriptedModel(
        responses=[],
        model_kwargs={
            "prompt_cache_key": "old-key",
            "conversation": "stale-model-conversation",
            "previous_response_id": "stale-model-response",
        },
        conversation=object(),
        conversation_id="stale-model-conversation-id",
    )

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        return ctx.messages[-1:] if ctx.request_index == 1 else ctx.messages

    adapter = LangChainAdapter(model, history_processor=processor)
    runner = adapter._history_processor_runner
    assert runner is not None
    middleware = LangChainRunMiddleware(
        archive=None,
        history_processor=runner,
        model_id="test-model",
        tool_retries=0,
    )
    conversation = AdapterConversationContext(
        session_id="session",
        cache_epoch=0,
        cache_key="old-key",
        lane_identity="langchain",
    )
    seen = []

    async def handler(request):
        seen.append(request)
        return "ok"

    async def run() -> None:
        with runner.run_scope(conversation):
            first_messages = [
                HumanMessage(content="obsolete"),
                AIMessage(content="obsolete answer"),
                HumanMessage(content="current"),
            ]
            first_update = await middleware.abefore_model(
                {"messages": first_messages},
                runtime=None,
            )
            assert first_update is not None
            first_processed = first_update["messages"].value
            first_request = SimpleNamespace(
                messages=first_processed,
                model=model,
                model_settings={
                    "prompt_cache_key": "request-key",
                    "conversation": "stale-setting-conversation",
                    "conversation_id": "stale-setting-conversation-id",
                    "previous_response_id": "stale-setting-response",
                },
            )
            first_request.override = lambda **changes: SimpleNamespace(
                **{**vars(first_request), **changes}
            )
            await middleware.awrap_model_call(first_request, handler)

            second_messages = [
                *first_processed,
                AIMessage(content="first answer"),
                HumanMessage(content="next"),
            ]
            assert (
                await middleware.abefore_model(
                    {"messages": second_messages},
                    runtime=None,
                )
                is None
            )
            second_request = SimpleNamespace(
                messages=second_messages,
                model=model,
                model_settings={},
            )
            second_request.override = lambda **changes: SimpleNamespace(
                **{**vars(second_request), **changes}
            )
            await middleware.awrap_model_call(second_request, handler)

    asyncio.run(run())

    assert seen[0].model.use_previous_response_id is False
    assert seen[1].model.use_previous_response_id is True
    assert seen[0].model.conversation is None
    assert seen[0].model.conversation_id is None
    assert seen[0].model.model_kwargs == {"prompt_cache_key": "old-key"}
    assert "conversation" not in seen[0].model_settings
    assert "conversation_id" not in seen[0].model_settings
    assert "previous_response_id" not in seen[0].model_settings
    assert seen[0].model_settings["prompt_cache_key"].endswith("_e1")
    assert seen[1].model_settings["prompt_cache_key"].endswith("_e1")


def test_langchain_rewrite_without_cache_context_only_resets_continuation() -> None:
    model = HistoryScriptedModel(responses=[])

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        return ctx.messages[-1:]

    adapter = LangChainAdapter(model, history_processor=processor)
    runner = adapter._history_processor_runner
    assert runner is not None
    middleware = LangChainRunMiddleware(
        archive=None,
        history_processor=runner,
        model_id="test-model",
        tool_retries=0,
    )
    seen = []

    async def handler(request):
        seen.append(request)
        return "ok"

    async def run() -> None:
        with runner.run_scope(None):
            update = await middleware.abefore_model(
                {
                    "messages": [
                        HumanMessage(content="obsolete"),
                        AIMessage(content="obsolete answer"),
                        HumanMessage(content="current"),
                    ]
                },
                runtime=None,
            )
            assert update is not None
            request = SimpleNamespace(
                messages=update["messages"].value,
                model=model,
                model_settings={},
            )
            request.override = lambda **changes: SimpleNamespace(**{**vars(request), **changes})
            await middleware.awrap_model_call(request, handler)

    asyncio.run(run())

    assert len(seen) == 1
    assert seen[0].model.use_previous_response_id is False
    assert seen[0].model_settings == {}


def test_langchain_rewrite_rejects_model_that_cannot_clear_stale_continuation() -> None:
    model = SimpleNamespace(use_previous_response_id=True)

    with pytest.raises(HistoryProcessingError, match="cannot disable its stale"):
        _without_stale_continuation(model, reset_previous_response=True)


def test_langchain_processor_failure_stops_before_model_and_preserves_conversation() -> None:
    model = HistoryScriptedModel(
        responses=[AIMessage(content="unexpected")],
        model_kwargs={"prompt_cache_key": "cache-e5"},
    )

    def processor(_ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        raise ValueError("redaction policy failed")

    adapter = LangChainAdapter(model, history_processor=processor)
    conversation = AdapterConversationContext(
        session_id="failure",
        cache_epoch=5,
        cache_key="cache-e5",
        resume_state={
            "langchain_messages": [
                message_to_dict(HumanMessage(content="obsolete")),
                message_to_dict(AIMessage(content="obsolete answer")),
            ]
        },
        lane_identity="langchain",
    )

    with (
        adapter.conversation_scope(conversation),
        pytest.raises(HistoryProcessingError, match="redaction policy failed"),
    ):
        adapter.invoke_sync(prompt="current task")

    assert model.seen == []
    assert conversation.cache_epoch == 5
    assert not conversation.cache_prefix_changed
    assert conversation.result_state == {}


def test_langchain_archive_runs_before_user_processor(tmp_path: Path) -> None:
    saw_checkpoint: list[bool] = []

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        saw_checkpoint.append(any(entry.kind == "checkpoint" for entry in ctx.entries))
        return ctx.messages

    model = HistoryScriptedModel(responses=[], stop_on_checkpoint=True)
    adapter = LangChainAdapter(
        model,
        history_archive=HistoryArchiveSettings(
            threshold_tokens=2_000,
            preserve_recent_tokens=500,
            minimum_reduction_tokens=500,
        ),
        history_processor=processor,
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
            adapter.tool_scope([_tool_spec("x" * 900)]),
        ):
            assert adapter.invoke_sync(prompt="current task") == "done"
    finally:
        manager.close()

    assert saw_checkpoint[-1]
