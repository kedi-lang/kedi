from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from kedi.agent_adapter import (
    HistoryProcessingError,
    HistoryProcessingValidationError,
    HistoryProcessorContext,
)
from kedi.agent_adapter.adapters.langchain.history_processing import (
    LangChainHistoryMessageCodec,
)
from kedi.agent_adapter.adapters.pydantic.history_processing import (
    PydanticHistoryMessageCodec,
)
from kedi.agent_adapter.conversation import AdapterConversationContext, ConversationState
from kedi.agent_adapter.history_processing import (
    HistoryProcessorRunner,
    rebase_history_cache_key,
)


def _pydantic_history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content="old task")]),
        ModelResponse(parts=[ToolCallPart("read_file", {"path": "a.py"}, tool_call_id="call-1")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read_file",
                    content="old result",
                    tool_call_id="call-1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="old answer")]),
        ModelRequest(parts=[UserPromptPart(content="current task")]),
    ]


def _langchain_history() -> list[BaseMessage]:
    return [
        HumanMessage(content="old task"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "call-1"}],
        ),
        ToolMessage(content="old result", tool_call_id="call-1"),
        AIMessage(content="old answer"),
        HumanMessage(content="current task"),
    ]


@pytest.mark.parametrize(
    ("messages", "codec"),
    [
        (_pydantic_history(), PydanticHistoryMessageCodec()),
        (_langchain_history(), LangChainHistoryMessageCodec()),
    ],
)
def test_equivalent_native_copy_is_a_noop(messages, codec) -> None:
    contexts: list[HistoryProcessorContext] = []

    async def processor(ctx: HistoryProcessorContext):
        contexts.append(ctx)
        await asyncio.sleep(0)
        return list(ctx.messages)

    runner = HistoryProcessorRunner(processor, codec)
    conversation = AdapterConversationContext(
        session_id="session",
        cache_epoch=4,
        cache_key="cache-e4",
        lane_identity=codec.adapter_shortname,
    )

    async def run():
        with runner.run_scope(conversation):
            return await runner.process(messages, model_id="test-model")

    outcome = asyncio.run(run())

    assert not outcome.changed
    assert conversation.cache_epoch == 4
    assert not conversation.cache_prefix_changed
    assert outcome.messages == messages
    assert contexts[0].request_index == 1
    assert contexts[0].cache_epoch == 4
    assert contexts[0].model_id == "test-model"
    assert contexts[0].estimated_tokens > 0
    assert contexts[0].groups


@pytest.mark.parametrize(
    ("messages", "codec"),
    [
        (_pydantic_history(), PydanticHistoryMessageCodec()),
        (_langchain_history(), LangChainHistoryMessageCodec()),
    ],
)
def test_complete_old_group_can_be_removed_and_stages_cache_generation(messages, codec) -> None:
    runner = HistoryProcessorRunner(lambda ctx: ctx.messages[-1:], codec)
    conversation = AdapterConversationContext(
        session_id="session",
        cache_epoch=2,
        cache_key="old-key",
        lane_identity=codec.adapter_shortname,
    )

    async def run():
        with runner.run_scope(conversation):
            outcome = await runner.process(messages, model_id=None)
            directive = runner.prepare_transport()
            return outcome, directive

    outcome, directive = asyncio.run(run())

    assert outcome.changed
    assert len(outcome.messages) == 1
    assert conversation.cache_epoch == 3
    assert conversation.cache_prefix_changed
    assert conversation.cache_key.endswith("_e3")
    assert directive.reset_continuation
    assert directive.cache_key == conversation.cache_key


@pytest.mark.parametrize(
    ("messages", "codec"),
    [
        (_pydantic_history(), PydanticHistoryMessageCodec()),
        (_langchain_history(), LangChainHistoryMessageCodec()),
    ],
)
def test_partial_tool_group_is_rejected_before_transport(messages, codec) -> None:
    def processor(ctx: HistoryProcessorContext):
        return [ctx.messages[0], ctx.messages[1], ctx.messages[3], ctx.messages[4]]

    runner = HistoryProcessorRunner(processor, codec)

    async def run():
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="atomic history group"):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("messages", "codec"),
    [
        (_pydantic_history(), PydanticHistoryMessageCodec()),
        (_langchain_history(), LangChainHistoryMessageCodec()),
    ],
)
def test_current_request_tail_cannot_be_removed_or_replaced(messages, codec) -> None:
    runner = HistoryProcessorRunner(lambda ctx: ctx.messages[:-1], codec)

    async def run():
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="current request tail"):
        asyncio.run(run())


@pytest.mark.parametrize("result", ["bad", b"bad", bytearray(b"bad"), {"bad": "value"}, None])
def test_processor_rejects_non_message_sequences(result: object) -> None:
    runner = HistoryProcessorRunner(
        lambda _ctx: cast("Any", result),
        LangChainHistoryMessageCodec(),
    )

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(_langchain_history(), model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="sequence of native messages"):
        asyncio.run(run())


def test_processor_rejects_empty_history() -> None:
    runner = HistoryProcessorRunner(lambda _ctx: (), LangChainHistoryMessageCodec())

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(_langchain_history(), model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="cannot return empty history"):
        asyncio.run(run())


def test_processor_rejects_wrong_native_message_type() -> None:
    runner = HistoryProcessorRunner(
        lambda _ctx: cast("Any", [object()]),
        LangChainHistoryMessageCodec(),
    )

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(_langchain_history(), model_id=None)

    with pytest.raises(
        HistoryProcessingValidationError, match="not valid langchain native history"
    ):
        asyncio.run(run())


def test_processor_requires_run_scope_for_processing_and_transport() -> None:
    runner = HistoryProcessorRunner(lambda ctx: ctx.messages, LangChainHistoryMessageCodec())

    with pytest.raises(HistoryProcessingError, match="outside an adapter run scope"):
        asyncio.run(runner.process(_langchain_history(), model_id=None))
    with pytest.raises(HistoryProcessingError, match="transport prepared outside"):
        runner.prepare_transport()


def test_processor_configuration_must_be_callable() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        HistoryProcessorRunner(cast("Any", "not callable"), LangChainHistoryMessageCodec())


def test_codec_inspection_failures_are_wrapped_without_message_contents() -> None:
    class BrokenCodec:
        adapter_shortname = "broken"
        state_key = "messages"

        @staticmethod
        def clone(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
            return list(messages)

        @staticmethod
        def inspect(_messages: Sequence[BaseMessage]):
            raise ValueError("invalid payload")

    runner = HistoryProcessorRunner(lambda ctx: ctx.messages, BrokenCodec())

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(_langchain_history(), model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="source could not be inspected"):
        asyncio.run(run())


def test_processor_rejects_non_native_codec_snapshot() -> None:
    native_codec = LangChainHistoryMessageCodec()

    class SerializedCodec:
        adapter_shortname = "serialized"
        state_key = "messages"

        @staticmethod
        def clone(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
            return native_codec.clone(messages)

        @staticmethod
        def inspect(messages: Sequence[BaseMessage]):
            return replace(native_codec.inspect(messages), storage_format="serialized")

    runner = HistoryProcessorRunner(lambda ctx: ctx.messages, SerializedCodec())

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(_langchain_history(), model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="only native serialized messages"):
        asyncio.run(run())


def test_source_history_must_contain_current_request() -> None:
    runner = HistoryProcessorRunner(
        lambda _ctx: [HumanMessage(content="fabricated current request")],
        LangChainHistoryMessageCodec(),
    )

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process([], model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="source history must contain"):
        asyncio.run(run())


def test_protected_system_history_cannot_be_removed_or_reordered() -> None:
    messages: list[BaseMessage] = [
        SystemMessage(content="system"),
        HumanMessage(
            content="checkpoint",
            additional_kwargs={"kedi_compaction_checkpoint": "checkpoint-1"},
        ),
        HumanMessage(content="current task"),
    ]

    async def run(processor) -> None:
        runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="removed provider-required"):
        asyncio.run(run(lambda ctx: ctx.messages[1:]))
    with pytest.raises(HistoryProcessingValidationError, match="reordered protected"):
        asyncio.run(run(lambda ctx: [ctx.messages[1], ctx.messages[0], ctx.messages[2]]))
    with pytest.raises(HistoryProcessingValidationError, match="system-history prefix"):
        asyncio.run(run(lambda ctx: [AIMessage(content="summary"), *ctx.messages]))


def test_atomic_group_internal_order_cannot_change() -> None:
    messages = _langchain_history()

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        return [
            ctx.messages[3],
            *ctx.messages[0:3],
            ctx.messages[4],
        ]

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="ordering inside atomic group"):
        asyncio.run(run())


def test_processor_may_add_text_summary_but_not_tool_lifecycle() -> None:
    messages = _langchain_history()

    async def run(processor):
        runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())
        with runner.run_scope(None):
            return await runner.process(messages, model_id=None)

    summary = asyncio.run(
        run(lambda ctx: [AIMessage(content="summary of completed work"), ctx.messages[-1]])
    )
    assert summary.changed
    assert summary.messages[0].content == "summary of completed work"

    with pytest.raises(HistoryProcessingValidationError, match="cannot fabricate"):
        asyncio.run(
            run(
                lambda ctx: [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "read_file", "args": {}, "id": "forged"}],
                    ),
                    ctx.messages[-1],
                ]
            )
        )


def test_callback_receives_detached_history_and_failure_does_not_mutate_source() -> None:
    messages = _pydantic_history()
    original_length = len(messages[0].parts)

    def processor(ctx: HistoryProcessorContext[ModelMessage]) -> Sequence[ModelMessage]:
        assert isinstance(ctx.messages[0], ModelRequest)
        ctx.messages[0].parts.append(UserPromptPart(content="detached mutation"))
        raise ValueError("policy failed")

    runner = HistoryProcessorRunner(processor, PydanticHistoryMessageCodec())

    async def run():
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingError, match="policy failed"):
        asyncio.run(run())
    assert len(messages[0].parts) == original_length


def test_accepted_result_is_detached_from_late_callback_mutation() -> None:
    messages = _langchain_history()
    returned: list[BaseMessage] = []

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        returned.extend(ctx.messages[-1:])
        return returned

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run():
        with runner.run_scope(None):
            return await runner.process(messages, model_id=None)

    outcome = asyncio.run(run())
    returned[0].content = "mutated later"
    assert outcome.messages[0].content == "current task"


def test_same_processor_recursion_fails_promptly() -> None:
    messages = _langchain_history()
    runner: HistoryProcessorRunner[BaseMessage]

    async def processor(ctx: HistoryProcessorContext[BaseMessage]):
        await runner.process(ctx.messages, model_id=None)
        return ctx.messages

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run():
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingError, match="recursively invoke"):
        asyncio.run(run())


def test_sync_processor_runs_off_the_event_loop_with_context_propagation() -> None:
    messages = _langchain_history()
    callback_threads: list[int] = []

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        callback_threads.append(threading.get_ident())
        return ctx.messages

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run() -> tuple[int, bool]:
        loop_thread = threading.get_ident()
        with runner.run_scope(None):
            outcome = await runner.process(messages, model_id=None)
        return loop_thread, outcome.changed

    loop_thread, changed = asyncio.run(run())

    assert not changed
    assert callback_threads
    assert callback_threads[0] != loop_thread


def test_async_callable_object_is_awaited() -> None:
    messages = _langchain_history()

    class Processor:
        def __init__(self) -> None:
            self.called = False

        async def __call__(
            self,
            ctx: HistoryProcessorContext[BaseMessage],
        ) -> Sequence[BaseMessage]:
            await asyncio.sleep(0)
            self.called = True
            return ctx.messages

    processor = Processor()
    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run() -> bool:
        with runner.run_scope(None):
            return (await runner.process(messages, model_id=None)).changed

    assert not asyncio.run(run())
    assert processor.called


def test_sync_processor_returning_awaitable_is_awaited() -> None:
    messages = _langchain_history()

    def processor(ctx: HistoryProcessorContext[BaseMessage]):
        async def result() -> Sequence[BaseMessage]:
            await asyncio.sleep(0)
            return ctx.messages

        return result()

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run() -> bool:
        with runner.run_scope(None):
            return (await runner.process(messages, model_id=None)).changed

    assert not asyncio.run(run())


def test_nested_scope_can_reuse_the_active_request_counter() -> None:
    messages = _langchain_history()
    indices: list[int] = []

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        indices.append(ctx.request_index)
        return ctx.messages

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)
            with runner.run_scope(None, reuse_existing=True):
                await runner.process(messages, model_id=None)

    asyncio.run(run())

    assert indices == [1, 2]


def test_parallel_runs_keep_request_indices_and_state_isolated() -> None:
    messages = _langchain_history()
    indices: dict[str, list[int]] = {"first": [], "second": []}

    async def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        assert ctx.model_id is not None
        indices[ctx.model_id].append(ctx.request_index)
        await asyncio.sleep(0)
        return ctx.messages

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def one_run(model_id: str) -> None:
        with runner.run_scope(None):
            await runner.process(messages, model_id=model_id)
            await runner.process(messages, model_id=model_id)

    async def run() -> None:
        await asyncio.gather(one_run("first"), one_run("second"))

    asyncio.run(run())

    assert indices == {"first": [1, 2], "second": [1, 2]}


def test_cancelled_sync_processor_cannot_commit_a_late_result() -> None:
    messages = _langchain_history()
    started = threading.Event()
    release = threading.Event()

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        started.set()
        assert release.wait(timeout=2)
        return ctx.messages[-1:]

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())
    conversation = AdapterConversationContext(
        session_id="cancelled",
        cache_epoch=3,
        cache_key="cache-e3",
        lane_identity="langchain",
    )

    async def run() -> None:
        with runner.run_scope(conversation):
            task = asyncio.create_task(runner.process(messages, model_id=None))
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            await asyncio.sleep(0.01)

    asyncio.run(run())

    assert conversation.cache_epoch == 3
    assert not conversation.cache_prefix_changed


def test_completed_groups_cannot_cross_a_checkpoint_barrier() -> None:
    messages: list[BaseMessage] = [
        HumanMessage(content="first task"),
        AIMessage(content="first answer"),
        HumanMessage(
            content="checkpoint",
            additional_kwargs={"kedi_compaction_checkpoint": "checkpoint-1"},
        ),
        HumanMessage(content="second task"),
        AIMessage(content="second answer"),
        HumanMessage(content="current task"),
    ]

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        return [
            *ctx.messages[3:5],
            ctx.messages[2],
            *ctx.messages[0:2],
            ctx.messages[5],
        ]

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run() -> None:
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingValidationError, match="protected barrier"):
        asyncio.run(run())


def test_staged_cache_generation_commits_exactly_once_and_reuses_next_turn_key() -> None:
    conversation = ConversationState(session_id="history-processing")

    with conversation.adapter_turn("pydantic", lane_identity="lane") as context:
        context.advance_cache_generation()
        context.advance_cache_generation()
        staged_key = context.cache_key
        context.result_state = {"pydantic_messages": ["accepted"]}

    assert conversation.cache_epoch == 2
    with conversation.adapter_turn("pydantic", lane_identity="lane") as next_context:
        assert next_context.cache_epoch == 2
        assert next_context.cache_key == staged_key
        assert next_context.resume_state == {"pydantic_messages": ["accepted"]}


def test_failed_turn_discards_staged_cache_generation() -> None:
    conversation = ConversationState(session_id="history-processing-failure")

    with pytest.raises(RuntimeError, match="stop"):
        with conversation.adapter_turn("langchain") as context:
            context.advance_cache_generation()
            raise RuntimeError("stop")

    assert conversation.cache_epoch == 0
    assert conversation.adapter_state_snapshot("langchain").state == {}


def test_cache_generation_without_conversation_identity_has_no_cache_key() -> None:
    context = AdapterConversationContext()

    assert context.advance_cache_generation() == 1
    assert context.cache_prefix_changed
    assert context.cache_key == ""


def test_history_cache_key_rebase_preserves_only_bounded_archive_suffix() -> None:
    assert rebase_history_cache_key("staged-e2", "caller") == "staged-e2"
    assert rebase_history_cache_key("staged-e2", "caller_h12ab34cd") == "staged-e2_h12ab34cd"
    assert (
        rebase_history_cache_key("staged-e2", "caller_h12ab34cd_hdeadbeef") == "staged-e2_hdeadbeef"
    )


def test_pydantic_clone_preserves_typed_tool_payload_and_detaches_nested_values() -> None:
    @dataclass
    class Payload:
        coordinates: tuple[int, int]
        labels: list[str]

    payload = Payload((1, 2), ["original"])
    message = ModelRequest(parts=[ToolReturnPart("read", payload, tool_call_id="call")])
    cloned = PydanticHistoryMessageCodec().clone([message])[0]
    assert isinstance(cloned, ModelRequest)
    part = cloned.parts[0]
    assert isinstance(part, ToolReturnPart)
    assert isinstance(part.content, Payload)
    assert part.content.coordinates == (1, 2)
    part.content.labels.append("new")
    assert payload.labels == ["original"]


def test_langchain_clone_preserves_native_subclass_and_detaches_nested_values() -> None:
    class TaggedMessage(HumanMessage):
        tags: list[str]

    message = TaggedMessage(content="input", tags=["original"])
    cloned = LangChainHistoryMessageCodec().clone([message])[0]
    assert isinstance(cloned, TaggedMessage)
    cloned.tags.append("new")
    assert message.tags == ["original"]


@pytest.mark.parametrize("codec", [PydanticHistoryMessageCodec(), LangChainHistoryMessageCodec()])
def test_serialized_records_are_not_accepted_as_native_callback_results(codec) -> None:
    with pytest.raises(TypeError, match="all values must be"):
        codec.clone([{"type": "human", "data": {"content": "serialized"}}])


def test_nested_native_runs_do_not_share_pending_transport_reset() -> None:
    runner = HistoryProcessorRunner(lambda ctx: ctx.messages[-1:], LangChainHistoryMessageCodec())

    async def run() -> None:
        with runner.run_scope(None, reuse_existing=True):
            await runner.process(_langchain_history(), model_id=None)
            with runner.run_scope(None, reuse_existing=True):
                await runner.process([HumanMessage(content="child")], model_id=None)
                assert not runner.prepare_transport().reset_continuation
            assert runner.prepare_transport().reset_continuation

    asyncio.run(run())
