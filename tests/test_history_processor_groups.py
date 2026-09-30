from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart

from kedi.agent_adapter import HistoryOrigin
from kedi.agent_adapter.adapters.langchain.history_processing import LangChainHistoryMessageCodec
from kedi.agent_adapter.adapters.pydantic.history_processing import PydanticHistoryMessageCodec
from kedi.agent_adapter.compaction.classification import classify_snapshot
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.history_processing import HistoryProcessorRunner
from kedi.agent_adapter.history_provenance import (
    HistoryLineage,
    HistoryLineageEntry,
    HistoryProvenance,
    history_source_scope,
    record_history_tool_origin,
)
from kedi.lang.ast import Span


def _history():
    return [
        SystemMessage(content="rules"),
        HumanMessage(content="first"),
        AIMessage(content="one"),
        HumanMessage(content="second"),
        AIMessage(content="two"),
        HumanMessage(content="current"),
    ]


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, ["rules", "current"]),
        (1, ["rules", "second", "two", "current"]),
        (2, ["rules", "first", "one", "second", "two", "current"]),
        (20, ["rules", "first", "one", "second", "two", "current"]),
    ],
)
def test_keep_recent_groups_preserves_protected_and_current(count, expected):
    messages = _history()
    runner = HistoryProcessorRunner(
        lambda ctx: ctx.keep_recent_groups(count), LangChainHistoryMessageCodec()
    )

    async def run():
        with runner.run_scope(None):
            return await runner.process(messages, model_id=None)

    outcome = asyncio.run(run())
    assert [message.content for message in outcome.messages] == expected
    if count >= 2:
        assert outcome.messages is messages
        assert not outcome.changed


@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
@pytest.mark.parametrize("count", [0, 1])
def test_protected_recent_cycle_does_not_consume_unprotected_group_count(adapter, count):
    if adapter == "pydantic":
        codec = PydanticHistoryMessageCodec()
        messages = [
            ModelRequest(parts=[UserPromptPart("obsolete")]),
            ModelResponse(parts=[TextPart("acknowledged")]),
            ModelRequest(parts=[UserPromptPart("required context")]),
            ModelResponse(
                parts=[
                    ThinkingPart("opaque reasoning", signature="signed-replay-data"),
                    TextPart("ready"),
                ]
            ),
            ModelRequest(parts=[UserPromptPart("current")]),
        ]
    else:
        codec = LangChainHistoryMessageCodec()
        messages = [
            HumanMessage(content="obsolete"),
            AIMessage(content="acknowledged"),
            HumanMessage(content="required context"),
            AIMessage(
                content=[
                    {
                        "type": "reasoning",
                        "reasoning": "opaque reasoning",
                        "signature": "signed-replay-data",
                    },
                    {"type": "text", "text": "ready"},
                ]
            ),
            HumanMessage(content="current"),
        ]
    contexts = []
    runner = HistoryProcessorRunner(
        lambda ctx: contexts.append(ctx) or ctx.keep_recent_groups(count), codec
    )

    async def run():
        with runner.run_scope(None):
            return await runner.process(messages, model_id=None)

    outcome = asyncio.run(run())
    assert contexts[0].groups[1].protected_reasons == ("provider-required",)
    assert outcome.messages == (messages if count == 1 else messages[2:])
    assert outcome.changed == (count == 0)


@pytest.mark.parametrize(
    "count,error", [(True, TypeError), (-1, ValueError), (1.5, TypeError), ("1", TypeError)]
)
def test_keep_recent_groups_rejects_invalid_counts(count, error):
    def processor(ctx):
        with pytest.raises(error):
            ctx.keep_recent_groups(count)
        return ctx.messages

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())

    async def run():
        with runner.run_scope(None):
            await runner.process(_history(), model_id=None)

    asyncio.run(run())


def test_helper_retains_unfinished_tool_group_and_exact_native_origins():
    contexts = []
    messages = [
        *_history()[:-1],
        HumanMessage(content="pending"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "call-pending"}]),
    ]
    runner = HistoryProcessorRunner(
        lambda ctx: contexts.append(ctx) or ctx.keep_recent_groups(0),
        LangChainHistoryMessageCodec(),
    )

    async def run():
        with runner.run_scope(None):
            return await runner.process(messages, model_id=None)

    outcome = asyncio.run(run())
    assert len(outcome.messages) == 3
    origin = contexts[0].groups[-1].origins[-1]
    assert origin.tool_name == "read_file"
    assert origin.tool_call_id == "call-pending"
    assert origin.run_id is None
    with pytest.raises(FrozenInstanceError):
        origin.run_id = "changed"


def test_provenance_is_out_of_band_and_persists_only_with_successful_capture():
    contexts = []
    runner = HistoryProcessorRunner(
        lambda ctx: contexts.append(ctx) or ctx.messages, LangChainHistoryMessageCodec()
    )
    conversation = AdapterConversationContext()
    messages = [HumanMessage(content="current")]

    async def run():
        with history_source_scope("first-run", None), runner.run_scope(conversation):
            await runner.process(messages, model_id=None)
            record_history_tool_origin("read_file", "call-1")
            messages.extend(
                [
                    AIMessage(
                        content="", tool_calls=[{"name": "read_file", "args": {}, "id": "call-1"}]
                    ),
                    ToolMessage(content="value", tool_call_id="call-1"),
                ]
            )
            await runner.process(messages, model_id=None)
            messages.append(AIMessage(content="done"))
            runner.capture(messages, messages)
        following = AdapterConversationContext(resume_state=dict(conversation.result_state))
        with history_source_scope("second-run", None), runner.run_scope(following):
            await runner.process([*deepcopy(messages), HumanMessage(content="next")], model_id=None)

    asyncio.run(run())
    assert contexts[-1].groups[0].origins[0].run_id == "first-run"
    assert contexts[-1].groups[-1].origins[0].run_id == "second-run"
    tools = [o for o in contexts[1].groups[-1].origins if o.tool_call_id]
    assert all(o.tool_name == "read_file" and o.run_id == "first-run" for o in tools)
    assert all("first-run" not in str(message) for message in messages)


def test_reordered_identical_entries_do_not_mix_origins():
    codec = LangChainHistoryMessageCodec()
    message = HumanMessage(content="same")
    messages = [message, deepcopy(message), AIMessage(content="done")]
    snapshot = classify_snapshot(codec.inspect(messages))
    source = replace(
        snapshot,
        entries=tuple(
            replace(entry, origins=(HistoryOrigin(kind="user", run_id=str(i)),))
            for i, entry in enumerate(snapshot.entries)
        ),
    )
    provenance = HistoryProvenance()
    returned = [messages[1], messages[0], messages[2]]
    provenance.accept(source, returned, codec.inspect(deepcopy(returned)))
    assert [entry.origins[0].run_id for entry in provenance.snapshot.entries] == ["1", "0", "2"]
    provenance.accept(source, deepcopy(returned), codec.inspect(deepcopy(returned)))
    assert provenance.snapshot.entries[0].origins == ()
    assert provenance.snapshot.entries[1].origins == ()


def test_unmatched_saved_history_and_new_summary_have_no_fabricated_lineage():
    codec = LangChainHistoryMessageCodec()
    provenance = HistoryProvenance(
        HistoryLineage(
            (HistoryLineageEntry("different", (HistoryOrigin(kind="user", run_id="old"),)),)
        )
    )
    snapshot = provenance.inspect(codec.inspect(_history()))
    assert snapshot.entries[1].origins == ()
    returned = [
        SystemMessage(content="rules"),
        AIMessage(content="new summary"),
        HumanMessage(content="current"),
    ]
    provenance.accept(snapshot, returned, codec.inspect(returned))
    assert provenance.snapshot.entries[1].origins == ()


def test_disabled_tracking_does_not_accumulate_tools_or_leak_into_parent():
    parent = HistoryProvenance()
    disabled = HistoryProvenance()
    with parent.bind():
        record_history_tool_origin("read_file", "parent")
        with disabled.bind(enabled=False):
            record_history_tool_origin("read_file", "disabled")
        record_history_tool_origin("read_file", "restored")
    record_history_tool_origin("read_file", "outside")
    assert set(parent.tools) == {"parent", "restored"}
    assert disabled.tools == {}


def test_callback_cannot_mutate_runtime_source_location():
    location = Span(0, 10, 1, 1, 1, 11, "history.kedi")
    conversation = AdapterConversationContext()

    def processor(ctx):
        ctx.groups[-1].origins[0].source_location.start_line = 999
        ctx.entries[-1].origins[0].source_location.start_line = 888
        return ctx.messages

    runner = HistoryProcessorRunner(processor, LangChainHistoryMessageCodec())
    messages = [HumanMessage(content="current")]

    async def run():
        with history_source_scope("run", location), runner.run_scope(conversation):
            await runner.process(messages, model_id=None)
            runner.capture(messages, messages)

    asyncio.run(run())
    assert location.start_line == 1
    lineage = conversation.result_state["history_processor_origins"]
    assert lineage.entries[0].origins[0].source_location.start_line == 1


def test_reused_tool_id_does_not_overwrite_previous_turn_origin():
    codec = LangChainHistoryMessageCodec()
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "same-id"}]),
        ToolMessage(content="old result", tool_call_id="same-id"),
        AIMessage(content="done"),
    ]
    provenance = HistoryProvenance()
    with history_source_scope("old-run", None), provenance.bind():
        record_history_tool_origin("read_file", "same-id")
        snapshot = provenance.inspect(codec.inspect(messages))
    saved = HistoryLineage(
        tuple(HistoryLineageEntry(e.fingerprint, e.origins) for e in snapshot.entries)
    )
    next_turn = HistoryProvenance(saved)
    with history_source_scope("new-run", None), next_turn.bind():
        next_turn.inspect(codec.inspect(messages + [HumanMessage(content="new")]))
        record_history_tool_origin("read_file", "same-id")
        result = next_turn.inspect(
            codec.inspect(
                messages
                + [
                    HumanMessage(content="new"),
                    AIMessage(
                        content="", tool_calls=[{"name": "read_file", "args": {}, "id": "same-id"}]
                    ),
                    ToolMessage(content="new result", tool_call_id="same-id"),
                ]
            )
        )
    assert result.entries[1].origins[0].run_id == "old-run"
    assert result.entries[-2].origins[0].run_id == "new-run"


def test_langchain_middleware_without_processor_keeps_native_messages():
    from kedi.agent_adapter.adapters.langchain.runtime import LangChainRunMiddleware

    middleware = LangChainRunMiddleware(archive=None, tool_retries=0)
    messages = _history()
    assert asyncio.run(middleware.abefore_model({"messages": messages}, None)) is None


def test_saved_current_message_keeps_origin_when_replayed_without_append():
    codec = LangChainHistoryMessageCodec()
    messages = [HumanMessage(content="same request")]
    snapshot = codec.inspect(messages)
    old_origin = HistoryOrigin(kind="user", run_id="original-run")
    provenance = HistoryProvenance(
        HistoryLineage((HistoryLineageEntry(snapshot.entries[0].fingerprint, (old_origin,)),))
    )
    with history_source_scope("replay-run", None):
        replayed = provenance.inspect(codec.inspect(deepcopy(messages)))
    assert replayed.entries[0].origins == (old_origin,)


def test_aliased_message_occurrences_do_not_acquire_a_guessed_origin():
    codec = LangChainHistoryMessageCodec()
    shared = HumanMessage(content="same")
    messages = [shared, shared, AIMessage(content="done")]
    snapshot = codec.inspect(messages)
    source = replace(
        snapshot,
        entries=tuple(
            replace(entry, origins=(HistoryOrigin(kind=entry.kind, run_id=str(i)),))
            for i, entry in enumerate(snapshot.entries)
        ),
    )
    provenance = HistoryProvenance()
    returned = [shared, messages[-1]]
    provenance.accept(source, returned, codec.inspect(deepcopy(returned)))
    assert provenance.snapshot.entries[0].origins == ()
    assert provenance.snapshot.entries[1].origins[0].run_id == "2"
