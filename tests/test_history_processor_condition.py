from __future__ import annotations

import asyncio
import threading
from dataclasses import FrozenInstanceError, replace

import pytest
import kedi.agent_adapter.history_processing as history_processing
from langchain_core.messages import AIMessage, HumanMessage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import (
    HistoryProcessingError,
    HistoryProcessorState,
    LangChainAdapter,
    PydanticAdapter,
)
from kedi.agent_adapter.adapters.langchain.history_processing import LangChainHistoryMessageCodec
from kedi.agent_adapter.adapters.pydantic.history_processing import PydanticHistoryMessageCodec
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.history_processing import HistoryProcessorRunner
from kedi.agent_profile import AgentProfile, HistoryProcessingSettings
from kedi.lang import compile_program, parse_program
from kedi.lsp.features.diagnostics import compute_diagnostics
from tests.lsp_helpers import hover_text_at
from tests.test_langchain_user_history_processor import HistoryScriptedModel


class _CountingCodec:
    def __init__(self, native):
        self.native = native
        self.adapter_shortname = native.adapter_shortname
        self.state_key = native.state_key
        self.estimate = 0
        self.clones = 0

    def clone(self, messages):
        self.clones += 1
        return self.native.clone(messages)

    def inspect(self, messages):
        return replace(self.native.inspect(messages), total_tokens=self.estimate)


@pytest.mark.parametrize(
    ("codec", "messages"),
    [
        (PydanticHistoryMessageCodec(), [ModelRequest(parts=[UserPromptPart(content="now")])]),
        (LangChainHistoryMessageCodec(), [HumanMessage(content="now")]),
    ],
)
def test_threshold_is_level_triggered_and_false_skips_clone(codec, messages):
    counting = _CountingCodec(codec)
    states: list[HistoryProcessorState] = []
    contexts = []

    def condition(state):
        states.append(state)
        return state.estimated_tokens >= 80_000

    def processor(ctx):
        contexts.append(ctx)
        return ctx.messages

    runner = HistoryProcessorRunner(processor, counting, condition)
    conversation = AdapterConversationContext(cache_epoch=3, cache_key="stable")

    async def run():
        with runner.run_scope(conversation):
            for estimate in (79_999, 80_000, 80_001):
                counting.estimate = estimate
                outcome = await runner.process(messages, model_id="test-model")
                assert outcome.messages is messages
                assert not outcome.changed
                assert not runner.prepare_transport().reset_continuation

    asyncio.run(run())
    assert [state.estimated_tokens for state in states] == [79_999, 80_000, 80_001]
    assert [state.request_index for state in states] == [1, 2, 3]
    assert [ctx.request_index for ctx in contexts] == [2, 3]
    assert [ctx.history_version for ctx in contexts] == [
        states[1].history_version,
        states[2].history_version,
    ]
    assert all(state.model_id == "test-model" and state.cache_epoch == 3 for state in states)
    assert counting.clones == 4
    assert conversation.cache_epoch == 3
    with pytest.raises(FrozenInstanceError):
        states[0].estimated_tokens = 1


def test_false_condition_skips_group_classification(monkeypatch):
    def unexpected(_snapshot):
        raise AssertionError("group classification should not run")

    monkeypatch.setattr(history_processing, "classify_snapshot", unexpected)
    runner = HistoryProcessorRunner(
        lambda ctx: ctx.messages,
        LangChainHistoryMessageCodec(),
        lambda _state: False,
    )

    async def run():
        with runner.run_scope(None):
            outcome = await runner.process([HumanMessage(content="task")], model_id=None)
            assert not outcome.changed

    asyncio.run(run())


@pytest.mark.parametrize("answer", [1, "yes", None])
def test_invalid_condition_result_fails_before_processor(answer):
    calls = []
    runner = HistoryProcessorRunner(
        lambda ctx: calls.append(ctx) or ctx.messages,
        LangChainHistoryMessageCodec(),
        lambda _state: answer,
    )

    async def run():
        with runner.run_scope(None):
            await runner.process([HumanMessage(content="task")], model_id=None)

    with pytest.raises(HistoryProcessingError, match="must return a bool"):
        asyncio.run(run())
    assert not calls


def test_condition_exception_is_not_ignored():
    def condition(_state):
        raise RuntimeError("bad predicate")

    runner = HistoryProcessorRunner(
        lambda ctx: ctx.messages, LangChainHistoryMessageCodec(), condition
    )

    async def run():
        with runner.run_scope(None):
            await runner.process([HumanMessage(content="task")], model_id=None)

    with pytest.raises(HistoryProcessingError, match="bad predicate"):
        asyncio.run(run())


def test_condition_existing_history_error_is_preserved():
    def condition(_state):
        raise HistoryProcessingError("policy failed")

    runner = HistoryProcessorRunner(
        lambda ctx: ctx.messages, LangChainHistoryMessageCodec(), condition
    )

    async def run():
        with runner.run_scope(None):
            await runner.process([HumanMessage(content="task")], model_id=None)

    with pytest.raises(HistoryProcessingError, match="^policy failed$"):
        asyncio.run(run())


def test_sync_condition_runs_off_event_loop_and_async_condition_is_awaited():
    main_thread = threading.get_ident()
    thread_ids = []
    runner = HistoryProcessorRunner(
        lambda ctx: ctx.messages,
        LangChainHistoryMessageCodec(),
        lambda _state: thread_ids.append(threading.get_ident()) or False,
    )

    async def run():
        with runner.run_scope(None):
            await runner.process([HumanMessage(content="task")], model_id=None)

    asyncio.run(run())
    assert thread_ids and thread_ids[0] != main_thread

    async def condition(_state):
        await asyncio.sleep(0)
        return True

    runner = HistoryProcessorRunner(
        lambda ctx: ctx.messages, LangChainHistoryMessageCodec(), condition
    )
    asyncio.run(run())


def test_cancelled_condition_does_not_advance_cache_or_run_processor():
    entered = asyncio.Event()
    calls = []

    async def condition(_state):
        entered.set()
        await asyncio.Future()

    runner = HistoryProcessorRunner(
        lambda ctx: calls.append(ctx) or ctx.messages,
        LangChainHistoryMessageCodec(),
        condition,
    )
    conversation = AdapterConversationContext(cache_epoch=8, cache_key="stable")

    async def run():
        with runner.run_scope(conversation):
            task = asyncio.create_task(
                runner.process([HumanMessage(content="task")], model_id=None)
            )
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert not calls
    assert conversation.cache_epoch == 8


def test_candidate_changed_during_condition_is_rechecked():
    messages = [HumanMessage(content="first")]
    versions = []
    contexts = []

    def condition(state):
        versions.append(state.history_version)
        if len(versions) == 1:
            messages[0] = HumanMessage(content="new")
            return False
        return True

    runner = HistoryProcessorRunner(
        lambda ctx: contexts.append(ctx) or ctx.messages,
        LangChainHistoryMessageCodec(),
        condition,
    )

    async def run():
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    asyncio.run(run())
    assert versions[0] != versions[1]
    assert contexts[0].history_version == versions[1]


def test_repeated_candidate_change_fails_without_processing():
    messages = [HumanMessage(content="first")]
    calls = []

    def condition(_state):
        messages[0] = HumanMessage(content=f"changed-{len(calls)}")
        calls.append(1)
        return True

    runner = HistoryProcessorRunner(
        lambda ctx: ctx.messages, LangChainHistoryMessageCodec(), condition
    )

    async def run():
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingError, match="changed repeatedly"):
        asyncio.run(run())
    assert len(calls) == 2


def test_clone_cannot_change_revision_after_condition():
    messages = [HumanMessage(content="first")]

    class MutatingCodec(LangChainHistoryMessageCodec):
        def clone(self, source):
            messages[0] = HumanMessage(content="changed")
            return super().clone(messages)

    runner = HistoryProcessorRunner(lambda ctx: ctx.messages, MutatingCodec(), lambda _s: True)

    async def run():
        with runner.run_scope(None):
            await runner.process(messages, model_id=None)

    with pytest.raises(HistoryProcessingError, match="between.*condition and processor"):
        asyncio.run(run())


def test_effective_pair_replacement_and_inheritance():
    def first(ctx):
        return ctx.messages

    def second(ctx):
        return ctx.messages

    def before(_state):
        return False

    def after(_state):
        return True

    default = HistoryProcessingSettings(processor=first, condition=before)
    inherit = HistoryProcessingSettings(condition=after, inherit_processor=True)

    assert default.merge(inherit).resolve(None) == (first, after)
    assert inherit.merge(inherit).resolve(first) == (first, after)
    assert default.merge(HistoryProcessingSettings(processor=second)).resolve(None) == (
        second,
        None,
    )
    assert default.merge(HistoryProcessingSettings(processor=None)).resolve(None) == (
        None,
        None,
    )
    assert default.merge(HistoryProcessingSettings(inherit_processor=True)).resolve(None) == (
        first,
        None,
    )
    with pytest.raises(ValueError, match="requires a processor"):
        inherit.resolve(None)
    with pytest.raises(ValueError, match="requires a processor"):
        HistoryProcessingSettings(condition=after)
    with pytest.raises(TypeError, match="condition must be callable"):
        HistoryProcessingSettings(processor=first, condition="yes")


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_constructor_condition_and_lexical_override(kind):
    seen = []

    def processor(ctx):
        seen.append(ctx.request_index)
        return ctx.messages

    if kind == "pydantic":
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("done")]))
        factory = PydanticAdapter
    else:
        model = HistoryScriptedModel(responses=[AIMessage(content="done") for _ in range(5)])
        factory = LangChainAdapter
    with pytest.warns(UserWarning, match="prefix-cache") as caught:
        adapter = factory(
            model,
            history_processor=processor,
            history_processor_condition=lambda _state: False,
        )
    assert len(caught) == 1
    assert adapter.invoke_sync(prompt="first") == "done"
    assert not seen
    with adapter.profile_override(
        AgentProfile(
            history_processing=HistoryProcessingSettings(
                condition=lambda _state: True,
                inherit_processor=True,
            )
        )
    ):
        assert adapter.invoke_sync(prompt="second") == "done"
    assert seen == [1]
    assert adapter.invoke_sync(prompt="third") == "done"
    assert seen == [1]


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_dsl_condition_only_inherits_constructor_processor(kind):
    seen = []

    def processor(ctx):
        seen.append(ctx)
        return ctx.messages

    if kind == "pydantic":
        adapter = PydanticAdapter(
            FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("done")])),
            history_processor=processor,
        )
    else:
        adapter = LangChainAdapter(
            HistoryScriptedModel(responses=[AIMessage(content="done")]),
            history_processor=processor,
        )
    source = """
@never(state: Any) -> bool:
    = `False`
> history:
    enabled: true
    processor_condition: `never`
[answer] << Complete the task.
= <answer>
"""
    assert compile_program(parse_program(source), adapter=adapter).run_main() == "done"
    assert not seen


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_condition_runs_on_each_tool_loop_request(kind):
    requests = []
    processed = []

    def condition(state):
        requests.append(state.request_index)
        return state.request_index == 2

    def processor(ctx):
        processed.append(ctx.request_index)
        return ctx.messages

    if kind == "pydantic":

        def respond(messages, _info):
            if any(
                isinstance(part, ToolReturnPart) for message in messages for part in message.parts
            ):
                return ModelResponse(parts=[TextPart("done")])
            return ModelResponse(parts=[ToolCallPart("read_file", {}, tool_call_id="call-1")])

        adapter = PydanticAdapter(FunctionModel(respond))
    else:
        adapter = LangChainAdapter(
            HistoryScriptedModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "read_file", "args": {}, "id": "call-1"}],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )
    source = """
@read_file() -> str:
    = file contents
> use: read_file
> approval: allow
> history:
    enabled: true
    processor: `inspect_history`
    processor_condition: `needs_processing`
[reply] << Read the file.
= <reply>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=adapter,
        runtime_globals={"inspect_history": processor, "needs_processing": condition},
    )
    assert runtime.run_main() == "done"
    assert requests == [1, 2]
    assert processed == [2]


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_concurrent_conditions_keep_invocation_local_state(kind):
    observed = []

    async def processor(ctx):
        observed.append(ctx.request_index)
        return ctx.messages

    if kind == "pydantic":
        adapter = PydanticAdapter(
            FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("done")])),
            history_processor=processor,
        )
    else:
        adapter = LangChainAdapter(
            HistoryScriptedModel(responses=[AIMessage(content="done") for _ in range(4)]),
            history_processor=processor,
        )
    conditions = []

    async def run(label):
        async def condition(state):
            await asyncio.sleep(0)
            conditions.append((label, state.request_index))
            return True

        policy = HistoryProcessingSettings(condition=condition, inherit_processor=True)
        with adapter.profile_override(AgentProfile(history_processing=policy)):
            return await adapter.invoke(prompt=label)

    async def both():
        return await asyncio.gather(run("first"), run("second"))

    assert asyncio.run(both()) == ["done", "done"]
    assert sorted(conditions) == [("first", 1), ("second", 1)]
    assert observed == [1, 1]


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_profile_condition_without_effective_processor_fails_before_transport(kind):
    if kind == "pydantic":
        calls = []
        adapter = PydanticAdapter(
            FunctionModel(
                lambda _messages, _info: calls.append(1) or ModelResponse(parts=[TextPart("done")])
            )
        )
    else:
        model = HistoryScriptedModel(responses=[AIMessage(content="done")])
        calls = model.seen
        adapter = LangChainAdapter(model)
    policy = HistoryProcessingSettings(condition=lambda _state: True, inherit_processor=True)
    with adapter.profile_override(AgentProfile(history_processing=policy)):
        with pytest.raises(ValueError, match="requires a processor"):
            adapter.invoke_sync(prompt="task")
    assert not calls


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("raises", [False, True])
def test_condition_failure_stops_before_model_transport(kind, raises):
    calls = []

    def condition(_state):
        if raises:
            raise RuntimeError("predicate failed")
        return 1

    if kind == "pydantic":
        model = FunctionModel(
            lambda _messages, _info: calls.append(1) or ModelResponse(parts=[TextPart("done")])
        )
        factory = PydanticAdapter
    else:
        model = HistoryScriptedModel(responses=[AIMessage(content="done")])
        factory = LangChainAdapter
    with pytest.warns(UserWarning, match="prefix-cache"):
        adapter = factory(
            model,
            history_processor=lambda ctx: ctx.messages,
            history_processor_condition=condition,
        )
    with pytest.raises(
        HistoryProcessingError, match="predicate failed" if raises else "must return a bool"
    ):
        adapter.invoke_sync(prompt="task")
    assert not calls
    if kind == "langchain":
        assert not model.seen


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_condition_without_processor_is_configuration_error(kind):
    if kind == "pydantic":
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("done")]))
        factory = PydanticAdapter
    else:
        model = HistoryScriptedModel(responses=[AIMessage(content="done")])
        factory = LangChainAdapter
    with pytest.raises(ValueError, match="requires a history_processor"):
        factory(model, history_processor_condition=lambda _state: True)


def test_condition_configuration_is_validated_in_runner_and_scope():
    codec = LangChainHistoryMessageCodec()
    with pytest.raises(TypeError, match="condition must be callable"):
        HistoryProcessorRunner(lambda ctx: ctx.messages, codec, "yes")
    runner = HistoryProcessorRunner(lambda ctx: ctx.messages, codec, lambda _state: False)
    with pytest.raises(ValueError, match="requires a history_processor"):
        with runner.run_scope(None, processor=None):
            pass


def test_lsp_warning_is_at_declaration_not_each_use():
    source = """
> profile: inspect:
    > history:
        enabled: true
        processor: `external_callback`
        processor_condition: `gate`
> use: inspect
> use: inspect
> history:
    enabled: true
    processor: `None`
"""
    warnings = [
        diagnostic
        for diagnostic in compute_diagnostics(source, "history.kedi")
        if diagnostic.code == "K-HISTORY-PREFIX-CACHE"
    ]
    assert len(warnings) == 1
    assert "neither option guarantees" in warnings[0].message


def test_statically_disabled_processor_has_no_cache_warning():
    source = "> history:\n    enabled: true\n    processor: `None`\n"
    assert not [
        item
        for item in compute_diagnostics(source, "history.kedi")
        if item.code == "K-HISTORY-PREFIX-CACHE"
    ]


def test_condition_requires_inline_callable_syntax():
    with pytest.raises(Exception, match="inline Python callable reference"):
        parse_program("> history:\n    enabled: true\n    processor_condition: gate\n")


def test_condition_hover_explains_state_and_cache_limit():
    source = (
        "> history:\n    enabled: true\n    processor: `keep`\n"
        "    processor_condition: `needs_compaction`\n"
    )
    hover = hover_text_at(source, "history.kedi", "processor_condition", offset=3)
    assert "HistoryProcessorState" in hover
    assert "does not guarantee a cache hit" in hover


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_adapter_condition_cancellation_stops_before_transport(kind):
    calls = []
    entered = asyncio.Event()

    async def condition(_state):
        entered.set()
        await asyncio.Future()

    if kind == "pydantic":
        model = FunctionModel(
            lambda _messages, _info: calls.append(1) or ModelResponse(parts=[TextPart("done")])
        )
        factory = PydanticAdapter
    else:
        model = HistoryScriptedModel(responses=[AIMessage(content="done")])
        calls = model.seen
        factory = LangChainAdapter
    with pytest.warns(UserWarning, match="prefix-cache"):
        adapter = factory(
            model,
            history_processor=lambda ctx: ctx.messages,
            history_processor_condition=condition,
        )
    conversation = AdapterConversationContext(cache_epoch=7, cache_key="stable")

    async def run():
        with adapter.conversation_scope(conversation):
            task = asyncio.create_task(adapter.invoke(prompt="task"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert not calls
    assert conversation.cache_epoch == 7
    assert not conversation.result_state


@pytest.mark.parametrize("in_profile", [False, True])
def test_lsp_condition_only_requires_history_processing(in_profile):
    source = "> agent: codex\n> history:\n    enabled: false\n    processor_condition: `gate`\n"
    if in_profile:
        source = (
            "> profile: gated:\n"
            + "\n".join("    " + line for line in source.splitlines())
            + "\n> use: gated\n"
        )
    diagnostics = compute_diagnostics(source, "history.kedi")
    assert any(item.code == "K-ADAPTER-HISTORY-PROCESSOR" for item in diagnostics)


@pytest.mark.parametrize("reset", ["processor_condition", "processor"])
def test_lsp_explicit_none_clears_inherited_condition_requirement(reset):
    source = (
        "> profile: gated:\n    > history:\n"
        "        enabled: false\n        processor_condition: `gate`\n"
        "> use: gated\n> history:\n    enabled: false\n"
        f"    {reset}: `None`\n> agent: codex\n"
    )
    diagnostics = compute_diagnostics(source, "history.kedi")
    assert not any(item.code == "K-ADAPTER-HISTORY-PROCESSOR" for item in diagnostics)
