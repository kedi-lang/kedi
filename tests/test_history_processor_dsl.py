from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.history_provenance import LINEAGE_STATE_KEY, history_source_scope
from kedi.agent_profile import AgentProfile, HistoryProcessingSettings
from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter
from tests.test_langchain_user_history_processor import HistoryScriptedModel


def _adapter(kind: str, processor=None):
    if kind == "pydantic":
        model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("done")]))
        return PydanticAdapter(model, history_processor=processor)
    return LangChainAdapter(
        HistoryScriptedModel(responses=[AIMessage(content="done") for _ in range(20)]),
        history_processor=processor,
    )


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("in_profile", [False, True])
def test_native_processor_overrides_constructor_callback(kind, in_profile):
    seen = []
    adapter = _adapter(kind, lambda ctx: seen.append(ctx) or ctx.messages)
    directive = "> history:\n    enabled: true\n    processor: `keep_recent`\n"
    if in_profile:
        directive = (
            "> profile: compact:\n"
            + "\n".join("    " + line for line in directive.splitlines())
            + "\n> use: compact\n"
        )
    source = (
        "@keep_recent(ctx: Any) -> list[Any]:\n"
        "    = `ctx.keep_recent_groups(0)`\n\n"
        + directive
        + "[first] << First task.\n> show: <first>\n"
        + "[second] << Second task.\n= <second>\n"
    )
    runtime = compile_program(parse_program(source), adapter=adapter)
    assert runtime.run_main() == "done"
    assert not seen, "lexical callback must replace, not compose with, constructor callback"
    if kind == "langchain":
        assert "First task" not in str(adapter._model.seen[-1])


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_dsl_only_callback_and_source_origins(kind):
    adapter = _adapter(kind)
    source = """
@inspect_history(ctx: Any) -> list[Any]:
    ```
    assert ctx.groups[-1].origins
    origin = ctx.groups[-1].origins[-1]
    assert origin.run_id is not None
    assert origin.source_location is not None
    ```
    = `list(ctx.messages)`

> history:
    enabled: true
    processor: `inspect_history`

[answer] << Complete the task.
= <answer>
"""
    assert compile_program(parse_program(source), adapter=adapter).run_main() == "done"


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_explicit_disable_restores_constructor_processor_on_scope_exit(kind):
    seen = []
    adapter = _adapter(kind, lambda ctx: seen.append(ctx.request_index) or ctx.messages)
    disabled = AgentProfile(history_processing=HistoryProcessingSettings(processor=None))
    with adapter.profile_override(disabled):
        assert adapter.invoke_sync(prompt="disabled") == "done"
    assert not seen
    assert adapter.invoke_sync(prompt="default") == "done"
    assert seen == [1]


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_concurrent_profile_callbacks_are_run_local(kind):
    adapter = _adapter(kind)
    seen = []

    async def run(label):
        async def processor(ctx):
            await asyncio.sleep(0)
            seen.append((label, ctx.request_index, str(ctx.messages)))
            return ctx.messages

        with adapter.profile_override(
            AgentProfile(history_processing=HistoryProcessingSettings(processor=processor))
        ):
            return await adapter.invoke(prompt=label)

    async def both():
        return await asyncio.gather(run("first-job"), run("second-job"))

    assert asyncio.run(both()) == ["done", "done"]
    assert len(seen) == 2
    for label, index, messages in seen:
        assert index == 1
        assert label in messages
        assert ("second-job" if label == "first-job" else "first-job") not in messages
    assert adapter.active_history_processor is None


@pytest.mark.parametrize("value", [0, False, "callback", object()])
def test_invalid_history_processor_configuration(value):
    with pytest.raises(TypeError, match="callable or None"):
        HistoryProcessingSettings(processor=value)


def test_profile_omission_and_disable_are_distinct():
    def callback(ctx):
        return ctx.messages

    parent = AgentProfile(history_processing=HistoryProcessingSettings(processor=callback))
    assert parent.merge(AgentProfile(history=False)).history_processing.processor is callback
    assert (
        parent.merge(
            AgentProfile(history_processing=HistoryProcessingSettings())
        ).history_processing.processor
        is None
    )
    assert parent.frozen_copy().history_processing is parent.history_processing


@pytest.mark.parametrize("field", ["processor", "processor_condition"])
def test_unsupported_adapter_rejects_processor_before_model_io(field):
    runtime = compile_program(
        parse_program(
            """
@keep(ctx: Any) -> list[Any]:
    = `list(ctx.messages)`
> history:
    enabled: false
    processor: `keep`
[answer] << Complete the task.
= <answer>
""".replace("processor:", f"{field}:")
        ),
        adapter=MockAdapter(),
    )
    with pytest.raises(Exception, match="history_processing"):
        runtime.run_main()


def test_dsl_processor_conflicts_with_constructor_native_history_capability():
    model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("done")]))
    adapter = PydanticAdapter(model, capabilities=[ProcessHistory(lambda _ctx, messages: messages)])
    profile = AgentProfile(
        history_processing=HistoryProcessingSettings(processor=lambda ctx: ctx.messages)
    )
    with (
        adapter.profile_override(profile),
        pytest.raises(Exception, match="another Pydantic ProcessHistory"),
    ):
        adapter.run_sync("task")


def test_explicit_none_dsl_disables_and_does_not_inherit():
    seen = []
    adapter = _adapter("pydantic", lambda ctx: seen.append(ctx) or ctx.messages)
    runtime = compile_program(
        parse_program("""
> history:
    enabled: true
    processor: `None`
[answer] << Complete the task.
= <answer>
"""),
        adapter=adapter,
    )
    assert runtime.run_main() == "done"
    assert not seen


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("skip_first", [False, True])
def test_adapter_serialization_preserves_origin_across_turns(kind, skip_first):
    seen = []
    adapter = _adapter(kind, lambda ctx: seen.append(ctx) or ctx.messages)
    first = AdapterConversationContext()
    policy = HistoryProcessingSettings(condition=lambda _state: False, inherit_processor=True)
    first_profile = AgentProfile(history_processing=policy) if skip_first else AgentProfile()
    with (
        history_source_scope("first-run", None),
        adapter.conversation_scope(first),
        adapter.profile_override(first_profile),
    ):
        assert adapter.invoke_sync(prompt="first") == "done"
    assert LINEAGE_STATE_KEY in first.result_state
    first_origin = next(
        origin
        for entry in first.result_state[LINEAGE_STATE_KEY].entries
        for origin in entry.origins
        if origin.kind == "user"
    )
    assert first_origin.run_id is not None
    second = AdapterConversationContext(resume_state=dict(first.result_state))
    with history_source_scope("second-run", None), adapter.conversation_scope(second):
        assert adapter.invoke_sync(prompt="second") == "done"
    assert seen[-1].groups[0].origins[0] == first_origin
    assert seen[-1].groups[-1].origins[0].run_id != first_origin.run_id
    assert first_origin.run_id not in str(second.result_state[f"{kind}_messages"])


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_native_procedure_captures_processor_and_restores_caller(kind):
    adapter = _adapter(kind)
    seen = []
    source = """
@keep(ctx: Any) -> list[Any]:
    = `ctx.keep_recent_groups(0)`
> history:
    enabled: true
    processor: `keep`
@worker() -> str:
    [reply] << Complete the inner task.
    = <reply>
> history:
    enabled: true
    processor: `None`
> show: <worker()>
[reply] << Complete the outer task.
= <reply>
"""
    runtime = compile_program(parse_program(source), adapter=adapter)
    runner = adapter._history_processor_runner
    original_process = runner.process

    async def observe(messages, *, model_id):
        seen.append(adapter.active_history_processor)
        return await original_process(messages, model_id=model_id)

    runner.process = observe
    assert runtime.run_main() == "done"
    assert callable(seen[0])
    assert seen[1] is None
    assert adapter.active_history_processor is None


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_lexical_processor_receives_exact_tool_boundary_and_request_source(kind):
    seen = []

    def processor(ctx):
        seen.append(ctx)
        return ctx.keep_recent_groups(1)

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
                        content="", tool_calls=[{"name": "read_file", "args": {}, "id": "call-1"}]
                    ),
                    AIMessage(content="done"),
                ]
            )
        )
    runtime = compile_program(
        parse_program(
            "@read_file() -> str:\n    = file contents\n"
            "> use: read_file\n> approval: allow\n"
            "> history:\n    enabled: true\n    processor: `inspect_history`\n"
            "[reply] << Read the file.\n= <reply>\n"
        ),
        adapter=adapter,
        runtime_globals={"inspect_history": processor},
    )
    assert runtime.run_main() == "done"
    assert [ctx.request_index for ctx in seen] == [1, 2]
    origins = [origin for origin in seen[-1].groups[-1].origins if origin.tool_call_id]
    assert {origin.kind for origin in origins} == {"tool-call", "tool-result"}
    assert all(
        origin.tool_name == "read_file" and origin.tool_call_id == "call-1" for origin in origins
    )
    assert all(
        origin.run_id is not None and origin.source_location is not None for origin in origins
    )


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_captured_processor_ignores_nested_same_name_procedure(kind):
    source = """
@keep(ctx: Any) -> list[Any]:
    = `list(ctx.messages)`
> history:
    enabled: true
    processor: `keep`
@worker() -> str:
    @keep(ctx: Any) -> list[Any]:
        ```
        raise AssertionError("wrong callback binding")
        ```
        = `list(ctx.messages)`
    [reply] << Complete the task.
    = <reply>
= <worker()>
"""
    assert compile_program(parse_program(source), adapter=_adapter(kind)).run_main() == "done"


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_disabled_history_processor_cannot_read_outer_conversation(kind):
    seen = []
    source = """
> history:
    enabled: true
    processor: `inspect_history`
[first] << Outer-only message.
> show: <first>
@isolated() -> str:
    > history: disabled
    [reply] << Isolated message.
    = <reply>
= <isolated()>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=_adapter(kind),
        runtime_globals={"inspect_history": lambda ctx: seen.append(ctx) or ctx.messages},
    )
    assert runtime.run_main() == "done"
    assert len(seen) == 2
    assert "Outer-only" in str(seen[0].messages)
    assert "Isolated message" in str(seen[1].messages)
    assert "Outer-only" not in str(seen[1].messages)


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_nested_processor_keeps_definition_scope_when_deeper_procedure_shadows_it(kind):
    source = """
@worker() -> str:
    @keep(ctx: Any) -> list[Any]:
        = `list(ctx.messages)`
    > history:
        enabled: true
        processor: `keep`
    @nested() -> str:
        @keep(ctx: Any) -> list[Any]:
            ```
            raise AssertionError("wrong nested callback binding")
            ```
            = `list(ctx.messages)`
        [reply] << Complete the task.
        = <reply>
    = <nested()>
= <worker()>
"""
    assert compile_program(parse_program(source), adapter=_adapter(kind)).run_main() == "done"
