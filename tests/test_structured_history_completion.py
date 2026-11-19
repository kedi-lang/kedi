from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_from_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_adapter.adapters.langchain.history_processing import LangChainHistoryMessageCodec
from kedi.agent_adapter.adapters.pydantic.history_processing import PydanticHistoryMessageCodec
from kedi.agent_adapter.adapters.pydantic.messages import _capture_conversation_messages
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.history_processing import HistoryProcessorRunner
from kedi.agent_adapter.subagents.control import LiveInput, RunInterrupted, live_input_scope
from kedi.agent_adapter.tools.specs import ToolSpec


@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("keep", [0, 1])
def test_successful_output_tool_cycle_can_be_removed(adapter, signed, keep):
    if adapter == "pydantic":
        reasoning = [ThinkingPart("reason", signature="signed")] if signed else []
        messages = [
            ModelRequest(parts=[UserPromptPart("previous task")]),
            ModelResponse(parts=[*reasoning, ToolCallPart("custom_output", {"value": 3}, "out")]),
            ModelRequest(parts=[ToolReturnPart("custom_output", "processed", "out")]),
        ]
        conversation = AdapterConversationContext()
        _capture_conversation_messages(
            conversation,
            SimpleNamespace(all_messages=lambda: messages, new_messages=lambda: messages),
            response_complete=True,
        )
        saved = ModelMessagesTypeAdapter.validate_python(
            conversation.result_state["pydantic_messages"]
        )
        current = ModelRequest(parts=[UserPromptPart("current task")])
        codec = PydanticHistoryMessageCodec()
    else:
        content = (
            [{"type": "reasoning", "reasoning": "reason", "signature": "signed"}] if signed else ""
        )
        messages = [
            HumanMessage(content="previous task"),
            AIMessage(
                content=content,
                tool_calls=[{"name": "custom_output", "args": {"value": 3}, "id": "out"}],
            ),
            ToolMessage(content="processed", tool_call_id="out"),
        ]
        conversation = AdapterConversationContext()
        adapter_instance = LangChainAdapter(model="test-model")
        token = adapter_instance._conversation_var.set(conversation)
        try:
            adapter_instance._capture_conversation_messages(
                {"messages": messages}, response_complete=True
            )
        finally:
            adapter_instance._conversation_var.reset(token)
        saved = messages_from_dict(conversation.result_state["langchain_messages"])
        current = HumanMessage(content="current task")
        codec = LangChainHistoryMessageCodec()
    contexts = []
    runner = HistoryProcessorRunner(
        lambda ctx: contexts.append(ctx) or ctx.keep_recent_groups(keep), codec
    )

    async def process():
        with runner.run_scope(None):
            return await runner.process([*saved, current], model_id=None)

    outcome = asyncio.run(process())
    assert len(contexts[0].groups) == 2
    assert contexts[0].groups[0].closed
    assert contexts[0].groups[0].protected == signed
    assert outcome.messages == ([*saved, current] if keep or signed else [current])
    # Capture annotates a copy; SDK results and user-visible native messages stay intact.
    if adapter == "pydantic":
        assert messages[-1].metadata is None
    else:
        assert messages[-1].additional_kwargs == {}


class Outcome(BaseModel):
    value: str


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
def test_next_model_request_omits_completed_structured_turn(adapter_name):
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter import PydanticAdapter
    from tests.test_langchain_user_history_processor import HistoryScriptedModel

    if adapter_name == "pydantic":
        seen = []

        def respond(messages, info):
            seen.append(list(messages))
            return ModelResponse(
                parts=[
                    ThinkingPart("plain chat reasoning", id="reasoning", provider_name="openai"),
                    ToolCallPart(info.output_tools[0].name, {"value": "ok"}, f"out-{len(seen)}"),
                ]
            )

        adapter = PydanticAdapter(
            FunctionModel(respond), history_processor=lambda ctx: ctx.keep_recent_groups(0)
        )
    else:
        model = HistoryScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "Outcome", "args": {"value": "ok"}, "id": f"out-{i}"}],
                )
                for i in range(2)
            ]
        )
        seen = model.seen
        adapter = LangChainAdapter(model, history_processor=lambda ctx: ctx.keep_recent_groups(0))

    async def run():
        first = AdapterConversationContext()
        with adapter.conversation_scope(first):
            assert (
                await adapter.produce(template="obsolete task", output_type=Outcome)
            ).value == "ok"
        following = AdapterConversationContext(resume_state=first.result_state)
        with adapter.conversation_scope(following):
            assert (
                await adapter.produce(template="current task", output_type=Outcome)
            ).value == "ok"

    asyncio.run(run())
    assert len(seen) == 2
    assert "obsolete task" in str(seen[0])
    assert "obsolete task" not in str(seen[1])
    assert "current task" in str(seen[1])


@pytest.mark.parametrize(
    "part,required",
    [
        (ThinkingPart("plain", id="reasoning", provider_name="openai"), False),
        (ThinkingPart("plain", id="reasoning_content", provider_name="openai"), False),
        (ThinkingPart("plain", id="content", provider_name="openai"), False),
        (ThinkingPart("opaque", id="rs_provider_item", provider_name="openai"), True),
        (ThinkingPart("signed", id="reasoning", signature="opaque"), True),
        (ThinkingPart("opaque", id="reasoning", provider_details={"opaque": "state"}), True),
    ],
)
def test_plain_reasoning_field_ids_are_not_opaque_replay_items(part, required):
    snapshot = PydanticHistoryMessageCodec().inspect([ModelResponse(parts=[part])])
    assert snapshot.entries[0].provider_required == required


def test_mixed_user_and_tool_result_tail_is_not_marked_complete():
    messages = [
        ModelResponse(parts=[ToolCallPart("output", {}, "out")]),
        ModelRequest(
            parts=[ToolReturnPart("output", "processed", "out"), UserPromptPart("next task")]
        ),
    ]
    conversation = AdapterConversationContext()
    _capture_conversation_messages(
        conversation,
        SimpleNamespace(all_messages=lambda: messages, new_messages=lambda: messages),
        response_complete=True,
    )
    saved = ModelMessagesTypeAdapter.validate_python(conversation.result_state["pydantic_messages"])
    assert not PydanticHistoryMessageCodec().inspect(saved).entries[-1].response_complete
    assert saved[-1].parts == messages[-1].parts


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("stop", ["redirect", "cancel", "failure"])
async def test_interrupted_tool_cycle_retains_prompt_when_resumed(backend, stop):
    from pydantic_ai.messages import TextPart
    from pydantic_ai.models.function import FunctionModel

    from kedi.agent_adapter import PydanticAdapter
    from tests.test_langchain_user_history_processor import HistoryScriptedModel

    started, release = asyncio.Event(), asyncio.Event()
    requests, effects = [], []

    async def read_file():
        effects.append("read")
        return "retained evidence"

    async def wait_for_stop():
        started.set()
        await release.wait()
        raise RuntimeError("request failed")

    if backend == "pydantic":

        async def respond(messages, info):
            requests.append(list(messages))
            if len(requests) == 1:
                return ModelResponse(parts=[ToolCallPart("read_file", {}, "read-1")])
            if len(requests) == 2:
                await wait_for_stop()
            return ModelResponse(parts=[TextPart("done")])

        adapter = PydanticAdapter(
            FunctionModel(respond), history_processor=lambda ctx: ctx.keep_recent_groups(0)
        )
        codec = PydanticHistoryMessageCodec()
    else:

        class InterruptibleModel(HistoryScriptedModel):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                requests.append(list(messages))
                if len(requests) == 1:
                    response = AIMessage(
                        content="",
                        tool_calls=[{"name": "read_file", "args": {}, "id": "read-1"}],
                    )
                else:
                    if len(requests) == 2:
                        await wait_for_stop()
                    response = AIMessage(content="done")
                return ChatResult(generations=[ChatGeneration(message=response)])

        adapter = LangChainAdapter(
            InterruptibleModel(responses=[]),
            history_processor=lambda ctx: ctx.keep_recent_groups(0),
        )
        codec = LangChainHistoryMessageCodec()

    tool = ToolSpec(
        fn=read_file,
        name="read_file",
        description="Read evidence once.",
        json_schema={"type": "object", "properties": {}},
        risk="read_only",
        artifact_result="never",
    )
    control = LiveInput("worker")
    control.configure(input=True, interrupt=True)
    conversation = AdapterConversationContext()
    with (
        adapter.conversation_scope(conversation),
        adapter.tool_scope([tool]),
        live_input_scope(control),
    ):
        task = asyncio.create_task(adapter.invoke(prompt="unfinished task"))
    try:
        await asyncio.wait_for(started.wait(), 3)
        if stop == "redirect":
            control.admit("Continue with the evidence.", sender_id="parent", interrupt=True)
            expected = RunInterrupted
        elif stop == "cancel":
            task.cancel()
            expected = asyncio.CancelledError
        else:
            release.set()
            expected = RuntimeError
        with pytest.raises(expected):
            await asyncio.wait_for(task, 3)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    saved_state = conversation.result_state
    if backend == "pydantic":
        saved = ModelMessagesTypeAdapter.validate_python(saved_state["pydantic_messages"])
    else:
        saved = messages_from_dict(saved_state["langchain_messages"])
    assert not codec.inspect(saved).entries[-1].response_complete
    assert not conversation._completed
    assert effects == ["read"]

    following = AdapterConversationContext(resume_state=saved_state)
    with adapter.conversation_scope(following), adapter.tool_scope([tool]):
        assert await adapter.invoke(prompt="Continue the task.") == "done"
    assert "unfinished task" in str(requests[-1])
    assert effects == ["read"]
    assert following._completed
