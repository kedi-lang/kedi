"""Gateway findings in #204: list tool results reach the model as JSON text, a repeated value keeps
its artifact ID, and business tool calls in a rejected final-output response still run."""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.artifacts import ArtifactHandle, ArtifactManager, ArtifactPolicy
from kedi.lang import compile_program, parse_program


def _run(runtime):
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def _items_program(value: str) -> str:
    return (
        "@items() -> list:\n    ###\n    Matching item ids.\n    ###\n    > tool:\n"
        f"        risk: read_only\n    = `{value}`\n> use: items\n"
        "[n: int] << Count the matching items.\n= `n`\n"
    )


class _ToolResultModel(FakeMessagesListChatModel):
    """Calls `items` once, records the tool message content, then returns the output."""

    seen: list = []
    output: str = ""

    def bind_tools(self, tools, **kwargs):
        names = [getattr(tool, "name", None) for tool in tools]
        return self.model_copy(update={"output": names[-1]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        replies = [message for message in messages if isinstance(message, ToolMessage)]
        if not replies:
            call = {"name": "items", "args": {}, "id": "call-1"}
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))]
            )
        self.seen.append(replies[-1].content)
        call = {"name": self.output, "args": {"response": 0}, "id": "call-2"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])


@pytest.mark.parametrize(
    ("value", "sent"),
    [("[]", "[]"), ('["O-1", "O-2"]', '["O-1", "O-2"]'), ('[{"id": 1}]', '[{"id": 1}]')],
)
def test_langchain_sends_list_tool_results_as_json_text(value, sent):
    model = _ToolResultModel(responses=[], seen=[])
    runtime = compile_program(parse_program(_items_program(value)), adapter=LangChainAdapter(model))
    assert _run(runtime) == 0
    assert model.seen == [sent]


def test_pydantic_sends_an_empty_list_tool_result_as_json_text():
    seen: list = []

    def respond(messages, info):
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("items", {})])
        seen.append(returns[-1].content)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"response": 0})])

    runtime = compile_program(
        parse_program(_items_program("[]")), adapter=PydanticAdapter(FunctionModel(respond))
    )
    assert _run(runtime) == 0
    assert seen == ["[]"]


def _policy() -> ArtifactPolicy:
    return ArtifactPolicy.patch({"enabled": True, "threshold": 1, "cleanup_interval": "1h"})


def test_admitting_the_same_value_again_keeps_its_artifact_id():
    manager = ArtifactManager(session_id="identity")
    try:
        policy = _policy()
        first = manager.artifactize("the same document", policy=policy, source="doc")
        again = manager.artifactize("the same document", policy=policy, source="doc")
        assert isinstance(first, ArtifactHandle) and isinstance(again, ArtifactHandle)
        assert again.ref.ref_id == first.ref.ref_id
        assert manager.stats()["active_artifacts"] == 1

        other_source = manager.artifactize("the same document", policy=policy, source="mail")
        sensitive = manager.artifactize(
            "the same document", policy=policy, source="doc", sensitive=True
        )
        derived = manager.artifactize(
            "the same document", policy=policy, source="doc", provenance={"from": ["x"]}
        )
        other_text = manager.artifactize("another document", policy=policy, source="doc")
        ids = {
            handle.ref.ref_id for handle in (first, other_source, sensitive, derived, other_text)
        }
        assert len(ids) == 5

        manager.release(first.ref.ref_id)
        after_release = manager.artifactize("the same document", policy=policy, source="doc")
        assert after_release.ref.ref_id not in ids
    finally:
        manager.close()


def test_business_tool_calls_run_before_the_output_correction():
    inspected: list[str] = []
    transcripts: list[list[str]] = []

    class ChatModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            transcripts.append([type(message).__name__ for message in messages])
            if not any(isinstance(m, ToolMessage) and m.name == "inspect" for m in messages):
                calls = [
                    {"name": "inspect", "args": {"role": "policy"}, "id": "c1"},
                    {"name": "Decision", "args": {"target": 5}, "id": "c2"},
                ]
            else:
                calls = [
                    {"name": "Decision", "args": {"target": "main", "allowed": True}, "id": "c3"}
                ]
            return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=calls))])

    def inspect_tool(role: str) -> str:
        inspected.append(role)
        return "proposal for " + role

    source = (
        "@inspect(role: str) -> str:\n    ###\n    Read the proposal.\n    ###\n    > tool:\n"
        "        risk: read_only\n    = `inspect_tool(role)`\n> use: inspect\n"
        "~Decision(target: str, allowed: bool)\n"
        "[d: Decision] << First inspect the policy, then decide.\n= `(d.target, d.allowed)`\n"
    )
    runtime = compile_program(
        parse_program(source),
        adapter=LangChainAdapter(ChatModel(responses=[])),
        runtime_globals={"inspect_tool": inspect_tool},
    )
    assert _run(runtime) == ("main", True)
    assert inspected == ["policy"]
    # The second request carries the inspect result, then the correction for the rejected output.
    assert transcripts[1][-2:] == ["ToolMessage", "HumanMessage"]
    assert len(transcripts) == 2


def test_a_deferred_correction_is_applied_once():
    from kedi.agent_adapter.adapters.langchain.runtime import (
        _DEFERRED_CORRECTION_KEY,
        _deferred_correction,
    )

    held = AIMessage(
        "",
        tool_calls=[{"name": "inspect", "args": {}, "id": "c1"}],
        response_metadata={_DEFERRED_CORRECTION_KEY: "fix it"},
    )
    reply = ToolMessage("done", tool_call_id="c1", name="inspect")
    assert _deferred_correction([held, reply]) == "fix it"
    assert _deferred_correction([held, reply, HumanMessage("fix it")]) is None
    assert _deferred_correction([AIMessage("plain")]) is None
    assert _deferred_correction([]) is None


def test_business_tool_names_come_from_tool_objects_or_schemas():
    from kedi.agent_adapter.adapters.langchain.runtime import _tool_name

    assert _tool_name({"type": "function", "function": {"name": "inspect"}}) == "inspect"
    assert _tool_name({"name": "search"}) == "search"
    assert _tool_name(type("Tool", (), {"name": "lookup"})()) == "lookup"
