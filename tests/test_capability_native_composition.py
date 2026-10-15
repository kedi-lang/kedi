from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.hooks import PreToolUseDecision
from kedi.lang import compile_program, parse_program


class ToolChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def adapter_for(kind):
    names = ["read_document", "write_document"]
    if kind == "pydantic":
        return PydanticAdapter(TestModel(call_tools=names))
    return LangChainAdapter(
        ToolChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": {}, "id": f"call_{index}"}
                        for index, name in enumerate(names)
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("reset", [False, True])
def test_dsl_owned_tools_keep_policies_hooks_and_registration_after_reset(kind, reset):
    executed = []
    events = []

    def read_document() -> str:
        executed.append("read")
        return "source evidence"

    def write_document() -> str:
        executed.append("write")
        return "saved"

    def observe_pre(event):
        events.append(("pre", event.tool_name))
        return PreToolUseDecision.continue_()

    def observe_post(event):
        events.append(("post", event.tool_name))

    source = (
        "> approval: deny\n"
        "> capability: reader:\n"
        "  > instructions: Read evidence before answering.\n"
        "  > approval: allow\n"
        "  > hooks:\n"
        "    pre_tool_use: `observe_pre`\n"
        "    post_tool_use: `observe_post`\n"
        "  > use: read_document\n"
        "> instructions: Host instructions.\n"
        "> use: reader\n> use: reader\n> use: write_document\n"
        + ("> instructions: Host instructions.\n" if reset else "")
        + "[reply] << Read and write the document.\n= <reply>\n"
    )
    runtime = compile_program(
        parse_program(source),
        adapter=adapter_for(kind),
        runtime_globals={
            "read_document": read_document,
            "write_document": write_document,
            "observe_pre": observe_pre,
            "observe_post": observe_post,
        },
    )
    try:
        runtime.run_main()
    finally:
        runtime.artifacts.close()
    assert executed == ["read"]
    assert events == [("pre", "read_document"), ("post", "read_document")]


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_native_adapters_honor_exact_artifact_exclusions(kind):
    def read_document() -> str:
        return "read evidence " * 100

    def write_document() -> str:
        return "write evidence " * 100

    runtime = compile_program(
        parse_program(
            "> approval: allow\n"
            '> artifacts:\n  threshold: 64b\n  exclude_tools: `["read_document"]`\n'
            "> use: read_document\n> use: write_document\n"
            "[reply] << Read and write the document.\n= <reply>\n"
        ),
        adapter=adapter_for(kind),
        runtime_globals={"read_document": read_document, "write_document": write_document},
    )
    try:
        runtime.run_main()
        results = {
            event.data["tool_name"]: event.data
            for event in runtime.artifacts.history.snapshot()
            if event.kind == "tool_result"
        }
        assert results["read_document"]["artifacted"] is False
        assert results["read_document"]["artifact_excluded"] is True
        assert results["write_document"]["artifacted"] is True
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_native_eager_procedure_capture_evaluates_use_only_once(kind):
    evaluations = []
    executed = []

    def choose_mode():
        evaluations.append(False)
        return False

    def read_document() -> str:
        executed.append("read")
        return "source evidence"

    def write_document() -> str:
        executed.append("write")
        return "saved"

    runtime = compile_program(
        parse_program(
            "> approval: allow\n"
            "> capability: documents:\n  > defer: `choose_mode()`\n"
            "  > use:\n    read_document\n    write_document\n"
            "> use: documents\n"
            "@review() -> str:\n  [reply] << Review the document.\n  = <reply>\n"
            "= <review()>\n"
        ),
        adapter=adapter_for(kind),
        runtime_globals={
            "choose_mode": choose_mode,
            "read_document": read_document,
            "write_document": write_document,
        },
    )
    try:
        assert evaluations == []
        runtime.run_main()
        assert evaluations == [False]
        assert sorted(executed) == ["read", "write"]
    finally:
        runtime.artifacts.close()
