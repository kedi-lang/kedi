from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


class CapabilityPydanticAdapter(PydanticAdapter):
    capabilities = replace(PydanticAdapter.capabilities, supports_capabilities=True)


class CapabilityLangChainAdapter(LangChainAdapter):
    capabilities = replace(LangChainAdapter.capabilities, supports_capabilities=True)


def scripted_adapter(kind, codemode, requests):
    steps = (
        [
            ("get_tool_schema", {"tool_names": ["access"]}),
            ("execute_code", {"code": "print(await access())"}),
        ]
        if codemode
        else [("access", {})]
    )
    if kind == "pydantic":

        def respond(messages, info):
            index = len(requests)
            requests.append(messages)
            if index < len(steps):
                name, arguments = steps[index]
                return ModelResponse([ToolCallPart(name, arguments, f"call_{index}")])
            return ModelResponse([TextPart("done")])

        return CapabilityPydanticAdapter(FunctionModel(respond), codemode=codemode)

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            requests.append(messages)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    return CapabilityLangChainAdapter(
        Model(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": arguments, "id": f"call_{index}"}],
                )
                for index, (name, arguments) in enumerate(steps)
            ]
            + [AIMessage(content="done")]
        ),
        codemode=codemode,
    )


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("codemode", [False, True])
@pytest.mark.parametrize("explicit_reuse", [False, True])
def test_python_rebinding_cannot_transfer_capability_authority(kind, codemode, explicit_reuse):
    requests = []
    executed = []

    def read() -> str:
        executed.append("read")
        return "source evidence"

    def write() -> str:
        executed.append("write")
        return "modified"

    runtime = compile_program(
        parse_program(
            "> approval: deny\n"
            "> capability: reader:\n  > approval: allow\n  > use: access\n"
            "> use: reader\n`access = write`\n"
            + ("> use: access\n" if explicit_reuse else "")
            + "[reply] << Read the document.\n= <reply>\n"
        ),
        adapter=scripted_adapter(kind, codemode, requests),
        runtime_globals={"access": read, "write": write},
    )
    try:
        if explicit_reuse:
            with pytest.raises(
                KediExecutionError, match="Conflicting capability tool binding.*access"
            ):
                runtime.run_main()
            assert requests == []
            assert executed == []
        else:
            runtime.run_main()
            assert executed == ["read"]
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("codemode", [False, True])
def test_nested_procedure_shadow_does_not_replace_a_capability_tool(kind, codemode):
    requests = []
    executed = []

    def record(operation: str) -> str:
        executed.append(operation)
        return operation

    runtime = compile_program(
        parse_program(
            "@access() -> str:\n  = `record('read')`\n"
            "> approval: deny\n"
            "> capability: reader:\n  > approval: allow\n  > use: access\n"
            "> use: reader\n"
            "@review() -> str:\n"
            "  @access() -> str:\n    = `record('write')`\n"
            "  [reply] << Read the document.\n  = <reply>\n"
            "= <review()>\n"
        ),
        adapter=scripted_adapter(kind, codemode, requests),
        runtime_globals={"record": record},
    )
    try:
        runtime.run_main()
        assert executed == ["read"]
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("deferred", [False, True])
def test_reusing_a_capability_cannot_replace_its_original_tool(deferred):
    runtime = compile_program(
        parse_program(
            "> capability: reader:\n"
            f"  > defer: {'true' if deferred else 'false'}\n"
            "  > use: access\n"
            "> use: reader\n`access = write`\n> use: reader\n= done\n"
        ),
        runtime_globals={"access": lambda: "evidence", "write": lambda: "modified"},
    )
    try:
        with pytest.raises(KediExecutionError, match="Conflicting capability tool binding.*access"):
            runtime.run_main()
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
def test_unchanged_tool_can_be_shared_and_explicitly_reused(kind):
    executed = []

    def access() -> str:
        executed.append("read")
        return "source evidence"

    runtime = compile_program(
        parse_program(
            "> approval: deny\n"
            "> capability: first:\n  > approval: allow\n  > use: access\n"
            "> capability: second:\n  > approval: allow\n  > use: access\n"
            "> use: first\n> use: second\n> use: first\n> use: access\n"
            "[reply] << Read the document.\n= <reply>\n"
        ),
        adapter=scripted_adapter(kind, False, []),
        runtime_globals={"access": access},
    )
    try:
        runtime.run_main()
        assert executed == ["read"]
    finally:
        runtime.artifacts.close()


def test_unrelated_profile_tool_override_is_not_blocked_by_a_capability():
    from kedi.agent_adapter.tools.specs import ToolSpec
    from kedi.agent_profile import AgentProfile
    from kedi.agent_profile.capabilities import AppliedCapability

    old = ToolSpec(fn=lambda: "old", name="write")
    new = ToolSpec(fn=lambda: "new", name="write")
    profile = AgentProfile(
        tools=("write",),
        tool_bindings={"write": old},
        capabilities=(
            AppliedCapability(
                identity="reader",
                name="reader",
                description=None,
                profile=AgentProfile(tools=("read",)),
            ),
        ),
    )
    updated = profile.merge(AgentProfile(tools=("write",), tool_bindings={"write": new}))
    assert updated.tool_bindings["write"] is new


def test_capability_subagent_binding_survives_registry_replacement():
    from kedi.agent_adapter.subagents.ownership import with_subagent_ownership
    from kedi.agent_profile import AgentProfile
    from kedi.agent_profile.evaluation import eval_profile_definition
    from kedi.agent_profile.definitions import ProfileSystemMember

    runtime = compile_program(
        parse_program(
            "> profile: researcher:\n  > instructions: Read evidence.\n"
            "> capability: research:\n  > subagent: researcher\n"
        )
    )
    try:
        original = runtime._profile_registry["researcher"]
        profile = eval_profile_definition(runtime, runtime._profile_registry["research"], {})
        assert profile.subagent_bindings["researcher"] == original
        runtime._profile_registry["researcher"] = replace(
            original, members=(ProfileSystemMember(plain_value="Modify evidence."),)
        )
        with_subagent_ownership((), runtime, profile)
        rebound = profile.merge(AgentProfile(subagents=("researcher",)))
        with pytest.raises(KediExecutionError, match="Conflicting capability subagent binding"):
            with_subagent_ownership((), runtime, rebound)
    finally:
        runtime.artifacts.close()
