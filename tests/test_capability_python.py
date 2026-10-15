from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import re
from inspect import unwrap
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import kedi
from kedi import Capability
from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.hooks import PreToolUseDecision
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, McpServerSpec
from kedi.agent_profile.capabilities import capability_tool_policy
from kedi.agent_profile.definitions import ProfileDefinition, ProfileSystemMember
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program


class CapabilityPydanticAdapter(PydanticAdapter):
    capabilities = replace(
        PydanticAdapter.capabilities,
        supports_capabilities=True,
        supports_deferred_tools=True,
    )


class CapabilityLangChainAdapter(LangChainAdapter):
    capabilities = replace(LangChainAdapter.capabilities, supports_capabilities=True)


class RecordingChatModel(FakeMessagesListChatModel):
    requests: list = Field(default_factory=list)
    tool_names: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tool_names.append(tuple(tool.name for tool in tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"name": "not a name"}, ValueError, "identifier"),
        ({"identity": " "}, ValueError, "nonempty"),
        ({"instructions": 5}, TypeError, "instructions"),
        ({"description": 5}, TypeError, "description"),
        ({"defer": "false"}, TypeError, "bool"),
        ({"approval": "allow"}, TypeError, "ApprovalPolicy"),
        ({"tools": (42,)}, TypeError, "callables or ToolSpec"),
        ({"capabilities": (AgentProfile(),)}, TypeError, "Capability"),
        ({"requires": "tool_registration"}, TypeError, "sequence"),
        ({"requires": ("not_a_requirement",)}, ValueError, "unknown adapter requirement"),
        ({"mcp_servers": ("stdio",)}, TypeError, "McpServerSpec"),
        ({"hooks": {"user_prompt_submit": lambda event: None}}, ValueError, "owned tool hooks"),
        ({"subagents": {"child": AgentProfile()}}, TypeError, "ProfileDefinition"),
        ({"model": "unavailable"}, TypeError, "model"),
        ({"settings": {"temperature": 0}}, TypeError, "settings"),
        ({"artifacts": False}, TypeError, "artifacts"),
    ],
)
def test_python_capability_rejects_invalid_or_host_owned_configuration(kwargs, error, message):
    with pytest.raises(error, match=message):
        Capability(**{"name": "reader", **kwargs})


def test_python_capability_normalizes_requirements_at_construction():
    value = Capability(name="reader", requires=["tool_registration", "tool_registration"])
    assert value.requires == ("tool_registration",)


def test_python_capability_snapshots_inputs_and_each_lowered_profile():
    class Reader:
        def __call__(self) -> str:
            return "revision 7"

    reader = Reader()
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    metadata = {"labels": ["source"]}
    env = {"MODE": "read"}
    tools = [ToolSpec(fn=reader, name="read_document", json_schema=schema, metadata=metadata)]
    child = ProfileDefinition(
        name="child", members=(ProfileSystemMember(plain_value="Read carefully."),)
    )
    children = {"child": child}
    capability = Capability(
        name="reader",
        tools=tools,
        mcp_servers=[McpServerSpec(command="fixture", env=env)],
        subagents=children,
    )
    tools.clear()
    children.clear()
    schema["properties"]["path"]["type"] = "integer"
    metadata["labels"].append("changed")
    env["MODE"] = "write"
    first = capability.as_profile()
    assert first.tool_bindings["read_document"].fn is reader
    assert (
        first.tool_bindings["read_document"].json_schema["properties"]["path"]["type"] == "string"
    )
    assert first.tool_bindings["read_document"].metadata == {"labels": ["source"]}
    assert first.mcp_servers[0].env == {"MODE": "read"}
    assert first.subagent_bindings["child"].members[0].plain_value == "Read carefully."
    first.tool_bindings["read_document"].metadata["labels"].append("first run")
    first.mcp_servers[0].env["MODE"] = "first run"
    second = capability.as_profile()
    assert second.tool_bindings["read_document"].metadata == {"labels": ["source"]}
    assert second.mcp_servers[0].env == {"MODE": "read"}
    copied = deepcopy(capability).as_profile()
    assert copied.tool_bindings["read_document"].fn is reader


def test_python_capability_composition_uses_common_reset_and_ownership_rules():
    def read_document() -> str:
        return "revision 7"

    child = Capability(
        name="citations", identity="example.citations", instructions="Cite the source."
    )
    reader = Capability(
        name="reader",
        instructions="Read the document.",
        capabilities=(child, child),
        tools=(read_document,),
        approval=ApprovalPolicy.allow(),
    )
    base = AgentProfile(system="Host guidance.", approval=ApprovalPolicy.deny())
    active = base.merge(reader.as_profile()).merge(reader.as_profile())
    assert active.system == "Host guidance."
    assert [part.content for part in active.capability_instructions] == [
        "Read the document.",
        "Cite the source.",
    ]
    assert active.tools == ("read_document",)
    assert capability_tool_policy(active, "read_document").approval == ApprovalPolicy.allow()
    reset = active.merge(AgentProfile(system="Host guidance."))
    assert reset.capability_instructions == ()
    assert reset.tools == active.tools
    assert (
        reset.merge(reader.as_profile()).capability_instructions == active.capability_instructions
    )
    assert reader.as_profile().capabilities[0].instructions_enabled


def test_python_capability_snapshots_child_definitions_without_cloning_executable_tools():
    class Reader:
        def __call__(self) -> str:
            return "revision 7"

    reader = Reader()
    tool = ToolSpec(fn=reader, name="read_document", metadata={"labels": ["source"]})
    child = ProfileDefinition(name="child", tool_bindings={"read_document": tool})
    capability = Capability(name="reader", subagents={"child": child})
    tool.metadata["labels"].append("changed")
    first = capability.as_profile().subagent_bindings["child"]
    assert first.tool_bindings["read_document"].metadata == {"labels": ["source"]}
    first.tool_bindings["read_document"].metadata["labels"].append("first run")
    second = deepcopy(capability).as_profile().subagent_bindings["child"]
    assert second.tool_bindings["read_document"].fn is reader
    assert second.tool_bindings["read_document"].metadata == {"labels": ["source"]}


def test_python_capability_preserves_named_tool_schema_and_explicit_policy():
    def read_document() -> str:
        """Read the current document revision."""
        return "revision 7"

    read_document.__kedi_tool_name__ = "lookup_revision"
    capability = Capability(name="reader", tools=(read_document,))
    tool = capability.as_profile().tool_bindings["lookup_revision"]
    assert unwrap(tool.fn) is read_document
    assert tool.description == "Read the current document revision."
    assert tool.return_schema["type"] == "string"
    with pytest.raises(ValueError, match="duplicate capability tool name"):
        Capability(name="reader", tools=(read_document, read_document))
    explicit = replace(tool, approval=ApprovalPolicy.deny())
    patch = Capability(
        name="reader", tools=(explicit,), approval=ApprovalPolicy.allow()
    ).as_profile()
    from kedi.agent_adapter.tools.specs import tool_approval_policy

    assert (
        tool_approval_policy(patch.tool_bindings["lookup_revision"], ApprovalPolicy.allow())
        == ApprovalPolicy.deny()
    )


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("reset", [False, True])
def test_query_uses_python_capability_tools_hooks_and_instructions(kind, reset):
    events = []

    def read_document() -> str:
        events.append("read")
        return "revision 7"

    def before(event):
        events.append(event.tool_name)
        return PreToolUseDecision.continue_()

    reader = Capability(
        name="reader",
        instructions="Use exact source revisions.",
        tools=(read_document,),
        approval=ApprovalPolicy.allow(),
        hooks={"pre_tool_use": before},
    )
    requests = []
    if kind == "pydantic":

        def respond(messages, info):
            requests.append(info.instructions)
            assert "read_document" in {tool.name for tool in info.function_tools}
            if len(requests) % 2:
                return ModelResponse([ToolCallPart("read_document", {}, "read")])
            return ModelResponse([TextPart("revision 7")])

        adapter = CapabilityPydanticAdapter(FunctionModel(respond))
    else:
        model = RecordingChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "read_document", "args": {}, "id": "read"}]
                ),
                AIMessage(content="revision 7"),
            ]
        )
        adapter = CapabilityLangChainAdapter(model)

    def review() -> str:
        pass

    review.__doc__ = (
        "kedi\n> approval: deny\n> instructions: Host guidance.\n"
        "> use: reader\n> use: reader\n"
        + ("> instructions: Host guidance.\n" if reset else "")
        + "[reply] << Retrieve the document revision.\n= <reply>\n"
    )
    review = kedi.query(adapter=adapter, env={"reader": reader})(review)
    assert review() == "revision 7"
    assert review() == "revision 7"
    assert events == ["read_document", "read", "read_document", "read"]
    if kind == "langchain":
        requests = [str(messages[0].content) for messages in model.requests]
        assert all(names.count("read_document") == 1 for names in model.tool_names)
    assert len(requests) == 4
    assert all(text.count("Host guidance.") == 1 for text in requests)
    assert all(text.count("Use exact source revisions.") == int(not reset) for text in requests)


def test_python_deferred_capability_loads_natively_and_isolates_parallel_runs():
    def read_document() -> str:
        return "revision 7"

    reader = Capability(
        name="reader",
        instructions="Use exact source revisions.",
        tools=(read_document,),
        defer=True,
    )

    def respond(messages, info):
        names = {tool.name for tool in info.function_tools}
        if len(messages) == 1:
            assert "read_document" not in names
            assert "Use exact source revisions." not in info.instructions
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        assert "read_document" in names
        assert "Use exact source revisions." in str(messages)
        return ModelResponse([TextPart("ready")])

    @kedi.query(adapter=CapabilityPydanticAdapter(FunctionModel(respond)), env={"reader": reader})
    def review() -> str:
        """kedi
        > use: reader
        [reply] << Load the reader.
        = <reply>
        """

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: review(), range(8))) == ["ready"] * 8


def test_python_deferred_capability_fails_before_langchain_inference():
    model = RecordingChatModel(responses=[AIMessage(content="unexpected")])

    @kedi.query(
        adapter=CapabilityLangChainAdapter(model),
        env={"reader": Capability(name="reader", defer=True)},
    )
    def review() -> str:
        """kedi
        > use: reader
        [reply] << Review the document.
        = <reply>
        """

    with pytest.raises(KediExecutionError, match="deferred_tools"):
        review()
    assert model.requests == []


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("documented", [False, True])
def test_python_capability_supports_typed_template_captures(kind, documented):
    calls = []

    def read_revision() -> int:
        calls.append("read")
        return 7

    reader = Capability(
        name="reader", tools=(read_revision,), instructions="Read the current revision."
    )
    if kind == "pydantic":

        def respond(messages, info):
            assert "Read the current revision." in info.instructions
            if not calls:
                return ModelResponse([ToolCallPart("read_revision", {}, "read")])
            return ModelResponse([ToolCallPart(info.output_tools[0].name, {"revision": 7}, "out")])

        adapter = CapabilityPydanticAdapter(FunctionModel(respond))
    else:

        class CaptureModel(FakeMessagesListChatModel):
            output_name: str = ""

            def bind_tools(self, tools, **kwargs):
                for tool in tools:
                    spec = convert_to_openai_tool(tool)["function"]
                    if "revision" in spec["parameters"].get("properties", {}):
                        self.output_name = spec["name"]
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                assert "Read the current revision." in messages[0].content
                call = (
                    {"name": "read_revision", "args": {}, "id": "read"}
                    if not calls
                    else {"name": self.output_name, "args": {"revision": 7}, "id": "out"}
                )
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
                )

        adapter = CapabilityLangChainAdapter(CaptureModel(responses=[]))

    if documented:
        path = Path(__file__).resolve().parents[1] / "kedi-docs/docs/python-api/capabilities.md"
        if not path.is_file():
            pytest.skip("documentation submodule is not checked out")
        source = re.search(r"```python\n(.*?)\n```", path.read_text(), re.DOTALL).group(1)
        namespace = {"__name__": "capability_docs_example"}
        exec(compile(source, str(path), "exec", dont_inherit=True), namespace)

        def record(event):
            if event.tool_name == "read_revision":
                calls.append("read")

        with kedi.context(adapter=adapter, hooks={"post_tool_use": record}):
            assert namespace["current_revision"]() == 7
    else:

        @kedi.query(adapter=adapter, env={"reader": reader}, approval="allow")
        def revision() -> int:
            """kedi
            > use: reader
            >> The current source revision is [revision: int].
            = `revision`
            """

        assert revision() == 7
    assert calls == ["read"]


@pytest.mark.parametrize("kind", ["profile", "capability"])
@pytest.mark.parametrize("reexport", [False, True])
@pytest.mark.parametrize("shadow", ["capability", "callable", "absent"])
def test_exported_definition_retains_private_python_capability(tmp_path, kind, reexport, shadow):
    (tmp_path / "bundle.kedi").write_text(
        "```\nfrom kedi import Capability\n"
        'reader = Capability(name="reader", instructions="Private source guidance.")\n```\n'
        f"> {kind}: bundle:\n  > use: reader\n> export:\n  bundle\n"
    )
    if reexport:
        (tmp_path / "forward.kedi").write_text("> import: bundle\n> export:\n  bundle\n")
    observed = []

    def respond(messages, info):
        observed.append(info.instructions)
        return ModelResponse([TextPart("done")])

    source = tmp_path / "main.kedi"
    runtime = compile_program(
        parse_program(
            f"> import: {'forward' if reexport else 'bundle'}\n"
            "> use: bundle\n[reply] << Review.\n= <reply>\n",
            source_path=str(source),
        ),
        adapter=CapabilityPydanticAdapter(FunctionModel(respond)),
        runtime_globals=(
            {}
            if shadow == "absent"
            else {
                "reader": Capability(name="reader", instructions="Unrelated caller guidance.")
                if shadow == "capability"
                else lambda: "unrelated tool"
            }
        ),
    )
    try:
        assert runtime.run_main() == "done"
    finally:
        runtime.artifacts.close()
    assert len(observed) == 1
    assert "Private source guidance." in observed[0]
    assert "Unrelated caller guidance." not in observed[0]


def test_python_capability_remains_definition_site_bound_in_exported_procedure():
    seen = []

    def respond(messages, info):
        seen.append(info.instructions)
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            "> instructions: Original guidance.\n> use: reader\n"
            "@review() -> str:\n  [reply] << Review the document.\n  = <reply>\n"
            "> instructions: Later guidance.\n= <review()>\n"
        ),
        adapter=CapabilityPydanticAdapter(FunctionModel(respond)),
        runtime_globals={"reader": Capability(name="reader", instructions="Read source evidence.")},
    )
    try:
        assert runtime.run_main() == "done"
        assert runtime._invoke_procedure(runtime._get_proc("review"), []) == "done"
    finally:
        runtime.artifacts.close()
    assert len(seen) == 2
    assert all("Original guidance." in text and "Read source evidence." in text for text in seen)
    assert all("Later guidance." not in text for text in seen)


def test_native_capabilities_keyword_remains_native_in_python_query():
    events = []

    class NativeCapability(AbstractCapability):
        async def before_run(self, ctx):
            events.append("native")

    @kedi.query
    def review() -> str:
        """kedi
        [reply] << Review the document.
        = <reply>
        """

    with kedi.context(
        adapter="pydantic",
        model="test",
        artifacts=False,
        capabilities=[NativeCapability()],
    ):
        assert isinstance(review(), str)
    assert events == ["native"]
