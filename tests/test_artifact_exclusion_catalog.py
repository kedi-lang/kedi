"""Exclusion diagnostics use real catalogs without activating deferred tools."""

import asyncio
from dataclasses import replace
import re
import warnings

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from mcp.server.fastmcp import FastMCP
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.tools.conversion import python_callable_tool_spec
from kedi.agent_adapter.tools.scope import artifact_admission_scope
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.capabilities import AppliedCapability
from kedi.agent_profile.mcp import McpServerSpec
from kedi.artifacts import ArtifactManager, ArtifactPolicy
from kedi.artifacts.policy import warn_unmatched_tool_exclusions
from kedi.lang import compile_program, parse_program


class ChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def manager():
    value = ArtifactManager()
    try:
        yield value
    finally:
        value.close()


@pytest.mark.parametrize("kind", ["pydantic", "langchain"])
@pytest.mark.parametrize("codemode", [False, True])
@pytest.mark.parametrize("mcp", [False, True])
def test_exclusions_use_underlying_native_catalog(kind, codemode, mcp, monkeypatch, manager):
    def lookup() -> str:
        """Read source evidence."""
        pytest.fail("Catalog validation must not execute the tool")

    if mcp:
        server = FastMCP("catalog")
        server.tool()(lookup)
        monkeypatch.setattr(
            "kedi.agent_adapter.adapters.pydantic.tools._mcp_toolset_from_config",
            lambda name, config: MCPToolset(server),
        )

        class Client:
            def __init__(self, configs):
                pass

            async def get_tools(self, *, server_name):
                return [StructuredTool.from_function(lookup)]

        monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)

    adapter = (
        PydanticAdapter(
            FunctionModel(lambda messages, info: ModelResponse([TextPart("done")])),
            codemode=codemode,
        )
        if kind == "pydantic"
        else LangChainAdapter(ChatModel(responses=[AIMessage(content="done")]), codemode=codemode)
    )
    policy = ArtifactPolicy(exclude_tools=("lookup", "lookup_typo"))
    with (
        adapter.profile_override(
            AgentProfile(mcp_servers=(McpServerSpec(command="catalog"),) if mcp else ())
        ),
        adapter.tool_scope(() if mcp else (python_callable_tool_spec("lookup", lookup),)),
        artifact_admission_scope(manager=manager, policy=policy),
        warnings.catch_warnings(record=True) as recorded,
    ):
        warnings.simplefilter("always")
        assert asyncio.run(adapter.invoke(prompt="No lookup needed.")) == "done"
    diagnostics = [
        str(item.message) for item in recorded if "Artifact exclude_tools" in str(item.message)
    ]
    assert diagnostics == [
        "Artifact exclude_tools 'lookup_typo' did not match any resolved tool; "
        "check the exact exposed tool names."
    ]


@pytest.mark.parametrize("load", [False, True])
@pytest.mark.parametrize("codemode", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_deferred_mcp_exclusions_wait_for_real_catalog(
    load, codemode, dynamic, monkeypatch, manager
):
    server = FastMCP("deferred")

    @server.tool()
    def lookup() -> str:
        """Read source evidence."""
        pytest.fail("Catalog validation must not execute tools")

    entered = []
    exited = []

    class ObservedToolset(MCPToolset):
        async def __aenter__(self):
            entered.append(True)
            return await super().__aenter__()

        async def __aexit__(self, *args):
            try:
                return await super().__aexit__(*args)
            finally:
                exited.append(True)

        async def get_tools(self, ctx):
            if dynamic and not any(
                name.startswith("kedi_reader_") for name in ctx.active_capability_ids
            ):
                return {}
            return await super().get_tools(ctx)

    monkeypatch.setattr(
        "kedi.agent_adapter.adapters.pydantic.tools._mcp_toolset_from_config",
        lambda name, config: ObservedToolset(server),
    )
    requests = []

    def respond(messages, info):
        assert not [item for item in recorded if "Artifact exclude_tools" in str(item.message)]
        requests.append(info)
        if len(requests) == 1:
            assert "lookup" not in {tool.name for tool in info.function_tools}
        if load and len(requests) == 1:
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        return ModelResponse([TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), codemode=codemode)
    capability = AppliedCapability(
        identity="reader",
        name="reader",
        description="Read document evidence",
        profile=AgentProfile(mcp_servers=(McpServerSpec(command="catalog"),)),
        deferred=True,
    )
    with (
        adapter.profile_override(AgentProfile(capabilities=(capability,))),
        artifact_admission_scope(manager=manager, policy=ArtifactPolicy(exclude_tools=("lookup",))),
        warnings.catch_warnings(record=True) as recorded,
    ):
        warnings.simplefilter("always")
        assert adapter.run_sync("Read only when needed.").output == "done"
    diagnostics = [
        str(item.message) for item in recorded if "Artifact exclude_tools" in str(item.message)
    ]
    assert entered and entered == exited
    if dynamic and not load:
        assert diagnostics == [
            "Artifact exclude_tools 'lookup' could not be verified against resolved tools; "
            "some deferred capabilities were not activated."
        ]
    else:
        assert diagnostics == []


def test_disabled_artifacts_do_not_warn_about_retained_exclusions():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warn_unmatched_tool_exclusions(ArtifactPolicy(enabled=False, exclude_tools=("later",)), ())
    assert recorded == []


@pytest.mark.parametrize("codemode", [False, True])
def test_deferred_python_tool_is_valid_before_activation(codemode):
    class Adapter(PydanticAdapter):
        capabilities = replace(
            PydanticAdapter.capabilities,
            supports_capabilities=True,
            supports_deferred_tools=True,
        )

    def lookup() -> str:
        pytest.fail("Excluded tool was activated for validation")

    def respond(messages, info):
        assert "lookup" not in {tool.name for tool in info.function_tools}
        return ModelResponse([TextPart("done")])

    runtime = compile_program(
        parse_program(
            '> artifacts:\n  exclude_tools: `["lookup"]`\n'
            "> capability: reader:\n  > defer: true\n  > use: lookup\n"
            "> use: reader\n[reply] << No lookup needed.\n= <reply>\n"
        ),
        adapter=Adapter(FunctionModel(respond), codemode=codemode),
        runtime_globals={"lookup": lookup},
    )
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            assert runtime.run_main() == "done"
        assert not [item for item in recorded if "Artifact exclude_tools" in str(item.message)]
    finally:
        runtime.artifacts.close()


def test_native_catalog_diagnostics_are_once_per_run_and_not_shared(manager):
    requests = []

    def lookup() -> str:
        return "source evidence"

    def respond(messages, info):
        requests.append(info)
        if len(requests) <= 2:
            return ModelResponse([ToolCallPart("lookup", {}, f"lookup-{len(requests)}")])
        return ModelResponse([TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond))
    policy = ArtifactPolicy(exclude_tools=("lookup", "missing"))
    with (
        artifact_admission_scope(manager=manager, policy=policy),
        warnings.catch_warnings(record=True) as recorded,
    ):
        warnings.simplefilter("always")
        with adapter.tool_scope((python_callable_tool_spec("lookup", lookup),)):
            assert adapter.run_sync("Read twice.").output == "done"
        assert adapter.run_sync("No tools this time.").output == "done"
    diagnostics = [
        str(item.message) for item in recorded if "Artifact exclude_tools" in str(item.message)
    ]
    assert len(diagnostics) == 2
    assert "'lookup'" not in diagnostics[0]
    assert "'lookup', 'missing'" in diagnostics[1]


@pytest.mark.parametrize("codemode", [False, True])
def test_native_prefixed_toolsets_use_exposed_names(codemode, manager):
    def lookup() -> str:
        """Read evidence."""
        pytest.fail("Catalog validation executed a tool")

    adapter = PydanticAdapter(
        FunctionModel(lambda messages, info: ModelResponse([TextPart("done")])),
        toolsets=[FunctionToolset(tools=[lookup]).prefixed("source")],
        codemode=codemode,
    )
    with (
        artifact_admission_scope(
            manager=manager, policy=ArtifactPolicy(exclude_tools=("lookup", "source_lookup"))
        ),
        warnings.catch_warnings(record=True) as recorded,
    ):
        warnings.simplefilter("always")
        assert adapter.run_sync("No lookup needed.").output == "done"
    diagnostics = [
        str(item.message) for item in recorded if "Artifact exclude_tools" in str(item.message)
    ]
    assert diagnostics == [
        "Artifact exclude_tools 'lookup' did not match any resolved tool; "
        "check the exact exposed tool names."
    ]
