"""Discovery errors must not leave partial capability resources behind."""

import asyncio
from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from mcp.server.fastmcp import FastMCP
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.capabilities import AppliedCapability
from kedi.agent_profile.mcp import McpServerSpec
from kedi.errors import KediExecutionError


def owned_profile(*servers):
    return AgentProfile(
        mcp_servers=servers,
        capabilities=(
            AppliedCapability(
                identity="fixture::catalog",
                name="catalog",
                description=None,
                profile=AgentProfile(mcp_servers=servers, approval=ApprovalPolicy.allow()),
            ),
        ),
    )


@pytest.mark.parametrize("cancel", [False, True])
def test_langchain_discovery_closes_siblings_before_error_escapes(monkeypatch, cancel):
    async def scenario():
        entered = asyncio.Event()
        closed = []

        class Client:
            def __init__(self, configs):
                pass

            async def get_tools(self, *, server_name):
                try:
                    if server_name == "kedi_mcp_0":
                        entered.set()
                        await asyncio.Event().wait()
                    else:
                        await entered.wait()
                        if not cancel:
                            raise RuntimeError("MCP discovery failed")
                        await asyncio.Event().wait()
                finally:
                    closed.append(server_name)

        monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)
        adapter = LangChainAdapter(
            FakeMessagesListChatModel(responses=[AIMessage(content="unused")])
        )
        profile = owned_profile(McpServerSpec(command="reader"), McpServerSpec(command="writer"))
        task = asyncio.create_task(adapter._mcp_tools_for_profile(profile))
        await asyncio.wait_for(entered.wait(), 2)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await asyncio.wait_for(task, 2)
        assert sorted(closed) == ["kedi_mcp_0", "kedi_mcp_1"]

    asyncio.run(scenario())


def test_langchain_conflicting_ownership_rejects_before_discovery(monkeypatch):
    calls = []

    class Client:
        def __init__(self, configs):
            pass

        async def get_tools(self, *, server_name):
            calls.append(server_name)
            return []

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)
    profile = owned_profile(McpServerSpec(command="reader"))
    original = profile.capabilities[0]
    conflicting = replace(
        original,
        identity="fixture::conflicting",
        name="conflicting",
        profile=replace(original.profile, approval=ApprovalPolicy.deny()),
    )
    profile = replace(profile, capabilities=(original, conflicting))
    adapter = LangChainAdapter(FakeMessagesListChatModel(responses=[AIMessage(content="unused")]))
    with pytest.raises(KediExecutionError, match="Conflicting capability approval"):
        asyncio.run(adapter._mcp_tools_for_profile(profile))
    assert calls == []


@pytest.mark.parametrize("codemode", [False, True])
def test_langchain_duplicate_mcp_catalog_fails_before_model(monkeypatch, codemode):
    calls = []

    def lookup() -> str:
        calls.append("tool")
        return "unused"

    class Client:
        def __init__(self, configs):
            pass

        async def get_tools(self, *, server_name):
            return [StructuredTool.from_function(lookup, description="Find a revision.")]

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            calls.append("model")
            return self

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)
    adapter = LangChainAdapter(Model(responses=[AIMessage(content="unused")]), codemode=codemode)
    profile = owned_profile(McpServerSpec(command="reader"), McpServerSpec(command="writer"))
    with adapter.profile_override(profile), pytest.raises(ValueError, match="Duplicate.*lookup"):
        asyncio.run(adapter.invoke(prompt="Read the revision."))
    assert calls == []


@pytest.mark.parametrize("codemode", [False, True])
def test_pydantic_duplicate_mcp_catalog_closes_sessions_before_model(monkeypatch, codemode):
    events = []
    servers = {name: FastMCP(name) for name in ("reader", "writer")}

    def lookup() -> str:
        pytest.fail("An ambiguous tool was executed")

    for server in servers.values():
        server.tool()(lookup)

    class ObservedMCPToolset(MCPToolset):
        async def __aenter__(self):
            value = await super().__aenter__()
            events.append(("enter", id(self)))
            return value

        async def __aexit__(self, *args):
            try:
                return await super().__aexit__(*args)
            finally:
                events.append(("exit", id(self)))

    monkeypatch.setattr(
        "kedi.agent_adapter.adapters.pydantic.tools._mcp_toolset_from_config",
        lambda name, config: ObservedMCPToolset(servers[config["command"]]),
    )

    def respond(messages, info):
        pytest.fail("An ambiguous catalog reached the model")

    adapter = PydanticAdapter(FunctionModel(respond), codemode=codemode)
    profile = owned_profile(McpServerSpec(command="reader"), McpServerSpec(command="writer"))
    with adapter.profile_override(profile), pytest.raises(UserError, match="name conflicts"):
        adapter.run_sync("Read the revision.")
    entered = [identity for action, identity in events if action == "enter"]
    exited = [identity for action, identity in events if action == "exit"]
    assert len(set(entered)) == 2
    assert sorted(entered) == sorted(exited)
