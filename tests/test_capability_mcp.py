"""MCP ownership is independent of the host policy and exposed tool spelling."""

import asyncio
from dataclasses import replace
import re

import pytest
from mcp.server.fastmcp import FastMCP
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.hooks import PreToolUseDecision
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.capabilities import (
    AppliedCapability,
    capability_mcp_policy,
    mcp_server_identity,
)
from kedi.agent_profile.hooks import HookSettings
from kedi.agent_profile.mcp import McpServerSpec
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.errors import KediExecutionError


def mcp_profile(events, *, deferred=False, deny_read=False):
    reader = McpServerSpec(command="reader")
    writer = McpServerSpec(command="writer")

    def before(event):
        events.append(("pre", event.tool_name))
        if deny_read:
            return PreToolUseDecision.deny("Source access is denied for this request.")
        return PreToolUseDecision.continue_()

    def after(event):
        events.append(("post", event.tool_name))

    capability = AppliedCapability(
        identity="fixture::reader",
        name="reader",
        description="Read document evidence",
        profile=AgentProfile(
            mcp_servers=(reader,),
            approval=ApprovalPolicy.allow(),
            hooks=HookSettings(pre_tool_use=(before,), post_tool_use=(after,)),
        ),
        deferred=deferred,
    )
    return AgentProfile(
        approval=ApprovalPolicy.deny(),
        capabilities=(capability,),
        mcp_servers=(writer,) if deferred else (reader, writer),
    )


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("codemode", [False, True])
def test_native_pydantic_mcp_keeps_capability_policy_and_hooks(monkeypatch, deferred, codemode):
    executed = []
    events = []
    servers = {name: FastMCP(name) for name in ("reader", "writer")}

    @servers["reader"].tool()
    def read_document() -> str:
        executed.append("read")
        return "Revision 7"

    @servers["writer"].tool()
    def write_document() -> str:
        executed.append("write")
        return "written"

    def toolset(name, config):
        return MCPToolset(servers[config["command"]])

    monkeypatch.setattr(
        "kedi.agent_adapter.adapters.pydantic.tools._mcp_toolset_from_config", toolset
    )
    requests = []

    def respond(messages, info):
        requests.append(info)
        names = {tool.name for tool in info.function_tools}
        if deferred and len(requests) == 1:
            assert "read_document" not in names
            identity = re.search(r"kedi_reader_[a-f0-9]{12}", info.instructions).group()
            return ModelResponse([ToolCallPart("load_capability", {"id": identity}, "load")])
        step = len(requests) - int(deferred)
        if codemode:
            assert not names & {"read_document", "write_document"}
            assert {"get_tool_schema", "execute_code"} <= names
            if step == 1:
                return ModelResponse(
                    [
                        ToolCallPart(
                            "get_tool_schema", {"tool_names": ["read_document", "write_document"]}
                        )
                    ]
                )
            if step == 2:
                return ModelResponse(
                    [
                        ToolCallPart(
                            "execute_code",
                            {"code": "await read_document()\nawait write_document()"},
                        )
                    ]
                )
        elif step == 1:
            assert {"read_document", "write_document"} <= names
            return ModelResponse(
                [
                    ToolCallPart("read_document", {}, "read"),
                    ToolCallPart("write_document", {}, "write"),
                ]
            )
        assert "denied" in str(messages)
        return ModelResponse([TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), codemode=codemode)
    with adapter.profile_override(mcp_profile(events, deferred=deferred)):
        assert adapter.run_sync("Read and write the document.").output == "done"
    assert executed == ["read"]
    assert events == [("pre", "read_document"), ("post", "read_document")]


@pytest.mark.parametrize("codemode", [False, True])
@pytest.mark.parametrize("host_hooks", [False, True])
@pytest.mark.parametrize("deny_read", [False, True])
def test_langchain_mcp_keeps_server_ownership_through_native_middleware(
    monkeypatch, codemode, host_hooks, deny_read
):
    executed = []
    events = []

    def read_document() -> str:
        executed.append("read")
        return "Revision 7"

    def write_document() -> str:
        executed.append("write")
        return "written"

    class Client:
        def __init__(self, configs):
            self.configs = configs

        async def get_tools(self, *, server_name):
            function = (
                read_document
                if self.configs[server_name]["command"] == "reader"
                else write_document
            )
            return [StructuredTool.from_function(function, description=function.__name__)]

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.i == len(self.responses) - 1:
                denied = [
                    message
                    for message in messages
                    if isinstance(message, ToolMessage) and message.status == "error"
                ]
                assert denied and "denied" in str(denied[-1].content)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)
    responses = (
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_tool_schema",
                        "args": {"tool_names": ["read_document", "write_document"]},
                        "id": "schema",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute_code",
                        "args": {"code": "await read_document()\nawait write_document()"},
                        "id": "execute",
                    }
                ],
            ),
        ]
        if codemode
        else [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_document", "args": {}, "id": "read"},
                    {"name": "write_document", "args": {}, "id": "write"},
                ],
            ),
        ]
    )
    adapter = LangChainAdapter(
        Model(responses=[*responses, AIMessage(content="done")]), codemode=codemode
    )
    profile = mcp_profile(events, deny_read=deny_read)
    if host_hooks:
        profile = replace(
            profile,
            hooks=HookSettings(pre_tool_use=(lambda event: PreToolUseDecision.continue_(),)),
        )
    with adapter.profile_override(profile):
        assert asyncio.run(adapter.invoke(prompt="Read and write the document.")) == "done"
    assert executed == ([] if deny_read else ["read"])
    assert events == (
        [("pre", "read_document")]
        if deny_read
        else [("pre", "read_document"), ("post", "read_document")]
    )


def test_conflicting_mcp_owners_fail_without_disclosing_server_credentials():
    server = McpServerSpec(
        transport="http", url="https://example.invalid/mcp", headers={"secret": "test-secret"}
    )
    first = AppliedCapability(
        identity="first",
        name="first",
        description=None,
        profile=AgentProfile(mcp_servers=(server,), approval=ApprovalPolicy.allow()),
    )
    second = replace(
        first,
        identity="second",
        name="second",
        profile=replace(first.profile, approval=ApprovalPolicy.deny()),
    )
    with pytest.raises(KediExecutionError, match="Conflicting capability approval") as caught:
        capability_mcp_policy(
            AgentProfile(capabilities=(first, second)), mcp_server_identity(server)
        )
    assert "test-secret" not in str(caught.value)
    assert "example.invalid" not in str(caught.value)
