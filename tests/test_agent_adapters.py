from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter.adapters import DSPyAdapter, PydanticAdapter, make_repr
from kedi.agent_adapter.tool_override import ToolScope
from kedi.agent_adapter.utils import (
    dspy_sync_tool_function,
    kedi_procedure_tool_spec,
    python_callable_tool_spec,
)
from kedi.lang import compile_program, parse_program
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.mcp import McpServerSpec


class DirectOutput(BaseModel):
    value: str


def test_make_repr_uses_class_name_for_standard_class_repr() -> None:
    class Custom:
        pass

    assert make_repr(Custom) == "Custom"


def test_pydantic_adapter_builds_types_and_produces_with_schema_or_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized_models: list[Any] = []

    def fake_agent_init(self: object, model: object = None, **_: Any) -> None:
        initialized_models.append(model)

    monkeypatch.setattr(adapters_module.Agent, "__init__", fake_agent_init)

    adapter = cast(Any, PydanticAdapter(model="openai/gpt-4o-mini"))
    PydanticAdapter(model=TestModel())

    async def fake_run(template: str, *, output_type: type[Any], **kwargs: Any) -> Any:
        assert template == "template"
        assert kwargs == {"metadata": "ok"}
        if output_type is DirectOutput:
            return SimpleNamespace(output=DirectOutput(value="direct"))
        return SimpleNamespace(output=output_type(answer=3))

    monkeypatch.setattr(adapter, "run", fake_run)

    assert initialized_models[0] == "openai:gpt-4o-mini"
    assert isinstance(initialized_models[1], TestModel)

    built = PydanticAdapter.type_builder({"answer": int})
    assert issubclass(built, BaseModel)
    assert built(answer=1).answer == 1

    schema_result = asyncio.run(
        adapter.produce("template", output_schema={"answer": int}, metadata="ok")
    )
    direct_result = asyncio.run(
        adapter.produce("template", output_type=DirectOutput, metadata="ok")
    )

    assert schema_result.answer == 3
    assert direct_result == DirectOutput(value="direct")
    assert adapter.produce_sync(
        "template", output_type=DirectOutput, metadata="ok"
    ) == DirectOutput(value="direct")

    with pytest.raises(ValueError, match="Output schema cannot be empty"):
        PydanticAdapter.type_builder({})
    with pytest.raises(ValueError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("template"))


def test_pydantic_adapter_builds_mcp_toolsets(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic_ai.mcp as pydantic_mcp

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fake_toolset(*args: Any, **kwargs: Any) -> str:
        calls.append(("toolset", args, kwargs))
        return f"toolset:{len(calls)}"

    monkeypatch.setattr(pydantic_mcp, "MCPToolset", fake_toolset)

    adapter = object.__new__(PydanticAdapter)
    profile = AgentProfile(
        mcp_servers=(
            McpServerSpec(command="vsh", args=("run", "--mcp"), env={"A": "B"}),
            McpServerSpec(transport="sse", url="http://localhost/sse", headers={"H": "V"}),
            McpServerSpec(transport="http", url="http://localhost/mcp", headers={"H2": "V2"}),
        )
    )

    assert adapter._mcp_toolsets_for_profile(profile) == [
        "toolset:1",
        "toolset:2",
        "toolset:3",
    ]
    assert calls == [
        (
            "toolset",
            (
                {
                    "mcpServers": {
                        "kedi_mcp_0": {
                            "command": "vsh",
                            "args": ["run", "--mcp"],
                            "env": {"A": "B"},
                        }
                    }
                },
            ),
            {},
        ),
        (
            "toolset",
            (
                {
                    "mcpServers": {
                        "kedi_mcp_1": {
                            "url": "http://localhost/sse",
                            "transport": "sse",
                            "headers": {"H": "V"},
                        }
                    }
                },
            ),
            {},
        ),
        ("toolset", ("http://localhost/mcp",), {"headers": {"H2": "V2"}}),
    ]

    assert adapter._mcp_toolsets_for_profile(
        AgentProfile(
            mcp_servers=(
                McpServerSpec(command="vsh"),
                McpServerSpec(transport="sse", url="http://localhost/sse"),
            )
        )
    ) == ["toolset:4", "toolset:5"]
    assert calls[-2:] == [
        (
            "toolset",
            ({"mcpServers": {"kedi_mcp_0": {"command": "vsh", "args": []}}},),
            {},
        ),
        (
            "toolset",
            ({"mcpServers": {"kedi_mcp_1": {"url": "http://localhost/sse", "transport": "sse"}}},),
            {},
        ),
    ]

    broken = SimpleNamespace(transport="other")
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        adapter._mcp_toolsets_for_profile(
            cast(AgentProfile, SimpleNamespace(mcp_servers=(broken,)))
        )

    with pytest.raises(ValueError, match="stdio transport requires a command"):
        adapter._mcp_toolsets_for_profile(AgentProfile(mcp_servers=(McpServerSpec(),)))
    with pytest.raises(ValueError, match="sse transport requires a url"):
        adapter._mcp_toolsets_for_profile(
            AgentProfile(mcp_servers=(McpServerSpec(transport="sse"),))
        )
    with pytest.raises(ValueError, match="http transport requires a url"):
        adapter._mcp_toolsets_for_profile(
            AgentProfile(mcp_servers=(McpServerSpec(transport="http"),))
        )


def test_pydantic_adapter_profile_override_rejects_broken_model_profile() -> None:
    adapter = object.__new__(PydanticAdapter)
    adapter._profile_stack = []
    adapter._model = "openai:gpt-4o-mini"

    broken_profile = cast(
        AgentProfile,
        SimpleNamespace(
            model=None,
            effort=None,
            needs_model_override=lambda _default: True,
        ),
    )

    with pytest.raises(RuntimeError, match="requires a model override"):
        with adapter.profile_override(broken_profile):
            pass


def test_pydantic_adapter_includes_active_profile_mcp_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PydanticAdapter, "_mcp_toolsets_for_profile", lambda *_: ["mcp-toolset"])

    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()
    adapter._profile_stack = [AgentProfile(mcp_servers=(McpServerSpec(command="vsh"),))]

    assert adapter._toolsets_for_run() == ["mcp-toolset"]


def test_dspy_adapter_configures_lm_and_builds_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lm_calls: list[dict[str, Any]] = []
    configure_calls: list[dict[str, Any]] = []

    class FakeLM:
        def __init__(self, **kwargs: Any) -> None:
            lm_calls.append(kwargs)

    class FakeSettings:
        def configure(self, **kwargs: Any) -> None:
            configure_calls.append(kwargs)

    monkeypatch.setattr(adapters_module, "LM", FakeLM)
    monkeypatch.setattr(adapters_module.dspy, "settings", FakeSettings())

    DSPyAdapter(model="openai:gpt-4o-mini", max_tokens=12, temperature=0.25, provider=None)

    assert lm_calls == [
        {
            "model": "openai/gpt-4o-mini",
            "provider": None,
            "max_tokens": 12,
            "temperature": 0.25,
            "cache": False,
        }
    ]
    assert configure_calls and isinstance(configure_calls[0]["lm"], FakeLM)

    signature = DSPyAdapter.type_builder({"answer": int})
    assert issubclass(signature, adapters_module.dspy.Signature)

    with pytest.raises(ValueError, match="Output schema must be provided"):
        DSPyAdapter.type_builder({})


def test_dspy_adapter_produce_and_sync_use_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSignature:
        def __init__(self) -> None:
            self.instructions: str | None = None

        def with_instructions(self, instructions: str | None) -> "FakeSignature":
            self.instructions = instructions
            return self

    class FakePredict:
        def __init__(self, signature: FakeSignature) -> None:
            self.signature = signature

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"instructions": self.signature.instructions, **kwargs}

    def fake_type_builder(output_schema: dict[str, type], **_: Any) -> FakeSignature:
        assert output_schema == {"answer": str}
        return FakeSignature()

    monkeypatch.setattr(adapters_module.dspy, "Predict", FakePredict)
    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(fake_type_builder))

    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = None
    adapter._profile_stack = []
    result = asyncio.run(
        adapter.produce(
            "prompt",
            {"answer": str},
            user_prompt="ignored",
            instructions="use context",
            extra=1,
        )
    )

    assert result == {
        "instructions": "use context",
        "extra": 1,
        "user_prompt": "prompt",
    }
    assert adapter.produce_sync("prompt", {"answer": str}, instructions=None) == {
        "instructions": None,
        "user_prompt": "prompt",
    }

    with pytest.raises(ValueError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("prompt", {}))


def test_dspy_adapter_produce_sync_uses_mcp_stdio_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []

    @dataclass
    class FakeStdioServerParameters:
        command: str
        args: list[str]
        env: dict[str, str] | None

    class FakeStdioClient:
        def __init__(self, params: FakeStdioServerParameters) -> None:
            self.params = params

        async def __aenter__(self) -> tuple[str, str]:
            events.append(("stdio", self.params))
            return "read", "write"

        async def __aexit__(self, *_: object) -> None:
            events.append("stdio-closed")

    class FakeClientSession:
        def __init__(self, read: str, write: str) -> None:
            self.read = read
            self.write = write

        async def __aenter__(self) -> "FakeClientSession":
            events.append(("session", self.read, self.write))
            return self

        async def __aexit__(self, *_: object) -> None:
            events.append("session-closed")

        async def initialize(self) -> None:
            events.append("initialized")

        async def list_tools(self) -> SimpleNamespace:
            return SimpleNamespace(tools=[SimpleNamespace(name="remote_search")])

    mcp_module = ModuleType("mcp")
    setattr(mcp_module, "ClientSession", FakeClientSession)
    setattr(mcp_module, "StdioServerParameters", FakeStdioServerParameters)
    mcp_client_module = ModuleType("mcp.client")
    mcp_stdio_module = ModuleType("mcp.client.stdio")
    setattr(mcp_stdio_module, "stdio_client", FakeStdioClient)
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", mcp_client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", mcp_stdio_module)

    class FakeTool:
        @staticmethod
        def from_mcp_tool(session: FakeClientSession, tool: SimpleNamespace) -> str:
            assert isinstance(session, FakeClientSession)
            return f"mcp:{tool.name}"

    class FakeReAct:
        def __init__(self, signature: object, tools: list[Any]) -> None:
            self.signature = signature
            self.tools = tools

        async def acall(self, **kwargs: Any) -> dict[str, Any]:
            return {"tools": self.tools, **kwargs}

    monkeypatch.setattr(adapters_module.dspy, "Tool", FakeTool)
    monkeypatch.setattr(adapters_module.dspy, "ReAct", FakeReAct)

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    adapter._profile_stack = [
        AgentProfile(
            mcp_servers=(McpServerSpec(command="vsh", args=("run", "--mcp"), env={"A": "B"}),)
        )
    ]

    result = asyncio.run(adapter._produce_with_mcp(object(), ["local"], "prompt", {"extra": 1}))

    assert result == {"tools": ["local", "mcp:remote_search"], "extra": 1, "user_prompt": "prompt"}
    assert events[0][0] == "stdio"
    assert events[0][1] == FakeStdioServerParameters("vsh", ["run", "--mcp"], {"A": "B"})
    assert events[1:] == [
        ("session", "read", "write"),
        "initialized",
        "session-closed",
        "stdio-closed",
    ]


def test_dspy_adapter_produce_sync_routes_to_mcp_when_profile_has_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_produce_with_mcp(
        self: DSPyAdapter,
        sig: object,
        tools: list[Any],
        template: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return {"sig": sig, "tools": tools, "template": template, "kwargs": kwargs}

    class FakeSignature:
        def with_instructions(self, instructions: str | None) -> str:
            assert instructions is None
            return "sig"

    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(lambda _: FakeSignature()))
    monkeypatch.setattr(DSPyAdapter, "_produce_with_mcp", fake_produce_with_mcp)

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    adapter._profile_stack = [AgentProfile(mcp_servers=(McpServerSpec(command="vsh"),))]

    result = adapter.produce_sync("prompt", {"answer": str}, user_prompt="ignored", extra=1)

    assert result == {"sig": "sig", "tools": [], "template": "prompt", "kwargs": {"extra": 1}}


def test_dspy_adapter_mcp_rejects_non_stdio() -> None:
    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    adapter._profile_stack = [
        AgentProfile(mcp_servers=(McpServerSpec(transport="http", url="http://localhost"),))
    ]

    with pytest.raises(NotImplementedError, match="stdio"):
        asyncio.run(adapter._produce_with_mcp(object(), [], "prompt", {}))


def test_dspy_adapter_mcp_rejects_missing_stdio_command() -> None:
    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    adapter._profile_stack = [AgentProfile(mcp_servers=(McpServerSpec(),))]

    with pytest.raises(ValueError, match="stdio transport requires a command"):
        asyncio.run(adapter._produce_with_mcp(object(), [], "prompt", {}))


def test_dspy_adapter_profile_override_rejects_broken_model_profile() -> None:
    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = "openai/gpt-4o-mini"
    adapter._profile_stack = []

    broken_profile = cast(
        AgentProfile,
        SimpleNamespace(
            model=None,
            effort=None,
            needs_model_override=lambda _default: True,
        ),
    )

    with pytest.raises(RuntimeError, match="requires a model override"):
        with adapter.profile_override(broken_profile):
            pass


def test_dspy_adapter_register_tool_wraps_async_callable() -> None:
    async def lookup(query: str, limit: int = 3) -> str:
        """Lookup docs."""
        return f"{query}:{limit}"

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(python_callable_tool_spec("lookup", lookup))

    tool = adapter.scoped_tools[0]
    assert getattr(tool, "name") == "lookup"
    assert getattr(tool, "desc") == "Lookup docs."
    assert getattr(tool, "args") == {
        "query": {"type": "string"},
        "limit": {"default": 3, "type": "integer"},
    }
    assert inspect.signature(tool.func) == inspect.signature(lookup)
    assert tool(query="kedi", limit=2) == "kedi:2"


def test_dspy_sync_tool_function_awaits_inside_running_loop() -> None:
    async def lookup(query: str) -> str:
        return query.upper()

    wrapped = dspy_sync_tool_function(lookup)

    async def outer() -> str:
        return wrapped("kedi")

    assert asyncio.run(outer()) == "KEDI"


def test_dspy_adapter_register_tool_wraps_async_callable_instance() -> None:
    class WebFetchLocalTool:
        """Callable object doc should not replace __call__ metadata."""

        async def __call__(self, url: str, timeout: int = 10) -> str:
            """Fetch a URL."""
            return f"{url}:{timeout}"

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(python_callable_tool_spec("web_fetch", WebFetchLocalTool()))

    tool = adapter.scoped_tools[0]
    assert getattr(tool, "name") == "web_fetch"
    assert getattr(tool, "desc") == "Fetch a URL."
    assert getattr(tool, "args") == {
        "url": {"type": "string"},
        "timeout": {"default": 10, "type": "integer"},
    }
    assert inspect.signature(tool.func) == inspect.signature(WebFetchLocalTool().__call__)
    assert tool(url="https://example.com", timeout=2) == "https://example.com:2"


def test_dspy_adapter_register_tool_wraps_sync_callable_idempotently() -> None:
    def lookup(query: str) -> str:
        """Lookup docs."""
        return query.upper()

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(python_callable_tool_spec("lookup", lookup))
    tool = adapter.scoped_tools[0]

    assert tool(query="kedi") == "KEDI"
    assert dspy_sync_tool_function(tool.func) is tool.func


def test_pydantic_adapter_registers_kedi_and_python_tools() -> None:
    def lookup(query: str) -> str:
        return query

    src = "@search(query: str) -> str:\n  = <query>\n"
    prog = parse_program(src)
    rt = compile_program(prog, adapter=None)
    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()

    adapter.register_tool(kedi_procedure_tool_spec(rt, prog.procedures[0]))
    adapter.register_tool(python_callable_tool_spec("lookup", lookup))

    tools = adapter.scoped_tools
    assert [tool.name for tool in tools] == ["search", "lookup"]
    assert tools[0].metadata is not None
    assert tools[1].function is lookup
