from __future__ import annotations

import asyncio
import builtins
import inspect
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from re import Pattern
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel, EmailStr, FileUrl, HttpUrl
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import ACPAdapter as ExportedACPAdapter
from kedi.agent_adapter import AdapterCapabilities
from kedi.agent_adapter import ClaudeAdapter as ExportedClaudeAdapter
from kedi.agent_adapter import DSPyAdapter as ExportedDSPyAdapter
from kedi.agent_adapter import LangChainAdapter as ExportedLangChainAdapter
from kedi.agent_adapter import PydanticAdapter as ExportedPydanticAdapter
from kedi.agent_adapter import WebGPUAdapter
from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter.adapters import (
    ACPAdapter,
    ClaudeAdapter,
    DSPyAdapter,
    LangChainAdapter,
    PydanticAdapter,
)
from kedi.agent_adapter.adapters import claude as claude_adapter_module
from kedi.agent_adapter.adapters import dspy as dspy_adapter_module
from kedi.agent_adapter.adapters import make_repr
from kedi.agent_adapter.adapters import pydantic as pydantic_adapter_module
from kedi.agent_adapter.tool_override import ToolScope, ToolSpec
from kedi.agent_adapter.utils import (
    dspy_sync_tool_function,
    kedi_procedure_tool_spec,
    kedi_tool_call_context,
    python_callable_tool_spec,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.mcp import McpServerSpec
from kedi.lang import compile_program, parse_program


class DirectOutput(BaseModel):
    value: str


class DirectAnswer(BaseModel):
    answer: str


class FormattedClaudeOutput(BaseModel):
    created_at: datetime
    elapsed: timedelta
    day: date
    clock: time
    matcher: Pattern[str]
    email: EmailStr
    site: HttpUrl
    file: FileUrl


def _set_profile_stack(adapter: object, *profiles: AgentProfile) -> None:
    setattr(
        adapter,
        "_profile_stack_var",
        ContextVar(f"test_profile_stack_{id(adapter)}", default=tuple(profiles)),
    )


def test_make_repr_uses_class_name_for_standard_class_repr() -> None:
    class Custom:
        pass

    assert make_repr(Custom) == "Custom"


def test_agent_adapter_metadata_is_exposed_from_public_surfaces() -> None:
    expected = [
        (PydanticAdapter, ExportedPydanticAdapter, "pydantic"),
        (LangChainAdapter, ExportedLangChainAdapter, "langchain"),
        (DSPyAdapter, ExportedDSPyAdapter, "dspy"),
    ]
    for adapter_cls, exported_cls, shortname in expected:
        assert adapter_cls is exported_cls
        assert adapter_cls.kind == "agent-framework"
        assert adapter_cls.shortname == shortname
        assert isinstance(adapter_cls.capabilities, AdapterCapabilities)
        assert adapter_cls.capabilities.supports("structured_output")

    assert WebGPUAdapter.kind == "agent-framework"
    assert WebGPUAdapter.shortname == "webgpu"
    assert WebGPUAdapter.capabilities.supports("tool_registration")
    assert ACPAdapter is ExportedACPAdapter
    assert ACPAdapter.kind == "agent-harness"
    assert ACPAdapter.shortname == "acp"
    assert not ACPAdapter.capabilities.supports("structured_output")
    assert not ACPAdapter.capabilities.supports("tool_registration")
    assert ClaudeAdapter is ExportedClaudeAdapter
    assert ClaudeAdapter.kind == "agent-harness"
    assert ClaudeAdapter.shortname == "claude"
    assert ClaudeAdapter.capabilities.supports("mcp")


def test_adapter_capabilities_support_boolean_fields() -> None:
    empty = AdapterCapabilities()
    assert not empty.supports("structured_output")

    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_mcp=True,
        supports_profile_override=True,
        supports_model_override=True,
        supports_effort=True,
        supports_settings=True,
        supports_codemode=True,
    )

    assert capabilities.supports("structured_output")
    assert capabilities.supports("tool_registration")
    assert capabilities.supports("mcp")
    assert capabilities.supports("profile_override")
    assert capabilities.supports("model_override")
    assert capabilities.supports("effort")
    assert capabilities.supports("settings")
    assert capabilities.supports("codemode")


def test_pydantic_adapter_builds_types_and_produces_with_schema_or_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized_models: list[Any] = []
    initialized_kwargs: list[dict[str, Any]] = []

    def fake_agent_init(self: object, model: object = None, **kwargs: Any) -> None:
        initialized_models.append(model)
        initialized_kwargs.append(kwargs)

    monkeypatch.setattr(pydantic_adapter_module.Agent, "__init__", fake_agent_init)

    adapter = cast(Any, PydanticAdapter(model="openai/gpt-4o-mini"))
    PydanticAdapter(model=TestModel())
    PydanticAdapter(
        model="openai/gpt-4o-mini",
        model_settings={"temperature": 0.1, "num_retries": 2},
    )
    PydanticAdapter(
        model="openai/gpt-4o-mini",
        model_settings={"num_retries": 2},
    )

    async def fake_run(template: str, *, output_type: type[Any], **kwargs: Any) -> Any:
        assert template == "template"
        assert kwargs == {"metadata": "ok"}
        if output_type is DirectOutput:
            return SimpleNamespace(output=DirectOutput(value="direct"))
        if set(output_type.model_fields) == {"summary"}:
            return SimpleNamespace(output=output_type(summary="ok"))
        if set(output_type.model_fields) == {"payload"}:
            return SimpleNamespace(output=output_type(payload=DirectOutput(value="direct")))
        return SimpleNamespace(output=output_type(answer=3, note="ok"))

    monkeypatch.setattr(adapter, "run", fake_run)

    assert initialized_models[0] == "openai-chat:gpt-4o-mini"
    assert isinstance(initialized_models[1], TestModel)
    assert initialized_kwargs[2]["model_settings"] == {"temperature": 0.1}
    assert "model_settings" not in initialized_kwargs[3]

    built = PydanticAdapter.type_builder({"answer": int})
    assert issubclass(built, BaseModel)
    assert built(answer=1).answer == 1

    schema_result = asyncio.run(
        adapter.produce("template", output_schema={"answer": int}, metadata="ok")
    )
    direct_result = asyncio.run(
        adapter.produce("template", output_type=DirectOutput, metadata="ok")
    )
    string_result = asyncio.run(
        adapter.produce("template", output_schema={"summary": str}, metadata="ok")
    )
    custom_schema_result = asyncio.run(
        adapter.produce(
            "template",
            output_schema={"payload": DirectOutput},
            metadata="ok",
        )
    )
    multi_result = asyncio.run(
        adapter.produce(
            "template",
            output_schema={"answer": int, "note": str},
            metadata="ok",
        )
    )

    assert isinstance(schema_result, BaseModel)
    assert schema_result.answer == 3
    assert direct_result == DirectOutput(value="direct")
    assert isinstance(string_result, BaseModel)
    assert string_result.summary == "ok"
    assert isinstance(custom_schema_result, BaseModel)
    assert custom_schema_result.payload == DirectOutput(value="direct")
    assert multi_result.answer == 3
    assert multi_result.note == "ok"
    assert adapter.produce_sync(
        "template", output_type=DirectOutput, metadata="ok"
    ) == DirectOutput(value="direct")

    with pytest.raises(ValueError, match="Output schema cannot be empty"):
        PydanticAdapter.type_builder({})
    with pytest.raises(ValueError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("template"))


def test_pydantic_adapter_isolates_nested_template_runs_inside_kedi_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = PydanticAdapter(model="openai/gpt-4o-mini")
    src = "@lesser_known_city(country):\n  >> [x] is a less known city in <country>.\n  = <x>\n"
    prog = parse_program(src)
    rt = compile_program(prog, adapter=adapter)
    tool = kedi_procedure_tool_spec(rt, prog.procedures[0])
    adapter.register_tool(tool)

    async def reused_agent_run(*_: Any, **__: Any) -> Any:
        raise AssertionError("nested Kedi tool template reused the outer adapter agent")

    monkeypatch.setattr(adapter, "run", reused_agent_run)

    created_agents: list[dict[str, Any]] = []

    class FreshAgent:
        def __init__(self, model: Any = None, **kwargs: Any) -> None:
            created_agents.append({"model": model, "kwargs": kwargs})

        async def run(self, template: str, *, output_type: type[Any], **kwargs: Any) -> Any:
            assert "less known city" in template
            assert kwargs == {}
            return SimpleNamespace(output=output_type(x="Konya"))

    monkeypatch.setattr(pydantic_adapter_module, "Agent", FreshAgent)

    assert tool.fn(country="Turkey") == "Konya"
    assert created_agents == [{"model": "openai-chat:gpt-4o-mini", "kwargs": {}}]


def _install_fake_langchain_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    events: dict[str, Any] = {
        "models": [],
        "agents": [],
        "tools": [],
        "mcp_configs": [],
    }

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            events["payload"] = payload
            response_format = self.kwargs.get("response_format")
            if response_format is None:
                return {"messages": [SimpleNamespace(content="plain langchain text")]}
            return {"structured_response": response_format(answer="ok")}

    class FakeStructuredTool:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            self.metadata = None

        @classmethod
        def from_function(cls, **kwargs: Any) -> FakeStructuredTool:
            events["tools"].append(kwargs)
            return cls(**kwargs)

    class FakeMCPClient:
        def __init__(self, configs: dict[str, Any]) -> None:
            events["mcp_configs"].append(configs)

        async def get_tools(self) -> list[str]:
            return ["mcp-tool"]

    def fake_init_chat_model(model: str, **settings: Any) -> dict[str, Any]:
        call = {"model": model, "settings": settings}
        events["models"].append(call)
        return call

    def fake_create_agent(**kwargs: Any) -> FakeAgent:
        events["agents"].append(kwargs)
        return FakeAgent(**kwargs)

    langchain = ModuleType("langchain")
    chat_models = ModuleType("langchain.chat_models")
    agents = ModuleType("langchain.agents")
    langchain_core = ModuleType("langchain_core")
    langchain_core_tools = ModuleType("langchain_core.tools")
    langchain_mcp_adapters = ModuleType("langchain_mcp_adapters")
    langchain_mcp_client = ModuleType("langchain_mcp_adapters.client")

    setattr(chat_models, "init_chat_model", fake_init_chat_model)
    setattr(agents, "create_agent", fake_create_agent)
    setattr(langchain_core_tools, "StructuredTool", FakeStructuredTool)
    setattr(langchain_mcp_client, "MultiServerMCPClient", FakeMCPClient)

    monkeypatch.setitem(sys.modules, "langchain", langchain)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", chat_models)
    monkeypatch.setitem(sys.modules, "langchain.agents", agents)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", langchain_core_tools)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", langchain_mcp_adapters)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", langchain_mcp_client)
    return events


def test_langchain_adapter_produces_with_structured_output_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_langchain_modules(monkeypatch)
    adapter = LangChainAdapter(
        model="openrouter/google/gemini-3-flash-preview",
        model_settings={
            "temperature": 0.1,
            "parallel_tool_calls": False,
            "num_retries": 2,
        },
    )
    adapter.register_tool(
        python_callable_tool_spec(
            "lookup",
            lambda query: f"found {query}",
        )
    )

    with adapter.profile_override(
        AgentProfile(
            model="openai/gpt-4o-mini",
            effort="max",
            system="Follow instructions.",
            settings={"max_tokens": 12, "num_retries": 5},
            mcp_servers=(McpServerSpec(command="vsh", args=("run", "--mcp"), env={"A": "B"}),),
        )
    ):
        result = asyncio.run(
            adapter.produce(
                "template",
                output_schema={"answer": str},
                instructions="Use terse prose.",
                trace_id="abc",
            )
        )

    assert cast(Any, result).answer == "ok"
    assert events["models"] == [
        {
            "model": "openai:gpt-4o-mini",
            "settings": {
                "temperature": 0.1,
                "parallel_tool_calls": False,
                "max_tokens": 12,
                "reasoning_effort": "xhigh",
            },
        }
    ]
    assert events["agents"][0]["model"] == events["models"][0]
    assert len(events["agents"][0]["tools"]) == 2
    assert events["agents"][0]["system_prompt"] == "Use terse prose."
    assert events["payload"] == {
        "messages": [{"role": "user", "content": "template"}],
        "trace_id": "abc",
    }
    assert events["mcp_configs"] == [
        {
            "kedi_mcp_0": {
                "transport": "stdio",
                "command": "vsh",
                "args": ["run", "--mcp"],
                "env": {"A": "B"},
            }
        }
    ]


def test_langchain_adapter_uses_output_type_and_sync_produce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_langchain_modules(monkeypatch)
    adapter = LangChainAdapter(model={"fake": "model"})

    assert adapter.produce_sync("template", output_type=DirectAnswer).answer == "ok"
    assert adapter.default_model() is None
    with adapter.tool_scope(("manual-tool",)):
        assert adapter.scoped_tools == ("manual-tool",)
    assert adapter.scoped_tools == ()


def test_langchain_adapter_invoke_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _install_fake_langchain_modules(monkeypatch)
    adapter = LangChainAdapter(model="openai/gpt-4o-mini")

    with adapter.profile_override(AgentProfile(system="Be terse.")):
        assert adapter.invoke_sync(prompt="say hi", run_id="abc") == "plain langchain text"

    assert events["agents"][0]["system_prompt"] == "Be terse."
    assert "response_format" not in events["agents"][0]
    assert events["payload"] == {
        "messages": [{"role": "user", "content": "say hi"}],
        "run_id": "abc",
    }


def test_langchain_adapter_requires_output_spec() -> None:
    adapter = LangChainAdapter(model={"fake": "model"})

    with pytest.raises(ValueError, match="Output schema cannot be empty"):
        LangChainAdapter.type_builder({})
    with pytest.raises(ValueError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("template"))
    with pytest.raises(RuntimeError, match="requires a model"):
        LangChainAdapter()._model_for_run()


def test_langchain_adapter_register_tool_preserves_schema_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_langchain_modules(monkeypatch)

    def search(query: str, limit: int = 5) -> str:
        return f"{query}:{limit}"

    json_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    spec = ToolSpec(
        fn=search,
        name="search",
        description="Search docs.",
        json_schema=json_schema,
        metadata={"source": "kedi"},
    )
    adapter = LangChainAdapter(model={"fake": "model"})
    adapter.register_tool(spec)
    tool = adapter.scoped_tools[0]

    assert events["tools"][0]["name"] == "search"
    assert events["tools"][0]["description"] == "Search docs."
    assert events["tools"][0]["func"]("docs", limit=2) == "docs:2"
    assert events["tools"][0]["args_schema"] == json_schema
    assert tool.metadata == {"source": "kedi"}


def test_langchain_adapter_wraps_async_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _install_fake_langchain_modules(monkeypatch)

    async def lookup(query: str) -> str:
        return f"async:{query}"

    adapter = LangChainAdapter(model={"fake": "model"})
    adapter.register_tool(
        ToolSpec(
            fn=lookup,
            name="lookup",
            description=None,
            json_schema=None,
            metadata={},
        )
    )

    assert events["tools"][0]["description"] == "lookup"
    assert events["tools"][0]["func"]("docs") == "async:docs"


def test_langchain_mcp_configs_cover_transports_and_errors() -> None:
    configs = adapters_module._langchain_mcp_server_configs(
        AgentProfile(
            mcp_servers=(
                McpServerSpec(command="vsh", args=("run", "--mcp")),
                McpServerSpec(transport="sse", url="http://localhost/sse"),
                McpServerSpec(
                    transport="sse",
                    url="http://localhost/sse2",
                    headers={"S": "V"},
                ),
                McpServerSpec(
                    transport="http",
                    url="http://localhost/mcp",
                    headers={"H": "V"},
                ),
                McpServerSpec(transport="http", url="http://localhost/mcp2"),
            )
        )
    )

    assert configs == {
        "kedi_mcp_0": {
            "transport": "stdio",
            "command": "vsh",
            "args": ["run", "--mcp"],
        },
        "kedi_mcp_1": {
            "transport": "sse",
            "url": "http://localhost/sse",
        },
        "kedi_mcp_2": {
            "transport": "sse",
            "url": "http://localhost/sse2",
            "headers": {"S": "V"},
        },
        "kedi_mcp_3": {
            "transport": "streamable_http",
            "url": "http://localhost/mcp",
            "headers": {"H": "V"},
        },
        "kedi_mcp_4": {
            "transport": "streamable_http",
            "url": "http://localhost/mcp2",
        },
    }

    with pytest.raises(ValueError, match="stdio transport requires a command"):
        adapters_module._langchain_mcp_server_configs(
            AgentProfile(mcp_servers=(McpServerSpec(transport="stdio"),))
        )
    with pytest.raises(ValueError, match="sse transport requires a url"):
        adapters_module._langchain_mcp_server_configs(
            AgentProfile(mcp_servers=(McpServerSpec(transport="sse"),))
        )
    with pytest.raises(ValueError, match="http transport requires a url"):
        adapters_module._langchain_mcp_server_configs(
            AgentProfile(mcp_servers=(McpServerSpec(transport="http"),))
        )
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        adapters_module._langchain_mcp_server_configs(
            AgentProfile(mcp_servers=(McpServerSpec(transport=cast(Any, "websocket")),))
        )


def test_langchain_settings_merge_openrouter_effort() -> None:
    adapter = LangChainAdapter(
        model="openrouter/google/gemini-3-flash-preview",
        model_settings={"temperature": 0.2, "reasoning": {"summary": "auto"}},
    )

    with adapter.profile_override(
        AgentProfile(
            effort="high",
            settings={"max_tokens": 20, "num_retries": 3},
        )
    ):
        settings = adapter._settings_for_run("openrouter:google/gemini-3-flash-preview")

    assert settings == {
        "temperature": 0.2,
        "reasoning": {"summary": "auto", "effort": "high"},
        "max_tokens": 20,
    }


def test_langchain_structured_response_supports_mapping_and_object() -> None:
    answer = DirectAnswer(answer="ok")

    assert adapters_module._langchain_structured_response({"structured_response": answer}) is answer
    assert (
        adapters_module._langchain_structured_response(SimpleNamespace(structured_response=answer))
        is answer
    )
    with pytest.raises(ValueError, match="did not return structured_response"):
        adapters_module._langchain_structured_response({"output": answer})
    with pytest.raises(ValueError, match="did not return structured_response"):
        adapters_module._langchain_structured_response(SimpleNamespace(output=answer))
    assert adapters_module._langchain_text_response("plain") == "plain"
    assert adapters_module._langchain_text_response({"content": "content"}) == "content"
    assert adapters_module._langchain_text_response({"output": "output"}) == "output"
    assert adapters_module._langchain_text_response({"messages": [{"content": "message"}]}) == (
        "message"
    )
    assert adapters_module._langchain_text_response({"messages": ["string message"]}) == (
        "string message"
    )
    assert (
        adapters_module._langchain_text_response(SimpleNamespace(content="object content"))
        == "object content"
    )
    with pytest.raises(ValueError, match="text response"):
        adapters_module._langchain_text_response({"messages": []})


def _block_import(monkeypatch: pytest.MonkeyPatch, blocked_name: str) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == blocked_name:
            raise ImportError(f"blocked {blocked_name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_langchain_adapter_dependency_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as blocked:
        _block_import(blocked, "langchain.chat_models")
        with pytest.raises(ModuleNotFoundError, match="requires langchain"):
            LangChainAdapter(model="openai:gpt-4o-mini")._chat_model_for_run()

    adapter = LangChainAdapter(model={"fake": "model"})
    with monkeypatch.context() as blocked:
        _block_import(blocked, "langchain.agents")
        with pytest.raises(ModuleNotFoundError, match="requires langchain"):
            adapter._create_agent(
                model={"fake": "model"},
                tools=(),
                response_format=DirectAnswer,
                system_prompt=None,
            )

    with monkeypatch.context() as blocked:
        _block_import(blocked, "langchain_mcp_adapters.client")
        with pytest.raises(ModuleNotFoundError, match="requires langchain-mcp-adapters"):
            asyncio.run(
                adapter._mcp_tools_for_profile(
                    AgentProfile(mcp_servers=(McpServerSpec(command="vsh"),))
                )
            )

    with monkeypatch.context() as blocked:
        _block_import(blocked, "langchain_core.tools")
        with pytest.raises(ModuleNotFoundError, match="requires langchain-core"):
            adapters_module._langchain_tool_from_spec(
                ToolSpec(
                    fn=lambda: "ok",
                    name="lookup",
                    description=None,
                    json_schema=None,
                    metadata={},
                )
            )

    assert asyncio.run(adapter._mcp_tools_for_profile(AgentProfile.empty())) == []


def _install_fake_claude_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    events: dict[str, Any] = {
        "options": [],
        "prompts": [],
        "tools": [],
        "servers": [],
        "prefix_message": object(),
        "structured_output": {"answer": "ok"},
        "text_result": "plain claude text",
    }

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            events["options"].append(kwargs)

    class FakeResultMessage:
        def __init__(
            self,
            structured_output: Any = None,
            *,
            is_error: bool = False,
            result: str = "failed",
        ) -> None:
            self.structured_output = structured_output
            self.is_error = is_error
            self.result = result

    async def fake_query(*, prompt: str, options: Any) -> Any:
        events["prompts"].append({"prompt": prompt, "options": options})
        prefix_message = events.get("prefix_message")
        if prefix_message is not None:
            yield prefix_message
        if "output_format" not in options.kwargs:
            yield FakeResultMessage(result=events["text_result"])
            return
        yield FakeResultMessage(events["structured_output"])

    def fake_tool(*, name: str, description: str, input_schema: dict[str, Any]) -> Any:
        def _decorate(fn: Any) -> Any:
            fn.name = name
            fn.description = description
            fn.input_schema = input_schema
            events["tools"].append(fn)
            return fn

        return _decorate

    def fake_create_sdk_mcp_server(*, name: str, tools: list[Any]) -> dict[str, Any]:
        server = {"name": name, "tools": tools}
        events["servers"].append(server)
        return server

    sdk = ModuleType("claude_agent_sdk")
    setattr(sdk, "ClaudeAgentOptions", FakeClaudeAgentOptions)
    setattr(sdk, "ResultMessage", FakeResultMessage)
    setattr(sdk, "query", fake_query)
    setattr(sdk, "tool", fake_tool)
    setattr(sdk, "create_sdk_mcp_server", fake_create_sdk_mcp_server)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    return events


def test_claude_adapter_produces_with_schema_profile_tools_and_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)

    async def lookup(query: str) -> dict[str, str]:
        return {"found": query}

    adapter = ClaudeAdapter(
        model="claude-sonnet-4",
        model_settings={"max_turns": 3, "temperature": 0.2},
    )
    adapter.register_tool(
        ToolSpec(
            fn=lookup,
            name="lookup",
            description="Look up docs.",
            json_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            metadata={"source": "kedi"},
        )
    )

    with adapter.profile_override(
        AgentProfile(
            model="claude-opus-4",
            effort="max",
            system="Profile system.",
            settings={"cwd": "/tmp/project", "max_tokens": 99},
            mcp_servers=(McpServerSpec(transport="http", url="http://localhost/mcp"),),
        )
    ):
        result = asyncio.run(
            adapter.produce(
                "template",
                output_schema={"answer": str},
                instructions="Runtime instructions.",
                allowed_tools=["existing-tool"],
                mcp_servers={"existing": {"type": "stdio", "command": "existing"}},
            )
        )

    assert cast(Any, result).answer == "ok"
    assert adapter.default_model() == "claude-sonnet-4"
    assert events["prompts"][0]["prompt"] == "template"
    options = events["options"][0]
    assert options["model"] == "claude-opus-4"
    assert options["tools"] == {"type": "preset", "preset": "claude_code"}
    assert options["permission_mode"] == "acceptEdits"
    assert options["max_turns"] == 3
    assert options["cwd"] == "/tmp/project"
    assert options["effort"] == "xhigh"
    assert options["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
        "append": "Runtime instructions.",
    }
    assert options["output_format"]["type"] == "json_schema"
    assert set(options["output_format"]["schema"]["properties"]) == {"answer"}
    assert "temperature" not in options
    assert options["mcp_servers"]["existing"] == {
        "type": "stdio",
        "command": "existing",
    }
    assert options["mcp_servers"]["kedi_mcp_0"] == {
        "type": "http",
        "url": "http://localhost/mcp",
    }
    assert options["mcp_servers"]["kedi"] == events["servers"][0]
    assert options["allowed_tools"] == ["existing-tool", "mcp__kedi__lookup"]

    tool_result = asyncio.run(events["tools"][0]({"query": "docs"}))
    assert tool_result == {"content": [{"type": "text", "text": '{"found": "docs"}'}]}


def test_claude_adapter_supports_output_type_sync_and_tool_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    events["structured_output"] = DirectAnswer(answer="sync")
    adapter = ClaudeAdapter(model=None)

    assert adapter.default_model() is None
    assert adapter.produce_sync("template", output_type=DirectAnswer).answer == "sync"
    assert events["options"][0]["output_format"]["schema"]["title"] == "DirectAnswer"
    assert "model" not in events["options"][0]

    events["structured_output"] = "plain"
    assert adapter.produce_sync("template", output_type=str) == "plain"

    spec = ToolSpec(fn=lambda: "ok", name="manual")
    with adapter.tool_scope((spec,)):
        assert adapter.scoped_tools == (spec,)
        assert adapter.produce_sync("template", output_type=str) == "plain"
    assert adapter.scoped_tools == ()
    assert asyncio.run(events["tools"][-1]({})) == {"content": [{"type": "text", "text": "ok"}]}

    assert adapter.invoke_sync(prompt="template") == "plain claude text"
    assert "output_format" not in events["options"][-1]
    assert events["options"][-1]["tools"] == {"type": "preset", "preset": "claude_code"}
    assert events["options"][-1]["permission_mode"] == "acceptEdits"
    assert events["options"][-1]["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
    }


def test_claude_adapter_allows_tools_setting_to_override_builtin_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    adapter = ClaudeAdapter(
        model="claude",
        model_settings={"tools": ["web_search"]},
    )

    adapter.produce_sync("template", output_schema={"answer": str})

    assert events["options"][0]["tools"] == ["web_search"]


def test_claude_adapter_allows_permission_mode_to_override_builtin_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    adapter = ClaudeAdapter(
        model="claude",
        model_settings={"permission_mode": "plan"},
    )

    adapter.produce_sync("template", output_schema={"answer": str})

    assert events["options"][0]["permission_mode"] == "plan"


def test_claude_adapter_preserves_supported_string_formats() -> None:
    properties = claude_adapter_module._output_json_schema(FormattedClaudeOutput)["properties"]

    assert properties["created_at"]["format"] == "date-time"
    assert properties["elapsed"]["format"] == "duration"
    assert properties["day"]["format"] == "date"
    assert properties["clock"]["format"] == "time"
    assert properties["matcher"]["format"] == "regex"
    assert properties["email"]["format"] == "email"
    assert properties["site"]["format"] == "uri"
    assert properties["file"]["format"] == "uri"


def test_claude_adapter_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_claude_sdk(monkeypatch)
    adapter = ClaudeAdapter(model="claude")

    with pytest.raises(ValueError, match="Output schema cannot be empty"):
        ClaudeAdapter.type_builder({})
    with pytest.raises(ValueError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("template"))

    sdk = sys.modules["claude_agent_sdk"]

    async def no_result_query(*, prompt: str, options: Any) -> Any:
        if False:
            yield prompt, options

    setattr(sdk, "query", no_result_query)
    with pytest.raises(RuntimeError, match="did not return a result"):
        asyncio.run(adapter.produce("template", output_schema={"answer": str}))
    with pytest.raises(RuntimeError, match="did not return a result"):
        asyncio.run(adapter.invoke(prompt="template"))

    class ErrorMessage:
        is_error = True
        result = "bad run"
        structured_output = None

    async def error_query(*, prompt: str, options: Any) -> Any:
        yield ErrorMessage()

    setattr(sdk, "query", error_query)
    with pytest.raises(RuntimeError, match="bad run"):
        asyncio.run(adapter.produce("template", output_schema={"answer": str}))
    with pytest.raises(RuntimeError, match="bad run"):
        asyncio.run(adapter.invoke(prompt="template"))

    class MissingStructuredOutput:
        is_error = False
        structured_output = None

    async def missing_output_query(*, prompt: str, options: Any) -> Any:
        yield MissingStructuredOutput()

    setattr(sdk, "query", missing_output_query)
    with pytest.raises(ValueError, match="structured_output"):
        asyncio.run(adapter.produce("template", output_schema={"answer": str}))


def test_claude_adapter_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
    monkeypatch.setattr(claude_adapter_module, "find_spec", lambda name: None)

    with pytest.raises(ModuleNotFoundError, match="claude-agent-sdk"):
        asyncio.run(ClaudeAdapter(model="claude").produce("template", output_schema={"x": str}))


def test_claude_sdk_import_uses_find_spec_path(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "claude_agent_sdk":
            return sentinel
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
    monkeypatch.setattr(claude_adapter_module, "find_spec", lambda name: object())
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert claude_adapter_module._claude_sdk() is sentinel


def test_claude_helpers_cover_mcp_and_serialization() -> None:
    configs = adapters_module._claude_external_mcp_servers(
        AgentProfile(
            mcp_servers=(
                McpServerSpec(command="plain"),
                McpServerSpec(command="vsh", args=("run",), env={"A": "B"}),
                McpServerSpec(transport="sse", url="http://localhost/plain-sse"),
                McpServerSpec(transport="sse", url="http://localhost/sse", headers={"S": "V"}),
                McpServerSpec(transport="http", url="http://localhost/plain-mcp"),
                McpServerSpec(transport="http", url="http://localhost/mcp", headers={"H": "V"}),
            )
        )
    )

    assert configs == {
        "kedi_mcp_0": {
            "type": "stdio",
            "command": "plain",
            "args": [],
        },
        "kedi_mcp_1": {
            "type": "stdio",
            "command": "vsh",
            "args": ["run"],
            "env": {"A": "B"},
        },
        "kedi_mcp_2": {"type": "sse", "url": "http://localhost/plain-sse"},
        "kedi_mcp_3": {
            "type": "sse",
            "url": "http://localhost/sse",
            "headers": {"S": "V"},
        },
        "kedi_mcp_4": {"type": "http", "url": "http://localhost/plain-mcp"},
        "kedi_mcp_5": {
            "type": "http",
            "url": "http://localhost/mcp",
            "headers": {"H": "V"},
        },
    }

    assert adapters_module._output_json_schema(str) == {"type": "string"}
    assert adapters_module._output_json_schema(int) == {"type": "integer"}
    assert adapters_module._output_json_schema(float) == {"type": "number"}
    assert adapters_module._output_json_schema(bool) == {"type": "boolean"}
    assert adapters_module._output_json_schema(object) == {"type": "object"}

    @dataclass
    class Payload:
        value: str

    assert claude_adapter_module._tool_result_text("already text") == "already text"
    assert claude_adapter_module._tool_result_text(DirectAnswer(answer="ok")) == '{"answer": "ok"}'
    assert claude_adapter_module._tool_result_text(Payload("ok")) == '{"value": "ok"}'
    assert claude_adapter_module._tool_result_text([DirectAnswer(answer="ok")]) == (
        '[{"answer": "ok"}]'
    )
    assert claude_adapter_module._claude_text_response(SimpleNamespace(result="text")) == "text"
    assert claude_adapter_module._claude_text_response(SimpleNamespace(content="content")) == (
        "content"
    )
    assert claude_adapter_module._claude_text_response(
        SimpleNamespace(result={"text": "dict"})
    ) == ("dict")
    with pytest.raises(ValueError, match="text response"):
        claude_adapter_module._claude_text_response(SimpleNamespace(result=None))
    with pytest.raises(ValueError, match="text response"):
        claude_adapter_module._claude_text_response(SimpleNamespace(result={"text": 1}))

    with pytest.raises(ValueError, match="stdio transport requires a command"):
        adapters_module._claude_external_mcp_servers(
            AgentProfile(mcp_servers=(McpServerSpec(transport="stdio"),))
        )
    with pytest.raises(ValueError, match="sse transport requires a url"):
        adapters_module._claude_external_mcp_servers(
            AgentProfile(mcp_servers=(McpServerSpec(transport="sse"),))
        )
    with pytest.raises(ValueError, match="http transport requires a url"):
        adapters_module._claude_external_mcp_servers(
            AgentProfile(mcp_servers=(McpServerSpec(transport="http"),))
        )
    with pytest.raises(ValueError, match="Unsupported MCP transport"):
        adapters_module._claude_external_mcp_servers(
            AgentProfile(mcp_servers=(McpServerSpec(transport=cast(Any, "websocket")),))
        )


def test_claude_adapter_is_available_from_runtime_builders() -> None:
    from kedi import api as api_module
    from kedi import app as app_module

    runtime_adapter = app_module._build_runtime_adapter("claude", "claude-sonnet-4")
    assert isinstance(runtime_adapter, ClaudeAdapter)

    lazy = api_module._lazy_adapter("claude", "claude-sonnet-4", {"max_turns": 2})
    assert lazy.default_model() == "claude-sonnet-4"


def test_pydantic_adapter_profile_stack_is_context_local() -> None:
    adapter = PydanticAdapter(model=None)
    seen: list[dict[str, Any]] = []
    entered_b = threading.Event()
    finish = threading.Event()

    def run_a() -> None:
        with adapter.profile_override(AgentProfile(settings={"temperature": 0.1})):
            entered_b.wait(1)
            seen.append(dict(adapter.active_profile.settings))
            finish.set()

    def run_b() -> None:
        with adapter.profile_override(AgentProfile(settings={"temperature": 0.9})):
            entered_b.set()
            finish.wait(1)

    thread_a = threading.Thread(target=run_a)
    thread_b = threading.Thread(target=run_b)
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    assert seen == [{"temperature": 0.1}]


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
            (
                {
                    "mcpServers": {
                        "kedi_mcp_1": {
                            "url": "http://localhost/sse",
                            "transport": "sse",
                        }
                    }
                },
            ),
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


def test_pydantic_adapter_filters_active_profile_model_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(
        adapter,
        AgentProfile(
            settings={
                "temperature": 0.2,
                "parallel_tool_calls": False,
                "num_retries": 2,
            }
        ),
    )
    seen_kwargs: list[dict[str, Any]] = []

    async def fake_run(template: str, **kwargs: Any) -> SimpleNamespace:
        assert template == "template"
        seen_kwargs.append(kwargs)
        return SimpleNamespace(output=DirectOutput(value="ok"))

    monkeypatch.setattr(adapter, "run", fake_run)

    assert asyncio.run(adapter.produce("template", output_type=DirectOutput)) == DirectOutput(
        value="ok"
    )
    assert seen_kwargs == [
        {
            "output_type": DirectOutput,
            "model_settings": {
                "temperature": 0.2,
                "parallel_tool_calls": False,
            },
        }
    ]

    seen_kwargs.clear()
    assert asyncio.run(
        adapter.produce(
            "template",
            output_type=DirectOutput,
            model_settings={"seed": 1},
        )
    ) == DirectOutput(value="ok")
    assert seen_kwargs == [
        {
            "output_type": DirectOutput,
            "model_settings": {
                "seed": 1,
                "temperature": 0.2,
                "parallel_tool_calls": False,
            },
        }
    ]


def test_pydantic_adapter_uses_profile_system_as_default_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = PydanticAdapter("test")
    captured: dict[str, Any] = {}

    async def fake_run(*_: Any, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(output="ok")

    monkeypatch.setattr(adapter, "run", fake_run)
    with adapter.profile_override(AgentProfile(system="Be exact.")):
        result = asyncio.run(adapter.produce("template", output_type=str))

    assert result == "ok"
    assert captured["instructions"] == "Be exact."

    captured.clear()
    with adapter.profile_override(AgentProfile(system="Raw system.")):
        assert adapter.invoke_sync(prompt="raw") == "ok"
    assert captured["instructions"] == "Raw system."


def test_pydantic_adapter_invoke_uses_fresh_agent_inside_kedi_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = PydanticAdapter("openai:gpt-4o-mini")
    created_agents: list[dict[str, Any]] = []

    class FreshAgent:
        def __init__(self, model: Any = None, **kwargs: Any) -> None:
            created_agents.append({"model": model, "kwargs": kwargs})

        async def run(self, prompt: str, **kwargs: Any) -> Any:
            assert prompt == "raw"
            assert kwargs["instructions"] == "Profile system."
            assert kwargs["model"] == "openai-chat:gpt-4o"
            return SimpleNamespace(output="fresh raw")

    monkeypatch.setattr(pydantic_adapter_module, "Agent", FreshAgent)

    with adapter.profile_override(AgentProfile(model="openai:gpt-4o", system="Profile system.")):
        with kedi_tool_call_context():
            assert adapter.invoke_sync(prompt="raw") == "fresh raw"

    assert created_agents == [{"model": "openai-chat:gpt-4o-mini", "kwargs": {}}]


def test_pydantic_adapter_profile_override_rejects_broken_model_profile() -> None:
    adapter = object.__new__(PydanticAdapter)
    _set_profile_stack(adapter)
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
    _set_profile_stack(adapter, AgentProfile(mcp_servers=(McpServerSpec(command="vsh"),)))

    assert adapter._toolsets_for_run() == ["mcp-toolset"]


def test_dspy_adapter_configures_lm_and_builds_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lm_calls: list[dict[str, Any]] = []
    configure_calls: list[dict[str, Any]] = []

    class FakeTSLM:
        """Stand-in for ThinkingStrippingLM that records init args."""

        def __init__(self, **kwargs: Any) -> None:
            lm_calls.append(kwargs)

    class FakeSettings:
        def configure(self, **kwargs: Any) -> None:
            configure_calls.append(kwargs)

    monkeypatch.setattr(dspy_adapter_module, "ThinkingStrippingLM", FakeTSLM)
    monkeypatch.setattr(dspy_adapter_module.dspy, "settings", FakeSettings())

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
    assert configure_calls and isinstance(configure_calls[0]["lm"], FakeTSLM)

    signature = DSPyAdapter.type_builder({"answer": int})
    assert issubclass(signature, dspy_adapter_module.dspy.Signature)

    with pytest.raises(ValueError, match="Output schema must be provided"):
        DSPyAdapter.type_builder({})


def test_dspy_adapter_filters_lm_settings() -> None:
    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = "openai/gpt-4o-mini"
    adapter._lm_provider = None
    adapter._lm_max_tokens = 4000
    adapter._lm_temperature = 1.0
    adapter._lm_extra_kwargs = {}
    adapter._lm_settings = {"max_tokens": 64}

    provider, kwargs = adapter._lm_kwargs(
        {
            "temperature": 0.2,
            "num_retries": 2,
            "parallel_tool_calls": False,
        }
    )

    assert provider is None
    assert kwargs == {
        "max_tokens": 64,
        "temperature": 0.2,
        "cache": False,
        "num_retries": 2,
    }


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

        def __call__(self, **kwargs: Any) -> dict[str, Any] | SimpleNamespace:
            if kwargs["user_prompt"] == "raw prompt":
                return SimpleNamespace(response="raw response")
            return {"instructions": self.signature.instructions, **kwargs}

    def fake_type_builder(output_schema: dict[str, type], **_: Any) -> FakeSignature:
        assert output_schema == {"answer": str}
        return FakeSignature()

    monkeypatch.setattr(dspy_adapter_module.dspy, "Predict", FakePredict)
    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(fake_type_builder))
    monkeypatch.setattr(DSPyAdapter, "invoke_signature", staticmethod(FakeSignature))

    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = None
    _set_profile_stack(adapter)
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
    assert adapter.invoke_sync(prompt="raw prompt", user_prompt="ignored") == "raw response"
    assert asyncio.run(adapter.invoke(prompt="raw prompt")) == "raw response"

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

    monkeypatch.setattr(dspy_adapter_module.dspy, "Tool", FakeTool)
    monkeypatch.setattr(dspy_adapter_module.dspy, "ReAct", FakeReAct)

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(
        adapter,
        AgentProfile(
            mcp_servers=(McpServerSpec(command="vsh", args=("run", "--mcp"), env={"A": "B"}),)
        ),
    )

    result = asyncio.run(adapter._produce_with_mcp(object(), ["local"], "prompt", {"extra": 1}))

    assert result == {
        "tools": ["local", "mcp:remote_search"],
        "extra": 1,
        "user_prompt": "prompt",
    }
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
        if template == "raw":
            return {"response": "raw mcp"}
        return {"sig": sig, "tools": tools, "template": template, "kwargs": kwargs}

    class FakeSignature:
        def with_instructions(self, instructions: str | None) -> str:
            assert instructions is None
            return "sig"

    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(lambda _: FakeSignature()))
    monkeypatch.setattr(DSPyAdapter, "_produce_with_mcp", fake_produce_with_mcp)

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    adapter._lm_model = None
    _set_profile_stack(adapter, AgentProfile(mcp_servers=(McpServerSpec(command="vsh"),)))

    result = adapter.produce_sync("prompt", {"answer": str}, user_prompt="ignored", extra=1)
    raw = asyncio.run(adapter.invoke(prompt="raw"))

    assert result == {
        "sig": "sig",
        "tools": [],
        "template": "prompt",
        "kwargs": {"extra": 1},
    }
    assert raw == "raw mcp"


def test_dspy_adapter_mcp_rejects_non_stdio() -> None:
    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(
        adapter,
        AgentProfile(mcp_servers=(McpServerSpec(transport="http", url="http://localhost"),)),
    )

    with pytest.raises(NotImplementedError, match="stdio"):
        asyncio.run(adapter._produce_with_mcp(object(), [], "prompt", {}))


def test_dspy_adapter_mcp_rejects_missing_stdio_command() -> None:
    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(adapter, AgentProfile(mcp_servers=(McpServerSpec(),)))

    with pytest.raises(ValueError, match="stdio transport requires a command"):
        asyncio.run(adapter._produce_with_mcp(object(), [], "prompt", {}))


def test_dspy_adapter_profile_override_rejects_broken_model_profile() -> None:
    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = "openai/gpt-4o-mini"
    _set_profile_stack(adapter)

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
