from __future__ import annotations

import asyncio
import builtins
import inspect
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter.adapters import DSPyAdapter, LangChainAdapter, PydanticAdapter, make_repr
from kedi.agent_adapter.tool_override import ToolScope, ToolSpec
from kedi.agent_adapter.utils import (
    dspy_code_act,
    dspy_code_act_bridge_source,
    dspy_code_act_tools,
    dspy_sync_tool_function,
    installed_package_version,
    kedi_procedure_tool_spec,
    monty_compat_capabilities,
    monty_compat_prompt_context,
    python_callable_tool_spec,
)
from kedi.agent_adapter import utils as adapter_utils
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.mcp import McpServerSpec
from kedi.lang import compile_program, parse_program


class DirectOutput(BaseModel):
    value: str


class DirectAnswer(BaseModel):
    answer: str


class FakeLangChainTool:
    name = "search_docs"
    description = "Search docs."
    args_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query."}},
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        return f"found:{args['query']}"


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


def test_pydantic_adapter_builds_types_and_produces_with_schema_or_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized_models: list[Any] = []
    initialized_kwargs: list[dict[str, Any]] = []

    def fake_agent_init(self: object, model: object = None, **kwargs: Any) -> None:
        initialized_models.append(model)
        initialized_kwargs.append(kwargs)

    monkeypatch.setattr(adapters_module.Agent, "__init__", fake_agent_init)

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
    PydanticAdapter(model="openai/gpt-4o-mini", codemode=True)
    PydanticAdapter(
        model="openai/gpt-4o-mini",
        codemode=True,
        capabilities=("existing",),
    )

    async def fake_run(template: str, *, output_type: type[Any], **kwargs: Any) -> Any:
        assert template == "template"
        assert kwargs == {"metadata": "ok"}
        if output_type is DirectOutput:
            return SimpleNamespace(output=DirectOutput(value="direct"))
        return SimpleNamespace(output=output_type(answer=3))

    monkeypatch.setattr(adapter, "run", fake_run)

    assert initialized_models[0] == "openai-chat:gpt-4o-mini"
    assert isinstance(initialized_models[1], TestModel)
    assert initialized_kwargs[2]["model_settings"] == {"temperature": 0.1}
    assert "model_settings" not in initialized_kwargs[3]
    assert initialized_kwargs[4]["capabilities"][0].__class__.__name__ == "CodeMode"
    assert initialized_kwargs[5]["capabilities"][0] == "existing"
    assert initialized_kwargs[5]["capabilities"][1].__class__.__name__ == "CodeMode"

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
            response_format = self.kwargs["response_format"]
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

    chat_models.init_chat_model = fake_init_chat_model  # type: ignore[attr-defined]
    agents.create_agent = fake_create_agent  # type: ignore[attr-defined]
    langchain_core_tools.StructuredTool = FakeStructuredTool  # type: ignore[attr-defined]
    langchain_mcp_client.MultiServerMCPClient = FakeMCPClient  # type: ignore[attr-defined]

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
        model_settings={"temperature": 0.1, "parallel_tool_calls": False, "num_retries": 2},
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

    assert result.answer == "ok"
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


def test_langchain_adapter_codemode_wraps_tools_as_run_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_langchain_modules(monkeypatch)
    monkeypatch.setattr(
        adapters_module,
        "monty_compat_prompt_context",
        lambda: "## Monty Sandbox — Supported Python Features\nfake context",
    )
    adapter = LangChainAdapter(model={"fake": "model"}, codemode=True)
    adapter._tool_scope.register_tool(FakeLangChainTool())

    assert asyncio.run(adapter.produce("template", output_type=DirectAnswer)).answer == "ok"

    tools = events["agents"][0]["tools"]
    assert [tool.name for tool in tools] == ["search_schema", "run_code"]
    assert events["tools"][0]["name"] == "search_schema"
    assert events["tools"][1]["name"] == "run_code"
    assert events["tools"][1]["args_schema"]["properties"]["code"]["type"] == "string"
    assert "def search_docs" not in events["tools"][1]["description"]
    assert "query: str" not in events["tools"][1]["description"]
    assert tools[1].metadata == {"code_arg_name": "code", "code_arg_language": "python"}

    schema = events["tools"][0]["func"](names=["search_docs"])
    assert schema["available_tools"] == ["search_docs"]
    assert schema["matches"][0]["signature"].startswith("def search_docs")
    assert schema["matches"][0]["parameters_json_schema"]["properties"]["query"]["type"] == "string"
    assert events["tools"][1]["func"]("search_docs(query='kedi')") == "found:kedi"
    assert events["tools"][1]["func"]("search_docs('positional')") == "found:positional"
    error = events["tools"][1]["func"]("import bs4")
    assert error["error"].startswith("MontyRuntimeError:")
    assert "search_schema" in error["hint"]
    assert "Available run_code function names: search_docs." in events["agents"][0]["system_prompt"]
    assert "Use `search_schema`" in events["agents"][0]["system_prompt"]
    assert "do not assume third-party packages" in events["agents"][0]["system_prompt"]
    assert "## Monty Sandbox — Supported Python Features" in events["agents"][0]["system_prompt"]


def test_langchain_adapter_profile_codemode_wraps_tools_as_run_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_langchain_modules(monkeypatch)
    monkeypatch.setattr(
        adapters_module,
        "monty_compat_prompt_context",
        lambda: "## Monty Sandbox — Supported Python Features\nfake context",
    )
    adapter = LangChainAdapter(model={"fake": "model"})
    adapter._tool_scope.register_tool(FakeLangChainTool())

    with adapter.profile_override(AgentProfile(settings={"codemode": True})):
        profile_tools = asyncio.run(adapter._tools_for_run())
        assert asyncio.run(adapter.produce("template", output_type=DirectAnswer)).answer == "ok"

    tools = events["agents"][0]["tools"]
    assert [tool.name for tool in tools] == ["search_schema", "run_code"]
    assert [tool.name for tool in profile_tools] == ["search_schema", "run_code"]
    assert events["tools"][0]["func"](query="docs")["matches"][0]["name"] == "search_docs"
    assert events["tools"][1]["func"]("search_docs(query='profile')") == "found:profile"


def test_langchain_adapter_uses_output_type_and_sync_produce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_langchain_modules(monkeypatch)
    adapter = LangChainAdapter(model={"fake": "model"})

    assert adapter.produce_sync("template", output_type=DirectAnswer).answer == "ok"
    assert adapter.default_model() is None
    with adapter.tool_scope(("manual-tool",)):
        assert adapter.scoped_tools == ("manual-tool",)
        assert asyncio.run(adapter._tools_for_run()) == ["manual-tool"]
    assert adapter.scoped_tools == ()


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


def test_langchain_code_mode_helper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = adapters_module._LangChainCodeModeRunner()

    assert runner.run(code=" \n ", functions={}) == ""
    assert runner.run(code="def helper():\n    return 1", functions={}) == ""

    class FakeMonty:
        streams: list[tuple[str, str]] = []
        result: Any = None

        def __init__(self, code: str, inputs: list[str]) -> None:
            self.code = code
            self.inputs = inputs

        def run(
            self,
            *,
            inputs: dict[str, Any],
            external_functions: dict[str, Any],
            print_callback: Any,
        ) -> Any:
            assert inputs == external_functions
            for stream, value in self.streams:
                print_callback(stream, value)
            return self.result

    monkeypatch.setattr(adapters_module, "_monty_class", lambda: FakeMonty)

    FakeMonty.streams = [("stderr", "ignored"), ("stdout", "printed\n")]
    FakeMonty.result = 3
    assert runner.run(code="1 + 2", functions={}, restart=True) == {
        "output": "printed",
        "result": 3,
    }

    FakeMonty.result = None
    assert runner.run(code="print('x')", functions={}) == "printed"

    FakeMonty.streams = []
    assert runner.run(code="None", functions={}) == ""

    def plain_callable(**kwargs: Any) -> str:
        return f"plain:{kwargs['value']}"

    assert adapters_module._langchain_tool_name(plain_callable) == "plain_callable"
    assert adapters_module._langchain_tool_name(object()) is None
    assert adapters_module._langchain_code_mode_functions([object()]) == {}
    assert adapters_module._langchain_code_mode_tool_definitions([object()]) == {}
    assert (
        adapters_module._langchain_tool_callable(
            plain_callable,
            "plain_callable",
            ("value",),
        )("ok")
        == "plain:ok"
    )
    with pytest.raises(TypeError, match="only 1 schema parameters"):
        adapters_module._langchain_tool_callable(
            plain_callable,
            "plain_callable",
            ("value",),
        )("ok", "extra")
    with pytest.raises(TypeError, match="multiple values"):
        adapters_module._langchain_tool_callable(
            plain_callable,
            "plain_callable",
            ("value",),
        )("ok", value="duplicate")

    async def async_callable(**kwargs: Any) -> str:
        return f"async:{kwargs['value']}"

    assert (
        adapters_module._langchain_tool_callable(async_callable, "async_callable", ("value",))(
            value="ok"
        )
        == "async:ok"
    )
    with pytest.raises(TypeError, match="not callable"):
        adapters_module._langchain_tool_callable(object(), "bad")()

    class ArgsSchema:
        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            return {"type": "object", "properties": {"query": {"type": "string"}}}

    assert adapters_module._langchain_tool_parameters_schema(
        SimpleNamespace(args_schema=ArgsSchema), "lookup"
    )["properties"] == {"query": {"type": "string"}}
    assert adapters_module._langchain_tool_parameters_schema(
        SimpleNamespace(args={"query": {"type": "string"}}), "lookup"
    )["properties"] == {"query": {"type": "string"}}
    assert (
        adapters_module._langchain_tool_parameter_names(
            SimpleNamespace(args_schema={"type": "object", "properties": []})
        )
        == ()
    )
    assert (
        adapters_module._langchain_tool_parameters_schema(
            SimpleNamespace(args_schema={"type": "array"}), "lookup"
        )["type"]
        == "object"
    )
    assert adapters_module._langchain_tool_description(SimpleNamespace(description=1)) is None
    assert adapters_module._langchain_search_schema(
        adapters_module._langchain_code_mode_tool_definitions([FakeLangChainTool()]),
        query="missing",
    ) == {"available_tools": ["search_docs"], "matches": []}
    all_schemas = adapters_module._langchain_search_schema(
        adapters_module._langchain_code_mode_tool_definitions([FakeLangChainTool()])
    )
    assert all_schemas["matches"][0]["name"] == "search_docs"
    assert (
        adapters_module._langchain_code_mode_system_prompt(
            base="Base instructions.",
            tool_defs={},
            monty_context="## Monty fake context",
        )
        == "Base instructions.\n\n"
        "LangChain code mode is active. The only exposed tools are `search_schema` and "
        "`run_code`. Do not ask for or rely on native tool calls for the underlying "
        "application tools.\n"
        "Available run_code function names: none.\n"
        "Use `search_schema` when you need a function's arguments, JSON schema, or exact "
        "signature. After inspecting the schema, call `run_code` with Python that invokes "
        "the selected function directly by name. Prefer keyword arguments exactly as shown "
        "in the schema, e.g. `tool_name(arg_name=value)`, and do not guess argument names "
        "for tools whose schema you have not inspected. Keep code small and task-focused; "
        "do not assume third-party packages are installed in the sandbox; prefer available "
        "tools and Python standard-library parsing. Code passed to `run_code` must be "
        "Monty-compatible; follow the Monty capability context below before choosing "
        "imports, builtins, exceptions, or language constructs. Use `restart=True` only "
        "when previous sandbox state could interfere.\n\n"
        "## Monty fake context"
    )

    assert adapters_module._sanitize_python_identifier("1-search") == "_1_search"
    assert adapters_module._sanitize_python_identifier("class") == "class_"
    assert adapters_module._sanitize_python_identifier("") == "_"
    assert adapters_module._single_python_definition_name("def ok():\n    return 1") == "ok"
    assert adapters_module._single_python_definition_name("@") is None
    assert adapters_module._single_python_definition_name("x = 1\ny = 2") is None
    assert adapters_module._single_python_definition_name("1 + 2") is None
    assert "Monty fake context" not in adapters_module._langchain_code_mode_system_prompt(
        base=None,
        tool_defs={},
        monty_context="",
    )


def test_langchain_code_mode_monty_compat_prompt_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monty_compat_prompt_context.cache_clear()

    monkeypatch.setattr(adapter_utils.importlib_metadata, "version", lambda name: f"{name}:1.2.3")
    assert installed_package_version("pydantic-monty") == "pydantic-monty:1.2.3"

    def missing_version(_name: str) -> str:
        raise adapter_utils.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(adapter_utils.importlib_metadata, "version", missing_version)
    assert installed_package_version("pydantic-monty") == "unknown"

    calls: list[tuple[str, str, bool]] = []

    class FakeCacheModule:
        @staticmethod
        def get_capabilities(*, cache: str, version: str, only_released: bool) -> object:
            calls.append((cache, version, only_released))
            return object()

    monkeypatch.setattr(
        adapter_utils.importlib,
        "import_module",
        lambda name: FakeCacheModule,
    )
    assert monty_compat_capabilities("0.0.18") is not None
    assert calls == [("auto", "0.0.18", True)]

    class FakeCapabilities:
        def to_prompt_context(self) -> str:
            return "## Monty Sandbox — Supported Python Features\nsupports loops"

    capability_calls: list[str] = []
    monkeypatch.setattr(adapter_utils, "installed_package_version", lambda name: "0.0.18")
    monkeypatch.setattr(
        adapter_utils,
        "monty_compat_capabilities",
        lambda version: capability_calls.append(version) or FakeCapabilities(),
    )
    prompt = monty_compat_prompt_context()
    assert "Installed pydantic-monty version: 0.0.18." in prompt
    assert "`cache='auto'`" in prompt
    assert "supports loops" in prompt
    assert monty_compat_prompt_context() == prompt
    assert capability_calls == ["0.0.18"]

    monty_compat_prompt_context.cache_clear()

    def broken_capabilities(_version: str) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(adapter_utils, "monty_compat_capabilities", broken_capabilities)
    fallback = monty_compat_prompt_context()
    assert "Compatibility Context Unavailable" in fallback
    assert "RuntimeError: boom" in fallback

    monty_compat_prompt_context.cache_clear()

    class EmptyCapabilities:
        def to_prompt_context(self) -> str:
            return "  "

    monkeypatch.setattr(
        adapter_utils, "monty_compat_capabilities", lambda version: EmptyCapabilities()
    )
    assert "Compatibility Context Unavailable" in monty_compat_prompt_context()
    monty_compat_prompt_context.cache_clear()


def test_langchain_code_mode_dependency_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as blocked:
        _block_import(blocked, "langchain_core.tools")
        with pytest.raises(ModuleNotFoundError, match="code mode requires langchain-core"):
            adapters_module._langchain_code_mode_tools(
                [],
                adapters_module._LangChainCodeModeRunner(),
            )

    with monkeypatch.context() as blocked:
        _block_import(blocked, "pydantic_monty")
        with pytest.raises(RuntimeError, match="requires pydantic-monty"):
            adapters_module._monty_class()


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
                "codemode": True,
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
            "capabilities": [seen_kwargs[0]["capabilities"][0]],
        }
    ]
    assert seen_kwargs[0]["capabilities"][0].__class__.__name__ == "CodeMode"

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
            "capabilities": [seen_kwargs[0]["capabilities"][0]],
        }
    ]
    assert seen_kwargs[0]["capabilities"][0].__class__.__name__ == "CodeMode"


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

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"instructions": self.signature.instructions, **kwargs}

    def fake_type_builder(output_schema: dict[str, type], **_: Any) -> FakeSignature:
        assert output_schema == {"answer": str}
        return FakeSignature()

    monkeypatch.setattr(adapters_module.dspy, "Predict", FakePredict)
    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(fake_type_builder))

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

    with pytest.raises(ValueError, match="Output spec must be provided"):
        asyncio.run(adapter.produce("prompt", {}))


def test_dspy_adapter_codemode_uses_codeact_with_bridge_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: dict[str, Any] = {}

    class FakeSignature:
        def __init__(self) -> None:
            self.instructions: str | None = None

        def with_instructions(self, instructions: str | None) -> "FakeSignature":
            self.instructions = instructions
            return self

    class FakeCodeAct:
        def __init__(self, signature: FakeSignature, tools: list[Any], interpreter: Any) -> None:
            events["signature"] = signature
            events["tools"] = tools
            events["interpreter"] = interpreter

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            return {"kind": "codeact", **kwargs}

    monkeypatch.setattr(adapters_module.dspy, "CodeAct", FakeCodeAct)
    monkeypatch.setattr(DSPyAdapter, "type_builder", staticmethod(lambda _: FakeSignature()))
    monkeypatch.setattr(
        adapter_utils,
        "monty_compat_prompt_context",
        lambda: "## Monty Sandbox — Supported Python Features\nfake context",
    )

    adapter = object.__new__(DSPyAdapter)
    adapter._tool_scope = ToolScope()
    adapter._lm_model = None
    adapter._codemode = False
    _set_profile_stack(adapter, AgentProfile(settings={"codemode": True}))

    def lookup(query: str) -> str:
        """Lookup docs."""
        return query.upper()

    adapter.register_tool(python_callable_tool_spec("lookup", lookup))
    result = adapter.produce_sync("prompt", {"answer": str}, instructions="use code", extra=1)

    assert result == {"kind": "codeact", "extra": 1, "user_prompt": "prompt"}
    assert "use code" in events["signature"].instructions
    assert "DSPy CodeAct is active" in events["signature"].instructions
    assert "## Monty Sandbox — Supported Python Features" in events["signature"].instructions
    bridge_tool = events["tools"][0]
    assert bridge_tool.name == "lookup"
    assert bridge_tool.desc == "Lookup docs."
    assert bridge_tool.args == {"query": {"type": "string"}}
    bridge_source = inspect.getsource(bridge_tool.func)
    assert "def lookup(**kwargs):" in bridge_source
    assert bridge_source.strip() in events["interpreter"].skip_sources
    assert events["interpreter"].tools["lookup"] is adapter.scoped_tools[0].func


def test_dspy_code_act_leaves_signature_unchanged_without_monty_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: dict[str, Any] = {}

    class FakeSignature:
        def with_instructions(self, _instructions: str) -> "FakeSignature":
            raise AssertionError("signature should not be rewritten without Monty context")

    class FakeCodeAct:
        def __init__(self, signature: FakeSignature, tools: list[Any], interpreter: Any) -> None:
            events["signature"] = signature
            events["tools"] = tools
            events["interpreter"] = interpreter

    monkeypatch.setattr(adapter_utils, "monty_compat_prompt_context", lambda: "")
    monkeypatch.setattr(adapter_utils.dspy, "CodeAct", FakeCodeAct)

    signature = FakeSignature()

    dspy_code_act(signature, [])

    assert events["signature"] is signature
    assert events["tools"] == []


def test_dspy_code_act_tool_helpers_sanitize_and_preserve_host_tools() -> None:
    def lookup(query: str) -> str:
        return query

    tool = SimpleNamespace(
        name="1-search",
        desc=None,
        args={"query": {"type": "string"}},
        arg_types={"query": str},
        func=lookup,
    )
    keyword_tool = SimpleNamespace(
        name="class",
        desc=None,
        args={},
        arg_types={},
        func=lambda: "keyword",
    )

    bridge_tools, host_tools, skip_sources = dspy_code_act_tools([tool, keyword_tool])

    assert bridge_tools[0].name == "_1_search"
    assert bridge_tools[1].name == "class_"
    assert host_tools["_1_search"](query="ok") == "ok"
    assert host_tools["class_"]() == "keyword"
    assert skip_sources == {
        dspy_code_act_bridge_source("_1_search").strip(),
        dspy_code_act_bridge_source("class_").strip(),
    }
    assert "def _1_search(**kwargs):" in inspect.getsource(bridge_tools[0].func)


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

    assert result == {
        "sig": "sig",
        "tools": [],
        "template": "prompt",
        "kwargs": {"extra": 1},
    }


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
