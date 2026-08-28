from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import subprocess
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from re import Pattern
from types import ModuleType, SimpleNamespace
from typing import Annotated, Any, cast
from uuid import uuid4

import pytest
from pydantic import BaseModel, EmailStr, FileUrl, HttpUrl
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, ToolApproved, ToolDenied
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset

from kedi.agent_adapter import ACPAdapter as ExportedACPAdapter
from kedi.agent_adapter import AdapterCapabilities, WebGPUAdapter
from kedi.agent_adapter import ClaudeAdapter as ExportedClaudeAdapter
from kedi.agent_adapter import DSPyAdapter as ExportedDSPyAdapter
from kedi.agent_adapter import LangChainAdapter as ExportedLangChainAdapter
from kedi.agent_adapter import PydanticAdapter as ExportedPydanticAdapter
from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter._schema import _portable_output_json_schema, output_field_metadata
from kedi.agent_adapter.adapters import (
    ACPAdapter,
    ClaudeAdapter,
    CodexAdapter,
    DSPyAdapter,
    LangChainAdapter,
    PydanticAdapter,
    make_repr,
)
from kedi.agent_adapter.adapters import claude as claude_adapter_module
from kedi.agent_adapter.adapters import dspy as dspy_adapter_module
from kedi.agent_adapter.adapters import langchain as langchain_adapter_module
from kedi.agent_adapter.adapters import pydantic as pydantic_adapter_module
from kedi.agent_adapter.approval import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.codemode import CodeModeExecutionError
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.events import (
    AgentMessageEvent,
    AgentRunStateEvent,
    observe_agent_events,
)
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations, report_invocation
from kedi.agent_adapter.subagents import (
    SubagentConversationContext,
    SubagentExecutionRequest,
)
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_adapter.tool_override import (
    SupportsToolRegistration,
    ToolScope,
    ToolSpec,
    approval_request_for_tool,
    arguments_target_secret_file,
    artifact_admission_scope,
)
from kedi.agent_adapter.utils import (
    dspy_sync_tool_function,
    kedi_procedure_tool_spec,
    kedi_tool_call_context,
    python_callable_tool_spec,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.compaction import CompactionSettings, CompactionSettingsError
from kedi.agent_profile.mcp import McpServerSpec
from kedi.artifacts import ArtifactManager, ArtifactPolicy
from kedi.errors import KediExecutionError
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


def test_artifact_capabilities_require_structured_tool_adapters() -> None:
    artifact_adapters = (
        PydanticAdapter,
        LangChainAdapter,
        DSPyAdapter,
        ClaudeAdapter,
        CodexAdapter,
        WebGPUAdapter,
    )

    for adapter in artifact_adapters:
        capabilities = adapter.capabilities
        assert capabilities.supports("artifacts")
        assert capabilities.supports("structured_output")
        assert capabilities.supports("tool_registration")
        assert isinstance(adapter, type)
        assert issubclass(adapter, SupportsToolRegistration)

    assert PydanticAdapter.capabilities.supports("native_artifacts")
    assert LangChainAdapter.capabilities.supports("native_artifacts")

    with pytest.raises(ValueError, match="structured output and tool registration"):
        AdapterCapabilities(supports_artifacts=True)
    with pytest.raises(ValueError, match="requires artifact support"):
        AdapterCapabilities(supports_native_artifacts=True)


def test_pydantic_native_telemetry_ownership_follows_operation_and_instrumentation() -> None:
    enabled = SimpleNamespace(instrument=True)
    disabled = SimpleNamespace(instrument=False)

    for operation in ("run_agent", "chat", "call_tool"):
        assert PydanticAdapter.native_telemetry_enabled(enabled, operation)
        assert not PydanticAdapter.native_telemetry_enabled(disabled, operation)
    assert not PydanticAdapter.native_telemetry_enabled(enabled, "delegate_task")


def _set_profile_stack(adapter: object, *profiles: AgentProfile) -> None:
    setattr(
        adapter,
        "_profile_stack_var",
        ContextVar(f"test_profile_stack_{id(adapter)}", default=tuple(profiles)),
    )
    setattr(
        adapter,
        "_subagent_conversation_var",
        ContextVar(f"test_subagent_conversation_{id(adapter)}", default=None),
    )
    setattr(
        adapter,
        "_run_tool_specs_var",
        ContextVar(f"test_run_tool_specs_{id(adapter)}", default=None),
    )
    setattr(adapter, "approval_resolution", "kedi")


@pytest.mark.parametrize(
    ("adapter", "context_var_name"),
    [
        (PydanticAdapter(model=TestModel()), "_subagent_conversation_var"),
        (ClaudeAdapter(model="claude"), "_conversation_var"),
    ],
)
def test_stateful_adapters_bind_and_reset_conversation_scope(
    adapter: Any,
    context_var_name: str,
) -> None:
    conversation = AdapterConversationContext()
    context_var = getattr(adapter, context_var_name)

    assert context_var.get() is None
    with adapter.conversation_scope(conversation):
        assert context_var.get() is conversation
    assert context_var.get() is None


def test_make_repr_uses_class_name_for_standard_class_repr() -> None:
    class Custom:
        pass

    assert make_repr(Custom) == "Custom"


def test_output_field_metadata_uses_first_annotated_string_as_description() -> None:
    assert output_field_metadata(int) == (int, None)
    assert output_field_metadata(Annotated[str, "Documented answer", "ignored"]) == (
        str,
        "Documented answer",
    )
    assert output_field_metadata(Annotated[str, 1]) == (str, None)


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
    assert ClaudeAdapter.capabilities.supports("subagents")
    assert ClaudeAdapter.capabilities.supports("background_subagents")
    assert CodexAdapter.capabilities.supports("subagents")
    assert CodexAdapter.capabilities.supports("background_subagents")
    assert LangChainAdapter.capabilities.supports("subagents")
    assert LangChainAdapter.capabilities.supports("background_subagents")
    assert DSPyAdapter.capabilities.supports("subagents")
    assert not DSPyAdapter.capabilities.supports("background_subagents")
    assert LangChainAdapter.capabilities.supports("native_approvals")
    assert LangChainAdapter.capabilities.supports("native_approval_handler")
    for adapter in (PydanticAdapter, LangChainAdapter, ClaudeAdapter, CodexAdapter, ACPAdapter):
        assert adapter.capabilities.supports("stream_events")
    for adapter in (PydanticAdapter, LangChainAdapter, ClaudeAdapter, CodexAdapter):
        assert adapter.capabilities.supports("codemode")
    assert not DSPyAdapter.capabilities.supports("codemode")
    assert not DSPyAdapter.capabilities.supports("stream_events")
    assert not WebGPUAdapter.capabilities.supports("stream_events")


@pytest.mark.parametrize(
    "adapter_type",
    [
        PydanticAdapter,
        LangChainAdapter,
        DSPyAdapter,
        ClaudeAdapter,
        CodexAdapter,
        WebGPUAdapter,
    ],
)
def test_structured_output_adapters_accept_schema_and_prebuilt_type(
    adapter_type: type[Any],
) -> None:
    for method_name in ("produce", "produce_sync"):
        parameters = inspect.signature(getattr(adapter_type, method_name)).parameters
        assert parameters["output_schema"].default is None
        assert parameters["output_type"].default is None


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
        supports_background_subagents=True,
    )

    assert capabilities.supports("structured_output")
    assert capabilities.supports("tool_registration")
    assert capabilities.supports("mcp")
    assert capabilities.supports("profile_override")
    assert capabilities.supports("model_override")
    assert capabilities.supports("effort")
    assert capabilities.supports("settings")
    assert capabilities.supports("codemode")
    assert capabilities.supports("background_subagents")


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
    monkeypatch.setattr(adapter, "_tool_specs_for_run", lambda _kwargs: ())

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


@pytest.mark.parametrize(
    "builder",
    (
        PydanticAdapter.type_builder,
        LangChainAdapter.type_builder,
        ClaudeAdapter.type_builder,
        CodexAdapter.type_builder,
    ),
)
def test_adapter_output_models_use_portable_fixed_tuple_schema(builder: Any) -> None:
    model = builder({"pair": tuple[int, int]})
    pair_schema = model.model_json_schema()["properties"]["pair"]

    assert "prefixItems" not in pair_schema
    assert pair_schema["items"] == {"type": "integer"}
    assert pair_schema["minItems"] == pair_schema["maxItems"] == 2
    assert model.model_validate({"pair": [3, 5]}).pair == (3, 5)

    mixed_schema = builder({"pair": tuple[int, str]}).model_json_schema()["properties"]["pair"]
    assert mixed_schema["items"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}

    empty_schema = builder({"pair": tuple[()]}).model_json_schema()["properties"]["pair"]
    assert "prefixItems" not in empty_schema
    assert empty_schema["minItems"] == empty_schema["maxItems"] == 0


def test_portable_output_schema_handles_empty_and_predefined_tuple_items() -> None:
    schema = {
        "empty": {"prefixItems": []},
        "predefined": {
            "prefixItems": [{"type": "integer"}],
            "items": {"type": "number"},
        },
    }

    _portable_output_json_schema(schema)

    assert schema["empty"] == {"items": {}}
    assert schema["predefined"] == {"items": {"type": "number"}}


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
            assert set(kwargs) == {"capabilities", "instructions", "usage"}
            assert any(
                type(capability).__name__ == "_KediArtifactCapability"
                for capability in kwargs["capabilities"]
            )
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

        async def get_tools(self) -> list[Any]:
            return events.get("mcp_tools", ["mcp-tool"])

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
    middleware = ModuleType("langchain.agents.middleware")
    langchain_core = ModuleType("langchain_core")
    langchain_core_tools = ModuleType("langchain_core.tools")
    messages = ModuleType("langchain.messages")
    langchain_mcp_adapters = ModuleType("langchain_mcp_adapters")
    langchain_mcp_client = ModuleType("langchain_mcp_adapters.client")

    setattr(chat_models, "init_chat_model", fake_init_chat_model)
    setattr(agents, "create_agent", fake_create_agent)
    setattr(middleware, "AgentMiddleware", type("AgentMiddleware", (), {}))
    setattr(
        messages,
        "ToolMessage",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    setattr(langchain_core_tools, "StructuredTool", FakeStructuredTool)
    setattr(langchain_mcp_client, "MultiServerMCPClient", FakeMCPClient)

    monkeypatch.setitem(sys.modules, "langchain", langchain)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", chat_models)
    monkeypatch.setitem(sys.modules, "langchain.agents", agents)
    monkeypatch.setitem(sys.modules, "langchain.agents.middleware", middleware)
    monkeypatch.setitem(sys.modules, "langchain.messages", messages)
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


def test_langchain_codemode_projects_and_executes_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_langchain_modules(monkeypatch)
    calls: list[int] = []

    def double(*, value: int) -> int:
        calls.append(value)
        return value * 2

    adapter = LangChainAdapter(model={"fake": "model"}, codemode=True)
    adapter.register_tool(
        ToolSpec(
            fn=double,
            name="double",
            description="Double one integer.",
            json_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        )
    )

    async def exercise_projection() -> None:
        with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
            async with adapter._projected_tools_for_run() as projection:
                tools = {
                    tool.name: tool
                    for tool in langchain_adapter_module._langchain_projection_tools(projection)
                }
                assert set(tools) == {"search_tools", "get_tool_schema", "execute_code"}
                assert tools["search_tools"].func()["tool_names"] == ["double"]
                tools["get_tool_schema"].func(tool_names=["double"])
                result = await tools["execute_code"].coroutine(code="await double(value=4)")
                assert result == 8

    asyncio.run(exercise_projection())
    assert calls == [4]

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        assert adapter.invoke_sync(prompt="Use CodeMode.") == "plain langchain text"
    exposed = {tool.name for tool in events["agents"][-1]["tools"]}
    assert exposed == {"search_tools", "get_tool_schema", "execute_code"}
    assert "CodeMode" in events["agents"][-1]["system_prompt"]
    assert events["agents"][-1]["middleware"] == []


def test_langchain_codemode_materializes_profile_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_langchain_modules(monkeypatch)
    calls: list[dict[str, Any]] = []

    class FakeMCPTool:
        name = "mcp_lookup"
        description = "Look up MCP data."
        args_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        metadata = {"source": "mcp"}

        async def ainvoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
            calls.append(arguments)
            return {"found": arguments["query"]}

    events["mcp_tools"] = [FakeMCPTool()]
    adapter = LangChainAdapter(model={"fake": "model"}, codemode=True)
    profile = AgentProfile(
        approval=ApprovalPolicy.allow(),
        mcp_servers=(McpServerSpec(command="fake-mcp"),),
    )

    async def run() -> None:
        with adapter.profile_override(profile):
            async with adapter._projected_tools_for_run() as projection:
                tools = {
                    tool.name: tool
                    for tool in langchain_adapter_module._langchain_projection_tools(projection)
                }
                assert tools["search_tools"].func()["tool_names"] == ["mcp_lookup"]
                tools["get_tool_schema"].func(tool_names=["mcp_lookup"])
                result = await tools["execute_code"].coroutine(
                    code="await mcp_lookup(query='kedi')"
                )
                assert result == {"found": "kedi"}

    asyncio.run(run())
    assert calls == [{"query": "kedi"}]


def test_langchain_codemode_returns_sandbox_errors_to_the_model() -> None:
    async def fail_code(*, code: str, restart: bool = False) -> None:
        del code, restart
        raise CodeModeExecutionError("mapping methods are unavailable")

    tool = langchain_adapter_module._langchain_tool_from_spec(
        ToolSpec(
            fn=fail_code,
            name="execute_code",
            description="Execute CodeMode code.",
            json_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "restart": {"type": "boolean", "default": False},
                },
                "required": ["code"],
            },
            metadata={"kind": "codemode_control"},
        )
    )

    result = asyncio.run(tool.ainvoke({"code": "mapping.get('key')"}))

    assert result == "mapping methods are unavailable"

    def fail_sync(*, code: str, restart: bool = False) -> None:
        del code, restart
        raise CodeModeExecutionError("sync mapping failure")

    sync_tool = langchain_adapter_module._langchain_tool_from_spec(
        ToolSpec(
            fn=fail_sync,
            name="execute_code",
            description="Execute CodeMode code.",
            json_schema={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            metadata={"kind": "codemode_control"},
        )
    )
    assert sync_tool.invoke({"code": "mapping.get('key')"}) == "sync mapping failure"


def test_langchain_native_codemode_tool_variants() -> None:
    class Arguments(BaseModel):
        value: int

    with pytest.raises(TypeError, match="non-empty name"):
        langchain_adapter_module._langchain_tool_spec_from_native(
            SimpleNamespace(name="", args_schema=None)
        )

    immediate = langchain_adapter_module._langchain_tool_spec_from_native(
        SimpleNamespace(
            name="immediate",
            description=None,
            args_schema=Arguments,
            metadata=None,
            ainvoke=lambda arguments: arguments["value"] + 1,
        )
    )
    assert immediate.json_schema == Arguments.model_json_schema()
    assert immediate.description == "immediate"
    assert immediate.metadata == {"source": "langchain"}
    assert asyncio.run(immediate.fn(value=2)) == 3

    async def delayed() -> int:
        return 5

    awaiting = langchain_adapter_module._langchain_tool_spec_from_native(
        SimpleNamespace(
            name="awaiting",
            description="Await a value.",
            args_schema=None,
            metadata={"source": "native"},
            invoke=lambda _arguments: delayed(),
        )
    )
    assert awaiting.json_schema == {"type": "object", "properties": {}}
    assert asyncio.run(awaiting.fn()) == 5

    synchronous = langchain_adapter_module._langchain_tool_spec_from_native(
        SimpleNamespace(
            name="synchronous",
            description="Return a value.",
            args_schema={"type": "object"},
            metadata={},
            invoke=lambda _arguments: 7,
        )
    )
    assert asyncio.run(synchronous.fn()) == 7

    missing = langchain_adapter_module._langchain_tool_spec_from_native(
        SimpleNamespace(name="missing", description="Missing.", args_schema=None)
    )
    with pytest.raises(TypeError, match="not invokable"):
        asyncio.run(missing.fn())


def test_langchain_adapter_emits_completed_model_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAgent:
        async def ainvoke(
            self,
            _payload: dict[str, Any],
            *,
            config: dict[str, Any],
        ) -> dict[str, Any]:
            callback = config["callbacks"][0]
            await callback.on_llm_end(
                SimpleNamespace(
                    generations=[
                        [
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="Checking the repository.",
                                    tool_calls=[{"name": "read_file"}],
                                )
                            )
                        ]
                    ]
                ),
                run_id=uuid4(),
            )
            await callback.on_tool_start({}, "README.md", run_id=uuid4())
            await callback.on_llm_end(
                SimpleNamespace(
                    generations=[
                        [SimpleNamespace(message=SimpleNamespace(content="Done.", tool_calls=[]))]
                    ]
                ),
                run_id=uuid4(),
            )
            return {"messages": [SimpleNamespace(content="Done.")]}

    adapter = LangChainAdapter(model={"fake": "model"})
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    streamed: list[AgentMessageEvent | AgentRunStateEvent] = []

    with observe_agent_events(streamed.append):
        assert asyncio.run(adapter.invoke(prompt="inspect")) == "Done."

    messages = [event for event in streamed if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [
        ("commentary", "Checking the repository."),
        ("final", "Done."),
    ]


def test_langchain_stream_handler_accepts_supported_content_shapes() -> None:
    streamed: list[AgentMessageEvent | AgentRunStateEvent] = []

    async def scenario() -> None:
        with observe_agent_events(streamed.append):
            with pydantic_adapter_module.agent_event_run("langchain") as resolver:
                assert resolver is not None
                callback = langchain_adapter_module._langchain_event_handler(resolver)
                await callback.on_llm_end(
                    SimpleNamespace(generations=object()),
                    run_id="invalid",
                )
                await callback.on_llm_end(
                    SimpleNamespace(
                        generations=[
                            object(),
                            [
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=[
                                            "First ",
                                            {"text": "message."},
                                            {"value": "ignored"},
                                            object(),
                                        ],
                                        tool_calls=[],
                                    )
                                ),
                                SimpleNamespace(
                                    message=SimpleNamespace(content=object(), tool_calls=[{}])
                                ),
                            ],
                        ]
                    ),
                    run_id="shapes",
                )

    asyncio.run(scenario())
    messages = [event for event in streamed if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [
        ("commentary", "First message.")
    ]


def test_langchain_completed_message_text_handles_scalar_and_empty_content() -> None:
    assert (
        langchain_adapter_module._langchain_completed_message_text(
            SimpleNamespace(content=" done ")
        )
        == "done"
    )
    assert (
        langchain_adapter_module._langchain_completed_message_text(SimpleNamespace(content=None))
        == ""
    )


def test_langchain_subagent_execution_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_langchain_modules(monkeypatch)

    class FakeToolStrategy:
        def __init__(self, schema: dict[str, Any]) -> None:
            self.schema = schema

    structured_output = ModuleType("langchain.agents.structured_output")
    setattr(structured_output, "ToolStrategy", FakeToolStrategy)
    monkeypatch.setitem(sys.modules, "langchain.agents.structured_output", structured_output)

    seen: dict[str, Any] = {}

    class FakeAgent:
        async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            seen["payload"] = payload
            return {
                "structured_response": {
                    "task_summary": "done",
                    "final_result": {"value": 9},
                },
                "messages": [
                    SimpleNamespace(
                        usage_metadata={"input_tokens": 2, "outputTokens": 3},
                        tool_calls=[{"name": "lookup"}],
                    )
                ],
            }

    def create_agent(**kwargs: Any) -> FakeAgent:
        seen["agent_kwargs"] = kwargs
        return FakeAgent()

    adapter = LangChainAdapter(model={"fake": "model"})
    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    request = SubagentExecutionRequest(
        profile=AgentProfile(system="child system", approval=ApprovalPolicy.allow()),
        tools=(
            ToolSpec(
                fn=lambda: "ok",
                name="lookup",
                metadata={"kind": "subagent"},
            ),
        ),
        instructions="child system",
        approval_ceiling=(ApprovalPolicy.allow(),),
        usage_limits=SubagentUsageLimits(request_limit=4),
    )
    usages: list[InvocationUsage] = []

    with adapter.subagent_scope(request):
        with observe_invocations(usages.append):
            result = asyncio.run(
                adapter.produce_subagent(
                    template="delegate",
                    output_schema={"type": "object"},
                    instructions=request.instructions,
                    **adapter.subagent_run_kwargs(request.usage_limits),
                )
            )

    assert result["final_result"] == {"value": 9}
    assert seen["payload"]["recursion_limit"] == 9
    assert isinstance(seen["agent_kwargs"]["response_format"], FakeToolStrategy)
    assert seen["agent_kwargs"]["system_prompt"] == "child system"
    assert adapter.telemetry_model_name() is None
    assert usages == [
        InvocationUsage(
            requests=1,
            tool_calls=1,
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
        )
    ]

    class BadAgent:
        async def ainvoke(self, _payload: dict[str, Any]) -> dict[str, Any]:
            return {"structured_response": "bad"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: BadAgent())
    with pytest.raises(TypeError, match="must be an object"):
        asyncio.run(adapter.produce_subagent(template="bad", output_schema={"type": "object"}))

    with pytest.raises(ValueError, match="Unknown subagent failure policy"):
        LangChainAdapter(
            model={"fake": "model"},
            subagent_failure_policy=cast(Any, "invalid"),
        )


def test_langchain_subagent_tool_errors_and_usage_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_langchain_modules(monkeypatch)

    class ToolException(Exception):
        pass

    setattr(sys.modules["langchain_core.tools"], "ToolException", ToolException)

    def fail() -> str:
        raise KediExecutionError("child failed")

    subagent = ToolSpec(fn=fail, name="delegate_task", metadata={"kind": "subagent"})
    with pytest.raises(ToolException, match="child failed"):
        langchain_adapter_module._langchain_sync_tool_function(subagent)()

    async def async_fail() -> str:
        raise KediExecutionError("async child failed")

    async_subagent = ToolSpec(
        fn=async_fail,
        name="delegate_async",
        metadata={"kind": "subagent"},
    )
    with pytest.raises(ToolException, match="async child failed"):
        asyncio.run(langchain_adapter_module._langchain_async_tool_function(async_subagent)())

    ordinary = ToolSpec(fn=fail, name="ordinary")
    with pytest.raises(KediExecutionError, match="child failed"):
        langchain_adapter_module._langchain_sync_tool_function(ordinary)()
    with pytest.raises(KediExecutionError, match="async child failed"):
        asyncio.run(
            langchain_adapter_module._langchain_async_tool_function(
                ToolSpec(fn=async_fail, name="ordinary_async")
            )()
        )

    assert langchain_adapter_module._langchain_invocation_usage("plain") == InvocationUsage(
        requests=1
    )
    assert langchain_adapter_module._langchain_invocation_usage({"messages": "bad"}) == (
        InvocationUsage(requests=1)
    )
    assert langchain_adapter_module._langchain_invocation_usage(
        {
            "messages": [
                SimpleNamespace(usage_metadata=None, tool_calls="invalid"),
                SimpleNamespace(
                    usage_metadata={"totalTokens": 9},
                    tool_calls=[],
                ),
            ]
        }
    ) == InvocationUsage(requests=1, total_tokens=9)
    assert langchain_adapter_module._langchain_invocation_usage(
        {
            "messages": [
                SimpleNamespace(
                    usage_metadata={
                        "input_tokens": 120,
                        "input_token_details": {
                            "cache_read": 90,
                            "cache_creation": 15,
                        },
                        "output_tokens": 8,
                    },
                    tool_calls=[],
                )
            ]
        }
    ) == InvocationUsage(
        requests=1,
        input_tokens=120,
        cache_read_tokens=90,
        cache_write_tokens=15,
        output_tokens=8,
        total_tokens=128,
    )

    historical = SimpleNamespace(
        type="ai",
        content="old answer",
        usage_metadata={"input_tokens": 100, "output_tokens": 10},
        tool_calls=[{"name": "old_tool"}],
    )
    current = SimpleNamespace(
        type="ai",
        content="new answer",
        usage_metadata={
            "input_tokens": 125,
            "input_token_details": {"cache_read": 100},
            "output_tokens": 5,
        },
        tool_calls=[],
    )
    assert langchain_adapter_module._langchain_invocation_usage(
        {
            "messages": [
                SimpleNamespace(type="human", content="prompt"),
                SimpleNamespace(
                    type="ai",
                    content="old answer",
                    usage_metadata=historical.usage_metadata,
                    tool_calls=historical.tool_calls,
                ),
                current,
            ]
        },
        input_messages=[{"role": "user", "content": "prompt"}, historical],
    ) == InvocationUsage(
        requests=1,
        input_tokens=125,
        cache_read_tokens=100,
        output_tokens=5,
        total_tokens=130,
    )

    assert langchain_adapter_module._langchain_current_messages(
        [current], [historical, current]
    ) == [current]
    assert langchain_adapter_module._langchain_current_messages(
        [current], [{"role": "assistant", "content": "different"}]
    ) == [current]
    assert langchain_adapter_module._same_langchain_message(current, current)
    assert langchain_adapter_module._same_langchain_message(
        {"role": "assistant", "content": "same"},
        {"role": "assistant", "content": "same"},
    )
    assert langchain_adapter_module._usage_int({"value": False}, "value") == 0


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
    assert asyncio.run(events["tools"][0]["coroutine"]("docs")) == "async:docs"


def test_langchain_approval_middleware_applies_allow_deny_and_edits() -> None:
    from langgraph.prebuilt.tool_node import ToolCallRequest

    calls: list[dict[str, Any]] = []

    def update_count(*, count: int) -> int:
        return count

    async def handler(request: Any) -> str:
        calls.append(dict(request.tool_call))
        return "tool result"

    def request_for(*, name: str, arguments: dict[str, Any]) -> Any:
        return ToolCallRequest(
            tool_call={"name": name, "args": arguments, "id": "call-1", "type": "tool_call"},
            tool=None,
            state={},
            runtime=cast(Any, None),
        )

    tool = python_callable_tool_spec("update_count", update_count)
    edit_policy = ApprovalPolicy.dynamic(lambda _request: ApprovalDecision.edit({"count": 3}))
    edit_middleware = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile(approval=edit_policy),
        tools=(tool,),
    )
    assert (
        asyncio.run(
            edit_middleware.awrap_tool_call(
                request_for(name="update_count", arguments={"count": 1}), handler
            )
        )
        == "tool result"
    )
    assert calls == [
        {
            "name": "update_count",
            "args": {"count": 3},
            "id": "call-1",
            "type": "tool_call",
        }
    ]

    deny_middleware = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile(approval=ApprovalPolicy.deny()),
        tools=(tool,),
    )
    denied = asyncio.run(
        deny_middleware.awrap_tool_call(
            request_for(name="update_count", arguments={"count": 1}), handler
        )
    )
    assert denied.status == "error"
    assert "denied" in denied.content
    assert len(calls) == 1

    external = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile.empty(),
        tools=(),
    )
    denied_external = asyncio.run(
        external.awrap_tool_call(
            request_for(name="external_mcp_tool", arguments={"path": ".env"}), handler
        )
    )
    assert denied_external.status == "error"
    assert len(calls) == 1

    allowed_external = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile(approval=ApprovalPolicy.allow()),
        tools=(),
    )
    assert (
        asyncio.run(
            allowed_external.awrap_tool_call(
                request_for(name="external_mcp_tool", arguments={"path": "notes.txt"}),
                handler,
            )
        )
        == "tool result"
    )
    assert len(calls) == 2

    inherited_ceiling = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile.empty(),
        tools=(tool,),
        approval_ceiling=(ApprovalPolicy.allow(),),
    )
    assert (
        asyncio.run(
            inherited_ceiling.awrap_tool_call(
                request_for(name="update_count", arguments={"count": 4}),
                handler,
            )
        )
        == "tool result"
    )
    assert len(calls) == 3


def test_langchain_native_results_cross_artifact_boundary_once_and_keep_call_id() -> None:
    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest

    def request_for(name: str, call_id: str) -> Any:
        return ToolCallRequest(
            tool_call={"name": name, "args": {}, "id": call_id, "type": "tool_call"},
            tool=cast(Any, SimpleNamespace(metadata={"kedi_origin": "mcp"})),
            state={},
            runtime=cast(Any, None),
        )

    async def successful_handler(request: Any) -> ToolMessage:
        return ToolMessage(
            content="native-result" * 20,
            tool_call_id=request.tool_call["id"],
        )

    manager = ArtifactManager(session_id="langchain-native")
    policy = ArtifactPolicy(threshold_bytes=10)
    middleware = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile(approval=ApprovalPolicy.allow()),
        tools=(),
    )
    with artifact_admission_scope(manager=manager, policy=policy):
        result = asyncio.run(
            middleware.awrap_tool_call(
                request_for("native_tool", "native-call"),
                successful_handler,
            )
        )

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "native-call"
    assert json.loads(cast(str, result.content))["ref_id"] == "tool_call_result_1"
    assert manager.get_value("tool_call_result_1") == "native-result" * 20
    events = manager.history.snapshot()
    assert events[0].data["call_id"] == "native-call"
    assert events[-1].data["call_id"] == "native-call"

    async def failed_handler(request: Any) -> ToolMessage:
        return ToolMessage(
            content="failure" * 20,
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    with artifact_admission_scope(manager=manager, policy=policy):
        failed = asyncio.run(
            middleware.awrap_tool_call(
                request_for("native_tool", "failed-call"),
                failed_handler,
            )
        )
    assert failed.status == "error"
    assert len(manager.history.snapshot()) == len(events)

    wrapped_tool = ToolSpec(
        fn=lambda: "unused",
        name="wrapped_tool",
        metadata={"kedi_artifact_wrapped": True},
    )
    wrapped_middleware = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile(approval=ApprovalPolicy.allow()),
        tools=(wrapped_tool,),
    )
    with artifact_admission_scope(manager=manager, policy=policy):
        wrapped = asyncio.run(
            wrapped_middleware.awrap_tool_call(
                request_for("wrapped_tool", "wrapped-call"),
                successful_handler,
            )
        )
    assert wrapped.content == "native-result" * 20
    assert len(manager.history.snapshot()) == len(events)

    with artifact_admission_scope(manager=manager, policy=policy):
        raw = asyncio.run(
            langchain_adapter_module._langchain_admit_native_tool_result(
                "raw-result" * 20,
                tool_name="raw_tool",
                call_id="raw-call",
            )
        )
    assert raw["ref_id"] == "tool_call_result_2"
    assert manager.get_value("tool_call_result_2") == "raw-result" * 20


def test_langchain_command_results_preserve_graph_updates_and_tool_causality() -> None:
    from langchain_core.messages import ToolMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

    def request_for(call_id: str) -> Any:
        return ToolCallRequest(
            tool_call={
                "name": "native_tool",
                "args": {},
                "id": call_id,
                "type": "tool_call",
            },
            tool=cast(Any, SimpleNamespace(metadata={"kedi_origin": "mcp"})),
            state={},
            runtime=cast(Any, None),
        )

    async def successful_handler(request: Any) -> Command[Any]:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="command-result" * 20,
                        tool_call_id=request.tool_call["id"],
                    )
                ],
                "counter": 1,
            },
            goto="next-node",
            resume={"checkpoint": "keep"},
        )

    manager = ArtifactManager(session_id="langchain-command")
    policy = ArtifactPolicy(threshold_bytes=10)
    middleware = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile(approval=ApprovalPolicy.allow()),
        tools=(),
    )
    with artifact_admission_scope(manager=manager, policy=policy):
        result = asyncio.run(
            middleware.awrap_tool_call(request_for("command-call"), successful_handler)
        )

    assert isinstance(result, Command)
    assert result.goto == "next-node"
    assert result.resume == {"checkpoint": "keep"}
    assert result.update["counter"] == 1
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "command-call"
    assert json.loads(cast(str, message.content))["ref_id"] == "tool_call_result_1"
    assert manager.get_value("tool_call_result_1") == "command-result" * 20

    async def failed_handler(request: Any) -> Command[Any]:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="command-failure" * 20,
                        tool_call_id=request.tool_call["id"],
                        status="error",
                    )
                ]
            }
        )

    event_count = len(manager.history.snapshot())
    with artifact_admission_scope(manager=manager, policy=policy):
        failed = asyncio.run(
            middleware.awrap_tool_call(request_for("failed-command"), failed_handler)
        )
    failed_message = failed.update["messages"][0]
    assert failed_message.status == "error"
    assert len(manager.history.snapshot()) == event_count

    tuple_command = Command(
        update=(
            "preserved-metadata",
            ToolMessage(content="tuple-result" * 20, tool_call_id="tuple-call"),
        )
    )
    with artifact_admission_scope(manager=manager, policy=policy):
        tuple_result = asyncio.run(
            langchain_adapter_module._langchain_admit_native_tool_result(
                tuple_command,
                tool_name="native_tool",
                call_id="tuple-call",
            )
        )
    assert isinstance(tuple_result.update, tuple)
    assert tuple_result.update[0] == "preserved-metadata"
    assert json.loads(cast(str, tuple_result.update[1].content))["ref_id"] == ("tool_call_result_2")

    without_messages = Command(update={"counter": 2})
    with artifact_admission_scope(manager=manager, policy=policy):
        assert (
            asyncio.run(
                langchain_adapter_module._langchain_admit_native_tool_result(
                    without_messages,
                    tool_name="native_tool",
                    call_id="unused-call",
                )
            )
            is without_messages
        )

    unmatched = Command(
        update={"messages": [ToolMessage(content="small", tool_call_id="other-call")]}
    )
    with artifact_admission_scope(manager=manager, policy=policy):
        assert (
            asyncio.run(
                langchain_adapter_module._langchain_admit_native_tool_result(
                    unmatched,
                    tool_name="native_tool",
                    call_id="expected-call",
                )
            )
            is unmatched
        )

    expected_message = ToolMessage(content="expected", tool_call_id="expected-call")
    mixed_messages = Command(
        update={
            "messages": [
                ToolMessage(content="other", tool_call_id="other-call"),
                expected_message,
            ]
        }
    )
    assert (
        langchain_adapter_module._langchain_tool_message_for_call(
            mixed_messages,
            "expected-call",
        )
        is expected_message
    )
    assert (
        langchain_adapter_module._langchain_tool_message_for_call(
            unmatched,
            "expected-call",
        )
        is None
    )
    assert (
        langchain_adapter_module._langchain_tool_message_for_call(
            Command(update={"messages": None}),
            "expected-call",
        )
        is None
    )
    manager.close()


def test_langchain_raw_message_admission_preserves_model_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.messages import AIMessage

    admitted_values = iter(({"ref_id": "tool_call_result_1"}, "compact scalar"))

    async def admit(_result: Any, *, tool_name: str, call_id: str) -> Any:
        assert tool_name == "native_tool"
        assert call_id == "native-call"
        return next(admitted_values)

    monkeypatch.setattr(
        langchain_adapter_module,
        "admit_current_artifact_result",
        admit,
    )

    original = AIMessage(content="large native result")
    mapped = asyncio.run(
        langchain_adapter_module._langchain_admit_native_tool_result(
            original,
            tool_name="native_tool",
            call_id="native-call",
        )
    )
    scalar = asyncio.run(
        langchain_adapter_module._langchain_admit_native_tool_result(
            original,
            tool_name="native_tool",
            call_id="native-call",
        )
    )

    assert isinstance(mapped, AIMessage)
    assert json.loads(cast(str, mapped.content))["ref_id"] == "tool_call_result_1"
    assert isinstance(scalar, AIMessage)
    assert scalar.content == "compact scalar"


def test_langchain_native_approval_helpers_cover_validation_and_async_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_langchain_modules(monkeypatch)

    def lookup(query: str) -> str:
        return query

    spec = python_callable_tool_spec("lookup", lookup)
    adapter = LangChainAdapter(model={"fake": "model"})
    with adapter.tool_scope((spec,)):
        assert adapter.scoped_tool_specs == (spec,)
    assert events["tools"][0]["name"] == "lookup"

    class SyncCallableReturningAwaitable:
        def __call__(self, query: str):  # type: ignore[no-untyped-def]
            async def result() -> str:
                return f"async:{query}"

            return result()

    sync_wrapper = langchain_adapter_module._langchain_sync_tool_function(
        ToolSpec(fn=SyncCallableReturningAwaitable(), name="sync_wrapper")
    )
    assert sync_wrapper(query="docs") == "async:docs"

    async_wrapper = langchain_adapter_module._langchain_async_tool_function(
        ToolSpec(fn=lambda *, query: query, name="async_wrapper")
    )
    assert asyncio.run(async_wrapper(query="docs")) == "docs"

    middleware = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile.empty(), tools=()
    )
    malformed = asyncio.run(
        middleware.awrap_tool_call(
            SimpleNamespace(tool_call={"name": None}),
            lambda _request: pytest.fail("malformed calls must not reach the tool"),
        )
    )
    assert malformed.status == "error"

    def invalid_arguments(_arguments: Any) -> dict[str, Any]:
        raise ValueError("invalid arguments")

    failing_middleware = langchain_adapter_module._langchain_approval_middleware(
        profile=AgentProfile.empty(),
        tools=(ToolSpec(fn=lookup, name="lookup", argument_validator=invalid_arguments),),
    )
    rejected = asyncio.run(
        failing_middleware.awrap_tool_call(
            SimpleNamespace(tool_call={"name": "lookup", "args": {"query": "docs"}, "id": "call"}),
            lambda _request: pytest.fail("invalid calls must not reach the tool"),
        )
    )
    assert rejected.status == "error"
    assert "invalid arguments" in rejected.content

    assert arguments_target_secret_file([{"path": ".env"}, 1])
    assert not arguments_target_secret_file(1)

    monkeypatch.setitem(sys.modules, "langchain.agents.middleware", None)
    with pytest.raises(ModuleNotFoundError, match="approval support requires langchain"):
        langchain_adapter_module._langchain_approval_middleware(
            profile=AgentProfile.empty(), tools=()
        )


def test_claude_native_approval_helper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    class FallbackHookMatcher:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    with pytest.raises(RuntimeError, match="does not expose"):
        claude_adapter_module._claude_hook_matcher_class(SimpleNamespace())
    assert (
        claude_adapter_module._claude_hook_matcher_class(
            SimpleNamespace(HookMatcher=FallbackHookMatcher)
        )
        is FallbackHookMatcher
    )

    with pytest.raises(TypeError, match="mapping"):
        claude_adapter_module._with_kedi_approval_hook(
            SimpleNamespace(HookMatcher=FallbackHookMatcher),
            "invalid",
            profile=AgentProfile.empty(),
            tools=(),
        )
    with pytest.raises(TypeError, match="lists or tuples"):
        claude_adapter_module._with_kedi_approval_hook(
            SimpleNamespace(HookMatcher=FallbackHookMatcher),
            {"PreToolUse": "invalid"},
            profile=AgentProfile.empty(),
            tools=(),
        )

    preserved = claude_adapter_module._with_kedi_approval_hook(
        SimpleNamespace(HookMatcher=FallbackHookMatcher),
        {"PostToolUse": ("existing",)},
        profile=AgentProfile.empty(),
        tools=(),
    )
    assert preserved["PostToolUse"] == ["existing"]

    hook = claude_adapter_module._claude_pre_tool_use_hook(
        profile=AgentProfile.empty(),
        tools=(),
    )
    assert (
        asyncio.run(hook("invalid", None, None))["hookSpecificOutput"]["permissionDecision"]
        == "deny"
    )
    assert (
        asyncio.run(hook({"tool_name": 1, "tool_input": {}}, None, None))["hookSpecificOutput"][
            "permissionDecision"
        ]
        == "deny"
    )

    def lookup(query: str) -> str:
        return query

    def invalid_arguments(_arguments: Any) -> dict[str, Any]:
        raise ValueError("invalid arguments")

    invalid_hook = claude_adapter_module._claude_pre_tool_use_hook(
        profile=AgentProfile.empty(),
        tools=(ToolSpec(fn=lookup, name="lookup", argument_validator=invalid_arguments),),
    )
    rejected = asyncio.run(
        invalid_hook({"tool_name": "lookup", "tool_input": {"query": "docs"}}, None, None)
    )
    assert rejected["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "invalid arguments" in rejected["hookSpecificOutput"]["permissionDecisionReason"]
    assert arguments_target_secret_file([{"path": ".env.local"}, 1])
    assert not arguments_target_secret_file(1)


def test_pydantic_native_approval_helper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(adapter)

    combined = adapter._with_deferred_approval_capability(("existing",))
    assert combined[0] == "existing"
    with pytest.raises(TypeError, match="list or tuple"):
        adapter._with_deferred_approval_capability("invalid")

    class BadCall:
        tool_call_id = "bad-call"
        tool_name = "lookup"

        @staticmethod
        def args_as_dict() -> dict[str, Any]:
            raise ValueError("bad JSON")

    class Requests:
        approvals = [BadCall()]

        @staticmethod
        def build_results(*, approvals: dict[str, Any]) -> dict[str, Any]:
            return approvals

    denied = asyncio.run(adapter._handle_deferred_tool_calls(None, Requests()))
    assert isinstance(denied["bad-call"], ToolDenied)

    def sync_tool(*, query: str) -> str:
        return query

    sync_wrapper = pydantic_adapter_module._validated_tool_function(
        ToolSpec(fn=sync_tool, name="sync_tool")
    )
    with pytest.raises(TypeError, match="only accepts keyword"):
        sync_wrapper("docs")

    monkeypatch.setattr(pydantic_adapter_module, "_is_async_callable", lambda _fn: True)
    async_wrapper = pydantic_adapter_module._validated_tool_function(
        ToolSpec(fn=sync_tool, name="async_tool")
    )
    assert asyncio.run(async_wrapper(query="docs")) == "docs"
    with pytest.raises(TypeError, match="only accepts keyword"):
        asyncio.run(async_wrapper("docs"))

    async def async_tool(*, query: str) -> str:
        return f"async:{query}"

    monkeypatch.setattr(
        pydantic_adapter_module,
        "_is_async_callable",
        pydantic_adapter_module._is_async_callable,
    )
    actual_async_wrapper = pydantic_adapter_module._validated_tool_function(
        ToolSpec(fn=async_tool, name="actual_async_tool")
    )
    assert asyncio.run(actual_async_wrapper(query="docs")) == "async:docs"
    assert pydantic_adapter_module._validated_tool_arguments(
        ToolSpec(fn=sync_tool, name="plain"), {"query": "docs"}
    ) == {"query": "docs"}


def test_pydantic_artifact_code_errors_are_model_retries() -> None:
    from pydantic_ai import ModelRetry

    from kedi.artifacts import ArtifactCodeError

    def fail_artifact_code(*, code: str) -> None:
        raise ArtifactCodeError(f"invalid artifact code: {code}")

    wrapped = pydantic_adapter_module._validated_tool_function(
        ToolSpec(
            fn=fail_artifact_code,
            name="run_artifact_code",
            metadata={"kind": "artifact_management"},
        )
    )
    with pytest.raises(ModelRetry, match="documented helper signatures"):
        wrapped(code="bad()")

    def fail_unrelated(*, code: str) -> None:
        raise ValueError(f"unrelated failure: {code}")

    unrelated = pydantic_adapter_module._validated_tool_function(
        ToolSpec(
            fn=fail_unrelated,
            name="run_artifact_code",
            metadata={"kind": "artifact_management"},
        )
    )
    with pytest.raises(ValueError, match="unrelated failure"):
        unrelated(code="bad()")


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
    response_blocks = [
        {"type": "reasoning", "summary": []},
        {"type": "text", "text": "block response", "phase": "final_answer"},
    ]
    assert (
        adapters_module._langchain_text_response(
            {"messages": [SimpleNamespace(content=response_blocks)]}
        )
        == "block response"
    )
    assert (
        langchain_adapter_module._langchain_completed_message_text(
            SimpleNamespace(content=response_blocks)
        )
        == "block response"
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


def test_langchain_aws_dependency_is_loaded_only_for_enabled_bedrock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bedrock = type(
        "ChatBedrockConverse",
        (),
        {"__module__": "langchain_aws.chat_models"},
    )()

    with monkeypatch.context() as blocked:
        _block_import(blocked, "langchain_aws.middleware.prompt_caching")
        assert adapters_module.langchain._langchain_cache_middleware(bedrock, enabled=False) == []
        assert adapters_module.langchain._langchain_cache_middleware(object(), enabled=True) == []
        with pytest.raises(ModuleNotFoundError, match=r"kedi\[langchain-aws\]"):
            adapters_module.langchain._langchain_cache_middleware(bedrock, enabled=True)


def test_core_import_does_not_load_optional_backend_packages() -> None:
    code = """
import builtins
import sys

blocked = {
    "dspy",
    "groq",
    "langchain",
    "langchain_aws",
    "langchain_mcp_adapters",
    "langchain_openai",
    "langchain_openrouter",
    "openinference",
}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] in blocked:
        raise ModuleNotFoundError(f"blocked optional dependency: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import kedi
from kedi.agent_adapter import PydanticAdapter

assert PydanticAdapter.__name__ == "PydanticAdapter"
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_adapter_exports_reject_unknown_lazy_attribute() -> None:
    with pytest.raises(AttributeError, match="missing_adapter"):
        adapters_module.__getattr__("missing_adapter")


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

    class FakeHookMatcher:
        def __init__(
            self,
            *,
            matcher: str | None = None,
            hooks: list[Any] | None = None,
        ) -> None:
            self.matcher = matcher
            self.hooks = hooks or []

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
            self.session_id = "claude-session"

    @dataclass
    class FakeTextBlock:
        text: str

    @dataclass
    class FakeToolUseBlock:
        name: str
        input: dict[str, Any]

    @dataclass
    class FakeAssistantMessage:
        content: list[Any]

    @dataclass
    class FakeUserMessage:
        content: list[Any]

    @dataclass
    class FakeSystemMessage:
        subtype: str
        data: dict[str, Any]

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
    setattr(sdk, "HookMatcher", FakeHookMatcher)
    setattr(sdk, "ResultMessage", FakeResultMessage)
    setattr(sdk, "SystemMessage", FakeSystemMessage)
    setattr(sdk, "TextBlock", FakeTextBlock)
    setattr(sdk, "ToolUseBlock", FakeToolUseBlock)
    setattr(sdk, "AssistantMessage", FakeAssistantMessage)
    setattr(sdk, "UserMessage", FakeUserMessage)
    setattr(sdk, "query", fake_query)
    setattr(sdk, "tool", fake_tool)
    setattr(sdk, "create_sdk_mcp_server", fake_create_sdk_mcp_server)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    events["TextBlock"] = FakeTextBlock
    events["ToolUseBlock"] = FakeToolUseBlock
    events["AssistantMessage"] = FakeAssistantMessage
    events["UserMessage"] = FakeUserMessage
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
    assert len(options["hooks"]["PreToolUse"]) == 1

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


def test_claude_codemode_system_prompt_shapes() -> None:
    unchanged: dict[str, Any] = {"system_prompt": "existing"}
    claude_adapter_module._set_claude_system_prompt(
        unchanged,
        "ignored",
        codemode_enabled=False,
    )
    assert unchanged == {"system_prompt": "existing"}

    string_prompt: dict[str, Any] = {"system_prompt": "existing"}
    claude_adapter_module._set_claude_system_prompt(
        string_prompt,
        None,
        codemode_enabled=True,
    )
    assert string_prompt["system_prompt"].startswith("existing\n\n")
    assert "CodeMode" in string_prompt["system_prompt"]

    preset: dict[str, Any] = {
        "system_prompt": {"type": "preset", "preset": "claude_code", "append": "base"}
    }
    claude_adapter_module._set_claude_system_prompt(
        preset,
        None,
        codemode_enabled=True,
    )
    assert preset["system_prompt"]["append"].startswith("base\n\n")

    invalid_append: dict[str, Any] = {"system_prompt": {"append": 3}}
    with pytest.raises(TypeError, match="append must be a string"):
        claude_adapter_module._set_claude_system_prompt(
            invalid_append,
            None,
            codemode_enabled=True,
        )

    with pytest.raises(TypeError, match="string or preset mapping"):
        claude_adapter_module._set_claude_system_prompt(
            {"system_prompt": 3},
            None,
            codemode_enabled=True,
        )


def test_claude_codemode_projects_sdk_controls_and_executes_nested_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]
    calls: list[int] = []

    def double(*, value: int) -> int:
        calls.append(value)
        return value * 2

    async def query(*, prompt: str, options: Any) -> Any:
        events["prompts"].append({"prompt": prompt, "options": options})
        server = options.kwargs["mcp_servers"]["kedi"]
        tools = {tool.name: tool for tool in server["tools"]}
        await tools["search_tools"]({})
        await tools["get_tool_schema"]({"tool_names": ["double"]})
        events["codemode_result"] = await tools["execute_code"]({"code": "await double(value=6)"})
        yield sdk.ResultMessage(result="done")

    setattr(sdk, "query", query)
    adapter = ClaudeAdapter(model="claude", codemode=True)
    adapter.register_tool(
        ToolSpec(
            fn=double,
            name="double",
            json_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        )
    )

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        assert adapter.invoke_sync(prompt="Use CodeMode.") == "done"

    assert calls == [6]
    assert json.loads(events["codemode_result"]["content"][0]["text"]) == 12
    options = events["options"][-1]
    assert set(options["allowed_tools"]) == {
        "mcp__kedi__search_tools",
        "mcp__kedi__get_tool_schema",
        "mcp__kedi__execute_code",
    }
    assert "CodeMode" in options["system_prompt"]["append"]
    controls = {tool.name for tool in options["mcp_servers"]["kedi"]["tools"]}
    assert controls == {"search_tools", "get_tool_schema", "execute_code"}

    hook = options["hooks"]["PreToolUse"][-1].hooks[0]
    allowed = asyncio.run(
        hook(
            {
                "tool_name": "mcp__kedi__execute_code",
                "tool_input": {"code": "1 + 1"},
            },
            "call-1",
            None,
        )
    )
    assert allowed["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_claude_codemode_routes_profile_mcp_through_local_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]
    materialized: list[tuple[McpServerSpec, ...]] = []

    async def materialize(
        _stack: Any,
        specs: tuple[McpServerSpec, ...],
        *,
        adapter_shortname: str,
    ) -> tuple[ToolSpec, ...]:
        assert adapter_shortname == "claude"
        materialized.append(tuple(specs))

        async def lookup(*, query: str) -> dict[str, str]:
            return {"source": "mcp", "query": query}

        return (
            ToolSpec(
                fn=lookup,
                name="mcp_lookup",
                json_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        )

    async def query(*, prompt: str, options: Any) -> Any:
        events["prompts"].append({"prompt": prompt, "options": options})
        assert set(options.kwargs["mcp_servers"]) == {"kedi"}
        tools = {tool.name: tool for tool in options.kwargs["mcp_servers"]["kedi"]["tools"]}
        await tools["get_tool_schema"]({"tool_names": ["mcp_lookup"]})
        events["mcp_codemode_result"] = await tools["execute_code"](
            {"code": "await mcp_lookup(query='kedi')"}
        )
        yield sdk.ResultMessage(result="done")

    monkeypatch.setattr(claude_adapter_module, "materialize_mcp_tool_specs", materialize)
    setattr(sdk, "query", query)
    server = McpServerSpec(command="fake-mcp")
    adapter = ClaudeAdapter(model="claude", codemode=True)
    with adapter.profile_override(
        AgentProfile(
            approval=ApprovalPolicy.allow(),
            mcp_servers=(server,),
        )
    ):
        assert adapter.invoke_sync(prompt="Use MCP through CodeMode.") == "done"

    assert materialized == [(server,)]
    assert json.loads(events["mcp_codemode_result"]["content"][0]["text"]) == {
        "source": "mcp",
        "query": "kedi",
    }


def test_claude_adapter_emits_completed_assistant_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    events["prefix_message"] = events["AssistantMessage"](
        [
            events["TextBlock"]("Inspecting the requested file."),
            events["ToolUseBlock"]("Read", {"file_path": "README.md"}),
        ]
    )
    events["text_result"] = "Inspection complete."
    streamed: list[AgentMessageEvent | AgentRunStateEvent] = []

    with observe_agent_events(streamed.append):
        assert (
            asyncio.run(ClaudeAdapter(model="claude").invoke(prompt="inspect"))
            == "Inspection complete."
        )

    messages = [event for event in streamed if isinstance(event, AgentMessageEvent)]
    assert [(event.phase, event.content) for event in messages] == [
        ("commentary", "Inspecting the requested file."),
        ("final", "Inspection complete."),
    ]


def test_claude_stream_helpers_handle_message_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _install_fake_claude_sdk(monkeypatch)
    streamed: list[AgentMessageEvent | AgentRunStateEvent] = []

    with observe_agent_events(streamed.append):
        with pydantic_adapter_module.agent_event_run("claude") as resolver:
            assert resolver is not None
            assistant = sdk["AssistantMessage"](" direct ")
            assert (
                claude_adapter_module._observe_claude_message(
                    SimpleNamespace(**sdk), assistant, resolver, 0
                )
                == 1
            )
            tool_only = sdk["AssistantMessage"](
                [sdk["ToolUseBlock"]("Read", {"file_path": "README.md"})]
            )
            assert (
                claude_adapter_module._observe_claude_message(
                    SimpleNamespace(**sdk), tool_only, resolver, 1
                )
                == 1
            )
            user = sdk["UserMessage"](["tool result"])
            assert (
                claude_adapter_module._observe_claude_message(
                    SimpleNamespace(**sdk), user, resolver, 1
                )
                == 1
            )
            assert (
                claude_adapter_module._observe_claude_message(
                    SimpleNamespace(**sdk), object(), resolver, 1
                )
                == 1
            )

    fake_sdk = SimpleNamespace(TextBlock=None, ToolUseBlock=None)
    assert (
        claude_adapter_module._claude_assistant_text(fake_sdk, SimpleNamespace(content=object()))
        == ""
    )
    assert claude_adapter_module._claude_assistant_text(fake_sdk, SimpleNamespace(content=[])) == ""
    assert not claude_adapter_module._claude_has_tool_use(
        fake_sdk, SimpleNamespace(content=object())
    )
    assert not claude_adapter_module._claude_has_tool_use(fake_sdk, SimpleNamespace(content=[]))


def test_claude_subagent_continuation_resumes_captured_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    adapter = ClaudeAdapter(model="claude")
    conversation = SubagentConversationContext(conversation_id="conversation")

    with adapter.subagent_scope(
        SubagentExecutionRequest(
            profile=AgentProfile(name="worker"),
            tools=(),
            instructions=None,
            approval_ceiling=(),
            usage_limits=SubagentUsageLimits(),
            conversation=conversation,
        )
    ):
        assert asyncio.run(adapter.invoke(prompt="first")) == "plain claude text"
    assert conversation.result_state["claude_session_id"] == "claude-session"
    assert isinstance(conversation.result_state["claude_session_fingerprint"], str)

    follow_up = SubagentConversationContext(
        conversation_id="conversation",
        resume_state=dict(conversation.result_state),
    )
    with adapter.subagent_scope(
        SubagentExecutionRequest(
            profile=AgentProfile(name="worker"),
            tools=(),
            instructions=None,
            approval_ceiling=(),
            usage_limits=SubagentUsageLimits(),
            conversation=follow_up,
        )
    ):
        assert asyncio.run(adapter.invoke(prompt="second")) == "plain claude text"

    assert "resume" not in events["options"][0]
    assert events["options"][1]["resume"] == "claude-session"


def test_claude_history_restarts_native_session_when_profile_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    adapter = ClaudeAdapter(model="claude")
    first = AdapterConversationContext()

    with (
        adapter.profile_override(
            AgentProfile(
                history=True,
                system="First system",
                compaction=CompactionSettings(),
            )
        ),
        adapter.conversation_scope(first),
    ):
        assert asyncio.run(adapter.invoke(prompt="first")) == "plain claude text"

    second = AdapterConversationContext(resume_state=dict(first.result_state))
    with (
        adapter.profile_override(
            AgentProfile(
                history=True,
                system="First system",
                compaction=CompactionSettings(),
            )
        ),
        adapter.conversation_scope(second),
    ):
        assert asyncio.run(adapter.invoke(prompt="second")) == "plain claude text"

    changed = AdapterConversationContext(resume_state=dict(second.result_state))
    with (
        adapter.profile_override(AgentProfile(history=True, system="Changed system")),
        adapter.conversation_scope(changed),
    ):
        assert asyncio.run(adapter.invoke(prompt="changed")) == "plain claude text"

    assert "resume" not in events["options"][0]
    assert events["options"][1]["resume"] == "claude-session"
    assert "resume" not in events["options"][2]
    assert changed.cache_prefix_changed is True


def test_claude_native_compaction_rejects_unavailable_exact_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_claude_sdk(monkeypatch)
    adapter = ClaudeAdapter(model="claude")
    conversation = AdapterConversationContext()

    with (
        adapter.profile_override(
            AgentProfile(
                history=True,
                compaction=CompactionSettings(threshold=42_000),
            )
        ),
        adapter.conversation_scope(conversation),
        pytest.raises(CompactionSettingsError, match="does not expose an exact"),
    ):
        asyncio.run(adapter.invoke(prompt="hello"))


def test_claude_session_fingerprint_serializes_dataclasses_and_opaque_values() -> None:
    @dataclass(frozen=True)
    class CustomSetting:
        value: str

    assert claude_adapter_module._claude_fingerprint_default(CustomSetting("value")) == {
        "value": "value"
    }
    assert claude_adapter_module._claude_fingerprint_default(object()) == {
        "type": "builtins.object"
    }


def test_claude_adapter_fails_fast_on_non_retryable_api_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]
    events["prefix_message"] = sdk.SystemMessage(
        subtype="api_retry",
        data={"error_status": 401, "error": "authentication_failed"},
    )

    with pytest.raises(RuntimeError, match="check ANTHROPIC_API_KEY"):
        asyncio.run(ClaudeAdapter(model="claude").invoke(prompt="hello"))


def test_claude_adapter_allows_transient_api_retry_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]
    events["prefix_message"] = sdk.SystemMessage(
        subtype="api_retry",
        data={"error_status": 429, "error": "rate_limit"},
    )

    assert asyncio.run(ClaudeAdapter(model="claude").invoke(prompt="hello")) == "plain claude text"


def test_claude_adapter_fails_fast_on_non_authentication_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]
    events["prefix_message"] = sdk.SystemMessage(
        subtype="api_retry",
        data={"error_status": 400, "error": "invalid_request"},
    )

    with pytest.raises(RuntimeError, match="non-retryable HTTP 400: invalid_request"):
        asyncio.run(ClaudeAdapter(model="claude").invoke(prompt="hello"))


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


def test_claude_adapter_native_approval_hook_covers_builtin_mcp_and_kedi_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    calls: list[int] = []

    def save_count(*, count: int) -> str:
        calls.append(count)
        return f"saved:{count}"

    def edit_count(_request: object) -> ApprovalDecision:
        return ApprovalDecision.edit({"count": 2})

    adapter = ClaudeAdapter(model="claude")
    spec = python_callable_tool_spec("save_count", save_count)
    with (
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(edit_count))),
        adapter.tool_scope((spec,)),
    ):
        adapter.produce_sync("template", output_schema={"answer": str})

    options = events["options"][0]
    matcher = options["hooks"]["PreToolUse"][0]
    hook = matcher.hooks[0]
    result = asyncio.run(
        hook(
            {"tool_name": "mcp__kedi__save_count", "tool_input": {"count": 0}},
            "tool-call",
            {},
        )
    )
    assert result["hookSpecificOutput"] == {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": {"count": 2},
    }
    assert asyncio.run(events["tools"][0]({"count": 2})) == {
        "content": [{"type": "text", "text": "saved:2"}]
    }
    assert calls == [2]

    denied_adapter = ClaudeAdapter(model="claude")
    with denied_adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())):
        denied_adapter.invoke_sync(prompt="template")
    denied_hook = events["options"][1]["hooks"]["PreToolUse"][0].hooks[0]
    denied = asyncio.run(
        denied_hook(
            {"tool_name": "mcp__external__write", "tool_input": {"path": "notes.txt"}},
            "tool-call",
            {},
        )
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    read_only_adapter = ClaudeAdapter(model="claude")
    read_only_adapter.invoke_sync(prompt="template")
    read_only_hook = events["options"][2]["hooks"]["PreToolUse"][0].hooks[0]
    allowed_read = asyncio.run(
        read_only_hook(
            {"tool_name": "Read", "tool_input": {"file_path": "notes.txt"}},
            "tool-call",
            {},
        )
    )
    denied_secret_read = asyncio.run(
        read_only_hook(
            {"tool_name": "Read", "tool_input": {"file_path": ".env"}},
            "tool-call",
            {},
        )
    )
    assert allowed_read["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert denied_secret_read["hookSpecificOutput"]["permissionDecision"] == "deny"

    allowed_adapter = ClaudeAdapter(model="claude")
    with allowed_adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        allowed_adapter.invoke_sync(prompt="template")
    allowed_hook = events["options"][3]["hooks"]["PreToolUse"][0].hooks[0]
    allowed_external = asyncio.run(
        allowed_hook(
            {"tool_name": "mcp__external__write", "tool_input": {"path": "notes.txt"}},
            "tool-call",
            {},
        )
    )
    assert allowed_external["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert ClaudeAdapter.capabilities.supports("native_approvals")
    assert ClaudeAdapter.capabilities.supports("native_approval_handler")


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


def test_claude_subagent_execution_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    events["structured_output"] = {
        "task_summary": "done",
        "final_result": {"value": 7},
    }
    adapter = ClaudeAdapter(model="claude")
    tool = ToolSpec(fn=lambda query: query, name="lookup")
    request = SubagentExecutionRequest(
        profile=AgentProfile(
            model="child",
            system="child system",
            approval=ApprovalPolicy.allow(),
        ),
        tools=(tool,),
        instructions="child system",
        approval_ceiling=(ApprovalPolicy.allow(),),
        usage_limits=SubagentUsageLimits(),
    )
    usages: list[InvocationUsage] = []

    with adapter.subagent_scope(request):
        assert adapter.active_profile is request.profile
        assert adapter.scoped_tools == (tool,)
        with observe_invocations(usages.append):
            result = asyncio.run(
                adapter.produce_subagent(
                    template="delegate",
                    output_schema={
                        "type": "object",
                        "properties": {"task_summary": {"type": "string"}},
                    },
                    **adapter.subagent_run_kwargs(request.usage_limits),
                )
            )

    assert result["final_result"] == {"value": 7}
    assert adapter.active_profile == AgentProfile.empty()
    assert adapter.scoped_tools == ()
    assert adapter.telemetry_model_name() == "claude"
    assert events["options"][0]["model"] == "child"
    assert events["options"][0]["max_turns"] == request.usage_limits.request_limit
    assert events["options"][0]["output_format"]["schema"]["type"] == "object"
    assert usages == [InvocationUsage()]

    events["structured_output"] = "not-an-object"
    with pytest.raises(TypeError, match="must be an object"):
        asyncio.run(adapter.produce_subagent(template="bad", output_schema={"type": "object"}))

    with pytest.raises(ValueError, match="Unknown subagent failure policy"):
        ClaudeAdapter(model="claude", subagent_failure_policy=cast(Any, "invalid"))


def test_claude_subagent_tool_errors_and_usage_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]

    def fail() -> str:
        raise KediExecutionError("child details")

    subagent_tool = ToolSpec(
        fn=fail,
        name="delegate_task",
        metadata={"kind": "subagent"},
    )
    native = claude_adapter_module._claude_sdk_tool(sdk, subagent_tool)
    result = asyncio.run(native({}))
    assert result["is_error"] is True
    assert "child details" in result["content"][0]["text"]

    ordinary = claude_adapter_module._claude_sdk_tool(
        sdk,
        ToolSpec(fn=fail, name="ordinary"),
    )
    with pytest.raises(KediExecutionError, match="child details"):
        asyncio.run(ordinary({}))

    usage = claude_adapter_module._claude_invocation_usage(
        SimpleNamespace(
            usage={"inputTokens": 2, "output_tokens": 3, "totalTokens": 0},
            num_turns=4,
        )
    )
    assert usage == InvocationUsage(
        requests=4,
        input_tokens=2,
        output_tokens=3,
        total_tokens=5,
    )
    assert claude_adapter_module._claude_invocation_usage(
        SimpleNamespace(usage={"totalTokens": 9}, num_turns=1)
    ) == InvocationUsage(requests=1, total_tokens=9)
    assert claude_adapter_module._usage_int({"value": True}, "value") == 0

    inherited_hook = claude_adapter_module._claude_pre_tool_use_hook(
        profile=AgentProfile.empty(),
        tools=(),
        approval_ceiling=(ApprovalPolicy.allow(),),
    )
    inherited = asyncio.run(
        inherited_hook(
            {"tool_name": "Read", "tool_input": {"file_path": "notes.txt"}},
            None,
            None,
        )
    )
    assert inherited["hookSpecificOutput"]["permissionDecision"] == "allow"


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
    assert lazy.capabilities.supports("subagents")
    assert lazy.default_model() == "claude-sonnet-4"


@pytest.mark.parametrize(
    "adapter_factory",
    [
        pytest.param(lambda: PydanticAdapter(model=None), id="pydantic"),
        pytest.param(lambda: ClaudeAdapter(model=None), id="claude"),
        pytest.param(lambda: LangChainAdapter(model={"fake": "model"}), id="langchain"),
        pytest.param(
            lambda: CodexAdapter(command=("codex", "app-server")),
            id="codex",
        ),
    ],
)
def test_native_adapter_profile_stacks_are_nested_and_context_local(
    adapter_factory: Any,
) -> None:
    adapter = adapter_factory()
    seen: list[dict[str, Any]] = []
    entered_b = threading.Event()
    finish = threading.Event()

    outer = AgentProfile(settings={"temperature": 0.1})
    inner = AgentProfile(settings={"temperature": 0.2})
    with adapter.profile_override(outer):
        assert adapter.active_profile is outer
        with adapter.profile_override(inner):
            assert adapter.active_profile is inner
        assert adapter.active_profile is outer
    assert adapter.active_profile == AgentProfile.empty()

    def run_a() -> None:
        with adapter.profile_override(outer):
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


def test_pydantic_adapter_filters_active_profile_model_settings() -> None:
    seen_settings: list[dict[str, Any]] = []

    async def respond(_messages: list[object], info: AgentInfo) -> ModelResponse:
        seen_settings.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"value": "ok"})])

    adapter = PydanticAdapter(FunctionModel(respond))
    profile = AgentProfile(
        settings={
            "temperature": 0.2,
            "parallel_tool_calls": False,
            "num_retries": 2,
        }
    )
    with adapter.profile_override(profile):
        assert asyncio.run(adapter.produce("template", output_type=DirectOutput)) == DirectOutput(
            value="ok"
        )
        assert asyncio.run(
            adapter.produce(
                "template",
                output_type=DirectOutput,
                model_settings={"seed": 1},
            )
        ) == DirectOutput(value="ok")

    assert seen_settings == [
        {"temperature": 0.2, "parallel_tool_calls": False},
        {"seed": 1, "temperature": 0.2, "parallel_tool_calls": False},
    ]


def test_pydantic_adapter_uses_profile_system_as_default_instruction() -> None:
    captured: list[str | None] = []

    async def respond(messages: list[object], _info: AgentInfo) -> ModelResponse:
        request = next(message for message in messages if isinstance(message, ModelRequest))
        captured.append(request.instructions)
        return ModelResponse(parts=[TextPart("ok")])

    adapter = PydanticAdapter(FunctionModel(respond))
    with adapter.profile_override(AgentProfile(system="Be exact.")):
        result = asyncio.run(adapter.produce("template", output_type=str))

    assert result == "ok"
    assert captured == ["Be exact."]

    with adapter.profile_override(AgentProfile(system="Raw system.")):
        assert adapter.invoke_sync(prompt="raw") == "ok"
    assert captured == ["Be exact.", "Raw system."]


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
    class FakeMcpToolset:
        def approval_required(self) -> str:
            return "approval-required-mcp-toolset"

    monkeypatch.setattr(
        PydanticAdapter,
        "_mcp_toolsets_for_profile",
        lambda *_: [FakeMcpToolset()],
    )

    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(adapter, AgentProfile(mcp_servers=(McpServerSpec(command="vsh"),)))

    assert adapter._toolsets_for_run() == ["approval-required-mcp-toolset"]


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
    typed_signature = DSPyAdapter.output_type_builder(bool)
    assert typed_signature.model_fields["result"].annotation is bool

    documented = DSPyAdapter.type_builder({"answer": Annotated[str, "Documented answer"]})
    answer_field = documented.model_fields["answer"]
    assert answer_field.annotation is str
    assert answer_field.json_schema_extra["desc"] == "Documented answer"

    with pytest.raises(ValueError, match="Output schema must be provided"):
        DSPyAdapter.type_builder({})


@pytest.mark.parametrize(
    "builder",
    [
        PydanticAdapter.type_builder,
        LangChainAdapter.type_builder,
        ClaudeAdapter.type_builder,
        CodexAdapter.type_builder,
    ],
)
def test_structured_output_builders_only_use_explicit_annotated_descriptions(
    builder: Any,
) -> None:
    output_type = builder({"answer": Annotated[str, "Documented answer"]})
    undocumented_output_type = builder({"answer": str})

    assert output_type.model_json_schema()["properties"]["answer"]["description"] == (
        "Documented answer"
    )
    assert "description" not in undocumented_output_type.model_json_schema()["properties"]["answer"]


def test_dspy_adapter_reuses_existing_global_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSettings:
        lm = object()

        def configure(self, **_kwargs: Any) -> None:
            raise AssertionError("existing DSPy baseline must not be reconfigured")

    monkeypatch.setattr(dspy_adapter_module.dspy, "settings", FakeSettings())

    DSPyAdapter(model="openai:gpt-4o-mini")


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


def test_dspy_adapter_supports_prebuilt_output_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Review(BaseModel):
        accepted: bool
        reason: str

    class FakeSignature:
        def with_instructions(self, instructions: str | None) -> "FakeSignature":
            assert instructions == "Judge carefully."
            return self

    class FakePredict:
        def __init__(self, signature: FakeSignature) -> None:
            assert isinstance(signature, FakeSignature)

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs == {"user_prompt": "Evaluate this proposal."}
            return {"result": {"accepted": True, "reason": "Safe"}}

    def fake_output_type_builder(output_type: type[Any]) -> FakeSignature:
        assert output_type is Review
        return FakeSignature()

    monkeypatch.setattr(dspy_adapter_module.dspy, "Predict", FakePredict)
    monkeypatch.setattr(
        DSPyAdapter,
        "output_type_builder",
        staticmethod(fake_output_type_builder),
    )

    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = None
    _set_profile_stack(adapter)

    result = adapter.produce_sync(
        "Evaluate this proposal.",
        output_type=Review,
        instructions="Judge carefully.",
    )

    assert result == Review(accepted=True, reason="Safe")


def test_dspy_subagent_execution_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSignature:
        def __init__(self) -> None:
            self.instructions: str | None = None

        def with_instructions(self, instructions: str | None) -> "FakeSignature":
            self.instructions = instructions
            return self

    prediction_result: Any = {
        "task_summary": "done",
        "final_result": {"value": 13},
    }

    class FakePredict:
        def __init__(self, signature: FakeSignature) -> None:
            self.signature = signature

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            assert self.signature.instructions == "child system"
            assert kwargs["user_prompt"] == "delegate"
            return {"result": prediction_result}

    monkeypatch.setattr(
        dspy_adapter_module,
        "build_dynamic_model",
        lambda *_args, **_kwargs: FakeSignature(),
    )
    monkeypatch.setattr(dspy_adapter_module.dspy, "Predict", FakePredict)

    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = None
    adapter._tool_scope = ToolScope()
    adapter._tool_spec_scope = ToolScope()
    adapter._subagent_failure_policy = "fail_closed"
    _set_profile_stack(adapter)
    request = SubagentExecutionRequest(
        profile=AgentProfile(system="child system", approval=ApprovalPolicy.allow()),
        tools=(
            ToolSpec(
                fn=lambda: "ok",
                name="lookup",
                metadata={"kind": "subagent"},
            ),
        ),
        instructions="child system",
        approval_ceiling=(ApprovalPolicy.allow(),),
        usage_limits=SubagentUsageLimits(),
    )

    with adapter.subagent_scope(request):
        assert tuple(tool.name for tool in adapter.scoped_tool_specs) == ("lookup",)
    assert adapter.scoped_tools == ()
    inherited_request = SubagentExecutionRequest(
        profile=AgentProfile(system="child system"),
        tools=request.tools,
        instructions=request.instructions,
        approval_ceiling=request.approval_ceiling,
        usage_limits=request.usage_limits,
    )
    with adapter.subagent_scope(inherited_request):
        assert tuple(tool.name for tool in adapter.scoped_tool_specs) == ("lookup",)
    with adapter.tool_scope(("native",)):
        assert adapter.scoped_tools == ("native",)

    usages: list[InvocationUsage] = []
    with observe_invocations(usages.append):
        result = asyncio.run(
            adapter.produce_subagent(
                template="delegate",
                output_schema={"type": "object"},
                instructions="child system",
            )
        )
    assert result["final_result"] == {"value": 13}
    assert usages == [InvocationUsage(requests=1)]
    assert adapter.telemetry_model_name() is None
    assert adapter.subagent_run_kwargs(request.usage_limits) == {}

    prediction_result = "bad"
    with pytest.raises(TypeError, match="must be an object"):
        asyncio.run(
            adapter.produce_subagent(
                template="delegate",
                output_schema={"type": "object"},
                instructions="child system",
            )
        )

    with pytest.raises(ValueError, match="Unknown subagent failure policy"):
        DSPyAdapter(
            model="openai/gpt-4o-mini",
            subagent_failure_policy=cast(Any, "invalid"),
        )


def test_dspy_subagent_tool_failure_is_recoverable() -> None:
    async def fail() -> str:
        raise KediExecutionError("delegation failed")

    wrapped = dspy_adapter_module._dspy_recoverable_subagent_tool(
        ToolSpec(fn=fail, name="delegate_task", metadata={"kind": "subagent"})
    )
    assert asyncio.run(wrapped.fn()) == "delegation failed"

    sync = dspy_adapter_module._dspy_recoverable_subagent_tool(
        ToolSpec(fn=lambda: "ok", name="delegate_task", metadata={"kind": "subagent"})
    )
    assert asyncio.run(sync.fn()) == "ok"

    guarded = dspy_adapter_module._dspy_tool_with_approval_policies(
        ToolSpec(fn=lambda: "ok", name="delegate_task"),
        (ApprovalPolicy.allow(),),
    )
    assert guarded.fn() == "ok"


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

    async def structured_mcp(
        _signature: object,
        _tools: list[Any],
        _template: str,
        _kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "result": {
                "task_summary": "done",
                "final_result": {"value": 3},
            }
        }

    adapter._lm_model = None
    adapter._subagent_failure_policy = "fail_closed"
    monkeypatch.setattr(adapter, "_produce_with_mcp", structured_mcp)
    assert adapter._produce_subagent_sync(
        "delegate",
        {"type": "object"},
        {"instructions": "child system"},
    ) == {
        "task_summary": "done",
        "final_result": {"value": 3},
    }


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


def test_dspy_sync_tool_function_reports_tool_usage() -> None:
    usages: list[InvocationUsage] = []
    wrapped = dspy_sync_tool_function(lambda value: value.upper())

    with observe_invocations(usages.append):
        assert wrapped("kedi") == "KEDI"

    assert usages == [InvocationUsage(tool_calls=1)]


def test_nested_invocation_observers_compose_and_restore() -> None:
    outer: list[InvocationUsage] = []
    inner: list[InvocationUsage] = []
    nested_usage = InvocationUsage(requests=1, total_tokens=3)
    restored_usage = InvocationUsage(tool_calls=1)

    with observe_invocations(outer.append):
        with observe_invocations(inner.append):
            report_invocation(nested_usage)
        report_invocation(restored_usage)

    assert inner == [nested_usage]
    assert outer == [nested_usage, restored_usage]


def test_dspy_invocation_usage_reads_tracked_tokens() -> None:
    class Prediction:
        @staticmethod
        def get_lm_usage() -> dict[str, Any]:
            return {
                "invalid": "ignored",
                "first": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
                "second": {"input_tokens": 2, "output_tokens": 4},
            }

    usages: list[InvocationUsage] = []
    with observe_invocations(usages.append):
        dspy_adapter_module._report_dspy_invocation(Prediction())

    assert usages == [
        InvocationUsage(
            requests=1,
            input_tokens=5,
            output_tokens=9,
            total_tokens=14,
        )
    ]


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
    assert tools[1].function.__wrapped__ is lookup
    assert tools[1].function(query="kedi") == "kedi"


def test_pydantic_adapter_deferred_approval_revalidates_edited_kedi_tool_arguments() -> None:
    calls: list[int] = []

    def increment(*, count: int) -> int:
        calls.append(count)
        return count + 1

    def edit_count(_request: object) -> ApprovalDecision:
        return ApprovalDecision.edit({"count": 2})

    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(adapter, AgentProfile(approval=ApprovalPolicy.dynamic(edit_count)))
    adapter.register_tool(python_callable_tool_spec("increment", increment))

    requests = DeferredToolRequests(
        approvals=[ToolCallPart("increment", {"count": 0}, tool_call_id="increment-call")]
    )
    results = asyncio.run(adapter._handle_deferred_tool_calls(None, requests))

    approval = results.approvals["increment-call"]
    assert isinstance(approval, ToolApproved)
    assert approval.override_args == {"count": 2}
    assert adapter.scoped_tools[0].function(**approval.override_args) == 3
    assert calls == [2]


def test_pydantic_adapter_deferred_approval_denies_and_handles_external_tools() -> None:
    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(adapter, AgentProfile(approval=ApprovalPolicy.deny()))
    adapter.register_tool(
        ToolSpec(
            fn=lambda *, path: path,
            name="write_file",
            risk="mutating",
        )
    )
    denied_requests = DeferredToolRequests(
        approvals=[ToolCallPart("write_file", {"path": "notes.txt"}, tool_call_id="denied-call")]
    )

    denied = asyncio.run(adapter._handle_deferred_tool_calls(None, denied_requests))
    assert isinstance(denied.approvals["denied-call"], ToolDenied)

    _set_profile_stack(adapter, AgentProfile(approval=ApprovalPolicy.allow()))
    external_requests = DeferredToolRequests(
        approvals=[ToolCallPart("mcp_search", {"query": "kedi"}, tool_call_id="mcp-call")]
    )
    allowed = asyncio.run(adapter._handle_deferred_tool_calls(None, external_requests))

    external_approval = allowed.approvals["mcp-call"]
    assert isinstance(external_approval, ToolApproved)
    assert external_approval.override_args == {"query": "kedi"}


@pytest.mark.parametrize(
    ("arguments", "expected_risk"),
    [
        ({"path": "notes.txt"}, "mutating"),
        ({"path": ".env"}, "sensitive"),
        ({"target": {"path": ".env.local"}}, "sensitive"),
        ({"files": ["notes.txt", {"path": ".env.production"}]}, "sensitive"),
    ],
)
def test_pydantic_external_tool_approval_elevates_secret_paths(
    arguments: dict[str, Any], expected_risk: str
) -> None:
    request = approval_request_for_tool(
        pydantic_adapter_module._external_tool_spec("external_read"),
        arguments,
        "pydantic",
    )

    assert request.risk == expected_risk


def test_pydantic_adapter_registers_deferred_approval_capability_for_approval_toolsets() -> None:
    adapter = object.__new__(PydanticAdapter)
    adapter._tool_scope = ToolScope()
    _set_profile_stack(adapter)
    adapter.register_tool(ToolSpec(fn=lambda: "ok", name="read_status", risk="read_only"))

    run_kwargs = adapter._run_kwargs({}, tool_specs=adapter.scoped_tool_specs)

    assert isinstance(run_kwargs["toolsets"][0], ApprovalRequiredToolset)
    assert len(run_kwargs["capabilities"]) == 1
    assert isinstance(run_kwargs["capabilities"][0], HandleDeferredToolCalls)
    assert PydanticAdapter.capabilities.supports("native_approvals")
    assert PydanticAdapter.capabilities.supports("native_approval_handler")


def test_pydantic_adapter_runs_the_native_deferred_edit_flow() -> None:
    calls: list[str] = []

    def write_status(*, name: str) -> str:
        calls.append(name)
        return f"saved:{name}"

    def rename_tool_target(_request: object) -> ApprovalDecision:
        return ApprovalDecision.edit({"name": "approved"})

    adapter = PydanticAdapter(TestModel(call_tools=["write_status"]))
    with adapter.profile_override(
        AgentProfile(approval=ApprovalPolicy.dynamic(rename_tool_target))
    ):
        with adapter.tool_scope((python_callable_tool_spec("write_status", write_status),)):
            result = asyncio.run(adapter.invoke(prompt="Call write_status."))

    assert result == '{"write_status":"saved:approved"}'
    assert calls == ["approved"]


def test_pydantic_adapter_denies_constructor_native_tools_before_invocation() -> None:
    calls: list[int] = []

    def native_tool(x: int = 1) -> int:
        calls.append(x)
        return x + 1

    adapter = PydanticAdapter(TestModel(call_tools=["native_tool"]), tools=[native_tool])
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())):
        result = asyncio.run(adapter.invoke(prompt="Call native_tool."))

    assert "denied" in result
    assert calls == []


def test_pydantic_adapter_denies_inherited_toolsets_before_invocation() -> None:
    calls: list[int] = []

    def native_tool(x: int = 1) -> int:
        calls.append(x)
        return x + 1

    adapter = PydanticAdapter(
        TestModel(call_tools=["native_tool"]),
        toolsets=[FunctionToolset(tools=[native_tool])],
    )
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())):
        result = asyncio.run(adapter.invoke(prompt="Call native_tool."))

    assert "denied" in result
    assert calls == []


def test_pydantic_adapter_wraps_native_toolset_overrides_and_nested_runs() -> None:
    def native_tool() -> str:
        return "ok"

    adapter = PydanticAdapter(TestModel(), tools=[native_tool])

    nested_kwargs: dict[str, object] = {}
    adapter._add_nested_native_toolsets(nested_kwargs)
    assert len(nested_kwargs["toolsets"]) == 1
    assert isinstance(nested_kwargs["toolsets"][0], ApprovalRequiredToolset)

    nested_kwargs = {"toolsets": ()}
    adapter._add_nested_native_toolsets(nested_kwargs)
    assert len(nested_kwargs["toolsets"]) == 1

    with pytest.raises(TypeError, match="must be a sequence"):
        adapter._add_nested_native_toolsets({"toolsets": "invalid"})

    with adapter.override(tools=[native_tool]):
        assert adapter._has_native_approval_tools()

    inherited_adapter = PydanticAdapter(TestModel())
    with inherited_adapter.override(toolsets=[FunctionToolset(tools=[native_tool])]):
        assert inherited_adapter._application_toolset_count() == 2
        assert inherited_adapter._has_native_approval_tools()

    wrapped = pydantic_adapter_module._approval_required_toolset(FunctionToolset())
    assert pydantic_adapter_module._approval_required_toolset(wrapped) is wrapped


def test_runtime_leaves_pydantic_tools_unwrapped_for_native_approval() -> None:
    program = parse_program("@write_file(path: str) -> str:\n  = <path>\n")
    adapter = PydanticAdapter(TestModel())
    runtime = compile_program(program, adapter=adapter)
    profile = AgentProfile(tools=("write_file",), approval=ApprovalPolicy.deny())

    tool = runtime._resolve_tool_specs(profile, adapter)[0]

    assert tool.fn(path="notes.txt") == "notes.txt"
