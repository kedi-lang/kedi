from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from kedi.agent_adapter import PydanticAdapter, pydantic_tool_from_spec
from kedi.agent_adapter.adapters._pydantic_compat import (
    PydanticAICompatibilityError,
    application_toolset_count,
    build_base_toolset_list,
    has_application_tools,
    infer_agent_name,
)
from kedi.agent_adapter.approval import ApprovalPolicy
from kedi.agent_adapter.tool_override import (
    ToolSpec,
    current_required_tool_calls,
)
from kedi.agent_adapter.utils import python_callable_tool_spec
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.mcp import McpServerSpec


def _request(messages: list[object]) -> ModelRequest:
    return next(message for message in messages if isinstance(message, ModelRequest))


def test_native_run_surfaces_apply_active_profile() -> None:
    observed: list[tuple[str | None, dict[str, Any]]] = []

    def record(messages: list[object], info: AgentInfo) -> None:
        observed.append((_request(messages).instructions, dict(info.model_settings or {})))

    async def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        record(messages, info)
        return ModelResponse(parts=[TextPart("ok")])

    async def respond_stream(
        messages: list[object],
        info: AgentInfo,
    ) -> AsyncIterator[str]:
        record(messages, info)
        yield "ok"

    adapter = PydanticAdapter(FunctionModel(respond, stream_function=respond_stream))
    profile = AgentProfile(system="Profile instruction.", settings={"temperature": 0.25})

    async def run_async_surfaces() -> list[str]:
        run_output = (await adapter.run("run")).output
        async with adapter.iter("iter") as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        async with adapter.run_stream("stream") as stream:
            stream_output = await stream.get_output()
        return [str(run_output), str(agent_run.result.output), str(stream_output)]

    with adapter.profile_override(profile):
        assert asyncio.run(run_async_surfaces()) == ["ok", "ok", "ok"]
        assert adapter.run_sync("run_sync").output == "ok"
        assert adapter.run_stream_sync("run_stream_sync").get_output() == "ok"

    assert observed == [("Profile instruction.", {"temperature": 0.25})] * 5


def test_native_call_arguments_have_documented_precedence() -> None:
    observed: list[tuple[str | None, dict[str, Any]]] = []

    async def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        observed.append((_request(messages).instructions, dict(info.model_settings or {})))
        return ModelResponse(parts=[TextPart("ok")])

    adapter = PydanticAdapter(FunctionModel(respond))
    with adapter.profile_override(
        AgentProfile(system="Profile instruction.", settings={"temperature": 0.9})
    ):
        result = adapter.run_sync(
            "prompt",
            instructions="Explicit instruction.",
            model_settings={"temperature": 0.1, "seed": 7},
        )

    assert result.output == "ok"
    assert observed == [("Explicit instruction.", {"temperature": 0.9, "seed": 7})]


def test_native_run_preserves_pydantic_name_inference() -> None:
    named_adapter = PydanticAdapter(TestModel())

    named_adapter.run_sync("Infer the adapter name.")

    assert named_adapter.name == "named_adapter"


def test_native_iter_preserves_pydantic_name_inference() -> None:
    iter_adapter = PydanticAdapter(TestModel())

    async def run() -> None:
        async with iter_adapter.iter("Infer the adapter name."):
            pass

    asyncio.run(run())

    assert iter_adapter.name == "iter_adapter"


def test_native_callable_model_settings_merge_with_profile_settings() -> None:
    observed: list[dict[str, Any]] = []

    async def respond(_messages: list[object], info: AgentInfo) -> ModelResponse:
        observed.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart("ok")])

    def dynamic_settings(_ctx: RunContext[Any]) -> dict[str, Any]:
        return {"seed": 7, "temperature": 0.1}

    adapter = PydanticAdapter(FunctionModel(respond))
    with adapter.profile_override(AgentProfile(settings={"temperature": 0.8})):
        adapter.run_sync("prompt", model_settings=dynamic_settings)

    assert observed == [{"seed": 7, "temperature": 0.8}]


class _RecordingCapability(AbstractCapability[Any]):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def before_run(self, ctx: RunContext[Any]) -> None:
        del ctx
        self.events.append("before_run")


def test_native_run_appends_scoped_and_caller_tools_and_preserves_capabilities() -> None:
    calls: list[str] = []
    capability_events: list[str] = []

    def scoped_tool() -> str:
        calls.append("scoped")
        return "scoped-result"

    def caller_tool() -> str:
        calls.append("caller")
        return "caller-result"

    model = TestModel(call_tools=["scoped_tool", "caller_tool"])
    adapter = PydanticAdapter(model)
    caller_toolset = FunctionToolset(tools=[caller_tool])

    with (
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())),
        adapter.tool_scope((python_callable_tool_spec("scoped_tool", scoped_tool),)),
    ):
        adapter.run_sync(
            "Call both tools.",
            toolsets=[caller_toolset],
            capabilities=[_RecordingCapability(capability_events)],
        )

    assert calls == ["scoped", "caller"]
    assert capability_events == ["before_run"]
    assert model.last_model_request_parameters is not None
    assert {tool.name for tool in model.last_model_request_parameters.function_tools} == {
        "scoped_tool",
        "caller_tool",
    }


def test_native_caller_toolsets_cannot_bypass_kedi_approval() -> None:
    calls: list[str] = []

    def caller_tool() -> str:
        calls.append("called")
        return "result"

    adapter = PydanticAdapter(TestModel(call_tools=["caller_tool"]))
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())):
        result = adapter.run_sync(
            "Call the tool.",
            toolsets=[FunctionToolset(tools=[caller_tool])],
        )

    assert calls == []
    assert "denied" in result.output


def test_native_run_rejects_invalid_toolset_and_capability_containers() -> None:
    adapter = PydanticAdapter(TestModel())

    with pytest.raises(TypeError, match="toolsets must be a sequence"):
        adapter.run_sync("prompt", toolsets="not-a-toolset")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="capabilities must be a list or tuple"):
        adapter._with_deferred_approval_capability("not-capabilities")


def test_native_capability_injection_is_idempotent() -> None:
    adapter = PydanticAdapter(TestModel())

    capabilities = adapter._with_deferred_approval_capability(None)
    capabilities = adapter._with_deferred_approval_capability(capabilities)

    assert len(capabilities) == 1


def test_native_run_includes_active_profile_mcp_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def mcp_tool() -> str:
        calls.append("mcp")
        return "mcp-result"

    model = TestModel(call_tools=["mcp_tool"])
    adapter = PydanticAdapter(model)
    monkeypatch.setattr(
        adapter,
        "_mcp_toolsets_for_profile",
        lambda _profile: [FunctionToolset(tools=[mcp_tool])],
    )
    profile = AgentProfile(
        approval=ApprovalPolicy.allow(),
        mcp_servers=(McpServerSpec(transport="http", url="https://mcp.invalid"),),
    )

    with adapter.profile_override(profile):
        adapter.run_sync("Call the MCP tool.")

    assert calls == ["mcp"]
    assert model.last_model_request_parameters is not None
    assert [tool.name for tool in model.last_model_request_parameters.function_tools] == [
        "mcp_tool"
    ]


def test_native_run_requires_marked_tools_before_any_output() -> None:
    model_calls = 0
    tool_calls = 0

    async def respond(_messages: list[object], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[TextPart("fabricated")])
        if model_calls == 2:
            return ModelResponse(parts=[ToolCallPart("lookup", {})])
        return ModelResponse(parts=[TextPart("verified")])

    def lookup() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return "source"

    adapter = PydanticAdapter(FunctionModel(respond), retries=3)
    required_tool = ToolSpec(
        fn=lookup,
        name="lookup",
        description="Load the required source.",
        json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        metadata={"required_before_output": True},
    )

    with (
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())),
        adapter.tool_scope((required_tool,)),
    ):
        result = adapter.run_sync("Use the required source.")

    assert result.output == "verified"
    assert model_calls == 3
    assert tool_calls == 1
    assert current_required_tool_calls() is None


def test_native_run_cleans_required_tool_scope_after_model_error() -> None:
    async def fail(_messages: list[object], _info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model failed")

    adapter = PydanticAdapter(FunctionModel(fail))
    required_tool = ToolSpec(
        fn=lambda: "unused",
        name="required_tool",
        description="Required tool.",
        json_schema={"type": "object", "properties": {}},
        metadata={"required_before_output": True},
    )

    with adapter.tool_scope((required_tool,)):
        with pytest.raises(RuntimeError, match="model failed"):
            adapter.run_sync("fail")

    assert current_required_tool_calls() is None


def test_public_pydantic_tool_conversion_preserves_schema_and_validation() -> None:
    async def implementation(*, count: int) -> int:
        return count + 1

    spec = ToolSpec(
        fn=implementation,
        name="increment",
        description="Increment a validated count.",
        json_schema={
            "type": "object",
            "properties": {"count": {"type": "integer", "description": "Input count."}},
            "required": ["count"],
        },
        metadata={"source": "test"},
        argument_validator=lambda arguments: {"count": int(arguments["count"])},
        sequential=True,
    )

    tool = pydantic_tool_from_spec(spec)

    assert tool.name == "increment"
    assert tool.description == "Increment a validated count."
    assert tool.metadata == {"source": "test"}
    assert tool.sequential is True
    assert tool.tool_def.parameters_json_schema == spec.json_schema
    assert asyncio.run(tool.function(count="2")) == 3


def test_public_pydantic_tool_conversion_preserves_description_without_schema() -> None:
    spec = ToolSpec(
        fn=lambda value: value,
        name="identity",
        description="Return the supplied value.",
    )

    tool = pydantic_tool_from_spec(spec)

    assert tool.description == "Return the supplied value."


@pytest.mark.parametrize(
    "operation",
    [
        lambda agent: build_base_toolset_list(agent),
        lambda agent: application_toolset_count(agent),
        lambda agent: has_application_tools(agent),
        lambda agent: infer_agent_name(agent, None),
    ],
)
def test_pydantic_compat_reports_unsupported_private_layout(
    operation: Any,
) -> None:
    partial_agent = SimpleNamespace()

    with pytest.raises(
        PydanticAICompatibilityError,
        match="unsupported Agent tool layout",
    ):
        operation(partial_agent)
