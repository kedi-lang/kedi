from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai import ModelRetry, ModelSettings
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import (
    DeferredToolRequests,
    RunContext,
    Tool,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai.toolsets import AbstractToolset, ExternalToolset, FunctionToolset

from kedi.agent_adapter import (
    PydanticAdapter,
    pydantic_approval_request_from_deferred,
    pydantic_deferred_results_from_decisions,
    pydantic_tool_from_spec,
    pydantic_toolset_from_specs,
)
from kedi.agent_adapter.adapters._pydantic_compat import (
    PydanticAICompatibilityError,
    application_toolset_count,
    build_base_toolset_list,
    has_application_tools,
    infer_agent_name,
)
from kedi.agent_adapter.adapters import pydantic as pydantic_adapter_module
from kedi.agent_adapter.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
)
from kedi.agent_adapter.subagents import (
    SubagentConversationContext,
    SubagentExecutionRequest,
)
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_adapter.tool_override import (
    ToolSpec,
    current_required_tool_calls,
)
from kedi.agent_adapter.utils import python_callable_tool_spec
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.mcp import McpServerSpec


def _request(messages: list[ModelMessage]) -> ModelRequest:
    return next(message for message in messages if isinstance(message, ModelRequest))


def test_native_run_surfaces_apply_active_profile() -> None:
    observed: list[tuple[str | None, dict[str, Any]]] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> None:
        observed.append((_request(messages).instructions, dict(info.model_settings or {})))

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        record(messages, info)
        return ModelResponse(parts=[TextPart("ok")])

    async def respond_stream(
        messages: list[ModelMessage],
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

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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

    async def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart("ok")])

    def dynamic_settings(_ctx: RunContext[Any]) -> ModelSettings:
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
    caller_toolset: AbstractToolset[Any] = FunctionToolset(tools=[Tool(caller_tool)])

    with (
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())),
        adapter.tool_scope((python_callable_tool_spec("scoped_tool", scoped_tool),)),
    ):
        adapter.run_sync(
            "Call both tools.",
            toolsets=[caller_toolset],
            capabilities=[_RecordingCapability(capability_events)],
        )

    assert sorted(calls) == ["caller", "scoped"]
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
            toolsets=[FunctionToolset(tools=[Tool(caller_tool)])],
        )

    assert calls == []
    assert "denied" in result.output


@pytest.mark.parametrize("placement", ["constructor", "caller"])
def test_native_external_approval_resolution_preserves_deferred_requests(
    placement: str,
) -> None:
    calls: list[str] = []

    def write_file(*, path: str) -> str:
        calls.append(path)
        return path

    adapter_kwargs: dict[str, Any] = {"approval_resolution": "external"}
    if placement == "constructor":
        adapter_kwargs["tools"] = [Tool(write_file)]
    adapter = PydanticAdapter(TestModel(call_tools=["write_file"]), **adapter_kwargs)
    run_kwargs: dict[str, Any] = {
        "output_type": [str, DeferredToolRequests],
    }
    if placement == "caller":
        run_kwargs["toolsets"] = [FunctionToolset(tools=[Tool(write_file)])]
    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.deny())):
        result = adapter.run_sync("Write the file.", **run_kwargs)

    assert isinstance(result.output, DeferredToolRequests)
    assert calls == []
    assert [call.tool_name for call in result.output.approvals] == ["write_file"]


def test_external_approval_handoff_preserves_kedi_semantics_through_resume() -> None:
    calls: list[str] = []
    model_calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": " .env "})])
        return ModelResponse(parts=[TextPart("verified")])

    def validate(arguments: Any) -> dict[str, Any]:
        path = str(arguments["path"]).strip()
        if path == "blocked":
            raise ValueError("blocked path")
        return {"path": path}

    def read_file(*, path: str) -> str:
        calls.append(path)
        return path

    spec = ToolSpec(
        fn=read_file,
        name="read_file",
        description="Read one project file.",
        json_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        metadata={"required_before_output": True, "source": "native-parity"},
        risk="read_only",
        risk_resolver=lambda arguments: (
            "sensitive" if str(arguments["path"]).startswith(".") else "read_only"
        ),
        argument_validator=validate,
    )
    adapter = PydanticAdapter(FunctionModel(respond), approval_resolution="external", retries=2)
    output_type = [str, DeferredToolRequests]

    first = adapter.run_sync(
        "Read the file.",
        output_type=output_type,
        toolsets=[pydantic_toolset_from_specs([spec])],
    )

    assert isinstance(first.output, DeferredToolRequests)
    deferred = first.output
    call = deferred.approvals[0]
    assert call.args_as_dict() == {"path": ".env"}
    request = pydantic_approval_request_from_deferred(deferred, call.tool_call_id)
    assert request.arguments == {"path": ".env"}
    assert request.risk == "sensitive"
    assert request.description == "Read one project file."
    assert request.metadata == {
        "required_before_output": True,
        "source": "native-parity",
        "kedi_required_tool_wrapped": True,
    }

    with pytest.raises(ValueError, match="blocked path"):
        pydantic_deferred_results_from_decisions(
            deferred,
            {call.tool_call_id: ApprovalDecision.edit({"path": " blocked "})},
            tool_specs=[spec],
        )

    results = pydantic_deferred_results_from_decisions(
        deferred,
        {call.tool_call_id: ApprovalDecision.edit({"path": " README.md "})},
        tool_specs=[spec],
    )
    resumed = adapter.run_sync(
        None,
        output_type=output_type,
        message_history=first.all_messages(),
        deferred_tool_results=results,
        toolsets=[pydantic_toolset_from_specs([spec])],
    )

    assert resumed.output == "verified"
    assert calls == ["README.md"]
    assert current_required_tool_calls() is None


def test_external_approval_executes_canonical_read_only_tool_without_deferral() -> None:
    calls: list[str] = []

    async def respond(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        tool_returned = any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if tool_returned:
            return ModelResponse(parts=[TextPart("verified")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "read_file",
                    {"path": "README.md"},
                    tool_call_id="read-only-call",
                )
            ]
        )

    def read_file(*, path: str) -> str:
        calls.append(path)
        return path

    spec = ToolSpec(
        fn=read_file,
        name="read_file",
        risk="read_only",
    )
    adapter = PydanticAdapter(FunctionModel(respond), approval_resolution="external")

    result = adapter.run_sync(
        "Read the file.",
        output_type=[str, DeferredToolRequests],
        toolsets=[pydantic_toolset_from_specs([spec])],
    )

    assert result.output == "verified"
    assert calls == ["README.md"]


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "no Kedi approval metadata"),
        (
            {"kedi_approval": {"arguments": "invalid", "risk": "read_only"}},
            "invalid Kedi arguments",
        ),
        (
            {
                "kedi_approval": {
                    "arguments": {},
                    "risk": "read_only",
                    "metadata": "invalid",
                }
            },
            "invalid Kedi tool metadata",
        ),
        (
            {"kedi_approval": {"arguments": {}, "risk": "unknown"}},
            "invalid Kedi risk",
        ),
    ],
)
def test_external_approval_handoff_rejects_invalid_metadata(
    metadata: dict[str, Any],
    message: str,
) -> None:
    requests = DeferredToolRequests(metadata={"call": metadata})

    with pytest.raises(ValueError, match=message):
        pydantic_approval_request_from_deferred(requests, "call")


def test_external_approval_result_builder_supports_boolean_decisions() -> None:
    allow_call = ToolCallPart("native_tool", {}, tool_call_id="allow")
    deny_call = ToolCallPart("native_tool", {}, tool_call_id="deny")
    payload = {
        "tool_name": "native_tool",
        "arguments": {},
        "risk": "mutating",
        "adapter_shortname": None,
        "description": None,
        "metadata": None,
    }
    requests = DeferredToolRequests(
        approvals=[allow_call, deny_call],
        metadata={
            "allow": {"kedi_approval": payload},
            "deny": {"kedi_approval": payload},
        },
    )

    results = pydantic_deferred_results_from_decisions(
        requests,
        {"allow": True, "deny": False},
        tool_specs=[],
    )

    assert isinstance(results.approvals["allow"], ToolApproved)
    assert isinstance(results.approvals["deny"], ToolDenied)
    with pytest.raises(ValueError, match="Unknown deferred approval tool call ID"):
        pydantic_deferred_results_from_decisions(
            requests,
            {"missing": True},
            tool_specs=[],
        )


def test_native_rejects_unknown_approval_resolution_owner() -> None:
    with pytest.raises(ValueError, match="Unknown approval resolution owner"):
        PydanticAdapter(TestModel(), approval_resolution=cast(Any, "unknown"))

    with pytest.raises(ValueError, match="Unknown subagent approval resolution owner"):
        PydanticAdapter(
            TestModel(),
            subagent_approval_resolution=cast(Any, "unknown"),
        )


def test_subagent_approval_owner_can_differ_from_external_parent() -> None:
    calls: list[str] = []
    approval_requests: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approval_requests.append(request)
        return ApprovalDecision.allow()

    async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returned = any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if returned:
            return ModelResponse(parts=[TextPart("child complete")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "write_file",
                    {"path": "child.txt"},
                    tool_call_id="child-write-1",
                )
            ]
        )

    def write_file(*, path: str) -> str:
        calls.append(path)
        return path

    adapter = PydanticAdapter(
        FunctionModel(respond),
        approval_resolution="external",
        subagent_approval_resolution="kedi",
    )
    profile = AgentProfile(
        name="worker",
        approval=ApprovalPolicy.dynamic(approve),
    )
    request = SubagentExecutionRequest(
        profile=profile,
        tools=(ToolSpec(fn=write_file, name="write_file", risk="mutating"),),
        instructions=None,
        approval_ceiling=(),
        usage_limits=SubagentUsageLimits(),
        conversation=SubagentConversationContext(conversation_id="child-conversation"),
    )

    assert adapter.active_approval_resolution == "external"
    with adapter.subagent_scope(request):
        assert adapter.active_approval_resolution == "kedi"
        result = asyncio.run(adapter.invoke(prompt="Write the child file."))

    assert result == "child complete"
    assert calls == ["child.txt"]
    assert [approval.tool_name for approval in approval_requests] == ["write_file"]
    assert adapter.active_approval_resolution == "external"


def test_native_run_rejects_invalid_toolset_and_capability_containers() -> None:
    adapter = PydanticAdapter(TestModel())

    with pytest.raises(TypeError, match="toolsets must be a sequence"):
        adapter.run_sync("prompt", toolsets=cast(Any, "not-a-toolset"))

    with pytest.raises(TypeError, match="capabilities must be a list or tuple"):
        adapter._with_deferred_approval_capability("not-capabilities")


def test_native_capability_injection_is_idempotent() -> None:
    adapter = PydanticAdapter(TestModel())

    capabilities = adapter._with_deferred_approval_capability(None)
    capabilities = adapter._with_deferred_approval_capability(capabilities)

    assert len(capabilities) == 1


def test_approval_required_toolset_fails_closed_for_unknown_projected_tool() -> None:
    toolset = pydantic_toolset_from_specs([ToolSpec(lambda: "ok", "known", risk="read_only")])
    wrapped = pydantic_adapter_module._approval_required_toolset(toolset)

    assert wrapped.approval_required_func(
        cast(Any, SimpleNamespace()),
        ToolDefinition(name="unknown"),
        {},
    )


def test_native_run_includes_active_profile_mcp_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def mcp_tool() -> str:
        calls.append("mcp")
        return "mcp-result"

    model = TestModel(call_tools=["mcp_tool"])
    adapter = PydanticAdapter(model)
    mcp_toolset: AbstractToolset[Any] = FunctionToolset(tools=[Tool(mcp_tool)])
    monkeypatch.setattr(
        adapter,
        "_mcp_toolsets_for_profile",
        lambda _profile: [mcp_toolset],
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

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
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
    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
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


def test_native_iter_cleans_kedi_state_when_closed_before_consumption() -> None:
    spec = ToolSpec(
        fn=lambda: "unused",
        name="required_tool",
        metadata={"required_before_output": True},
    )
    adapter = PydanticAdapter(TestModel())

    async def close_early() -> None:
        async with adapter.iter(
            "prompt",
            toolsets=[pydantic_toolset_from_specs([spec])],
        ):
            assert current_required_tool_calls() is not None
        assert current_required_tool_calls() is None
        assert adapter._run_tool_specs_var.get() is None

    asyncio.run(close_early())


def test_native_iter_cleans_kedi_state_after_cancellation() -> None:
    spec = ToolSpec(
        fn=lambda: "unused",
        name="required_tool",
        metadata={"required_before_output": True},
    )

    async def cancel_run() -> None:
        started = asyncio.Event()
        never = asyncio.Event()

        async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            started.set()
            await never.wait()
            return ModelResponse(parts=[TextPart("unreachable")])

        adapter = PydanticAdapter(FunctionModel(respond))

        async def execute_run() -> None:
            async with adapter.iter(
                "prompt",
                toolsets=[pydantic_toolset_from_specs([spec])],
            ) as agent_run:
                async for _node in agent_run:
                    pass

        task = asyncio.create_task(execute_run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert current_required_tool_calls() is None
        assert adapter._run_tool_specs_var.get() is None

    asyncio.run(cancel_run())


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
    assert asyncio.run(cast(Any, tool.function)(count="2")) == 3


def test_public_pydantic_tool_conversion_preserves_description_without_schema() -> None:
    spec = ToolSpec(
        fn=lambda value: value,
        name="identity",
        description="Return the supplied value.",
    )

    tool = pydantic_tool_from_spec(spec)

    assert tool.description == "Return the supplied value."


def test_bundled_filesystem_path_errors_are_model_retries() -> None:
    def missing_file(*, file_path: str) -> str:
        raise FileNotFoundError(file_path)

    tool = pydantic_tool_from_spec(
        ToolSpec(
            fn=missing_file,
            name="read_text_file",
            metadata={"kedi_tool_kind": "filesystem"},
        )
    )

    with pytest.raises(ModelRetry, match="inspect unknown paths with get_file_info"):
        cast(Any, tool.function)(file_path="missing.kedi")


def test_filesystem_permission_errors_are_reported_without_bypassing_the_boundary() -> None:
    def denied(*, file_path: str) -> str:
        raise PermissionError(file_path)

    tool = pydantic_tool_from_spec(
        ToolSpec(
            fn=denied,
            name="read_text_file",
            metadata={"kedi_tool_kind": "filesystem"},
        )
    )

    with pytest.raises(ModelRetry, match="Use repository-relative paths only"):
        cast(Any, tool.function)(file_path="secret.env")


@pytest.mark.parametrize("placement", ["constructor", "caller"])
def test_public_pydantic_tool_conversion_preserves_required_tool_semantics(
    placement: str,
) -> None:
    model_calls = 0
    tool_calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
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

    spec = ToolSpec(
        fn=lookup,
        name="lookup",
        description="Load the required source.",
        json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        metadata={"required_before_output": True},
    )
    adapter_kwargs: dict[str, Any] = {"retries": 3}
    if placement == "constructor":
        adapter_kwargs["tools"] = [pydantic_tool_from_spec(spec)]
    adapter = PydanticAdapter(FunctionModel(respond), **adapter_kwargs)
    run_kwargs: dict[str, Any] = {}
    if placement == "caller":
        run_kwargs["toolsets"] = [pydantic_toolset_from_specs([spec])]

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.allow())):
        result = adapter.run_sync("Use the required source.", **run_kwargs)

    assert result.output == "verified"
    assert model_calls == 3
    assert tool_calls == 1
    assert current_required_tool_calls() is None


def test_public_pydantic_toolset_preserves_risk_and_edited_argument_validation() -> None:
    calls: list[str] = []
    approval_requests: list[ApprovalRequest] = []
    model_calls = 0

    async def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": ".env"})])
        return ModelResponse(parts=[TextPart("verified")])

    def read_file(*, path: str) -> str:
        calls.append(path)
        return path

    def approve_with_safe_path(request: ApprovalRequest) -> ApprovalDecision:
        approval_requests.append(request)
        return ApprovalDecision.edit({"path": " README.md "})

    spec = ToolSpec(
        fn=read_file,
        name="read_file",
        description="Read one project file.",
        json_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        metadata={"source": "native-parity"},
        risk="read_only",
        risk_resolver=lambda arguments: (
            "sensitive" if str(arguments["path"]).startswith(".") else "read_only"
        ),
        argument_validator=lambda arguments: {"path": str(arguments["path"]).strip()},
    )
    adapter = PydanticAdapter(FunctionModel(respond))

    with adapter.profile_override(
        AgentProfile(approval=ApprovalPolicy.dynamic(approve_with_safe_path))
    ):
        result = adapter.run_sync(
            "Read the file.",
            toolsets=[pydantic_toolset_from_specs([spec])],
        )

    assert result.output == "verified"
    assert calls == ["README.md"]
    assert len(approval_requests) == 1
    request = approval_requests[0]
    assert request.risk == "sensitive"
    assert request.description == "Read one project file."
    assert request.metadata == {"source": "native-parity"}


def test_public_pydantic_toolset_retains_specs_through_native_wrappers() -> None:
    spec = ToolSpec(
        fn=lambda: "source",
        name="lookup",
        description="Load a source.",
        metadata={"required_before_output": True},
    )
    wrapped_toolset = pydantic_toolset_from_specs([spec]).prepared(lambda _ctx, tools: tools)
    unrelated_toolset = ExternalToolset(
        [
            ToolDefinition(
                name="external_lookup",
                description="An unrelated externally executed tool.",
                parameters_json_schema={"type": "object", "properties": {}},
            )
        ]
    )
    adapter = PydanticAdapter(TestModel())

    retained_specs = adapter._tool_specs_for_run({"toolsets": [wrapped_toolset, unrelated_toolset]})

    assert len(retained_specs) == 1
    assert retained_specs[0].name == spec.name
    assert retained_specs[0].metadata is not None
    assert retained_specs[0].metadata["required_before_output"] is True


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
