from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.adapters.langchain import LangChainAdapter
from kedi.agent_adapter.adapters.langchain.middleware import _langchain_approval_middleware
from kedi.agent_adapter.adapters.langchain.reason import tools_with_reason
from kedi.agent_adapter.adapters.langchain.tools import _langchain_tool_from_spec
from kedi.agent_adapter.approvals.policy import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.approvals.reason import (
    TOOL_REASON_INSTRUCTIONS,
    accepts_tool_reason,
    reason_schema,
    split_tool_reason,
)
from kedi.agent_adapter.tools.execution import resolve_tool_arguments_async
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.mcp import McpServerSpec


def _spec(calls, *, risk="mutating", resolver=None):
    def write(value: str) -> str:
        calls.append(value)
        return value

    return ToolSpec(write, "write", risk=risk, risk_resolver=resolver)


@pytest.mark.parametrize("adapter", [PydanticAdapter, LangChainAdapter])
def test_reason_opt_in_boolean(adapter):
    assert adapter().tool_reason is False
    with pytest.raises(TypeError, match="tool_reason"):
        adapter(tool_reason="yes")


def test_reason_schema_is_optional_copy_and_rejects_collision():
    original = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    schema = reason_schema(original, name="write")
    assert schema["properties"]["reason"] == {"type": "string"}
    assert schema["required"] == ["value"]
    assert "reason" not in original["properties"]
    with pytest.raises(ValueError, match="reserves"):
        reason_schema(schema, name="write")


@pytest.mark.parametrize("reason", [None, 3, [], {}])
def test_non_string_reason_rejected(reason):
    with pytest.raises(ValueError, match="string"):
        split_tool_reason({"reason": reason})


def test_only_potentially_approved_tools():
    assert not accepts_tool_reason(_spec([], risk="read_only"))
    assert accepts_tool_reason(_spec([], risk="read_only", resolver=lambda _: "sensitive"))
    assert accepts_tool_reason(_spec([]))


@pytest.mark.parametrize(
    "metadata", [{"kedi_codemode_control": True}, {"kind": "codemode_control"}]
)
def test_codemode_control_tools_do_not_request_action_reasons(metadata):
    assert not accepts_tool_reason(replace(_spec([]), metadata=metadata))


@pytest.mark.parametrize("bad_args", ["[]", "{", {"value": "hello", "reason": 3}])
def test_pydantic_retries_invalid_arguments_before_approval_or_execution(bad_args):
    from pydantic_ai.messages import RetryPromptPart

    calls, approvals = [], []
    step = 0

    def approve(request):
        approvals.append(request)
        return ApprovalDecision.allow()

    def respond(messages, info):
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(parts=[ToolCallPart("write", bad_args, tool_call_id="invalid")])
        if step == 2:
            assert not calls and not approvals
            assert any(isinstance(part, RetryPromptPart) for m in messages for part in m.parts)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "write",
                        {"value": "hello", "reason": "Requested note"},
                        tool_call_id="valid",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), tool_reason=True)
    with (
        adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))),
        adapter.tool_scope([_spec(calls)]),
    ):
        assert adapter.run_sync("write").output == "done"
    assert calls == ["hello"]
    assert len(approvals) == 1
    assert approvals[0].reason == "Requested note"
    assert adapter._tool_reasons_var.get() == {}


def test_pydantic_rejects_provider_native_mcp_before_sending_a_request():
    from pydantic_ai.capabilities import NativeTool
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.native_tools import MCPServerTool

    def respond(messages, info):
        pytest.fail("Provider-native MCP must not bypass local approval")

    adapter = PydanticAdapter(
        FunctionModel(respond),
        tool_reason=True,
        capabilities=[NativeTool(MCPServerTool(id="remote", url="https://example.com/mcp"))],
    )
    with pytest.raises(UserError, match="provider-native MCP bypasses Kedi approval"):
        adapter.run_sync("write")


@pytest.mark.parametrize("reason", [None, "Write the requested note"])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
def test_pydantic_real_tool_execution_strips_reason(reason, external, as_json):
    calls, approvals = [], []
    spec = _spec(calls)
    step = 0

    def approve(request):
        approvals.append(request)
        return ApprovalDecision.allow()

    def respond(messages, info):
        nonlocal step
        step += 1
        schema = info.function_tools[0].parameters_json_schema
        assert schema["properties"]["reason"] == {"type": "string"}
        assert "reason" not in schema.get("required", [])
        assert any(
            TOOL_REASON_INSTRUCTIONS in (getattr(m, "instructions", None) or "") for m in messages
        )
        if step == 1:
            args = {"value": "hello"}
            if reason is not None:
                args["reason"] = reason
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "write", json.dumps(args) if as_json else args, tool_call_id="call"
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), tool_reason=True)
    profile = AgentProfile(approval=ApprovalPolicy.dynamic(approve))
    with adapter.profile_override(profile):
        if external:
            # The same local toolset boundary used by MCP: no Kedi ToolSpec on the tool.
            result = adapter.run_sync("write", toolsets=[FunctionToolset(tools=[spec.fn])])
        else:
            with adapter.tool_scope([spec]):
                result = adapter.run_sync("write")
    assert result.output == "done"
    assert calls == ["hello"]
    assert len(approvals) == 1
    assert approvals[0].arguments == {"value": "hello"}
    assert approvals[0].reason == reason
    assert approvals[0].tool_reason_enabled
    assert adapter._tool_reasons_var.get() == {}


@pytest.mark.parametrize("enabled,risk", [(False, "mutating"), (True, "read_only")])
def test_pydantic_leaves_unaffected_schemas_unchanged(enabled, risk):
    calls = []

    def respond(messages, info):
        assert "reason" not in info.function_tools[0].parameters_json_schema["properties"]
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAdapter(FunctionModel(respond), tool_reason=enabled)
    with adapter.tool_scope([_spec(calls, risk=risk)]):
        adapter.run_sync("inspect")


@pytest.mark.parametrize("reason", [None, "Requested write"])
@pytest.mark.parametrize("external", [False, True])
def test_langchain_projection_and_actual_tool_execution(reason, external):
    calls, approvals = [], []
    spec = _spec(calls)
    native = _langchain_tool_from_spec(spec)
    projected = tools_with_reason([native], [] if external else [spec])[0]
    assert "reason" not in native.args_schema.model_json_schema()["properties"]
    assert projected.args_schema["properties"]["reason"] == {"type": "string"}

    def approve(request):
        approvals.append(request)
        return ApprovalDecision.allow()

    middleware = _langchain_approval_middleware(
        profile=AgentProfile(approval=ApprovalPolicy.dynamic(approve)),
        tools=[] if external else [spec],
        tool_reason=True,
    )
    args = {"value": "hello"}
    if reason is not None:
        args["reason"] = reason
    request = SimpleNamespace(
        tool_call={"name": "write", "args": args, "id": "call"}, tool=projected
    )
    request.override = lambda **kw: SimpleNamespace(**{**vars(request), **kw})

    async def handler(req):
        assert "reason" not in req.tool_call["args"]
        return await projected.ainvoke(req.tool_call["args"])

    assert asyncio.run(middleware.awrap_tool_call(request, handler)) == "hello"
    assert calls == ["hello"]
    assert approvals[0].reason == reason
    assert approvals[0].arguments == {"value": "hello"}


def test_reason_survives_risk_increasing_edit():
    seen = []

    def approve(request):
        seen.append(request)
        return (
            ApprovalDecision.edit({"value": ".env"}) if len(seen) == 1 else ApprovalDecision.allow()
        )

    tool = _spec([], resolver=lambda args: "sensitive" if args["value"] == ".env" else "mutating")
    result = asyncio.run(
        resolve_tool_arguments_async(
            tool,
            {"value": "safe"},
            policy=ApprovalPolicy.dynamic(approve),
            adapter_shortname="test",
            reason="Original purpose",
            tool_reason_enabled=True,
        )
    )
    assert result == {"value": ".env"}
    assert [r.reason for r in seen] == ["Original purpose", "Original purpose"]
    assert [r.risk for r in seen] == ["mutating", "sensitive"]


def test_langchain_read_only_retains_business_reason_argument():
    def read(reason: str) -> str:
        return reason

    native = _langchain_tool_from_spec(ToolSpec(read, "read", risk="read_only"))
    assert tools_with_reason([native], [ToolSpec(read, "read", risk="read_only")]) == [native]


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
def test_real_mcp_server_never_receives_reason(backend):
    requests = []

    def approve(request):
        requests.append(request)
        return ApprovalDecision.allow()

    profile = AgentProfile(
        approval=ApprovalPolicy.dynamic(approve),
        mcp_servers=(
            McpServerSpec(
                command=sys.executable,
                args=(str(Path(__file__).parent / "fixtures" / "reason_mcp_server.py"),),
            ),
        ),
    )
    if backend == "pydantic":
        step = 0

        def respond(messages, info):
            nonlocal step
            step += 1
            tool = info.function_tools[0]
            assert "reason" in tool.parameters_json_schema["properties"]
            if step == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool.name,
                            {"value": "hello", "reason": "Requested note"},
                            tool_call_id="mcp-call",
                        )
                    ]
                )
            returns = [
                p
                for m in messages
                for p in m.parts
                if getattr(p, "part_kind", None) == "tool-return"
            ]
            assert returns and "hello" in str(returns[-1].content)
            assert "Approval metadata leaked" not in str(returns[-1].content)
            return ModelResponse(parts=[TextPart("done")])

        adapter = PydanticAdapter(FunctionModel(respond), tool_reason=True)
        with adapter.profile_override(profile):
            assert adapter.run_sync("write").output == "done"
    else:
        adapter = LangChainAdapter(tool_reason=True)

        async def run():
            with adapter.profile_override(profile):
                async with adapter._projected_tools_for_run() as projection:
                    tool = projection.tools[0]
                    assert "reason" in tool.args_schema["properties"]
                    middleware = adapter._approval_middleware_for_run(projection.tools)[0]
                    req = SimpleNamespace(
                        tool=tool,
                        tool_call={
                            "name": tool.name,
                            "id": "mcp-call",
                            "args": {"value": "hello", "reason": "Requested note"},
                        },
                    )
                    req.override = lambda **kw: SimpleNamespace(**{**vars(req), **kw})

                    async def handler(request):
                        assert request.tool_call["args"] == {"value": "hello"}
                        return await tool.ainvoke(request.tool_call["args"])

                    result = await middleware.awrap_tool_call(req, handler)
                    assert "hello" in str(result)
                    assert "Approval metadata leaked" not in str(result)

        asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].reason == "Requested note"
    assert requests[0].arguments == {"value": "hello"}


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
def test_codemode_nested_reason(backend):
    calls, requests = [], []

    def approve(request):
        requests.append(request)
        return ApprovalDecision.allow()

    profile = AgentProfile(approval=ApprovalPolicy.dynamic(approve))
    if backend == "pydantic":
        step = 0

        def respond(messages, info):
            nonlocal step
            step += 1
            if step == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "get_tool_schema", {"tool_names": ["write"]}, tool_call_id="schema"
                        )
                    ]
                )
            if step == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "execute_code",
                            {"code": "await write(value='hello', reason='Requested note')"},
                            tool_call_id="exec",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("done")])

        adapter = PydanticAdapter(FunctionModel(respond), tool_reason=True, codemode=True)
        with adapter.profile_override(profile), adapter.tool_scope([_spec(calls)]):
            adapter.run_sync("write")
    else:
        adapter = LangChainAdapter(tool_reason=True, codemode=True)

        async def run():
            with adapter.profile_override(profile), adapter.tool_scope([_spec(calls)]):
                async with adapter._projected_tools_for_run() as projection:
                    projection.bridge.get_tool_schema(tool_names=["write"])
                    await projection.bridge.execute_code(
                        code="await write(value='hello', reason='Requested note')"
                    )

        asyncio.run(run())
    assert calls == ["hello"]
    assert len(requests) == 1
    assert requests[0].reason == "Requested note"


def test_pydantic_concurrent_runs_and_calls_keep_reasons_isolated():
    calls, seen = [], []

    async def approve(request):
        await asyncio.sleep(0)
        seen.append((request.arguments["value"], request.reason))
        return ApprovalDecision.allow()

    async def respond(messages, info):
        if any(getattr(p, "part_kind", None) == "tool-return" for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("done")])
        prompt = next(
            p.content
            for m in messages
            for p in m.parts
            if getattr(p, "part_kind", None) == "user-prompt"
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "write",
                    {"value": f"{prompt}-{i}", "reason": f"reason-{prompt}-{i}"},
                    tool_call_id=f"call-{i}",
                )
                for i in range(2)
            ]
        )

    adapter = PydanticAdapter(FunctionModel(respond), tool_reason=True)

    async def run(prompt):
        with (
            adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))),
            adapter.tool_scope([_spec(calls)]),
        ):
            assert (await adapter.run(prompt)).output == "done"

    async def both():
        await asyncio.gather(run("first"), run("second"))

    asyncio.run(both())
    assert sorted(seen) == [
        (f"{p}-{i}", f"reason-{p}-{i}") for p in ("first", "second") for i in range(2)
    ]


def test_pydantic_external_approval_preserves_reason_in_metadata():
    from pydantic_ai.tools import DeferredToolRequests

    from kedi.agent_adapter import (
        pydantic_approval_request_from_deferred,
        pydantic_deferred_results_from_decisions,
    )

    calls = []

    def respond(messages, info):
        if any(getattr(p, "part_kind", None) == "tool-return" for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "write", {"value": "hello", "reason": "Requested note"}, tool_call_id="external"
                )
            ]
        )

    spec = _spec(calls)
    adapter = PydanticAdapter(
        FunctionModel(respond), tool_reason=True, approval_resolution="external"
    )
    with adapter.tool_scope([spec]):
        result = adapter.run_sync("write", output_type=[str, DeferredToolRequests])
        request = pydantic_approval_request_from_deferred(result.output, "external")
        assert request.reason == "Requested note"
        assert request.arguments == {"value": "hello"}
        assert request.tool_reason_enabled
        decisions = pydantic_deferred_results_from_decisions(
            result.output, {"external": True}, tool_specs=[spec]
        )
        final = adapter.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=decisions,
            output_type=[str, DeferredToolRequests],
        )
        assert final.output == "done"
    assert calls == ["hello"]


def test_langchain_full_agent_run():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    calls, seen = [], []

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            assert tools[0].args_schema["properties"]["reason"] == {"type": "string"}
            return self

    model = Model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write",
                        "args": {"value": "hello", "reason": "Requested note"},
                        "id": "lc-call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )

    def approve(request):
        seen.append(request)
        return ApprovalDecision.allow()

    adapter = LangChainAdapter(model, tool_reason=True)

    async def run():
        with (
            adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))),
            adapter.tool_scope([_spec(calls)]),
        ):
            return await adapter.invoke(prompt="write")

    assert asyncio.run(run()) == "done"
    assert calls == ["hello"]
    assert seen[0].reason == "Requested note"


@pytest.mark.parametrize("dsl_setting", [False, True])
def test_real_kedi_program_uses_reason_with_llm_approval(dsl_setting):
    from kedi.lang import compile_program, parse_program

    judge_prompts = []

    def respond(messages, info):
        prompt = str(messages)
        if info.output_tools:
            judge_prompts.append(prompt)
            assert "Requested note" in prompt
            return ModelResponse(
                parts=[
                    ToolCallPart(info.output_tools[0].name, {"risk": "low", "decision": "allow"})
                ]
            )
        if any(getattr(p, "part_kind", None) == "tool-return" for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "write_note",
                    {"content": "hello", "reason": "Requested note"},
                    tool_call_id="write",
                )
            ]
        )

    adapter = PydanticAdapter(FunctionModel(respond), tool_reason=not dsl_setting)
    source = """
> import: helpers

@write_note(content: str) -> str:
    = <content>

> approval: `llm_approval`
> use: write_note
[result] << Write a note.
= <result>
"""
    if dsl_setting:
        source = "> settings:\n    tool_reason: enabled\n" + source
    runtime = compile_program(parse_program(source), adapter=adapter)
    assert runtime.run_main() == "done"
    assert len(judge_prompts) == 1
