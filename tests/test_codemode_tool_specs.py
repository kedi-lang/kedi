from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any

import fastmcp
import pytest

from kedi.agent_adapter.approvals.policy import ApprovalDecision, ApprovalPolicy
from kedi.agent_adapter.codemode import CodeModeHydrationError, CodeModeLimits
from kedi.agent_adapter.codemode import mcp as codemode_mcp_module
from kedi.agent_adapter.codemode.tool_specs import (
    CONTROL_TOOL_NAMES,
    CodeModeProjection,
    CodeModeToolSpecBridge,
    append_codemode_instructions,
    approval_policies,
    codemode_projection,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import McpServerSpec


def test_tool_spec_bridge_exposes_only_uniform_controls() -> None:
    bridge = CodeModeToolSpecBridge(
        [
            ToolSpec(
                fn=lambda *, value: value,
                name="echo",
                json_schema={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            )
        ],
        limits=CodeModeLimits(),
        policies=(ApprovalPolicy.allow(),),
        adapter_shortname="test",
    )

    controls = {tool.name: tool for tool in bridge.control_tools}
    assert controls.keys() == CONTROL_TOOL_NAMES
    assert "notebook cell" in controls["execute_code"].description
    assert "variables persist across calls" in controls["execute_code"].description
    assert "output and the error in separate fields" in controls["execute_code"].description
    assert bridge.search_tools()["tool_names"] == ["echo"]
    assert bridge.get_tool_schema(tool_names=["echo"])["tools"][0]["name"] == "echo"
    asyncio.run(bridge.close())


@pytest.mark.parametrize(
    "preload_tools", [("lookup_release",), ("lookup_release", "lookup_release")]
)
def test_tool_spec_bridge_preloads_exact_exposed_name_without_running_body(preload_tools) -> None:
    calls = 0

    def lookup(*, version: str) -> str:
        nonlocal calls
        calls += 1
        return version

    bridge = CodeModeToolSpecBridge(
        [
            ToolSpec(
                fn=lookup,
                name="lookup_release",
                description="Read release metadata.",
                json_schema={
                    "type": "object",
                    "properties": {"version": {"type": "string"}},
                    "required": ["version"],
                },
                metadata={"kedi_source_name": "read_release"},
            )
        ],
        limits=CodeModeLimits(),
        policies=(ApprovalPolicy.allow(),),
        adapter_shortname="test",
        preload_tools=preload_tools,
    )

    assert calls == 0
    assert len(bridge.preloaded_tools) == 1
    assert bridge.preloaded_tools[0]["name"] == "lookup_release"
    assert bridge.preloaded_tools[0]["callable_name"] == "lookup_release"
    instructions = append_codemode_instructions(
        None,
        enabled=True,
        preloaded_tools=bridge.preloaded_tools,
    )
    assert instructions is not None
    assert '"description":"Read release metadata."' in instructions
    assert '"version":{"type":"string"}' in instructions

    async def run() -> None:
        assert await bridge.execute_code(code='await lookup_release(version="v1")') == "v1"
        await bridge.close()

    asyncio.run(run())
    assert calls == 1

    with pytest.raises(CodeModeHydrationError, match="Unknown"):
        CodeModeToolSpecBridge(
            [ToolSpec(fn=lookup, name="lookup_release")],
            limits=CodeModeLimits(),
            policies=(),
            adapter_shortname="test",
            preload_tools=("read_release",),
        )


def test_codemode_preload_is_run_local_and_respects_hydration_limit() -> None:
    first = CodeModeToolSpecBridge(
        [ToolSpec(lambda: "a", "alpha"), ToolSpec(lambda: "b", "beta")],
        limits=CodeModeLimits(max_hydrated_tools=1),
        policies=(),
        adapter_shortname="test",
        preload_tools=("alpha",),
    )
    second = CodeModeToolSpecBridge(
        [ToolSpec(lambda: "a", "alpha"), ToolSpec(lambda: "b", "beta")],
        limits=CodeModeLimits(max_hydrated_tools=1),
        policies=(),
        adapter_shortname="test",
        preload_tools=("beta",),
    )
    assert [tool["name"] for tool in first.preloaded_tools] == ["alpha"]
    assert [tool["name"] for tool in second.preloaded_tools] == ["beta"]
    with pytest.raises(CodeModeHydrationError, match="at most 1"):
        CodeModeToolSpecBridge(
            [ToolSpec(lambda: "a", "alpha"), ToolSpec(lambda: "b", "beta")],
            limits=CodeModeLimits(max_hydrated_tools=1),
            policies=(),
            adapter_shortname="test",
            preload_tools=("alpha", "beta"),
        )
    asyncio.run(first.close())
    asyncio.run(second.close())


def test_tool_spec_bridge_approves_edits_once_and_executes_nested_call_once() -> None:
    requests: list[dict[str, Any]] = []
    calls: list[int] = []

    def decide(request: Any) -> ApprovalDecision:
        requests.append(dict(request.arguments))
        return ApprovalDecision.edit({"value": 7})

    def validate(arguments: Any) -> dict[str, Any]:
        return {"value": int(arguments["value"])}

    def record(*, value: int) -> int:
        calls.append(value)
        return value * 2

    async def run() -> None:
        bridge = CodeModeToolSpecBridge(
            [
                ToolSpec(
                    fn=record,
                    name="record",
                    json_schema={"type": "object"},
                    argument_validator=validate,
                )
            ],
            limits=CodeModeLimits(),
            policies=(ApprovalPolicy.dynamic(decide),),
            adapter_shortname="test",
        )
        bridge.get_tool_schema(tool_names=["record"])
        assert await bridge.execute_code(code="await record(value='3')") == 14
        await bridge.close()

    asyncio.run(run())
    assert requests == [{"value": 3}]
    assert calls == [7]


def test_tool_spec_bridge_denial_does_not_execute_tool() -> None:
    calls = 0

    def mutate() -> str:
        nonlocal calls
        calls += 1
        return "called"

    async def run() -> None:
        bridge = CodeModeToolSpecBridge(
            [ToolSpec(fn=mutate, name="mutate")],
            limits=CodeModeLimits(),
            policies=(ApprovalPolicy.deny(),),
            adapter_shortname="test",
        )
        bridge.get_tool_schema(tool_names=["mutate"])
        with pytest.raises(Exception, match="denied"):
            await bridge.execute_code(code="await mutate()")
        await bridge.close()

    asyncio.run(run())
    assert calls == 0


def test_tool_spec_bridge_rejects_duplicate_and_split_artifact_boundaries() -> None:
    with pytest.raises(ValueError, match="Duplicate CodeMode tool names"):
        CodeModeToolSpecBridge(
            [ToolSpec(lambda: None, "same"), ToolSpec(lambda: None, "same")],
            limits=CodeModeLimits(),
            policies=(),
            adapter_shortname="test",
        )

    async def first(value: Any, _name: str) -> Any:
        return value

    async def second(value: Any, _name: str) -> Any:
        return value

    setattr(first, "_kedi_artifact_boundary_key", "first")
    setattr(second, "_kedi_artifact_boundary_key", "second")
    with pytest.raises(ValueError, match="multiple artifact result boundaries"):
        CodeModeToolSpecBridge(
            [
                ToolSpec(lambda: None, "first", result_admitter=first),
                ToolSpec(lambda: None, "second", result_admitter=second),
            ],
            limits=CodeModeLimits(),
            policies=(),
            adapter_shortname="test",
        )


def test_tool_spec_projection_disabled_close_and_final_admission() -> None:
    tool = ToolSpec(lambda: 4, "value")
    disabled = codemode_projection(
        [tool],
        limits=None,
        policies=(),
        adapter_shortname="test",
    )
    assert disabled == CodeModeProjection(tools=(tool,))
    asyncio.run(disabled.close())

    admitted: list[tuple[Any, str]] = []

    async def admit(value: Any, name: str) -> Any:
        admitted.append((value, name))
        return {"admitted": value}

    async def run() -> None:
        bridge = CodeModeToolSpecBridge(
            [ToolSpec(lambda: 4, "value", result_admitter=admit)],
            limits=CodeModeLimits(),
            policies=(),
            adapter_shortname="test",
        )
        bridge.get_tool_schema(tool_names=["value"])
        assert await bridge.execute_code(code="await value()") == {"admitted": 4}
        await bridge.close()

    asyncio.run(run())
    assert admitted == [(4, "execute_code")]

    with pytest.raises(ValueError, match="reserves tool name"):
        CodeModeToolSpecBridge(
            [ToolSpec(lambda: None, "search_tools")],
            limits=CodeModeLimits(),
            policies=(),
            adapter_shortname="test",
        )


def test_codemode_instruction_and_policy_helpers_are_stable() -> None:
    assert append_codemode_instructions("base", enabled=False) == "base"
    enabled = append_codemode_instructions("base", enabled=True)
    assert enabled is not None and enabled.startswith("base\n\n")
    assert "mapping[key]" in enabled
    assert ".get()" in enabled
    assert 'orders = await list_orders(user_id=user["id"])' in enabled
    assert "await asyncio.gather(...)" in enabled
    assert "persistent notebook cells" in enabled
    assert "Never repeat a successful application tool call" in enabled
    assert "printing or returning a variable does not consume it" in enabled
    assert "separate `output` and `error` fields" in enabled
    ceiling = ApprovalPolicy.deny()
    local = ApprovalPolicy.allow()
    assert approval_policies(local, (ceiling,)) == (ceiling, local)
    assert approval_policies(None, (ceiling,)) == (ceiling,)
    assert approval_policies(None, ()) == (None,)


def test_codemode_mcp_materializer_owns_client_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeClient:
        def __init__(self, config: dict[str, Any]) -> None:
            assert config["mcpServers"]["kedi_codemode_0"]["command"] == "fake-mcp"

        async def __aenter__(self) -> FakeClient:
            events.append("enter")
            return self

        async def __aexit__(self, *_args: Any) -> None:
            events.append("exit")

        async def list_tools(self) -> list[Any]:
            return [
                SimpleNamespace(
                    name="mcp_lookup",
                    description="MCP lookup.",
                    inputSchema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                    outputSchema={"type": "object"},
                )
            ]

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            assert name == "mcp_lookup"
            return SimpleNamespace(
                isError=False,
                structuredContent={"query": arguments["query"]},
                content=[],
            )

    monkeypatch.setattr(fastmcp, "Client", FakeClient)

    async def run() -> None:
        async with AsyncExitStack() as stack:
            tools = await codemode_mcp_module.materialize_mcp_tool_specs(
                stack,
                [McpServerSpec(command="fake-mcp")],
                adapter_shortname="claude",
            )
            assert len(tools) == 1
            assert await tools[0].fn(query="kedi") == {"query": "kedi"}
            assert events == ["enter"]
        assert events == ["enter", "exit"]

    asyncio.run(run())


def test_codemode_mcp_configs_and_result_normalization() -> None:
    from mcp.types import TextContent

    assert codemode_mcp_module._fastmcp_config(
        McpServerSpec(command="server", args=("--stdio",), env={"A": "B"}),
        1,
    ) == {
        "mcpServers": {
            "kedi_codemode_1": {
                "command": "server",
                "args": ["--stdio"],
                "env": {"A": "B"},
            }
        }
    }
    assert codemode_mcp_module._fastmcp_config(
        McpServerSpec(
            transport="sse",
            url="https://example.test/sse",
            headers={"X-Test": "yes"},
        ),
        2,
    )["mcpServers"]["kedi_codemode_2"] == {
        "transport": "sse",
        "url": "https://example.test/sse",
        "headers": {"X-Test": "yes"},
    }
    assert codemode_mcp_module._fastmcp_config(
        McpServerSpec(transport="http", url="https://example.test/mcp"),
        3,
    )["mcpServers"]["kedi_codemode_3"] == {
        "transport": "streamable-http",
        "url": "https://example.test/mcp",
    }

    results = [
        SimpleNamespace(isError=False, structuredContent={"value": 3}, content=[]),
        SimpleNamespace(
            isError=False,
            structuredContent=None,
            content=[
                TextContent(type="text", text="plain"),
                SimpleNamespace(model_dump=lambda **_kwargs: {"type": "image"}),
            ],
        ),
        SimpleNamespace(
            isError=True,
            structuredContent=None,
            content=[TextContent(type="text", text="failed")],
        ),
    ]

    class Client:
        async def call_tool(self, _name: str, _arguments: dict[str, Any]) -> Any:
            return results.pop(0)

    spec = codemode_mcp_module._tool_spec(
        Client(),
        SimpleNamespace(
            name="lookup",
            description=None,
            inputSchema={"type": "object"},
            outputSchema=None,
        ),
    )
    assert spec.risk_resolver is not None
    assert spec.risk_resolver({"path": ".env"}) == "sensitive"
    assert spec.risk_resolver({"path": "README.md"}) == "mutating"

    async def run() -> None:
        assert await spec.fn() == {"value": 3}
        assert await spec.fn() == 'plain\n{"type": "image"}'
        with pytest.raises(RuntimeError, match="failed"):
            await spec.fn()

    asyncio.run(run())
