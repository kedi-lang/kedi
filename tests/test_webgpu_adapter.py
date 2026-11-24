from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Iterator

import pytest
from pydantic import BaseModel, ValidationError

import kedi.agent_adapter.adapters.webgpu as webgpu_module
from kedi.agent_adapter.adapters.webgpu import WebGPUAdapter
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, McpServerSpec

WIKI_SERVER = Path(__file__).parent / "files" / "mcp_wiki_server.py"


class QueueBridge:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []
        self.timeouts: list[float] = []

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.payloads.append(payload)
        self.timeouts.append(timeout)
        return self.responses.pop(0)


class NamedOutput(BaseModel):
    answer: str


def test_webgpu_adapter_produces_schema_and_output_type() -> None:
    bridge = QueueBridge(
        {"kind": "final", "data": {"answer": "42"}},
        {"data": {"answer": "done"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai", supported_models=("bonsai",))

    assert adapter.default_model() == "bonsai"
    dynamic = adapter.produce_sync(
        template="answer",
        output_schema={"answer": Annotated[int, "Computed answer"]},
    )
    typed = asyncio.run(adapter.produce(template="answer", output_type=NamedOutput))

    assert dynamic.answer == 42
    assert typed == NamedOutput(answer="done")
    assert bridge.payloads[0]["outputSchema"]["properties"]["answer"]["type"] == "integer"
    assert bridge.payloads[0]["outputSchema"]["properties"]["answer"]["description"] == (
        "Computed answer"
    )
    output_prompt = bridge.payloads[0]["outputPrompt"]
    assert "<name>answer</name>" in output_prompt
    assert "<type>integer</type>" in output_prompt
    assert "<answer>integer</answer>" not in output_prompt
    assert "properties" not in output_prompt
    assert "required" not in output_prompt
    assert "title" not in output_prompt


def test_webgpu_adapter_invoke_returns_text_response() -> None:
    bridge = QueueBridge(
        {"kind": "final", "text": "plain text"},
        {"kind": "final", "data": "data text"},
        {"kind": "final", "data": {"text": "nested text"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")

    assert adapter.invoke_sync(prompt="say hi") == "plain text"
    assert adapter.invoke_sync(prompt="say hi") == "data text"
    assert asyncio.run(adapter.invoke(prompt="say hi")) == "nested text"
    assert all(payload["responseMode"] == "text" for payload in bridge.payloads)
    assert all("outputSchema" not in payload for payload in bridge.payloads)


def test_webgpu_adapter_validates_construction_and_output_contract() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        WebGPUAdapter(QueueBridge(), model="bonsai", max_tool_steps=0)
    with pytest.raises(ValueError, match="Unknown browser model"):
        WebGPUAdapter(QueueBridge(), model="missing", supported_models=("bonsai",))

    adapter = WebGPUAdapter(QueueBridge(), model="anything")
    with pytest.raises(ValueError, match="Output spec"):
        adapter.produce_sync(template="answer")
    with pytest.raises(TypeError, match="text response"):
        WebGPUAdapter(QueueBridge({"kind": "final", "data": {}}), model="anything").invoke_sync(
            prompt="answer"
        )
    with pytest.raises(TypeError, match="text response"):
        WebGPUAdapter(QueueBridge({"kind": "final", "data": 1}), model="anything").invoke_sync(
            prompt="answer"
        )


def test_webgpu_adapter_profile_scope_and_settings() -> None:
    bridge = QueueBridge({"kind": "final", "data": {"answer": "ok"}})
    adapter = WebGPUAdapter(
        bridge,
        model="bonsai",
        supported_models=("bonsai", "ternary"),
        model_settings={"top_p": 0.8, "seed": 7},
        timeout=40,
    )
    profile = AgentProfile(
        model="ternary",
        effort="high",
        system="Be exact.",
        settings={"temperature": 0.2, "timeout": 12},
    )

    with adapter.profile_override(profile):
        assert adapter.active_profile is profile
        result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "ok"
    assert adapter.active_profile == AgentProfile.empty()
    assert bridge.timeouts == [12]
    payload = bridge.payloads[0]
    assert payload["model"] == "ternary"
    assert payload["instructions"] == "Be exact."
    assert payload["effort"] == "high"
    assert payload["settings"] == {
        "temperature": 0.2,
        "top_p": 0.8,
        "seed": 7,
    }


def test_webgpu_adapter_capabilities_declare_only_supported_settings() -> None:  # 3.13
    from kedi.agent_adapter.adapters.webgpu import WEBGPU_SETTING_KEYS

    assert WebGPUAdapter.capabilities.settings == {
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop_sequences",
        "timeout",
    }
    assert WebGPUAdapter.capabilities.settings == WEBGPU_SETTING_KEYS

    adapter = WebGPUAdapter(QueueBridge(), model="bonsai")
    with adapter.profile_override(AgentProfile(settings={"thinking": True})):
        with pytest.raises(
            ValueError,
            match="Unsupported WebGPU model settings: thinking",
        ):
            adapter.produce_sync(template="answer", output_schema={"answer": str})


def test_webgpu_adapter_profile_rejects_unknown_model_and_setting() -> None:
    adapter = WebGPUAdapter(
        QueueBridge(),
        model="bonsai",
        supported_models=("bonsai",),
    )
    with pytest.raises(ValueError, match="Unknown browser model"):
        with adapter.profile_override(AgentProfile(model="missing")):
            pass
    with adapter.profile_override(AgentProfile(settings={"presence_penalty": 1})):
        with pytest.raises(
            ValueError,
            match="Unsupported WebGPU model settings: presence_penalty",
        ):
            adapter.produce_sync(template="answer", output_schema={"answer": str})


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"kind": "final", "data": "nope"}, "Input should be a valid dictionary"),
        ({"kind": "error", "error": "browser failed"}, "browser failed"),
        ({"kind": "other"}, "Unknown browser model response kind"),
    ],
)
def test_webgpu_adapter_rejects_bad_browser_responses(
    response: dict[str, Any],
    message: str,
) -> None:
    adapter = WebGPUAdapter(QueueBridge(response), model="bonsai")
    with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
        adapter.produce_sync(template="answer", output_schema={"answer": str})


def test_webgpu_adapter_executes_sync_and_async_tools() -> None:
    bridge = QueueBridge(
        {"kind": "tool_call", "name": "lookup", "arguments": {"query": "kedi"}},
        {"kind": "tool_call", "name": "async_lookup", "arguments": {"query": "docs"}},
        {"kind": "final", "data": {"answer": "found"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")

    async def async_lookup(query: str) -> str:
        return f"async:{query}"

    tools = (
        ToolSpec(
            fn=lambda query: f"sync:{query}",
            name="lookup",
            description="Lookup",
            json_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        ToolSpec(fn=async_lookup, name="async_lookup"),
    )
    with adapter.tool_scope(tools):
        result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "found"
    assert adapter.scoped_tools == ()
    assert bridge.payloads[0]["tools"][0]["description"] == "Lookup"
    assert '"result": "sync:kedi"' in bridge.payloads[1]["messages"][-1]["content"]
    assert '"result": "async:docs"' in bridge.payloads[2]["messages"][-1]["content"]


def test_webgpu_adapter_instruments_tool_execution() -> None:
    events: list[tuple[str, Any]] = []

    class Instrumentation:
        @contextmanager
        def tool_span(
            self,
            name: str,
            call_id: str,
            arguments: dict[str, Any],
        ) -> Iterator[Any]:
            events.append(("start", (name, call_id, arguments)))
            yield lambda result: events.append(("finish", result))

    bridge = QueueBridge(
        {
            "kind": "tool_call",
            "name": "lookup",
            "callId": "call-1",
            "arguments": {"query": "kedi"},
        },
        {"kind": "final", "data": {"answer": "found"}},
    )
    adapter = WebGPUAdapter(
        bridge,
        model="bonsai",
        instrumentation=Instrumentation(),
    )

    with adapter.tool_scope((ToolSpec(fn=lambda query: {"query": query}, name="lookup"),)):
        result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "found"
    assert events == [
        ("start", ("lookup", "call-1", {"query": "kedi"})),
        ("finish", {"query": "kedi"}),
    ]
    assert '"callId": "call-1"' in bridge.payloads[1]["messages"][-2]["content"]


def test_webgpu_adapter_requires_explicitly_named_tool_before_final() -> None:
    bridge = QueueBridge(
        {"kind": "final", "data": {"answer": "simulated"}},
        {"kind": "tool_call", "name": "say_hello", "arguments": {"name": "Mert"}},
        {"kind": "final", "data": {"answer": "done"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")
    tool = ToolSpec(
        fn=lambda name: f"Hello, {name}!",
        name="say_hello",
        json_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )

    with adapter.tool_scope((tool,)):
        result = adapter.produce_sync(
            template="Call the say_hello tool for Mert and return [answer].",
            output_schema={"answer": str},
        )

    assert result.answer == "done"
    assert bridge.payloads[0]["requiredTools"] == ["say_hello"]
    assert "Call the explicitly requested tool" in bridge.payloads[1]["messages"][-1]["content"]
    assert bridge.payloads[2]["requiredTools"] == []
    assert '"result": "Hello, Mert!"' in bridge.payloads[2]["messages"][-1]["content"]


def test_webgpu_adapter_retries_invalid_tool_protocol_and_requires_declared_tool() -> None:
    bridge = QueueBridge(
        {
            "kind": "retry",
            "requiredTool": "get_capital",
            "error": "Call get_capital using tool_call JSON.",
            "telemetry": {
                "outputText": "We should use get_capital.",
            },
        },
        {
            "kind": "tool_call",
            "name": "get_capital",
            "arguments": {"country": "Miyawland"},
        },
        {"kind": "final", "data": {"capital": "Miyawcity"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")

    with adapter.tool_scope(
        (
            ToolSpec(
                fn=lambda country: "Miyawcity",
                name="get_capital",
                json_schema={
                    "type": "object",
                    "properties": {"country": {"type": "string"}},
                    "required": ["country"],
                },
            ),
        )
    ):
        result = adapter.produce_sync(
            template="What's the capital of Miyawland?",
            output_schema={"capital": str},
        )

    assert result.capital == "Miyawcity"
    assert bridge.payloads[0]["requiredTools"] == []
    assert bridge.payloads[1]["requiredTools"] == ["get_capital"]
    assert bridge.payloads[1]["messages"][-2:] == [
        {"role": "assistant", "content": "We should use get_capital."},
        {
            "role": "user",
            "content": ("Call get_capital using tool_call JSON."),
        },
    ]
    assert bridge.payloads[2]["requiredTools"] == []
    assert '"result": "Miyawcity"' in bridge.payloads[2]["messages"][-1]["content"]


def test_webgpu_adapter_rejects_retry_for_unknown_tool() -> None:
    adapter = WebGPUAdapter(
        QueueBridge({"kind": "retry", "requiredTool": "missing"}),
        model="bonsai",
    )

    with pytest.raises(ValueError, match="retry named unknown tool"):
        adapter.produce_sync(template="answer", output_schema={"answer": str})


def test_webgpu_adapter_retries_without_optional_tool_metadata() -> None:
    bridge = QueueBridge(
        {"kind": "retry"},
        {"kind": "retry", "telemetry": {"outputText": ""}},
        {"kind": "final", "data": {"answer": "done"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")

    result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "done"
    assert bridge.payloads[1]["messages"][-1]["content"] == (
        "Return exactly one call_tool or final action JSON object."
    )
    assert bridge.payloads[2]["messages"][-1]["content"] == (
        "Return exactly one call_tool or final action JSON object."
    )


def test_webgpu_adapter_registers_tool_in_active_scope() -> None:
    bridge = QueueBridge(
        {"kind": "tool_call", "name": "lookup", "arguments": {}},
        {"kind": "final", "data": {"answer": "found"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")

    with adapter.tool_scope():
        adapter.register_tool(ToolSpec(fn=lambda: "registered", name="lookup"))
        result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "found"
    assert '"result": "registered"' in bridge.payloads[1]["messages"][-1]["content"]


def test_webgpu_adapter_returns_tool_failure_to_model() -> None:
    def lookup(capital: str) -> int:
        return 12_560_000

    bridge = QueueBridge(
        {
            "kind": "tool_call",
            "name": "get_capital_population",
            "arguments": {"country": "Miyawland"},
        },
        {"kind": "final", "data": {"population": 2_000_000}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")
    tool = ToolSpec(
        fn=lookup,
        name="get_capital_population",
        json_schema={
            "type": "object",
            "properties": {"capital": {"type": "string"}},
            "required": ["capital"],
        },
    )
    with adapter.tool_scope((tool, "ignored")):
        result = adapter.produce_sync(
            template="Find the population.",
            output_schema={"population": int},
        )

    # The failure is returned to the model, but the failed tool is not forced
    # on the model again: the next final answer is accepted as-is. (3.9)
    assert result.population == 2_000_000
    assert '"ok": false' in bridge.payloads[1]["messages"][-1]["content"]
    assert "unexpected keyword argument 'country'" in bridge.payloads[1]["messages"][-1]["content"]
    assert bridge.payloads[1]["requiredTools"] == []
    assert len(bridge.payloads) == 2
    assert all(
        "Call the explicitly requested tool" not in message.get("content", "")
        for payload in bridge.payloads
        for message in payload["messages"]
    )


def test_webgpu_adapter_validates_non_dict_final_data() -> None:  # 3.9
    bridge = QueueBridge({"kind": "final", "data": ["a", "b"]})
    adapter = WebGPUAdapter(bridge, model="bonsai")
    assert adapter.produce_sync(template="List two.", output_type=list[str]) == ["a", "b"]

    failing = WebGPUAdapter(QueueBridge({"kind": "final", "data": "nope"}), model="bonsai")
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        failing.produce_sync(template="List two.", output_type=list[str])


def test_webgpu_adapter_formats_tool_catalog_as_xml() -> None:
    bridge = QueueBridge({"kind": "final", "data": {"answer": "ok"}})
    adapter = WebGPUAdapter(bridge, model="bonsai")
    tool = ToolSpec(
        fn=lambda country: country,
        name="get_capital",
        json_schema={
            "type": "object",
            "properties": {"country": {"title": "Country", "type": "string"}},
            "required": ["country"],
        },
    )

    with adapter.tool_scope((tool,)):
        result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "ok"
    catalog = bridge.payloads[0]["toolsPrompt"]
    assert catalog.startswith("<Tools>\n  <Tool>")
    assert "<name>get_capital</name>" in catalog
    assert "<arguments>" in catalog
    assert "<name>country</name>" in catalog
    assert "<type>string</type>" in catalog
    assert "<country>string</country>" not in catalog
    assert "inputSchema" not in catalog
    assert "properties" not in catalog
    assert "required" not in catalog
    assert "title" not in catalog
    assert catalog.endswith("</Tool>\n</Tools>")


def test_webgpu_adapter_formats_schema_variants_as_xml() -> None:
    output = webgpu_module._format_output_contract(
        {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
                "status": {"enum": ["ready", "pending"]},
                "ignored": "invalid",
            },
            "required": ["item"],
        }
    )
    catalog = webgpu_module._format_tool_catalog(
        (
            {
                "name": "inspect",
                "description": "Inspect values",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "integer"}},
                        "filter": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        },
                        "payload": {"type": "object", "title": "Payload"},
                    },
                    "required": ["values"],
                },
                "returnSchema": {
                    "oneOf": [{"type": "boolean"}, {"type": "null"}],
                },
            },
        )
    )

    assert "<name>item</name>" in output
    assert "<type>Item</type>" in output
    assert "<name>status</name>" in output
    assert "<type>ready | pending</type>" in output
    assert "<name>values</name>" in catalog
    assert "<type>list[integer]</type>" in catalog
    assert "<name>filter</name>" in catalog
    assert "<type>string | null</type>" in catalog
    assert "<name>payload</name>" in catalog
    assert "<type>Payload</type>" in catalog
    assert "<returns>boolean | null</returns>" in catalog
    assert webgpu_module._format_output_contract({"type": ["string", "null"]}) == (
        "<Output>\n  <type>string | null</type>\n</Output>"
    )
    assert webgpu_module._schema_type_name({"type": "array"}, {}) == "list[value]"
    assert webgpu_module._schema_type_name({"type": "object"}, {}) == "object"
    assert webgpu_module._schema_type_name({"title": "Untyped"}, {}) == "Untyped"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"kind": "tool_call", "name": 1}, "name must be a string"),
        (
            {"kind": "tool_call", "name": "tool", "arguments": []},
            "arguments must be an object",
        ),
        ({"kind": "tool_call", "name": "missing"}, "unknown tool"),
    ],
)
def test_webgpu_adapter_validates_tool_calls(
    response: dict[str, Any],
    message: str,
) -> None:
    adapter = WebGPUAdapter(QueueBridge(response), model="bonsai")
    with adapter.tool_scope((ToolSpec(fn=lambda: None, name="tool"),)):
        with pytest.raises((TypeError, ValueError), match=message):
            adapter.produce_sync(template="answer", output_schema={"answer": str})


def test_webgpu_adapter_enforces_tool_step_limit() -> None:
    bridge = QueueBridge({"kind": "tool_call", "name": "again"})
    adapter = WebGPUAdapter(bridge, model="bonsai", max_tool_steps=1)
    with adapter.tool_scope((ToolSpec(fn=lambda: "ok", name="again"),)):
        with pytest.raises(RuntimeError, match="1-step tool limit"):
            adapter.produce_sync(template="answer", output_schema={"answer": str})


def test_webgpu_adapter_rejects_duplicate_local_and_mcp_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_mcp_tools(*_: Any, **_kwargs: Any) -> tuple[ToolSpec, ...]:
        return (ToolSpec(fn=lambda: None, name="same", metadata={"source": "mcp"}),)

    monkeypatch.setattr(webgpu_module, "materialize_mcp_tool_specs", fake_mcp_tools)
    adapter = WebGPUAdapter(QueueBridge(), model="bonsai")
    profile = AgentProfile(mcp_servers=(McpServerSpec(command="tool"),))
    with (
        adapter.profile_override(profile),
        adapter.tool_scope((ToolSpec(fn=lambda: None, name="same"),)),
    ):
        with pytest.raises(ValueError, match="unique"):
            adapter.produce_sync(template="answer", output_schema={"answer": str})


def test_webgpu_adapter_rejects_disallowed_mcp_transport() -> None:
    adapter = WebGPUAdapter(
        QueueBridge(),
        model="bonsai",
        allowed_mcp_transports={"http", "sse"},
    )
    with adapter.profile_override(AgentProfile(mcp_servers=(McpServerSpec(command="server"),))):
        with pytest.raises(ValueError, match="does not allow stdio MCP transport"):
            adapter.produce_sync(template="answer", output_schema={"answer": str})


def test_webgpu_adapter_allowed_mcp_policy_accepts_empty_profile() -> None:
    adapter = WebGPUAdapter(
        QueueBridge({"kind": "final", "data": {"answer": "ok"}}),
        model="bonsai",
        allowed_mcp_transports={"http", "sse"},
    )

    result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "ok"


class McpWikiBridge:
    """Call one MCP tool, then answer with the tool result the adapter sent back."""

    def __init__(self, tool: str) -> None:
        self.tool = tool

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        results = [m for m in payload["messages"] if m.get("role") == "tool"]
        if not results:
            arguments = {"page": "Home"}
            return {"kind": "tool_call", "name": self.tool, "arguments": arguments, "callId": "c1"}
        return {"kind": "final", "text": results[-1]["content"]}


@pytest.mark.parametrize(
    ("approval", "deleted"),
    [
        (ApprovalPolicy.deny(), False),
        (None, False),  # MCP tools are mutating, so no policy denies them as on other adapters
        (ApprovalPolicy.allow(), True),
    ],
)
def test_webgpu_mcp_tools_follow_the_profile_approval(
    tmp_path: Path, approval: ApprovalPolicy | None, deleted: bool
) -> None:
    log = tmp_path / "wiki.log"
    server = McpServerSpec(command=sys.executable, args=(str(WIKI_SERVER), str(log)))
    adapter = WebGPUAdapter(McpWikiBridge("delete_page"), model="bonsai")

    with adapter.profile_override(AgentProfile(mcp_servers=(server,), approval=approval)):
        content = json.loads(adapter.invoke_sync(prompt="Delete the Home page."))

    if deleted:
        assert content == {"ok": True, "result": {"result": "deleted Home"}}
        assert log.read_text(encoding="utf-8") == "deleted Home\n"
    else:
        assert content["ok"] is False
        assert content["error"].startswith("ToolApprovalDenied: Tool 'delete_page' was denied")
        assert "Kedi stack trace" not in content["error"]
        assert not log.exists()


def test_webgpu_mcp_tool_errors_reach_the_model(tmp_path: Path) -> None:
    server = McpServerSpec(command=sys.executable, args=(str(WIKI_SERVER), str(tmp_path / "log")))
    adapter = WebGPUAdapter(McpWikiBridge("broken_page"), model="bonsai")

    with adapter.profile_override(
        AgentProfile(mcp_servers=(server,), approval=ApprovalPolicy.allow())
    ):
        content = json.loads(adapter.invoke_sync(prompt="Open the Home page."))

    assert content["ok"] is False
    assert "cannot open Home" in content["error"]
