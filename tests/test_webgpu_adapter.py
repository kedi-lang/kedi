from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Iterator

import pytest
from pydantic import BaseModel

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
    # Kedi renders the output contract; the bridge forwards it and authors none.
    instructions = bridge.payloads[0]["instructions"]
    assert instructions.startswith("<OutputProtocol>\n")
    assert '"description": "Computed answer"' in instructions
    assert "outputPrompt" not in bridge.payloads[0]
    assert "toolsPrompt" not in bridge.payloads[0]


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
    assert payload["instructions"].startswith("Be exact.\n\n<OutputProtocol>")
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


def test_webgpu_adapter_requires_declared_tools_before_final() -> None:
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
        metadata={"required_before_output": True},
    )

    with adapter.tool_scope((tool,)):
        result = adapter.produce_sync(
            template="Greet Mert as [answer].", output_schema={"answer": str}
        )

    assert result.answer == "done"
    assert "requiredTools" not in bridge.payloads[0]
    # Kedi's correction, as the Pydantic AI adapter sends it.
    assert bridge.payloads[1]["messages"][-1]["content"] == (
        "Call required tool say_hello and use its result before returning the parent response."
    )
    assert '"result": "Hello, Mert!"' in bridge.payloads[2]["messages"][-1]["content"]


def test_webgpu_adapter_never_infers_required_tools_from_the_prompt() -> None:
    bridge = QueueBridge({"kind": "final", "data": {"answer": "done"}})
    adapter = WebGPUAdapter(bridge, model="bonsai")
    with adapter.tool_scope((ToolSpec(fn=lambda: "hi", name="say_hello"),)):
        result = adapter.produce_sync(
            template="Call the say_hello tool and return [answer].", output_schema={"answer": str}
        )
    assert result.answer == "done"
    assert len(bridge.payloads) == 1


def test_webgpu_adapter_corrects_a_failed_required_tool() -> None:
    def say_hello(name: str) -> str:
        raise RuntimeError("greeter is down")

    bridge = QueueBridge(
        {"kind": "tool_call", "name": "say_hello", "arguments": {"name": "Mert"}},
        {"kind": "final", "data": {"answer": "simulated"}},
        {"kind": "tool_call", "name": "say_hello", "arguments": {"name": "Mert"}},
        {"kind": "final", "data": {"answer": "simulated"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai", max_tool_steps=4)
    tool = ToolSpec(fn=say_hello, name="say_hello", metadata={"required_before_output": True})
    with adapter.tool_scope((tool,)):
        with pytest.raises(RuntimeError, match="4-step tool limit"):
            adapter.produce_sync(template="Greet as [answer].", output_schema={"answer": str})
    assert bridge.payloads[2]["messages"][-1]["content"].startswith(
        "Required tool say_hello did not complete successfully."
    )


def test_webgpu_adapter_retries_invalid_tool_protocol_with_kedi_text() -> None:
    bridge = QueueBridge(
        {
            "kind": "retry",
            "requiredTool": "get_capital",
            "error": "Expected a JSON object.",
            "telemetry": {"outputText": "We should use get_capital."},
        },
        {"kind": "tool_call", "name": "get_capital", "arguments": {"country": "Miyawland"}},
        {"kind": "final", "data": {"capital": "Miyawcity"}},
    )
    adapter = WebGPUAdapter(bridge, model="bonsai")
    tool = ToolSpec(
        fn=lambda country: "Miyawcity",
        name="get_capital",
        json_schema={
            "type": "object",
            "properties": {"country": {"type": "string"}},
            "required": ["country"],
        },
    )
    with adapter.tool_scope((tool,)):
        result = adapter.produce_sync(
            template="What's the capital of Miyawland?", output_schema={"capital": str}
        )

    assert result.capital == "Miyawcity"
    # The bridge's parse error is quoted as detail; a requiredTool it names is ignored.
    assert bridge.payloads[1]["messages"][-2:] == [
        {"role": "assistant", "content": "We should use get_capital."},
        {
            "role": "user",
            "content": (
                "Return exactly one call_tool or final action JSON object.\n"
                "<ParseError>Expected a JSON object.</ParseError>"
            ),
        },
    ]
    assert '"result": "Miyawcity"' in bridge.payloads[2]["messages"][-1]["content"]


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
    assert len(bridge.payloads) == 2
    assert all(
        "Call required tool" not in message.get("content", "")
        for payload in bridge.payloads
        for message in payload["messages"]
    )


def test_webgpu_adapter_validates_non_dict_final_data() -> None:  # 3.9
    bridge = QueueBridge({"kind": "final", "data": ["a", "b"]})
    adapter = WebGPUAdapter(bridge, model="bonsai")
    assert adapter.produce_sync(template="List two.", output_type=list[str]) == ["a", "b"]

    # Invalid data gets Kedi's correction turn, within the step limit.
    corrected = QueueBridge({"kind": "final", "data": "nope"}, {"kind": "final", "data": ["c"]})
    assert WebGPUAdapter(corrected, model="bonsai").produce_sync(
        template="List one.", output_type=list[str]
    ) == ["c"]
    assert corrected.payloads[1]["messages"][-2:] == [
        {"role": "assistant", "content": '{"action": "final", "data": "nope"}'},
        {
            "role": "user",
            "content": (
                "The structured output does not match the required schema:\n"
                ": Input should be a valid list\n"
                "Return a corrected response that matches the schema."
            ),
        },
    ]
    exhausted = QueueBridge({"kind": "final", "data": "nope"}, {"kind": "final", "data": "no"})
    with pytest.raises(RuntimeError, match="2-step tool limit"):
        WebGPUAdapter(exhausted, model="bonsai", max_tool_steps=2).produce_sync(
            template="List two.", output_type=list[str]
        )


def test_webgpu_adapter_sends_kedis_tool_catalog() -> None:
    bridge = QueueBridge({"kind": "final", "data": {"answer": "ok"}})
    adapter = WebGPUAdapter(bridge, model="bonsai")
    tool = ToolSpec(
        fn=lambda country: country,
        name="get_capital",
        description="Look up a capital.",
        json_schema={
            "type": "object",
            "properties": {"country": {"title": "Country", "type": "string"}},
            "required": ["country"],
        },
    )

    with adapter.tool_scope((tool,)):
        result = adapter.produce_sync(template="answer", output_schema={"answer": str})

    assert result.answer == "ok"
    instructions = bridge.payloads[0]["instructions"]
    tools = instructions[instructions.index("<ToolProtocol>") :]
    assert tools.startswith("<ToolProtocol>\nYou can call the tools below.")
    assert '{"action": "final", "data": <a value matching the output schema>}' in tools
    assert '<Tool name="get_capital">\n<Description>Look up a capital.</Description>' in tools
    assert (
        '<InputSchema>{"properties": {"country": {"title": "Country", "type": "string"}}, '
        '"required": ["country"], "type": "object"}</InputSchema>'
    ) in tools


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
        self.payloads: list[dict[str, Any]] = []

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        self.payloads.append(payload)
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
    bridge = McpWikiBridge("delete_page")
    adapter = WebGPUAdapter(bridge, model="bonsai")

    with adapter.profile_override(AgentProfile(mcp_servers=(server,), approval=approval)):
        content = json.loads(adapter.invoke_sync(prompt="Delete the Home page."))
    adapter_payloads = bridge.payloads

    # The MCP tool is in the catalog Kedi rendered, though it exists only inside the run.
    assert '<Tool name="delete_page">' in adapter_payloads[0]["instructions"]
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
