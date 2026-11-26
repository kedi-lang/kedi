"""Every frame Kedi and a bridge exchange matches the protocol's JSON Schema.

The schema's source is kedi-lang/json-rpc (``schema/v1/protocol.json``); Kedi ships a copy
next to the adapter. These tests record the traffic of representative runs, both
directions, and validate each params and result object against its method's definition.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from kedi.agent_adapter import JsonRpcAdapter
from kedi.agent_adapter.adapters.jsonrpc import reference_bridge_command
from kedi.agent_adapter.adapters.jsonrpc.connection import Connection
from kedi.agent_adapter.conformance.endpoint import RecordingEndpoint, Script
from tests.test_jsonrpc_adapter import (
    MCP_SERVER,
    PIXEL,
    PROCESSED,
    keep_recent,
    lookup_tool,
    peer,
    run,
)

SCHEMA_PATH = (
    Path(__file__).parents[1] / "src/kedi/agent_adapter/adapters/jsonrpc/protocol.schema.json"
)
SCHEMA = json.loads(SCHEMA_PATH.read_text())
REGISTRY = Registry().with_resource(SCHEMA["$id"], Resource.from_contents(SCHEMA))


def _validator(ref: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"{SCHEMA['$id']}{ref}"}, registry=REGISTRY)


class Traffic:
    """The frames of one connection, as Kedi sent and received them."""

    def __init__(self) -> None:
        self.frames: list[tuple[str, dict[str, Any]]] = []

    def problems(self) -> list[str]:
        """Each frame that does not match its method's schema, described."""

        methods = SCHEMA["methods"]
        asked: dict[tuple[str, Any], str] = {}
        found: list[str] = []
        for direction, frame in self.frames:
            method = frame.get("method")
            if method is not None:
                definition = methods.get(method)
                if definition is None:
                    found.append(f"{direction} {method}: not a protocol method")
                    continue
                sender = "kedi" if direction == "sent" else "bridge"
                if not definition["direction"].startswith(sender):
                    found.append(f"{direction} {method}: wrong direction")
                if "id" in frame:
                    asked[(sender, frame["id"])] = method
                found += _check(f"{method} params", definition["params"], frame.get("params", {}))
                continue
            # A response answers the other side's request with the same id.
            asker = "bridge" if direction == "sent" else "kedi"
            method = asked.pop((asker, frame.get("id")), None)
            if method is None:
                continue
            if "error" in frame:
                data = frame["error"].get("data")
                if isinstance(data, dict):
                    found += _check(f"{method} error data", "#/$defs/ErrorData", data)
                continue
            result_ref = methods[method].get("result")
            if result_ref is not None:
                found += _check(f"{method} result", result_ref, frame.get("result"))
        return found


def _check(label: str, ref: str, value: Any) -> list[str]:
    errors = sorted(_validator(ref).iter_errors(value), key=lambda error: list(error.path))
    return [f"{label}: {error.message} at {list(error.path)}" for error in errors[:3]]


@pytest.fixture
def traffic(monkeypatch: pytest.MonkeyPatch) -> Iterator[Traffic]:
    recorded = Traffic()
    send, dispatch = Connection._send, Connection._dispatch

    def sent(self: Connection, message: dict[str, Any]) -> None:
        recorded.frames.append(("sent", json.loads(json.dumps(message, default=str))))
        send(self, message)

    def received(self: Connection, message: dict[str, Any]) -> None:
        recorded.frames.append(("received", message))
        dispatch(self, message)

    monkeypatch.setattr(Connection, "_send", sent)
    monkeypatch.setattr(Connection, "_dispatch", received)
    yield recorded
    assert recorded.frames, "no traffic was recorded"
    problems = recorded.problems()
    assert not problems, "\n".join(problems)


def reference(endpoint: RecordingEndpoint, adapters: list, **options: Any) -> JsonRpcAdapter:
    adapter = JsonRpcAdapter(reference_bridge_command(f"{endpoint.base_url}/v1", **options))
    adapters.append(adapter)
    return adapter


@pytest.fixture
def adapters() -> Iterator[list[JsonRpcAdapter]]:
    opened: list[JsonRpcAdapter] = []
    yield opened
    for adapter in opened:
        adapter.close()


def test_text_tools_and_structured_runs_match_the_schema(traffic: Traffic, adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="lookup", arguments={"query": "q"})
        adapter = reference(endpoint, adapters)
        with adapter.tool_scope((lookup_tool(),)):
            assert asyncio.run(adapter.invoke(prompt="Look it up.")) == "done"
    # A structured run, its answer scripted by a test model.
    structured = JsonRpcAdapter(
        reference_bridge_command("http://127.0.0.1:9/v1"),
        test_model=lambda call: json.dumps({"capital": "Ankara"}),
    )
    adapters.append(structured)
    assert run(">> The capital of Turkey is [capital: str].\n= <capital>\n", structured) == "Ankara"


def test_a_conversation_with_media_and_history_matches_the_schema(
    traffic: Traffic, adapters: list
) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True)
        adapter = reference(endpoint, adapters)
        source = (
            f'> import: media\n> history: enabled\n[photo: Image] = `image("{PIXEL}")`\n'
            "[a] << Look at <<photo>>.\n[b] << Again.\n= `[a, b]`\n"
        )
        assert run(source, adapter) == ["done", "done"]
        assert run(PROCESSED, adapter, recent=keep_recent) == ["done", "done", "done"]


def test_budgets_settings_and_fallbacks_match_the_schema(traffic: Traffic, adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True)
        adapter = reference(endpoint, adapters)
        source = (
            "> model: primary\n> fallback: backup\n> effort: low\n"
            "> settings:\n    temperature: 0.2\n> budget:\n    request_limit: 3\n"
            "[r] << Hi.\n= <r>\n"
        )
        assert run(source, adapter) == "done"


def test_mcp_calls_match_the_schema(traffic: Traffic, adapters: list) -> None:
    with RecordingEndpoint() as endpoint:
        endpoint.script = Script(final="done", text=True, tool="shout", arguments={"text": "hi"})
        adapter = reference(endpoint, adapters)
        adapter.on("pre_tool_use")(lambda event: None)
        server = f"    command: {sys.executable}\n    args: `[{str(MCP_SERVER)!r}]`\n"
        source = f"> approval: allow\n> mcp:\n{server}[r] << Shout hi.\n= <r>\n"
        assert run(source, adapter) == "done"


def test_a_test_model_run_matches_the_schema(traffic: Traffic, adapters: list) -> None:
    def script(call: Any) -> Any:
        if not any(message["role"] == "tool" for message in call.messages):
            return {
                "parts": [
                    {"type": "toolCall", "id": "t1", "name": "lookup", "arguments": {"query": "q"}}
                ]
            }
        return "found"

    adapter = JsonRpcAdapter(reference_bridge_command("http://127.0.0.1:9/v1"), test_model=script)
    adapters.append(adapter)
    with adapter.tool_scope((lookup_tool(),)):
        assert asyncio.run(adapter.invoke(prompt="Look it up.")) == "found"


def test_kedi_managed_children_match_the_schema(traffic: Traffic, adapters: list) -> None:
    from tests.test_jsonrpc_adapter import manager_run

    adapter = peer("subagents")
    adapters.append(adapter)
    assert manager_run(adapter, "Delegate in the background.")


def test_the_check_reports_what_breaks_the_schema() -> None:
    broken = Traffic()
    broken.frames = [
        ("sent", {"jsonrpc": "2.0", "id": 1, "method": "run", "params": {"runId": "r"}}),
        ("received", {"jsonrpc": "2.0", "id": 1, "result": {"nothing": True}}),
        ("received", {"jsonrpc": "2.0", "method": "teleport", "params": {}}),
        ("received", {"jsonrpc": "2.0", "id": 7, "method": "run", "params": {}}),
        ("sent", {"jsonrpc": "2.0", "id": 2, "method": "tool/call", "params": {}}),
        ("received", {"jsonrpc": "2.0", "id": 2, "error": {"code": -32002, "message": "x"}}),
        ("received", {"jsonrpc": "2.0", "id": 9, "method": "run/usage", "params": {"runId": "r"}}),
        (
            "sent",
            {
                "jsonrpc": "2.0",
                "id": 9,
                "error": {"code": 1, "message": "m", "data": {"harness": 3}},
            },
        ),
    ]
    problems = broken.problems()
    assert any(problem.startswith("run params") for problem in problems)
    assert any(problem.startswith("run result") for problem in problems)
    assert "received teleport: not a protocol method" in problems
    assert "received run: wrong direction" in problems
    assert "sent tool/call: wrong direction" in problems
    assert any(problem.startswith("run/usage error data") for problem in problems)
