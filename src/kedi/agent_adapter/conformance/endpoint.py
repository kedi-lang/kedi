"""A recording model endpoint for the adapter conformance kit.

It speaks OpenAI Chat Completions (``/v1/chat/completions``), Anthropic Messages
(``/v1/messages``) and the WebGPU bridge protocol (``/kedi/webgpu``), records every request
body, and answers from the current scenario's script: call one Kedi tool first, then give the
scripted final value in whatever shape the request asks for.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal

Protocol = Literal["openai", "anthropic", "webgpu"]


@dataclass(frozen=True)
class Script:
    """How the endpoint answers during one scenario."""

    final: Any
    text: bool
    kedi_tools: frozenset[str] = frozenset()
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Exchange:
    """One recorded request."""

    protocol: Protocol
    body: dict[str, Any]


class RecordingEndpoint:
    """A local HTTP model endpoint that records requests and answers from a script."""

    def __init__(self) -> None:
        self.exchanges: list[Exchange] = []
        self.script: Script | None = None
        self._lock = threading.Lock()
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                route = self.path.split("?", 1)[0].rstrip("/")
                protocol: Protocol | None = (
                    "openai"
                    if route.endswith("/chat/completions")
                    else "anthropic"
                    if route.endswith("/messages")
                    else "webgpu"
                    if route.endswith("/kedi/webgpu")
                    else None
                )
                if protocol is None:
                    self._send(404, {"error": f"unknown route {self.path}"})
                    return
                endpoint._record(Exchange(protocol, body))
                try:
                    answer = endpoint._answer(protocol, body)
                except Exception as error:  # noqa: BLE001 - a failed answer reaches the adapter
                    self._send(500, {"error": f"conformance endpoint: {error}"})
                    return
                self._send(200, answer)

            def _send(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> RecordingEndpoint:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._server.shutdown()
        self._server.server_close()

    def take(self) -> list[Exchange]:
        """The requests recorded since the last call."""

        with self._lock:
            taken, self.exchanges = self.exchanges, []
        return taken

    def _record(self, exchange: Exchange) -> None:
        with self._lock:
            self.exchanges.append(exchange)

    def _answer(self, protocol: Protocol, body: dict[str, Any]) -> dict[str, Any]:
        script = self.script
        if script is None:
            raise RuntimeError("no scenario is running")
        if body.get("stream"):
            raise RuntimeError("streamed requests are not supported; send stream=false")
        if protocol == "openai":
            return _openai_answer(script, body)
        if protocol == "anthropic":
            return _anthropic_answer(script, body)
        return _webgpu_answer(script, body)


Decision = tuple[str | None, tuple[str, Any] | None]
"""The answer: text content, or one tool call (name, arguments)."""


def _decide(
    script: Script,
    *,
    system: str,
    tool_result_seen: bool,
    wire_tools: list[tuple[str, dict[str, Any]]],
) -> Decision:
    """Call the scenario's tool once, then give the final value in the shape the request asks.

    A request carrying Kedi's ``<ToolProtocol>`` gets JSON actions as text. Otherwise the
    tool call is native, a structured final goes to the request's output tool (a wire tool
    that is not Kedi's) when there is one, and is JSON text otherwise.
    """

    tool_first = script.tool is not None and not tool_result_seen
    if "<ToolProtocol>" in system:
        if tool_first:
            action = {"action": "call_tool", "callId": "c1", "name": script.tool}
            return json.dumps({**action, "arguments": script.arguments}), None
        return json.dumps(
            {"action": "final", "text" if script.text else "data": script.final}
        ), None
    if tool_first:
        return None, (str(script.tool), script.arguments)
    if script.text:
        return script.final, None
    output_tools = [(name, schema) for name, schema in wire_tools if name not in script.kedi_tools]
    if output_tools:
        name, schema = output_tools[0]
        return None, (name, _output_arguments(script.final, schema))
    return json.dumps(script.final), None


def _output_arguments(final: Any, schema: dict[str, Any]) -> Any:
    """The final value as an output tool's arguments: as is, or under its one property."""

    properties = schema.get("properties") or {}
    if isinstance(final, dict) and set(final) == set(properties):
        return final
    (name,) = properties
    return {name: final}


def texts(content: Any) -> str:
    """The text of a message content: a string, or the text blocks of a content list."""

    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content or () if part.get("type") == "text")


def _openai_answer(script: Script, body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    text, call = _decide(
        script,
        system="\n".join(
            texts(message.get("content"))
            for message in messages
            if message.get("role") in {"system", "developer"}
        ),
        tool_result_seen=any(message.get("role") == "tool" for message in messages),
        wire_tools=[
            (tool["function"]["name"], tool["function"].get("parameters") or {})
            for tool in body.get("tools") or []
        ],
    )
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if call is not None:
        name, arguments = call
        message["tool_calls"] = [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    return {
        "id": "chatcmpl-conformance",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", "conformance"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop" if call is None else "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic_answer(script: Script, body: dict[str, Any]) -> dict[str, Any]:
    blocks = [
        block
        for message in body.get("messages") or []
        if isinstance(message.get("content"), list)
        for block in message["content"]
    ]
    text, call = _decide(
        script,
        system=texts(body.get("system")),
        tool_result_seen=any(block.get("type") == "tool_result" for block in blocks),
        wire_tools=[
            (tool["name"], tool.get("input_schema") or {}) for tool in body.get("tools") or []
        ],
    )
    if call is None:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    else:
        name, arguments = call
        content = [{"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": arguments}]
    return {
        "id": "msg_conformance",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "conformance"),
        "content": content,
        "stop_reason": "end_turn" if call is None else "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _webgpu_answer(script: Script, body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    if script.tool is not None and not any(message.get("role") == "tool" for message in messages):
        return {
            "kind": "tool_call",
            "name": script.tool,
            "arguments": script.arguments,
            "callId": "c1",
        }
    return {"kind": "final", "text" if script.text else "data": script.final}


__all__ = ["Exchange", "RecordingEndpoint", "Script", "texts"]
