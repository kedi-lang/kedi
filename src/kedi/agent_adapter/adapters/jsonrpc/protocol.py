"""The Kedi JSON-RPC adapter protocol, version 1.

Kedi talks to a backend integration (a "bridge") over JSON-RPC 2.0 on the bridge's stdio.
Kedi renders every model-visible text; the bridge transports it (#232, #197).

Framing: one JSON-RPC message per line, UTF-8, no embedded newlines, at most
``MAX_FRAME_BYTES`` per line. stdout carries only protocol frames; diagnostics go to stderr.

Kedi to bridge:

- ``initialize`` ``{protocolVersion, client}`` returns ``{protocolVersion, backend, kind,
  capabilities}``. Kedi sends nothing else before it completes.
- ``run`` ``{runId, callKind, promptKind, model, effort, settings, system, instructionParts,
  user, output, tools, continuation}`` returns ``{text}`` or ``{data}``, optionally with
  ``usage``.
- ``run/cancel`` (notification) ``{runId}``: stop the run; it answers with ``CANCELLED``.
- ``shutdown`` ``{}``: finish and exit; Kedi then closes stdin.

Bridge to Kedi:

- ``tool/call`` ``{runId, callId, name, arguments}`` returns ``{ok: true, result}`` or
  ``{ok: false, error}``. Only tools of a run in progress are callable.
"""

from __future__ import annotations

from typing import Any, Final, Literal, TypeAlias

PROTOCOL_VERSION: Final = 1
MAX_FRAME_BYTES: Final = 8 * 1024 * 1024

JsonObject: TypeAlias = dict[str, Any]
BridgeKind: TypeAlias = Literal["agent-framework", "agent-harness"]

# JSON-RPC 2.0 reserved codes, then Kedi's.
PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603
UNSUPPORTED: Final = -32001
"""The request needs a capability the peer did not negotiate."""
RUN_FAILED: Final = -32002
"""The backend could not complete the run."""
CANCELLED: Final = -32003
"""The run stopped after ``run/cancel``."""
OVERLOADED: Final = -32004
"""The receiver has no capacity for another request; the sender may retry later."""
UNKNOWN_RUN: Final = -32005
"""The ``runId`` names no run in progress on this connection."""

CAPABILITIES: Final = {
    "structuredOutput": "supports_structured_output",
    "tools": "supports_tool_registration",
    "nativeStructuredOutput": "supports_native_structured_output",
    "nativeToolCalling": "supports_native_tool_calling",
}
"""Capabilities a bridge may advertise in protocol version 1, and the Kedi flags they set.

Anything else, such as media, events, sessions or cancellation guarantees, is not part of
version 1 and stays unavailable whatever the bridge says.
"""


class JsonRpcError(RuntimeError):
    """A JSON-RPC error response, or a protocol failure Kedi raises with one of the codes."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data

    def to_json(self) -> JsonObject:
        return {"code": self.code, "message": str(self)}


class JsonRpcDisconnected(JsonRpcError):
    """The bridge exited, closed its stdout, or broke the framing; the connection is over."""

    def __init__(self, message: str) -> None:
        super().__init__(INTERNAL_ERROR, message)


__all__ = [
    "CANCELLED",
    "CAPABILITIES",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "MAX_FRAME_BYTES",
    "METHOD_NOT_FOUND",
    "OVERLOADED",
    "PARSE_ERROR",
    "PROTOCOL_VERSION",
    "RUN_FAILED",
    "UNKNOWN_RUN",
    "UNSUPPORTED",
    "BridgeKind",
    "JsonObject",
    "JsonRpcDisconnected",
    "JsonRpcError",
]
