"""The Kedi JSON-RPC adapter protocol, version 1.

Kedi talks to a backend integration (a "bridge") over JSON-RPC 2.0 on the bridge's stdio.
Kedi renders every model-visible text; the bridge transports it (#232, #197).

Framing: one JSON-RPC message per line, UTF-8, no embedded newlines, at most
``MAX_FRAME_BYTES`` per line. stdout carries only protocol frames; diagnostics go to stderr.

Kedi to bridge:

- ``initialize`` ``{protocolVersion, client}`` returns ``{protocolVersion, backend, kind,
  capabilities, mediaKinds?, settings?, effort?, mcp?}``: ``settings`` names the
  ``> settings:`` the backend applies, ``effort`` the ``> effort:`` levels it maps, ``mcp``
  ``{transports}`` the ``> mcp:`` transports it connects (with ``nativeTools``). Kedi sends nothing else before
  it completes.
- ``run`` ``{runId, callKind, promptKind, model, fallbackModels, settings, effort, system,
  instructionParts, user, userContent, output, tools, continuation, session, subagent, hooks,
  mcpServers, admitResults}``
  returns ``{text}``, ``{data}`` or ``{content}``, optionally
  with ``usage`` and, in a session, ``state``. ``output`` is ``null`` for text, ``{schema,
  toolName}`` for data, or, with ``imageOutput`` or ``audioOutput``, ``{media: "image",
  schema?, toolName?}`` (the model draws an image; with ``schema`` it may instead name an image
  of its request, as ``data``) or ``{media: "audio", voice}`` (the model speaks; the run has
  no tools). ``content`` lists the answer's parts, as ``userContent`` does: ``{type: "text",
  text}`` and ``{type: "media", kind, mediaType, data | url}``; a media output takes its one
  part of that kind. ``subagent`` is ``null``, or, for a child Kedi
  runs on the bridge, ``{id, parentRunId, conversationId, name, limits}``: Kedi's run id for
  the child, the bridge run whose tool call started it, the child conversation, the child
  profile and its remaining limits. ``session`` is ``null``, or ``{id, turn, state}``:
  the bridge's ``state`` from the conversation's last committed turn, ``null`` at first.
  ``userContent`` is ``null``, or the user turn with its media: ``{type: "text", text}`` and
  ``{type: "media", name, kind, mediaType, data | url}`` parts, ``data`` in base64.
- ``run/cancel`` (notification) ``{runId}``: stop the run; it answers with ``CANCELLED``.
- ``run/input`` ``{runId, items: [{id, text}]}`` (with ``liveInput``): input for a running
  run, from its parent or a child. Add each ``text`` as a user turn before the next model
  request, and answer ``{delivered: true}`` once a model request carries them, or
  ``{delivered: false}`` if the run ends first: Kedi then gives them to the next attempt.
- ``run/interrupt`` (notification) ``{runId}`` (with ``interrupt``): stop the current attempt
  and answer the run with ``INTERRUPTED``, its ``data`` holding ``{state}`` in a session. The
  conversation continues: Kedi starts another attempt with the new input.
- ``shutdown`` ``{}``: finish and exit; Kedi then closes stdin.

Bridge to Kedi:

- ``tool/call`` ``{runId, callId, name, arguments}`` returns ``{ok: true, result}`` or
  ``{ok: false, error}``. Only tools of a run in progress are callable.
- ``budget/admit`` ``{runId, dimension}`` (with the ``governed`` capability), before the step
  runs: ``dimension`` is ``request`` for every model request, retries included, and
  ``toolAttempt`` for each attempt of a bridge tool. Kedi charges the run budget and counts
  the step for a child's limits. It returns ``{admitted: true}``, or ``{admitted: false,
  dimension, reason}``, with ``limit``, ``consumed`` and ``scopeDepth`` for a run budget: the
  bridge then skips the step and ends the run, which fails with Kedi's error.
- ``approval/request`` ``{runId, callId?, tool, arguments, risk, description?, origin?}`` (with
  the ``nativeTools`` capability), before each call of a bridge tool; ``origin`` is
  ``adapter_native`` (the default), ``mcp`` or ``provider_builtin``. ``risk`` is ``read_only``,
  ``mutating`` or ``sensitive``. It returns ``{decision: "allow"}``, ``{decision: "deny",
  reason}``, or ``{decision: "edit", arguments, reason}``: the bridge runs the edited
  arguments, after another ``approval/request`` if their risk is higher.
- ``hook/invoke`` ``{runId, event, callId, tool, origin, arguments, inputSchema?, result?,
  error?, interrupted?}`` (with ``nativeTools``), around each call of a bridge tool, for the
  events the run's ``hooks`` lists (those with handlers): ``pre_tool_use`` before its
  ``approval/request``, then
  ``post_tool_use`` with its ``result`` or ``post_tool_use_failure`` with ``error`` ``{type,
  message}``. ``pre_tool_use`` returns ``{decision: "continue"}``, ``{decision: "edit",
  arguments}`` or ``{decision: "deny", reason}``; the others return ``{}``. Kedi runs the
  program's hooks for Kedi tools itself, at ``tool/call``.
- ``artifact/admit`` ``{runId, callId, tool, origin, result}`` (with ``nativeTools``, when the
  run's ``admitResults`` is true), for each result of a bridge tool, MCP tools included, before
  the model sees it: returns ``{result}``, the result itself or Kedi's artifact reference for a
  large one, which the bridge sends the model instead.
- ``run/usage`` (notification, with ``usage``) ``{runId, inputTokens, outputTokens,
  totalTokens?, cacheReadTokens?, cacheWriteTokens?}``: one model request's counts, sent when
  it completes. Kedi checks a child's token limits as each arrives; ``usage`` in the run's
  result is then ignored.
- ``run/event`` (notification, with the ``events`` capability) ``{runId, type, ...}``:
  ``message.delta`` ``{messageId, text, phase?}``, ``message.completed`` ``{messageId,
  text?, phase?}`` or ``tool.boundary``. ``phase`` is ``commentary`` or ``final``.
"""

from __future__ import annotations

from typing import Any, Final, Literal, TypeAlias

PROTOCOL_VERSION: Final = 1
MAX_FRAME_BYTES: Final = 8 * 1024 * 1024

JsonObject: TypeAlias = dict[str, Any]
MEDIA_KINDS: Final = frozenset({"image", "audio", "video", "document"})
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
INTERRUPTED: Final = -32006
"""The run stopped after ``run/interrupt``; its conversation continues in another attempt."""

CAPABILITIES: Final = {
    "structuredOutput": "supports_structured_output",
    "tools": "supports_tool_registration",
    "nativeStructuredOutput": "supports_native_structured_output",
    "nativeToolCalling": "supports_native_tool_calling",
    "events": "supports_stream_events",
    "mediaInput": "supports_media_input",
    "sessions": "supports_stateful_history",
    "governed": "supports_request_budget",
    "subagents": "supports_subagents",
    "liveInput": "supports_live_subagent_input",
    "interrupt": "supports_subagent_interrupt",
    "usage": "supports_token_usage",
    "modelFallback": "supports_model_fallback",
    "imageOutput": "supports_image_output",
    "audioOutput": "supports_audio_output",
}
"""Capabilities a bridge may advertise in protocol version 1, and the Kedi flags they set.

``subagents`` lets Kedi run its children on the bridge, concurrently when they run in the
background; it needs ``governed``, because Kedi admits each of a child's requests against its
limits before the bridge sends it. ``liveInput`` (``run/input``) needs ``subagents``, and
``interrupt`` (``run/interrupt``) needs ``liveInput``. ``usage`` promises a ``run/usage`` for
every model request, so a child's token limits hold. ``modelFallback`` tries ``fallbackModels``
in order when a model request fails, admitting each attempt.

``governed`` promises that every model request the bridge's backend makes goes through
``budget/admit`` first: SDK retries, tool-loop turns and internal calls too. A bridge that
cannot route one of them must disable it or not advertise ``governed``. ``nativeTools``
(``NATIVE_TOOLS``) says the backend runs tools of its own besides Kedi's; each call then
needs ``approval/request``, and with ``governed`` also ``budget/admit``.

Anything else, such as cancellation guarantees, is not part of version 1 and stays unavailable
whatever the bridge says. ``mediaInput`` needs ``mediaKinds``,
the kinds the bridge sends: ``image``, ``audio``, ``video``, ``document``.
"""

NATIVE_TOOLS: Final = "nativeTools"
MCP_TRANSPORTS: Final = frozenset({"stdio", "http", "sse"})
"""The ``> mcp:`` transports; ``streamable-http`` normalizes to ``http``."""
HOOK_EVENTS: Final = frozenset({"pre_tool_use", "post_tool_use", "post_tool_use_failure"})

BUDGET_DIMENSIONS: Final = {"request": "request", "toolAttempt": "tool_attempt"}
"""``budget/admit`` dimensions and Kedi's run budget dimensions."""
TOOL_RISKS: Final = frozenset({"read_only", "mutating", "sensitive"})


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
    "BUDGET_DIMENSIONS",
    "CANCELLED",
    "CAPABILITIES",
    "MEDIA_KINDS",
    "NATIVE_TOOLS",
    "TOOL_RISKS",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INTERRUPTED",
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
