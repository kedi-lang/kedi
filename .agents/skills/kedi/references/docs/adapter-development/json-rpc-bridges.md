# JSON-RPC Bridges

`JsonRpcAdapter` connects Kedi to a backend written in any language. The
backend integration, a *bridge*, is a process that speaks Kedi's JSON-RPC
protocol on its stdio. Kedi renders every piece of text the model reads; the
bridge transports it to its framework or harness and calls Kedi's tools back.

```python
from kedi.agent_adapter import JsonRpcAdapter
from kedi.lang import compile_program, parse_program

with JsonRpcAdapter(["node", "my_bridge.mjs"], model="my-model") as adapter:
    runtime = compile_program(parse_program(source), adapter=adapter)
    runtime.run_main()
```

From Kedi source, `> agent: rpc:` with a `command:` body launches a bridge (see
[Backend Selection](../agentic-engineering/backend-selection.md#json-rpc-bridges)).

The adapter launches the command and owns the process. Readiness is
structural: the constructor is the handshake, so an adapter object always has
its negotiated capabilities, and a bridge that fails `initialize` is stopped
and never exposed. `await JsonRpcAdapter.launch(command)` does the same from
async code. `close()` sends `shutdown`, closes stdin, and stops the process if
it has not exited within the timeout.

## Wire Format

- JSON-RPC 2.0, one message per line, UTF-8, at most 8 MiB per line.
- stdout carries only protocol frames. Write logs and diagnostics to stderr;
  Kedi drains it and quotes its last lines when the connection fails.
- A line that is not a JSON-RPC 2.0 object, or a line over the limit, ends the
  connection: every waiting call fails with `JsonRpcDisconnected`. A
  disconnected adapter refuses further calls; version 1 has no reconnect, so
  create a new adapter.
- Exit when stdin closes, so a bridge never outlives the Kedi process that owns
  it.

## Methods

| Method | Direction | Params | Result |
|---|---|---|---|
| `initialize` | Kedi → bridge | `protocolVersion`, `client` | `protocolVersion`, `backend` (`name`, `version`), `kind`, `capabilities`, `mediaKinds` |
| `run` | Kedi → bridge | see below | `{"text": ...}` or `{"data": ...}`, optional `usage`, and `state` in a session |
| `run/cancel` | Kedi → bridge (notification) | `runId` | — |
| `tool/call` | bridge → Kedi | `runId`, `callId`, `name`, `arguments` | `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}` |
| `run/event` | bridge → Kedi (notification) | `runId`, `type`, see [Events](#events) | — |
| `shutdown` | Kedi → bridge | `{}` | `{}`; then Kedi closes stdin |

The protocol version is `1`. A bridge that answers `initialize` with another
version, an unknown `kind` (`agent-framework` or `agent-harness`), or no
backend name is refused, and the process is stopped.

### Capabilities

| Capability | Meaning |
|---|---|
| `structuredOutput` | the bridge answers runs that have an `output` with `data` |
| `tools` | the bridge lets its model call the run's `tools` through `tool/call` |
| `nativeStructuredOutput` | the backend takes the output schema natively |
| `nativeToolCalling` | the backend takes tool definitions natively |
| `events` | the bridge sends `run/event` notifications |
| `mediaInput` | the bridge sends prompt media to its model; `mediaKinds` lists the kinds (`image`, `audio`, `video`, `document`) |
| `sessions` | the bridge continues conversations from the `state` Kedi hands back; enables `> history: enabled` |

Advertise what the bridge implements, not everything its SDK could do. A
capability left out is unavailable, and Kedi refuses an operation that needs it
before sending anything. Version 1 has no live input: a bridge cannot enable
it by advertising it.

Without `nativeStructuredOutput` or `nativeToolCalling`, Kedi adds its
`protocol.output` and `protocol.tools` parts to the system text (see
[Rendered Request](rendered-request.md)), and the model answers with JSON
actions the bridge parses.

### `run`

| Field | Contents |
|---|---|
| `runId` | the run; `tool/call` names it |
| `callKind`, `promptKind` | `invoke` or `produce`; `raw`, `template` or `claim` |
| `model` | the profile's model, or the adapter's default |
| `system` | the system text: send it verbatim as the system or developer message |
| `instructionParts` | the same text as parts, each with `identity`, `source` and `content` |
| `user` | the prompt: send it verbatim as the user turn |
| `userContent` | `null`, or the user turn with its media; see [Media](#media) |
| `output` | `null` for text, or `schema` (the output JSON Schema) and `toolName` |
| `tools` | each tool's `name`, `description`, `inputSchema`, `outputSchema` and `requiredBeforeOutput` |
| `continuation` | `null`, or `answer` (the bridge's previous result) and `message` (Kedi's correction) |
| `session` | `null`, or `id`, `turn` and `state`; see [Sessions](#sessions) |

Kedi validates every answer. When the data does not match the schema, a required
tool did not complete, or the bridge reports `{"invalid": "<model text>",
"error": "<parse error>"}`, Kedi sends the same run again with a
`continuation`: replay `answer` as the model's turn and send `message` as the
next user turn. The bridge never writes correction text and never infers
required tools from the prompt. `output_retries` (default 1) bounds the
corrections.

Report `usage` with the counts the backend measured: `requests`,
`inputTokens`, `outputTokens`, `totalTokens`. Kedi records a count the bridge
leaves out as zero, and `requests` as one.

### `tool/call`

A bridge may call only the tools of a run in progress, with that run's
`runId`. Kedi validates the arguments, applies the approval policy, runs the
tool and answers. A failure the model can act on, such as a wrong argument or
a denied approval, comes back as `{"ok": false, "error": ...}` for the model to
read. A failure that is not the model's to correct, such as an exhausted
budget, stops the run: Kedi answers the call with `-32002`, sends
`run/cancel`, and raises it. A call still running when its run ends is answered
with `-32003`. Calls run concurrently on
the run's event loop, so a tool may itself start a nested run on the same
bridge while the outer run waits.

### Events

With `events`, a bridge reports what its model writes during a run, and Kedi
publishes it to its event observers (see
[Events, Streaming, and Telemetry](events.md)) under the Kedi run it belongs to:

| `type` | Fields | Meaning |
|---|---|---|
| `message.delta` | `messageId`, `text`, optional `phase` | a chunk of a message |
| `message.completed` | `messageId`, optional `text` and `phase` | the message is complete; `text` replaces the chunks |
| `tool.boundary` | — | the model called a tool: messages so far are commentary |

`phase` is `commentary` or `final`. Kedi starts and ends the run itself, and
publishes the run's answer as its final message, so a bridge reports only what
the model wrote along the way. Events arrive in stream order before the run's
answer. Events for a run that is not in progress, of an unknown `type`, or
without the fields above are ignored, as are all events from a bridge that did
not negotiate `events`.

### Sessions

With `sessions`, a run in a conversation (`> history: enabled`) carries
`session`: the conversation's `id`, the `turn` number, and the `state` the
bridge returned for the conversation's last committed turn, `null` at first.
The bridge answers with the new `state`, any JSON value: the turn's messages
in the backend's own format, a remote thread id, or both.

Kedi owns the conversation: it orders the turns, commits `state` only after it
accepts the turn's answer, discards it after a failed or cancelled turn, and
persists it with interactive sessions and snapshots. The bridge keeps no
history of its own outside `state`, so the two sides never disagree. A turn
with another model, tools or settings is another lane (see
[Conversations and History](conversations.md)) and starts from `null`.

`state` travels in every `run`, so it counts against the 8 MiB frame. Keep large
values, such as media the conversation already sent, out of it: store an id the
backend can resolve instead of the bytes. A run whose frame would exceed the
limit fails with `-32600` before it is sent.

### Media

With `mediaInput`, a prompt's media arrives in `userContent`, laid out as
Kedi lays out every multimodal user turn (see
[Multimodal Input](multimodal-input.md)): text parts, and each attachment where
its label puts it.

```json
[
  {"type": "text", "text": "<attachments>\n<image id=\"chart\" type=\"image/png\">"},
  {"type": "media", "name": "chart", "kind": "image", "mediaType": "image/png", "data": "iVBORw0..."},
  {"type": "text", "text": "</image>\n</attachments>\nDescribe chart in one sentence."}
]
```

`data` is base64; a media value given by URL arrives as `url` instead. Send
`userContent` in place of `user`, each part as the backend's native part. Kedi
refuses, before the run, a kind outside `mediaKinds` and a provider file id,
which cannot cross a bridge. A frame holds at most 8 MiB, so send large media
by URL.

### Errors

| Code | Meaning |
|---|---|
| `-32700`…`-32603` | JSON-RPC parse, request, method, params and internal errors |
| `-32001` | unsupported: the operation needs a capability that was not negotiated |
| `-32002` | the run failed in the backend |
| `-32003` | the run stopped after `run/cancel` |
| `-32004` | overloaded: Kedi has no capacity for another request right now |
| `-32005` | unknown run: the `runId` names no run in progress |

Kedi admits 64 bridge requests at a time (`max_inbound_requests`); more are
answered with `-32004`.

### Cancellation and Timeouts

When a Kedi call is cancelled or exceeds `request_timeout`, Kedi sends
`run/cancel` and stops waiting. The notification is a request: it does not
prove that the backend stopped or that side effects were undone.

## Reference Bridge

`kedi.agent_adapter.adapters.jsonrpc.REFERENCE_BRIDGE` is a bridge in
JavaScript (Node.js 18 or later) for any OpenAI-compatible chat endpoint:

```python
from kedi.agent_adapter.adapters.jsonrpc import reference_bridge_command

adapter = JsonRpcAdapter(reference_bridge_command("http://localhost:11434/v1"), model="llama3")
```

`mode="protocol"` uses Kedi's protocol parts instead of native tools and
response formats, for endpoints without them. Start a new bridge from it.

## Conformance

`kedi adapter check kedi.agent_adapter:JsonRpcAdapter` runs the
[conformance kit](testing.md#conformance-check) against the reference bridge.
To check your bridge, give a class a `for_conformance(base_url)` that launches
it with `base_url` as its model endpoint:

```python
class MyBridge(JsonRpcAdapter):
    @classmethod
    def for_conformance(cls, base_url):
        return cls(["node", "my_bridge.mjs", "--base-url", f"{base_url}/v1"])
```

The reference bridge passes in both modes:

```python title="run_json_rpc_bridge.py"
from kedi.agent_adapter import JsonRpcAdapter
from kedi.agent_adapter.adapters.jsonrpc import reference_bridge_command
from kedi.agent_adapter.conformance import check_adapter

for mode in ("native", "protocol"):
    report = check_adapter(
        lambda base_url: JsonRpcAdapter(reference_bridge_command(f"{base_url}/v1", mode=mode))
    )
    assert report.passed, report.render()
    assert report.harness_text == []
```
