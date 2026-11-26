# JSON-RPC Bridges

!!! warning "Experimental"
    The protocol is still changing: version 1 gains the capabilities Kedi's
    in-process adapters have, each in its own message. Until Kedi declares it
    stable, a release may change it without a compatibility period.

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
- A line ends at LF (`\n`) only. Kedi writes U+2028 and U+2029 unescaped, and
  a text may hold CR. Do not split frames with a reader that also ends lines
  at those, such as Node's `readline` or Python's `str.splitlines()`: split on
  `\n` yourself.
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
| `initialize` | Kedi → bridge | `protocolVersion`, `client` | `protocolVersion`, `backend` (`name`, `version`), `kind`, `capabilities`, `mediaKinds`, `settings`, `effort`, `hooks`; see [Settings, Effort and Fallback](#settings-effort-and-fallback) and [Hooks](#hooks) |
| `run` | Kedi → bridge | see below | `{"text": ...}`, `{"data": ...}` or `{"content": [...]}`, optional `usage`, and `state` in a session |
| `run/cancel` | Kedi → bridge (notification) | `runId` | — |
| `run/input` | Kedi → bridge | `runId`, `items` (`id`, `text`) | `{"delivered": true}` or `{"delivered": false}`; see [Live Input and Interrupts](#live-input-and-interrupts) |
| `run/interrupt` | Kedi → bridge (notification) | `runId` | — ; the run answers with `-32006` |
| `run/usage` | bridge → Kedi (notification) | `runId`, `inputTokens`, `outputTokens`, optional `totalTokens`, `cacheReadTokens`, `cacheWriteTokens` | — |
| `tool/call` | bridge → Kedi | `runId`, `callId`, `name`, `arguments` | `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}` |
| `budget/admit` | bridge → Kedi | `runId`, `dimension` | `{"admitted": true}`, or `admitted: false` with `dimension` and `reason`, and for a run budget `limit`, `consumed` and `scopeDepth`; see [Budgets](#budgets) |
| `hook/invoke` | bridge → Kedi | `runId`, `event`, `callId`, `tool`, `origin`, `arguments`, optional `inputSchema`, `result`, `error` (`type`, `message`), `interrupted` | for `pre_tool_use`: `decision` (`continue`, `edit` with `arguments`, or `deny` with `reason`); otherwise `{}`; see [Hooks](#hooks) |
| `approval/request` | bridge → Kedi | `runId`, `tool`, `arguments`, `risk`, optional `callId` and `description` | `decision` (`allow`, `deny` or `edit`), optional `reason`, and `arguments` for `edit`; see [Approval](#approval) |
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
| `governed` | every model request the backend makes goes through `budget/admit` first; enables `request_limit` (see [Budgets](#budgets)) |
| `nativeTools` | the backend runs tools of its own besides Kedi's, and asks `approval/request` before each call (see [Approval](#approval)) |
| `subagents` | Kedi may run its children on the bridge (see [Subagents](#subagents)); needs `governed` |
| `liveInput` | the bridge takes input for a running run with `run/input`; needs `subagents` |
| `interrupt` | the bridge stops an attempt on `run/interrupt`; needs `liveInput` |
| `usage` | the bridge sends `run/usage` for every model request; enables `token_usage`, so children can have token limits |
| `modelFallback` | the bridge tries `fallbackModels` in order when a model request fails; enables `> fallback:` |
| `imageOutput` | the bridge's model draws images for `[img: Image] << ...`; see [Media Output](#media-output) |
| `audioOutput` | the bridge's model speaks for `[speech: Audio] << ...` |

Advertise what the bridge implements, not everything its SDK could do. A
capability left out is unavailable, and Kedi refuses an operation that needs it
before sending anything.

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
| `fallbackModels` | the profile's `> fallback:` models, in order; empty for none |
| `settings` | the profile's `> settings:` the bridge listed in `initialize`, as name and value |
| `effort` | the profile's `> effort:` level, or `null` |
| `system` | the system text: send it verbatim as the system or developer message |
| `instructionParts` | the same text as parts, each with `identity`, `source` and `content` |
| `user` | the prompt: send it verbatim as the user turn |
| `userContent` | `null`, or the user turn with its media; see [Media](#media) |
| `output` | `null` for text; `schema` (the output JSON Schema) and `toolName` for data; or `media` for an image or speech the model makes (see [Media Output](#media-output)) |
| `tools` | each tool's `name`, `description`, `inputSchema`, `outputSchema` and `requiredBeforeOutput` |
| `continuation` | `null`, or `answer` (the bridge's previous result) and `message` (Kedi's correction) |
| `session` | `null`, or `id`, `turn` and `state`; see [Sessions](#sessions) |
| `subagent` | `null`, or the child Kedi runs: `id`, `parentRunId`, `conversationId`, `name`, `limits`; see [Subagents](#subagents) |

Kedi validates every answer. When the data does not match the schema, a required
tool did not complete, or the bridge reports `{"invalid": "<model text>",
"error": "<parse error>"}`, Kedi sends the same run again with a
`continuation`: replay `answer` as the model's turn and send `message` as the
next user turn. The bridge never writes correction text and never infers
required tools from the prompt. `output_retries` (default 1) bounds the
corrections.

Report `usage` with the counts the backend measured: `requests`,
`inputTokens`, `outputTokens`, `totalTokens`. A token count the bridge leaves
out is unknown, not zero, and a run without `usage` has unknown counts; a
missing `requests` is one.

With `usage`, send `run/usage` as each model request completes, with that
request's counts. Kedi checks a child's token limits as each report arrives, so a
limit stops the child after the request that crosses it, as in-process. The
run's `usage` is then ignored. A governed bridge's requests are counted at
`budget/admit`, so `run/usage` carries tokens only.

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

### Media Output

With `imageOutput`, `[img: Image] << Draw a cat.` reaches the bridge with
`output` `{"media": "image"}`. Have the model draw, and answer with its
`content`: the parts it made, in the layout of `userContent`, such as
`[{"type": "text", "text": "Here it is."}, {"type": "media", "kind": "image",
"mediaType": "image/png", "data": "<base64>"}]`, or a `url` instead of `data`.
Kedi takes the one image part. When the request holds images, or a tool
may return one, `output` also has `schema` and `toolName`: the model may then
name one of the request's images instead, and the bridge answers
`{"data": "<name>"}`. Kedi resolves the name to that image, as it does for any
media-typed value a model writes.

With `audioOutput`, `[speech: Audio] << Read this aloud.` reaches the bridge
with `output` `{"media": "audio", "voice": ...}`: `voice` is the profile's
`> settings: voice`, or `null` for the model's own. The run has no tools and
no media. Answer with `content` holding one `audio` part, for example
`{"type": "media", "kind": "audio", "mediaType": "audio/wav", "data":
"<base64>"}`. Kedi fails a run whose `content` holds no part of the kind asked
for, or more than one.

Without the capability, Kedi has the model name a request image for
`[img: Image] <<`, as for any adapter that cannot draw, and refuses
`[speech: Audio] <<` before the run.

### Settings, Effort and Fallback

A bridge lists in `initialize` what it applies: `settings`, the `> settings:`
names it maps to its backend, and `effort`, the `> effort:` levels it accepts
(`minimal`, `low`, `medium`, `high`, `xhigh`, `max`). Each run then carries the
profile's values for those names in `settings`, and its level in `effort`.

Kedi applies its own settings (`context_limit`, `media_limit`,
`request_retries`, `tool_reason`, `voice`) and never sends them; a bridge may
not list them. Kedi refuses a setting or level the bridge did not list before
the run, so a program never runs with a configuration it did not get.

With `modelFallback`, a run carries the profile's `> fallback:` models in
`fallbackModels`. Try them in order, after `model`, when a model request
fails. A governed bridge asks `budget/admit` before each attempt: a fallback is
a request like any other. Without `modelFallback`, Kedi refuses `> fallback:`.

### Budgets

A [run budget](../runtime/run-budgets.md) stops a step before it starts, so
the bridge asks Kedi before each step it takes:

- `dimension: "request"` before every request to its model;
- `dimension: "toolAttempt"` before each attempt of one of its own tools, with
  `nativeTools`.

Kedi charges the step to the budget of the Kedi call that started the run,
counts it, and answers `{"admitted": true}`. When the budget is spent it answers
`{"admitted": false, ...}` with a `reason`, sends `run/cancel`, and fails the run
with `RunBudgetExceeded`. The bridge then skips the step and ends the run with an
error; if it takes the step anyway, the run still fails. Kedi admits the
attempts of its own tools at `tool/call`, so a bridge never asks for them.

`governed` is a promise about the whole backend: every model request goes
through `budget/admit`, including SDK retries, tool-loop turns, native
subagents and internal calls such as summarization. A path the bridge cannot
route that way must be disabled, or the bridge must not advertise `governed`.
This is the bridge's form of [Provider Retries Must Be
Observable](../runtime/run-budgets.md#provider-retries-must-be-observable).

Without `governed`, Kedi refuses `request_limit` before the run. Without
`governed`, a bridge with `nativeTools` also cannot enforce
`tool_attempt_limit`, so Kedi refuses it.

### Approval

With `nativeTools`, the bridge asks before each call of one of its own tools:

```json
{"runId": "r1", "callId": "c1", "tool": "shell", "arguments": {"command": "ls"},
 "risk": "mutating", "description": "Run a shell command."}
```

`risk` is `read_only`, `mutating` or `sensitive`, as for Kedi tools: Kedi
allows `read_only` calls and applies the run's approval policy to the rest.
Without a policy, the call is denied. The answer is `allow`, `deny` with a
`reason` for the model, or `edit` with the `arguments` to run instead. When the
edited arguments are riskier, ask again before running them. A cancelled
approval prompt or a failing approval handler stops the run, as for Kedi tools.

Kedi's own tools need no `approval/request`: Kedi applies the policy at
`tool/call`.

### Hooks

Kedi runs a program's hooks on every bridge: `user_prompt_submit` before the
run is assembled, and the tool events around each Kedi tool at `tool/call`.
The bridge needs nothing for these.

For its own tools, a bridge with `nativeTools` lists in `initialize` the
sources whose calls it reports, in `hooks` `{"origins": [...]}`:
`adapter_native` for the backend's tools, `mcp` for its MCP servers' tools and
`provider_builtin` for tools the model provider runs. It then sends
`hook/invoke` around each such call:

1. `pre_tool_use` with the call's `arguments`, before `approval/request`.
   Kedi answers `continue`, `edit` with the `arguments` to run instead (ask
   approval for those), or `deny` with a `reason` for the model; skip the call
   then.
2. After the call, `post_tool_use` with its `result`, or
   `post_tool_use_failure` with `error` `{type, message}` and `interrupted`
   when it was cancelled.

A hook that fails or times out stops the run, as for Kedi tools: a failing
hook is a failing safety or audit control. Without a source in
`hooks.origins`, Kedi refuses `hook/invoke` for it.

### Subagents

With `subagents`, a profile's children run on the bridge: Kedi decides to start
one, checks `max_agents`, depth and concurrency, and sends the child as a `run`
of its own. Background children are runs that overlap their parent's.

`subagent` says whose run it is:

| Field | Contents |
|---|---|
| `id` | Kedi's run id for the child; `delegate_task` returns it to the parent |
| `parentRunId` | the bridge run whose `tool/call` started the child, or `null` |
| `conversationId` | the child conversation; `continue_subagent` continues it |
| `name` | the child profile |
| `limits` | the child's remaining `requests`, `toolCalls`, `inputTokens`, `outputTokens` and `totalTokens`; `null` for no limit |

Kedi enforces the limits; the bridge may pass them to its backend as well, for
an earlier stop. Kedi counts each request it admits with `budget/admit` and
each tool call it serves, so a child's request or tool-call limit stops the step
before the bridge takes it: `budget/admit` answers `admitted: false` with the
limit in `reason`, and a `tool/call` fails. For a governed bridge, `requests`
in `usage` is ignored, because Kedi has already counted them. This is why
`subagents` needs `governed`.

A child sees exactly its own tools. Kedi applies the child's approval policy
and every ancestor's to Kedi tools; for the bridge's own tools,
`approval/request` answers with the same policies, ancestors first.

A structured child result has `output.schema`, the child's result schema with
`task_summary` and `final_result`. Kedi validates the answer and sends its
correction as a `continuation`, as for any run.

With `sessions`, `continue_subagent` sends the state the bridge returned for the
child's last completed run, in `session` with the child's `conversationId`; Kedi
commits it only after accepting the run. Without `sessions`, the new task
carries the earlier turns as text.

### Live Input and Interrupts

With `liveInput`, a parent's `send_subagent_input` and a child's
`send_parent_message` reach a run while it is running. Kedi sends `run/input`
with each item's `text`, Kedi's rendering of the message. Add each text as a
user turn before the next model request, and answer `{"delivered": true}` once a
model request carries it. Answer `{"delivered": false}` if the run ends first:
Kedi then starts another attempt of the same task with the input in its prompt,
and counts that input delivered when the attempt's first request is admitted.

With `interrupt`, `send_subagent_input` with `interrupt: true` stops the child's
current attempt. Kedi sends `run/interrupt`; stop the backend's turn and answer
the run with `-32006`. In a session, put the conversation so far in the error's
`data` as `{"state": ...}`: the next attempt, with the new input, resumes from
it. An interrupt is not a cancel: the task goes on.

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
| `-32006` | interrupted: the run stopped after `run/interrupt`; the task continues in another attempt |

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
response formats, for endpoints without them. It is `governed`: it asks
`budget/admit` before every request and makes no retries. It advertises
`subagents`, `liveInput`, `interrupt` and `usage`: it runs children as ordinary
runs, adds live input before its next request, stops a fetch on `run/interrupt`,
and reports each request's token counts. It lists the OpenAI chat settings
(`temperature`, `max_tokens`, `top_p`, `seed`, `stop_sequences`,
`presence_penalty`, `frequency_penalty`), sends `effort` as `reasoning_effort`,
and tries `fallbackModels` when a request fails. With `--image-output` it asks
image models to draw (OpenRouter's chat completions with `modalities`
`image`), and with `--audio-output` audio models to speak (OpenAI's chat
completions audio); `reference_bridge_command(..., image_output=True,
audio_output=True)` passes them. Start a new bridge from it.

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
