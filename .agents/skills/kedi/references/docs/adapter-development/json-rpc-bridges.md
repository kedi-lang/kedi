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
| `initialize` | Kedi → bridge | `protocolVersion`, `client`, optional `testModel` | `protocolVersion`, `backend` (`name`, `version`), `kind`, `capabilities`, `mediaKinds`, `settings`, `effort`, `mcp`; see [Settings, Effort and Fallback](#settings-effort-and-fallback) and [MCP Servers](#mcp-servers) |
| `run` | Kedi → bridge | see below | `{"text": ...}`, `{"data": ...}` or `{"content": [...]}`, optional `usage`, and `state` in a session |
| `run/cancel` | Kedi → bridge (notification) | `runId` | — |
| `run/input` | Kedi → bridge | `runId`, `items` (`id`, `text`) | `{"delivered": true}` or `{"delivered": false}`; see [Live Input and Interrupts](#live-input-and-interrupts) |
| `run/interrupt` | Kedi → bridge (notification) | `runId` | — ; the run answers with `-32006` |
| `run/usage` | bridge → Kedi (notification) | `runId`, `inputTokens`, `outputTokens`, optional `totalTokens`, `cacheReadTokens`, `cacheWriteTokens` | — |
| `tool/call` | bridge → Kedi | `runId`, `callId`, `name`, `arguments` | `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}` |
| `budget/admit` | bridge → Kedi | `runId`, `dimension` | `{"admitted": true}`, or `admitted: false` with `dimension` and `reason`, and for a run budget `limit`, `consumed` and `scopeDepth`; see [Budgets](#budgets) |
| `model/complete` | bridge → Kedi | `runId`, `model`, `system`, `messages` (Kedi's schema), `tools`, `output` | `message`: the assistant's message in Kedi's schema; see [Test Model](#test-model) |
| `history/process` | bridge → Kedi | `runId`, `messages` (the conversation so far, in Kedi's schema) | `messages`: what the program's history processor kept; see [History Processing](#history-processing) |
| `media/get` | bridge → Kedi | `runId`, `ref` | `kind`, `mediaType`, and `data` (base64) or `url`; see [Transcript](#transcript) |
| `artifact/admit` | bridge → Kedi | `runId`, `callId`, `tool`, `origin`, `result` | `result`: what the model sees, the result itself or Kedi's artifact reference; see [MCP Servers](#mcp-servers) |
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
| `transcript` | Kedi holds the conversation in its own schema; runs carry `history`, results return the turn's `messages`; enables `> history: enabled` and `history_replay` (see [Transcript](#transcript)) |
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
| `hooks` | the tool events that have handlers, for a bridge with `nativeTools`: send `hook/invoke` for these only (see [Hooks](#hooks)); empty otherwise |
| `mcpServers` | the profile's `> mcp:` servers: `transport`, `command`, `args`, `env`, `url`, `headers` |
| `admitResults` | for a bridge with `nativeTools`: whether each of its tools' results goes through `artifact/admit` before the model sees it |
| `system` | the system text: send it verbatim as the system or developer message |
| `instructionParts` | the same text as parts, each with `identity`, `source` and `content` |
| `user` | the prompt: send it verbatim as the user turn |
| `userContent` | `null`, or the user turn with its media; see [Media](#media) |
| `output` | `null` for text; `schema` (the output JSON Schema) and `toolName` for data; or `media` for an image or speech the model makes (see [Media Output](#media-output)) |
| `tools` | each tool's `name`, `description`, `inputSchema`, `outputSchema` and `requiredBeforeOutput` |
| `continuation` | `null`, or `answer` (the bridge's previous result) and `message` (Kedi's correction) |
| `session` | `null`, or `id`, `turn` and `state`; see [Sessions](#sessions) |
| `history` | in transcript mode, the conversation's earlier messages in Kedi's schema; absent otherwise |
| `processHistory` | in transcript mode, whether the program has a history processor: ask `history/process` before each model request |
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

A bridge with `nativeTools` runs tools of its own, so it reports their calls
to the program's hooks: that is part of `nativeTools`, as `approval/request`
is. Each run's `hooks` lists the tool events that have handlers; for those,
send `hook/invoke` around each call of a bridge tool, with `origin`
`adapter_native` for the backend's tools or `provider_builtin` for tools the
model provider runs. With no handlers, `hooks` is empty and nothing is sent.

1. `pre_tool_use` with the call's `arguments`, before `approval/request`.
   Kedi answers `continue`, `edit` with the `arguments` to run instead (ask
   approval for those), or `deny` with a `reason` for the model; skip the call
   then.
2. After the call, `post_tool_use` with its `result`, or
   `post_tool_use_failure` with `error` `{type, message}` and `interrupted`
   when it was cancelled.

A hook that fails or times out stops the run, as for Kedi tools: a failing
hook is a failing safety or audit control. Kedi refuses `hook/invoke` from a
bridge without `nativeTools`.

### MCP Servers

A bridge with `nativeTools` lists in `initialize` the `> mcp:` transports it
connects, in `mcp` `{"transports": [...]}`: `stdio`, `http` (streamable HTTP)
and `sse`. Kedi refuses, before the run, a server whose transport the bridge
did not list, and every server for a bridge that lists none. Each run carries
the profile's servers in `mcpServers`; connect them for the run, offer their
tools to the model, and close them when the run ends.

An MCP tool is the bridge's own tool, so each call goes through the steps a
Kedi tool's wrapper takes, in this order:

1. `hook/invoke` `pre_tool_use` with `origin` `mcp`, when the run's `hooks`
   lists it;
2. `approval/request` with `origin` `mcp` and `risk` `mutating`: a server
   cannot declare its tools read-only to Kedi;
3. `budget/admit` with `toolAttempt`, for a governed bridge;
4. the call itself, then `hook/invoke` `post_tool_use` or
   `post_tool_use_failure`;
5. `artifact/admit` with the result, when the run's `admitResults` is true:
   send the model the `result` Kedi returns. For a large result it is an
   artifact reference, and the model reads the rest with Kedi's artifact tools.

`artifact/admit` applies to every result of a bridge tool, not only MCP ones.
Kedi's tool-attempt budget still refuses profiles with MCP servers, as for
every adapter.

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

### Transcript

With `transcript`, Kedi holds the conversation instead of the bridge. A run in
a conversation carries `history`, the earlier turns' messages, and the bridge
answers with `messages`, this turn's new ones, the user turn included. Kedi
checks them against its schema and appends them when it commits the turn, as it
commits `state`. A transcript run that answers without `messages` fails.

```json
[{"role": "user", "parts": [{"type": "text", "text": "Look at "}, {"type": "media", "ref": "photo"}]},
 {"role": "assistant", "parts": [{"type": "toolCall", "id": "c1", "name": "lookup", "arguments": {}}],
  "providerData": {"...": "the bridge's, sent back as given"}},
 {"role": "tool", "parts": [{"type": "toolResult", "id": "c1", "result": "found"}]},
 {"role": "assistant", "parts": [{"type": "text", "text": "Done."}]}]
```

| Part | Fields |
|---|---|
| `text` | `text` |
| `media` | `ref`: the media's name in `userContent` |
| `toolCall` | `id`, `name`, `arguments` |
| `toolResult` | `id`, `result`, optional `isError` |
| `reasoning` | optional `text` |

`providerData` is opaque to Kedi: put there what the backend needs back exactly,
such as an encrypted reasoning block. Media is named by `ref` after the turn
that attached it, so its bytes are not sent again with each `history`. A bridge
that needs the bytes, for example after a restart, asks `media/get` with the
`ref`.

Kedi gives each recorded message an `id`, so two identical messages stay
distinct; send messages back with the `id` they came with.

### History Processing

With a `> history:` processor, a transcript run carries `processHistory: true`.
Before each model request, tool-loop turns included, send `history/process`
with the conversation so far in Kedi's schema: `history` followed by this
turn's messages. Kedi runs the program's processor, as the in-process adapters
do before each request, and answers with the messages it kept. Send the model
those instead, and continue the turn from them.

The processor may drop or rewrite earlier messages, so such a run answers with
`transcript`, the whole conversation as it stands after the turn, instead of
the turn's `messages`. Kedi stores it as the conversation. A processor that
fails fails the run, as in-process. `transcript` enables `history_processing`.

A bridge may support both modes. Kedi uses `transcript` for an
`agent-framework` bridge, and for an `agent-harness` only when the program
needs Kedi's copy of the conversation (`> requires: history_replay`, history
processing); a harness otherwise keeps its own `state`. `history` travels in
every run, so a long conversation counts against the 8 MiB frame.

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

When a run fails in the harness, put the harness's own error in the error's
`data`, so the user sees what went wrong behind the bridge:

```json
{"code": -32002, "message": "the harness failed",
 "data": {"harness": {"type": "ProviderError", "message": "429 rate limited",
                      "cause": {"type": "HTTPError", "message": "Too Many Requests"}}}}
```

Kedi quotes it, with the bridge's last five stderr lines, in the error it
raises. When Kedi stops the run itself, for example on an exhausted budget,
its own error comes first and the bridge's report follows it:

```text
Kedi error: RuntimeError
  Run request budget exhausted (limit 1, consumed 1)
  from the bridge: the harness failed
  harness error: ProviderError: 429 rate limited
    caused by: HTTPError: Too Many Requests
  bridge stderr:
    [harness] retrying the provider
```

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
`subagents`, `liveInput`, `interrupt`, `usage` and `transcript`: it runs children as ordinary
runs, adds live input before its next request, stops a fetch on `run/interrupt`,
and reports each request's token counts. It lists the OpenAI chat settings
(`temperature`, `max_tokens`, `top_p`, `seed`, `stop_sequences`,
`presence_penalty`, `frequency_penalty`), sends `effort` as `reasoning_effort`,
and tries `fallbackModels` when a request fails. With `--image-output` it asks
image models to draw (OpenRouter's chat completions with `modalities`
`image`), and with `--audio-output` audio models to speak (OpenAI's chat
completions audio); `reference_bridge_command(..., image_output=True,
audio_output=True)` passes them. In native mode it connects `stdio` MCP
servers for each run, taking each MCP call through the steps above. Start a new
bridge from it.

## Test Model

A bridge that declares `testModel` can have Kedi answer its model requests,
for tests with no model endpoint, no network and no cost. Kedi asks for it with
`testModel: true` in `initialize`; the bridge then sends each model request,
admitted as usual, as `model/complete` instead of calling its model:

| Field | Content |
|---|---|
| `messages` | the conversation as the model would get it, in Kedi's transcript schema; media may be inline (`kind`, `mediaType`, `data` or `url`) |
| `system` | the system text the bridge sends its model |
| `tools` | the tools the model may call: `name`, `description`, `inputSchema` |
| `output` | `schema` and `toolName` for a structured run, `null` for text |

Kedi answers with `message`, the assistant's reply in Kedi's schema (`text`,
`toolCall`, `reasoning` parts), which the bridge treats as its model's.
Without `testModel`, Kedi refuses a test model at the handshake.

In Python, `JsonRpcAdapter(command, test_model=script)` takes the script, a
function, sync or async, in the style of Pydantic AI's `FunctionModel`: it
receives a `ModelCall` and returns the assistant's text, or a message with
`parts`. Because it sees exactly what the bridge would send its model, a test
can check the bridge's prompt, tools and history as well as script its answers:

```python
from kedi.agent_adapter.adapters.jsonrpc import JsonRpcAdapter, ModelCall


def script(call: ModelCall):
    if not any(message["role"] == "tool" for message in call.messages):
        return {"parts": [{"type": "toolCall", "id": "t1", "name": "lookup",
                           "arguments": {"query": "Lyon"}}]}
    return "Found it."


adapter = JsonRpcAdapter(["node", "my_bridge.mjs"], test_model=script)
```

A harness usually has a scripted model of its own to build this on: pi's
`@earendil-works/pi-ai` faux provider takes a function of the transcript, and
a proxy that a bridge runs in front of a harness can answer from Kedi instead
of forwarding.

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
