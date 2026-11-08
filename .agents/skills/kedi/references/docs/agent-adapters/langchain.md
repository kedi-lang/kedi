# LangChain

## Installation

Install the optional LangChain integrations:

```bash
uv add 'kedi[langchain]'
```

Other providers may require their own LangChain integration package.

## Chat Models

```kedi
> adapter: langchain
> model: openai:gpt-6-luna
```

String models are passed through `langchain.chat_models.init_chat_model`.
An existing chat model object may be supplied directly to `LangChainAdapter`.
A missing model fails when the adapter first runs.

## Model Settings

Kedi supports common constructor/runtime fields including `temperature`,
`max_tokens`, `timeout`, `max_retries`, sampling penalties, `tool_choice`,
`parallel_tool_calls`, callbacks, metadata/tags, streaming controls,
OpenAI-compatible endpoint fields, and `reasoning`/`reasoning_effort`.

For `openrouter:` model names, or models with an existing `reasoning` mapping,
effort is written to `reasoning={"effort": ...}` while retaining the other
reasoning options. Other models receive `reasoning_effort`.

The Python constructor's `model_settings` accepts native options without the DSL
settings filter. An existing model is copied for each run: request fields such
as `reasoning` are applied, and its `model_kwargs` are merged with constructor
kwargs, constructor `model_settings`, then active profile settings. Original
objects are not mutated. Explicit cache keys override generated history keys.

Configure endpoint, credentials, timeout and retries when constructing a native
model. Changing those fields on an already initialized model would leave its
existing clients unchanged, so Kedi rejects that combination. String models can
receive those construction settings through the adapter.

## Codex Responses Sessions

```python
from kedi import codex_responses_model
from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.conversation import ConversationState

model = codex_responses_model(
    "gpt-6-luna", adapter="langchain", connection="websocket",
)
adapter = LangChainAdapter(model, model_settings={"reasoning": {"effort": "high"}})
conversation = ConversationState(session_id="release-review")
try:
    async with adapter.responses_session():
        for prompt in ("Review release amber.", "Recheck the selected release."):
            async with conversation.adapter_turn_async("langchain") as context:
                with adapter.conversation_scope(context):
                    answer = await adapter.invoke(prompt=prompt, instructions="Review the evidence.")
                context.record_exchange(prompt, answer)
finally:
    conversation.close()
```

Install `kedi[codex-model,langchain]` on Python 3.11 or newer. This selects
`codex-auth-helper[websocket]==1.13.0`. Authentication uses the existing Codex
login. See [Codex Models](codex-models.md) for factory parameters and the
distinction from the Codex App Server harness.

HTTP remains the default. Kedi owns one transport session for the complete
agent run, including tool turns. An explicit `responses_session()` shares the
connection across sequential calls, but does not manage message history itself.
Child Kedi runs use isolated sessions. Exceptions, cancellation and early stream
close release resources. Changes to history, instructions, schemas, tools or
settings invalidate response-ID continuation and send full input.

Updated helpers give each adapter run its own routing turn, matching Pydantic.
HTTP and WebSocket echo server-provided `x-codex-turn-state` within the run's tool
loop and clear it for the next run, without closing a shared connection. It is not
stored in history or metrics and does not guarantee a provider cache hit. Older
helpers remain compatible without this feature.

Explicit HTTP and handshake 401s have bounded same-account reload/refresh recovery.
Account changes require a new client, and permanent auth failures or exhausted
401s do not trigger HTTP fallback. Same-file refresh locking is in-process;
accepted requests and uncertain sends are not replayed by auth recovery.

The helper maps active system messages to Codex instructions and retains the
factory instruction when no active system message exists. Native LangChain tool,
structured-output and stream-event conversion remain in use. HTTP invocation
also consumes streaming responses internally.

Do not set `use_previous_response_id` or supply response IDs manually. WebSocket
requires async native model calls or Kedi's sync wrappers; HTTP response-header
capture is unavailable in this mode. With helper 1.8.0, ordinary
`ainvoke` buffers each model response and retries transient WebSocket failures
with full history. Completed local tools are not rerun by recovery. The client
`max_retries` controls reconnects (default two), with bounded backoff;
`fallback="http"` permits HTTP after retries are exhausted or the initial
handshake fails. Lost provider work can still incur unreported cost.

The WebSocket factory uses automatic streaming selection. `astream`, explicit
`streaming=True`, or streaming callbacks use the exposed-stream path, which does
not replay partial output. Cancellation, permanent auth/permission failures and
provider-hosted tools also prevent response replay. HTTP defaults are unchanged.
The model can independently request a tool again; recovery is not an exactly-once
tool guarantee. Install the pinned helper with `uv add 'kedi[codex-model]'`.
`transport_observer=` receives content-free transport metrics, including
receive failures and retry/fallback events.

Connection reuse supports cache reuse, but does not guarantee cache hits or a
fixed hit percentage. Measure actual provider usage.

## Structured Outputs

Kedi builds a Pydantic output model and calls `create_agent` with a response
format. It reads LangChain's `structured_response` field and fails if the agent
does not provide one.

Subagent JSON Schema uses LangChain `ToolStrategy`, preserving the child
schema rather than inventing a Kedi output format.

## Tool Binding

LangChain supports eager [capability bundles](../agentic-engineering/capabilities.md)
through its tool and middleware surfaces. `> use:` registers their tools, guidance,
and owned policies without a loading call. Deferred capabilities are rejected
before inference; `supports_deferred_tools` remains false.

Kedi tools become `StructuredTool` instances with JSON argument schemas.
Sync and async functions retain the correct invocation path. Kedi's approval
middleware guards projected tools and preserves tool metadata.

## MCP Tools

`MultiServerMCPClient` maps:

- Kedi stdio to LangChain `stdio`;
- Kedi SSE to `sse`;
- Kedi HTTP to `streamable_http`.

MCP tools are added to each agent run. External MCP tools are treated as
mutating by default and upgraded to sensitive when arguments target dotenv
secret files.

## Native Tool Artifacts

When artifacts are enabled, LangChain-native and MCP tool results cross Kedi's
artifact-admission middleware before their `ToolMessage` is committed to
history. Large successful content is replaced by the same compact
`ArtifactRef` representation used by Kedi-defined tools.

The original `tool_call_id` and message metadata remain attached to the result.
Error `ToolMessage` values remain native errors, approval and hook ordering are
unchanged, and tools already wrapped by Kedi are not admitted twice. Request-time
cache markers are projected only for the provider call and are not persisted in
canonical conversation history.

## CodeMode

`> codemode: enabled` replaces LangChain's model-facing application tools with
`search_tools`, `get_tool_schema`, and `execute_code`. Scoped Kedi tools and
`MultiServerMCPClient` tools enter one run-scoped catalog. Nested calls retain
Kedi argument validation and inline approval; the direct-call approval
middleware does not approve the three controls a second time.

LangChain receives Monty and boundary failures as failed tool results, allowing
the model to correct a snippet without terminating the complete agent run. See
[CodeMode](../agentic-engineering/codemode.md) for the shared contract.

## Subagent Lifecycle

LangChain supports foreground/background child runs. Request, tool-call and token
limits apply to the native child loop. Graph recursion is an additional guard,
not a substitute for a request limit. Usage is collected per model response,
independently of messages retained in history, and reported to Kedi's budget observer.

## Run Limits and Tool Corrections

`LangChainAdapter(default_usage_limits=UsageLimits(...))` uses the same limit
object as PydanticAdapter. A call's `usage_limits` overrides that default;
explicit `None` uses normal run defaults. Import `UsageLimits` from `pydantic_ai`.

Request and successful tool-call limits are checked before dispatch, including
reservations for parallel tools. Input, output, total and per-request input token
limits use provider-reported usage. These input limits include cached tokens;
Kedi's separate model-request budget uses uncached input. Provider-exact preflight
token counting and USD `cost_limit` are unsupported here and fail before sending
a model request rather than silently going unenforced.

Use `config={...}` for native LangGraph configuration. A `recursion_limit` keyword
is also forwarded into config; neither is inserted into the model's message state.
Caller callbacks are preserved when Kedi streaming callbacks are added.

Filesystem, terminal, dynamic-workflow, subagent and artifact errors use the same
Kedi correction messages as Pydantic AI. The model may correct a failed tool call
up to `tool_retries=3` times per tool per run by default. The adapter does not
automatically execute the same command again. Unclassified application exceptions
and cancellation still propagate.

Structured output that fails parsing or validation, including a subagent result's
JSON Schema, gets a correction turn like Pydantic AI's output retry: the failed
response and its validation feedback go back to the model, `output_retries=1` times
by default. Each correction is a counted model request. A truncated response is
reported as incomplete rather than corrected.

## Exact History Archives

Both framework adapters accept the same opt-in policy:

```python
from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.compaction import HistoryArchiveSettings

adapter = LangChainAdapter(
    history_archive=HistoryArchiveSettings(
        threshold_tokens=32_000,
        preserve_recent_tokens=12_000,
        minimum_reduction_tokens=8_000,
    ),
)
```

Supply the model normally and enable Kedi artifacts for the run. Without an
active artifact manager and enabled policy, archival does nothing.

Old completed tool exchanges are stored losslessly as native message records and
replaced by a checkpoint containing an artifact reference and tool-outcome counts.
This is not LLM summarization and does not introduce a DSL compaction mode.
The original user request, protected messages, unresolved tool-call pairs, earlier
checkpoints and recent tail remain intact. Provider input growth triggers the
policy when available; otherwise a conservative local estimate is used.

Graph history is persistently replaced, so archived messages do not return on the
next tool turn. Each checkpoint changes the cache epoch once. Token and cache-read
usage remain counted even after the corresponding messages leave history. Storage
or quota failures leave history unchanged. Exact archives can contain sensitive
tool output; apply the artifact store's retention/access policy accordingly.

For direct message processing, `LangChainArtifactHistoryProcessor` is exported
from `kedi.agent_adapter.compaction`; using it alone does not manage graph state.

For application-defined retention or summarization, pass a native
`history_processor=` callback. Kedi commits its accepted result through
LangGraph `Overwrite`, so removed history cannot reappear on a later tool step.
See [User-Defined History Processing](../runtime/history.md#user-defined-history-processing).

## Capability Limits

Backend-specific settings still depend on the selected chat model. Native
approval middleware covers tools represented in the LangChain agent; it cannot
grant capabilities the provider itself does not expose.
