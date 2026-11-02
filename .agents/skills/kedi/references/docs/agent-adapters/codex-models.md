# Codex-Authenticated Models

Codex Responses provides model access inside a framework adapter. This is different from [Codex App Server](codex.md), which runs an agent harness. Choosing a model route does not switch the surrounding agent framework.

## Codex Responses Connections

The factory signature is
`codex_responses_model(model_id, adapter, *, connection="http", fallback="error", transport_observer=None)`.
`adapter` must be `"pydantic"` or `"langchain"`. The identifier may include the
`codex/` prefix. `fallback="http"` is valid only with WebSocket.
`transport_observer` receives content-free transport events. The helper reads
existing Codex login credentials; `KEDI_CODEX_AUTH_FILE` selects an explicit
auth file. Do not include that file in source, logs, or published examples.

`codex_responses_model()` builds a Codex-authenticated Pydantic model. HTTP is
the default. Install `kedi[codex-model]` on Python 3.11 or newer for the pinned
`codex-auth-helper[websocket]==1.8.0`. Select WebSocket explicitly; it is not
enabled just by installing the extra.

```python
from kedi import codex_responses_model
from kedi.agent_adapter import PydanticAdapter

model = codex_responses_model(
    "gpt-6-luna",
    adapter="pydantic",
    connection="websocket",
    fallback="error",
)
adapter = PydanticAdapter(model)

async def review():
    async with adapter.responses_session():
        first = await adapter.run("The current release is amber. Acknowledge it.")
        second = await adapter.run(
            "Which release is current?",
            message_history=first.all_messages(),
        )
    return second.output
```

An ordinary adapter run keeps its connection across model/tool turns. The
explicit session above also retains it across sequential adapter calls.
It owns the connection, not the history: supply native message history as
shown or use Kedi's normal stateful-history surface. Concurrent child tasks
remain isolated, and exit, errors, and cancellation clean up the connection.

Updated helper builds scope a separate routing turn to every adapter run. HTTP
and WebSocket echo the first server-provided `x-codex-turn-state` within its tool
loop, but discard it before the next run, even on a shared connection. Kedi does
not put this opaque value in history or transport metrics. Prompt-cache keys and
native history remain unchanged; provider cache hits are not guaranteed.

Auth recovery reloads same-account credentials after an explicit 401, then allows
one refresh if necessary. Account changes require a new client; permanent auth
failures and exhausted 401s do not trigger HTTP fallback. Same-file refresh locking
is in-process only. Older helper builds work without the new turn-state feature.

Append-only requests can transmit only their new suffix. History edits or
changed request settings reset continuation. If the socket is already known
to be closed before the next request, a new connection receives full history.
Changing handshake headers also replaces the connection.

With helper 1.8.0, non-streaming model requests recover
transient WebSocket send/receive failures by discarding incomplete output and
rebuilding full history on a new connection. Completed local tools are not rerun
by recovery. The client `max_retries` controls reconnects (default two), with
bounded backoff. `fallback="http"` permits HTTP after retries are exhausted, as
well as initial-handshake fallback. The model can still choose another tool
call, and lost provider work may incur unreported cost.

Raw client calls and exposed streams are not replayed; cancellation, permanent
auth/permission failures and provider-hosted tools also prevent response replay.
Install the pinned helper with `uv add 'kedi[codex-model]'`. Stream consumers can stop on
`response.completed` without discarding the completed response; stopping before
completion abandons it. The closing handshake has a 250 ms timeout, separate
from generation and tool execution.

Connection reuse can improve provider cache reads, but does not guarantee a
particular hit rate or lower latency. Cached input is part of total input,
not an additional token count.
