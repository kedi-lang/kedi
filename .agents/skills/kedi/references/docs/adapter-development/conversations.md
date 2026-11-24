# Conversations and History

With `> history: enabled`, consecutive calls of a profile form one conversation:
each call sees the earlier turns. Kedi owns the conversation: turn order,
commit and rollback, persistence, and cache identity. The adapter owns the
native form of the history, the messages its backend needs to continue, and
stores it in state Kedi hands it.

## The Protocol

`kedi.agent_adapter.SupportsConversationState`:

```python
from contextlib import AbstractContextManager
from kedi.agent_adapter import AdapterConversationContext

def conversation_scope(
    self, conversation: AdapterConversationContext
) -> AbstractContextManager[None]: ...
```

Advertise `supports_stateful_history=True`. Without both, `> history: enabled`
fails before the call: `Adapter '<shortname>' does not support stateful
history`.

## One Turn

For each call in a conversation, Kedi:

1. reserves the next turn; calls of one conversation run strictly in order,
   even when the program runs blocks in parallel;
2. builds an `AdapterConversationContext` with the adapter's state from the
   last committed turn as `resume_state`;
3. enters `conversation_scope(context)` around the call (see [Call
   Lifecycle](call-lifecycle.md), step 10);
4. after a successful call, commits `context.result_state` as the adapter's
   state, together with the prompt and the result as portable turns;
5. after a failed or cancelled call, discards `result_state` and keeps the
   previous state.

The adapter reads `resume_state` at the start of the call and writes
`result_state` before it returns. It never commits anything itself.

## The Context

| Field | Use |
|---|---|
| `resume_state` | the adapter's state from the last committed turn of this lane; empty for the first turn |
| `result_state` | write the state the next turn should resume from |
| `session_id`, `turn_sequence` | the conversation and the turn number |
| `cache_key`, `cache_epoch` | pass `cache_key` to provider prompt caching; it changes when the epoch does |
| `cache_prefix_changed` | set to `True` when this turn rewrote earlier history (for example native compaction); Kedi then rotates the cache epoch |
| `cleanup` | a callable that releases a live resource kept across turns, such as a remote session; Kedi calls it when the state is replaced or the conversation closes |
| `require_explicit_completion()`, `mark_completed()` | for backends that report completion asynchronously: call the first at the start and the second when the backend confirms the turn, and Kedi commits only then |

### Lanes

Kedi keys the adapter's state by a lane: a fingerprint of the adapter class
and shortname, its default model, and the profile's model, fallback models,
effort, settings, MCP servers, and tool definitions. When any of them changes
between turns, the new turn gets an empty `resume_state`: native history
recorded for one model and tool set is not replayed into another. The
portable turns are still recorded and persisted by Kedi.

### What to Store

`resume_state` and `result_state` are persisted with interactive sessions and
snapshots. Store only JSON-compatible values: serialized messages, a remote
thread id, a digest of the instructions. Keep live clients, streams, and SDK
objects out; hold those in `cleanup`'s closure.

Store the complete causal message sequence, including tool calls and their
results in order. Provider-native tool calls and results stay native
messages; never flatten them into user-prompt text.

## Implementing It

```python title="stateful_adapter.py"
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from kedi.agent_adapter import AdapterConversationContext

from backend import Reply
from chat_adapter import ChatAdapter


class StatefulAdapter(ChatAdapter):
    capabilities = replace(ChatAdapter.capabilities, supports_stateful_history=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._conversation: ContextVar[AdapterConversationContext | None] = ContextVar(
            f"chat_conversation_{id(self)}", default=None
        )

    @contextmanager
    def conversation_scope(
        self, conversation: AdapterConversationContext
    ) -> Generator[None, None, None]:
        token = self._conversation.set(conversation)
        try:
            yield
        finally:
            self._conversation.reset(token)

    async def _run(
        self,
        *,
        prompt: str,
        instructions: str | None,
        output_schema: dict[str, Any] | None,
        history: Sequence[dict[str, Any]] = (),
        max_requests: int | None = None,
    ) -> tuple[Reply, list[dict[str, Any]]]:
        conversation = self._conversation.get()
        if conversation is not None and not history:
            history = conversation.resume_state.get("messages", [])
        reply, messages = await super()._run(
            prompt=prompt,
            instructions=instructions,
            output_schema=output_schema,
            history=history,
            max_requests=max_requests,
        )
        if conversation is not None:
            conversation.result_state["messages"] = messages
        return reply, messages
```

The class adds history at the one place every call passes through. Writing
`result_state` on the way out is safe even when Kedi later rejects the value
(a missing field, a required tool that never ran): Kedi commits it only after
accepting the result.

## History Processing

`> history:` may name a history processor, a callable that receives the
native message list before each model request and returns a shorter or
rewritten one. Advertise `supports_history_processing=True` only when the
adapter runs processors through Kedi's runner,
`kedi.agent_adapter.history_processing.HistoryProcessorRunner`:

| Step | Call |
|---|---|
| construct once | `HistoryProcessorRunner(processor, codec, condition)` with a `HistoryMessageCodec` for your message type |
| per call | `with runner.run_scope(conversation):` around the run |
| before each model request | `outcome = await runner.process(messages, model_id=model)`; send `outcome.messages` |
| after a successful run | `runner.capture(messages, stored_messages)` to record the lineage of the saved history |
| before sending | `runner.prepare_transport()` tells whether to reset a server-side continuation and which cache key to use |

The codec has two methods: `clone(messages)` returns an independent copy,
and `inspect(messages)` returns a `HistorySnapshot` describing each message's
role, tool-call pairing, and fingerprint, so the runner can protect tool-call
groups and unfinished turns from being cut apart. Kedi's Pydantic AI and
LangChain adapters (`PydanticHistoryMessageCodec`,
`LangChainHistoryMessageCodec`) are the reference implementations.

## Running It

```python title="run_conversations.py"
import asyncio

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from stateful_adapter import StatefulAdapter


class Recorder:
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return Reply(text=f"answer to {request.messages[-1]['content']}")


def run(source: str, backend: Recorder) -> None:
    runtime = compile_program(parse_program(source), adapter=StatefulAdapter(backend, model="m1"))
    try:
        runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def users(request: BackendRequest) -> list[str]:
    return [m["content"] for m in request.messages if m["role"] == "user"]


backend = Recorder()
run("> history: enabled\n[a] << First.\n[b] << Second.\n= <b>\n", backend)
first, second = backend.requests
assert users(first) == ["First."]
assert users(second) == ["First.", "Second."]
assert second.messages[1] == {"role": "assistant", "content": "answer to First."}

backend = Recorder()
run(
    "> history: enabled\n[a] << First.\n> model: m2\n[b] << Second.\n= <b>\n",
    backend,
)
assert users(backend.requests[1]) == ["Second."]
```

The second request carries the first turn. A different model starts a new
lane, so its first request carries no native history.
