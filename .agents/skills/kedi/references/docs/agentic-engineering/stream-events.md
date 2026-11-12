# Stream Events

Kedi exposes completed semantic agent messages as an adapter-neutral event
stream. This lets a terminal, editor, web UI, or orchestrator show useful
progress while the authoritative adapter call continues normally.

The stream is deliberately **not** a token stream. Provider deltas, mutable
snapshots, and hidden reasoning stay private. Kedi emits a message only after
the adapter has identified a complete assistant message and classified it as
commentary or final output.

## Event Lifecycle

Every observed run emits one start event and one terminal event:

```text
started
commentary  Inspecting the package manifest.
commentary  The dependency graph is consistent.
final       Validation passed with no blocking findings.
completed
```

The terminal state is exactly one of `completed`, `failed`, or `cancelled`.
Failed and cancelled runs do not publish unfinished private text as a final
message. The adapter's return value or exception remains authoritative; events
are an observation side channel.

### Commentary

`phase="commentary"` is a complete progress message that is not the run's final
answer. Kedi can identify commentary when:

- a provider or harness marks a completed message as commentary;
- a tool-call boundary seals completed assistant text produced before the
  tool runs;
- a later completed assistant message makes an earlier message intermediate.

Kedi does not expose chain-of-thought or infer commentary from partial token
chunks. If a backend provides only a final response, the run may contain no
commentary events.

### Final

`phase="final"` is the completed natural-language response returned by a
successful raw `invoke()` call. At most one final message is authoritative for
one run. Structured `produce()` results remain typed data and are not mirrored
as invented natural-language final messages.

Tool calls are not `AgentMessageEvent` values. Use Kedi's tool telemetry or the
backend's native event surface when a UI also needs tool-call cards. A tool
boundary only determines that preceding completed text is commentary.

## Observe With a Callback

The public types and observer are exported from `kedi`:

```python
import asyncio

from kedi import AgentMessageEvent, AgentRunStateEvent, observe_agent_events
from kedi.agent_adapter import PydanticAdapter


def render(event: AgentMessageEvent | AgentRunStateEvent) -> None:
    if isinstance(event, AgentMessageEvent):
        print(f"{event.phase}: {event.content}")
    else:
        print(f"run {event.run_id}: {event.state}")


adapter = PydanticAdapter("openai:gpt-6-luna")

with observe_agent_events(render):
    result = asyncio.run(adapter.invoke(prompt="Inspect the parser."))

print("result:", result)
```

The callback runs on Kedi's dispatcher thread. Keep it short and thread-safe.
For UI frameworks or asyncio applications, bridge through their event loop
instead of mutating UI state directly from the callback.

Event observation does not modify prompts, schemas, tools, message history,
usage accounting, return values, or exception behavior.

Unlike [hooks](hooks.md), observers cannot edit or deny a request. A progress
message is not a permission decision, a tool receipt, or proof that a task
succeeded. The completed adapter result remains the application contract.

## Consume From Async Code

`AsyncAgentEventQueue` is a thread-safe sink that forwards dispatcher events to
an `asyncio.Queue`:

```python
import asyncio

from kedi import AgentRunStateEvent, AsyncAgentEventQueue, observe_agent_events
from kedi.agent_adapter import PydanticAdapter


async def main() -> str:
    adapter = PydanticAdapter("openai:gpt-6-luna")
    events = AsyncAgentEventQueue()
    root_run_id = None

    with observe_agent_events(events):
        task = asyncio.create_task(adapter.invoke(prompt="Inspect the parser."))

        try:
            while True:
                event = await events.queue.get()
                if (
                    isinstance(event, AgentRunStateEvent)
                    and event.state == "started"
                    and event.parent_run_id is None
                ):
                    root_run_id = event.run_id

                if (
                    isinstance(event, AgentRunStateEvent)
                    and event.run_id == root_run_id
                    and event.state in {"completed", "failed", "cancelled"}
                ):
                    break

            return await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


print(asyncio.run(main()))
```

Construct the queue inside the target event loop. Pass `loop=` explicitly only
when construction and consumption occur in different setup code.

This example owns one root invocation. The `finally` block also cancels and
joins it if the consumer is cancelled. In a multi-run UI, keep separate run
identities instead of treating the most recently started root as the only run.

## Event Fields

`AgentMessageEvent` carries:

| Field | Meaning |
| --- | --- |
| `run_id` | Stable identity of the emitting agent run |
| `parent_run_id` | Parent run for subagents, otherwise `None` |
| `message_id` | Backend message identity normalized by the adapter |
| `adapter` | Adapter shortname such as `pydantic`, `langchain`, or `codex` |
| `sequence` | Monotonic event order within that run |
| `phase` | `commentary` or `final` |
| `content` | Complete immutable message text |
| `segment_index`, `segment_count` | Ordering metadata for a segmented completed message |

`AgentRunStateEvent` carries `run_id`, `parent_run_id`, `adapter`, `sequence`,
and `state`. Sequence numbers are scoped per run; use `(run_id, sequence)` when
merging concurrently emitted parent and subagent events.

## Subagents and Concurrent Runs

Subagent events use the coordinator-assigned child `run_id` and identify the
calling run through `parent_run_id`. Do not assume events from concurrent runs
are globally ordered. Maintain independent render state for each `run_id`, then
use the parent relationship to nest child activity.

A root run has `parent_run_id=None`. A terminal event closes only its own run;
a child completion must not be mistaken for root completion.

## Delivery and Backpressure

Kedi dispatches events away from provider and SDK reader threads so a slow
observer does not directly block model execution. Control and final events are
kept separate from the bounded commentary queue. Under sustained commentary
overflow, Kedi drops the oldest pending commentary and logs a warning rather
than allowing unbounded memory growth.

Observer failures are isolated from the model call. Repeated callback failures
detach that observer, and a callback that cannot drain during shutdown is
detached after a bounded wait. Observation is therefore suitable for progress
reporting, not durable audit storage. Persist authoritative traces through
[Telemetry](../runtime/telemetry.md).

## Adapter Support

Semantic message boundaries are capability-driven:

| Adapter | Stream events |
| --- | :---: |
| Pydantic AI | yes |
| LangChain | yes |
| Claude Agent SDK | yes |
| Codex App Server | yes |
| ACP | yes |
| WebGPU | no |

`LazyAdapter` reports the resolved adapter's capability. Unsupported adapters
set `supports_stream_events=False`; Kedi does not fabricate trustworthy message
boundaries from raw token chunks.

See the generated [Capability Matrix](../reference/capability-matrix.md) for the
current built-in adapter metadata.
