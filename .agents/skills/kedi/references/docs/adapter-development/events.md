# Events, Streaming, and Telemetry

A host can follow a run while it happens: the notebook shows a child's
progress, an A2A server forwards messages, a UI renders the model's text. Kedi
defines the event model and delivers events to the host. The adapter tells
Kedi when the backend produced a message and when a run changed state.

## The Events

`kedi.agent_adapter`:

| Event | Fields | Published |
|---|---|---|
| `AgentRunStateEvent` | `run_id`, `parent_run_id`, `agent_name`, `adapter`, `sequence`, `state` (`started`, `completed`, `failed`, `cancelled`) | when a run starts and ends |
| `AgentMessageEvent` | `run_id`, `parent_run_id`, `agent_name`, `adapter`, `sequence`, `message_id`, `phase` (`commentary`, `final`), `content`, `segment_index`, `segment_count` | when a model message is complete |

`commentary` is text the model produced on its way, such as "Let me look that
up" before a tool call; `final` is the answer. `run_id` and `parent_run_id`
tie a child's events to the delegation that started it.

A host observes events by binding a dispatcher around the work:

```python
from kedi.agent_adapter import AgentEventDispatcher, bind_agent_event_dispatcher

dispatcher = AgentEventDispatcher(print)  # a callback or an AgentEventSink
with bind_agent_event_dispatcher(dispatcher):
    runtime.run_main()
dispatcher.close()  # delivers what is still queued
```

The dispatcher delivers on its own thread, so a slow sink never blocks the
model loop; `AsyncAgentEventQueue` adapts it to an asyncio consumer.

## The Adapter's Part

Wrap each backend run in `agent_event_run(shortname)` from
`kedi.agent_adapter.events` and report messages to the resolver it yields:

| Resolver call | When |
|---|---|
| (entering the context) | publishes `started` |
| `append_delta(message_id, text)` | a streamed chunk of a message arrived |
| `complete_message(message_id, text=None)` | a message is complete; without `text` the deltas are joined |
| `mark_tool_boundary()` | the model called a tool: pending messages become `commentary` |
| `mark_message_boundary()` | a new message started without a tool call |
| `complete_run(final_text=)` | a text call succeeded: publishes the answer as `final`, then `completed` |
| (leaving the context) | publishes `completed`, `failed`, or `cancelled` from how the block exited |

The resolver is `None` when no host observes the run; skip the calls then.
Publish `final` only for text calls (`invoke`): a structured call's result is
a value, not a message. Advertise `supports_stream_events=True` when the
adapter publishes messages this way.

```python title="event_adapter.py"
import itertools
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from kedi.agent_adapter.events import AgentMessageResolver, agent_event_run

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter


class EventAdapter(ChatAdapter):
    capabilities = replace(ChatAdapter.capabilities, supports_stream_events=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._resolver: ContextVar[AgentMessageResolver | None] = ContextVar(
            f"chat_events_{id(self)}", default=None
        )
        self._message_ids = itertools.count(1)

    async def _run(self, **kwargs: Any) -> Any:
        with agent_event_run(self.shortname) as resolver:
            token = self._resolver.set(resolver)
            try:
                reply, messages = await super()._run(**kwargs)
            finally:
                self._resolver.reset(token)
            if resolver is not None and kwargs.get("output_schema") is None and reply.text:
                resolver.complete_run(final_text=reply.text)
            return reply, messages

    async def _complete(self, request: BackendRequest) -> Reply:
        reply = await super()._complete(request)
        resolver = self._resolver.get()
        if resolver is not None and reply.tool_calls:
            if reply.text:
                resolver.complete_message(f"message-{next(self._message_ids)}", reply.text)
            resolver.mark_tool_boundary()
        return reply
```

## Telemetry

Kedi creates the spans: the agent call, each model call, each tool call (from
the wrapped `fn`), approvals, hooks, and children. The adapter contributes:

- **usage**, through `report_invocation` (see [Budgets and Usage](budgets.md));
- **the model name**, through `default_model()` and, for children,
  `telemetry_model_name()`, used when the profile sets no model;
- **errors**, by raising the provider's exception, which the spans record.

Do not open spans for operations Kedi already traces.

## Running It

```python title="run_events.py"
import asyncio

from kedi.agent_adapter import (
    AgentEventDispatcher,
    AgentMessageEvent,
    AgentRunStateEvent,
    bind_agent_event_dispatcher,
)
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from event_adapter import EventAdapter

SOURCE = """
@population(city: str) -> str:
  ###
  Return the population of a city.
  ###
  > tool:
    risk: read_only
  = `"2.1M"`

> use: population
[answer] << How many people live in Paris?
= <answer>
"""


class Script:
    async def complete(self, request: BackendRequest) -> Reply:
        if not any(m["role"] == "tool" for m in request.messages):
            return Reply(
                text="Let me look that up.",
                tool_calls=(ToolCall("c1", "population", {"city": "Paris"}),),
            )
        return Reply(text="About 2.1 million.")


events: list[object] = []
dispatcher = AgentEventDispatcher(events.append)
runtime = compile_program(parse_program(SOURCE), adapter=EventAdapter(Script(), model="m1"))
try:
    with bind_agent_event_dispatcher(dispatcher):
        assert runtime.run_main() == "About 2.1 million."
finally:
    dispatcher.close()
    asyncio.run(runtime.aclose())

timeline = [
    (event.state,) if isinstance(event, AgentRunStateEvent) else (event.phase, event.content)
    for event in events
    if isinstance(event, (AgentRunStateEvent, AgentMessageEvent))
]
assert timeline == [
    ("started",),
    ("commentary", "Let me look that up."),
    ("final", "About 2.1 million."),
    ("completed",),
], timeline
assert len({event.run_id for event in events}) == 1
```
