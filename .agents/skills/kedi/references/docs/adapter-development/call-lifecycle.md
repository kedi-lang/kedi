# Call Lifecycle

Every model call a program makes goes through the same sequence. This page
lists it step by step, with the adapter methods Kedi calls at each point, and
ends with a test that records the order on the [reference
adapter](reference-adapter.md).

## Which Statement Calls What

| Kedi statement | Entry point | Arguments |
|---|---|---|
| `[x] << ...` (untyped raw invoke) | `invoke_sync` | `prompt`, `instructions` |
| `[x: T] << ...` (typed raw invoke) | `produce_sync` | `template`, `output_type=T`, `instructions` |
| `>> ... [x] ... [y: T] ...` (template capture) | `produce_sync` | `template`, `output_schema`, `instructions` |
| `> if: <claim>` and `> loop: <claim>` conditions | `produce_sync` | `template`, `output_type` (an annotated `bool`), `instructions` |
| a child run started by `delegate_task` | `invoke` or `produce_subagent` (async) | see [Subagents](subagents.md) |

Kedi calls only keyword arguments. Accept unknown ones with `**kwargs`: new
keywords are added over time, and an adapter must not fail on one it ignores.

- `prompt` and `template` are the prompt Kedi rendered, not the source
  template. A template capture's prompt starts with Kedi's field list; a
  claim's prompt is `Claim: <text>`.
- `instructions` is the complete instruction text for the call, or `None` when
  there is none. See [Prompts and Instructions](prompt-contract.md).
- `produce_sync` receives exactly one of `output_schema` (a mapping from each
  output name to its resolved Kedi type) and `output_type` (a Python type).
  See [Structured Output](structured-output.md).

## The Sequence

For one call, Kedi:

1. **Resolves the call.** It freezes the active profile, resolves the
   adapter from `> adapter:` / `> agent:` or the compiled default, and
   resolves the call's tools: the profile's Kedi tools, deferred capability
   tools, skill tools, artifact tools, and delegation tools, each wrapped for
   approval, required-tool tracking, artifact admission, and telemetry.
2. **Checks the adapter against the profile.** `> requires:` capabilities,
   `> fallback:`, history processors, `tool_reason`, native compaction,
   CodeMode, and hook events are checked against `capabilities`. If the
   adapter does not implement `SupportsProfileOverride`, a profile that sets a
   model, effort, MCP servers, settings, compaction, or CodeMode fails with
   `AgentProfileOverrideUnsupported`. These checks raise before any model work.
3. **Opens a tool scope.** For an adapter with `SupportsToolRegistration`, it
   enters `tool_scope()` and calls `register_tool(spec)` for each tool, even
   when the call has none. The empty scope keeps a nested call from
   inheriting its caller's tools.
4. **Enters the profile.** For an adapter with `SupportsProfileOverride`, it
   enters `profile_override(profile)`.
5. **Checks multimodal input.** A prompt with images, audio, video, or
   documents and an adapter without
   `supports_media_input` fails with `MediaError: Adapter '<shortname>' does
   not accept media input`.
6. **Runs the prompt hook.** For an adapter based on `HookRegistrationMixin`,
   `user_prompt_submit` hooks run now and may replace the prompt text. See
   [Hooks](hooks.md).
7. **Assembles the request.** The prompt, instruction parts, tool
   definitions, output contract, and conversation coordinates become one
   immutable `AgentRunRequest`.
8. **Opens the conversation turn.** With history enabled, it reserves the
   next turn in the conversation; `> history: enabled` on an adapter without
   `supports_stateful_history` and `SupportsConversationState` fails with
   `Adapter '<shortname>' does not support stateful history`. See
   [Conversations and History](conversations.md).
9. **Checks the run budget and request size.** An active `> budget:` needs
   the matching budget capability. Kedi estimates the request's tokens from
   the prompt, instructions, tool schemas, output schema, media, and history,
   and reserves them against the profile's context limit. See [Budgets and
   Usage](budgets.md).
10. **Enters the request scopes**: required-tool tracking, artifact admission,
    the adapter's `conversation_scope(conversation)` when history is on, and
    the model-call telemetry span.
11. **Observes usage.** Every `report_invocation` from the adapter now feeds
    the request-size reservation, telemetry, and checkpoints.
12. **Calls the adapter**: `invoke_sync(prompt=..., instructions=...)` or
    `produce_sync(template=..., output_schema=... | output_type=...,
    instructions=...)`.
13. **Checks required tools.** If a tool marked `required_before_output` did
    not return successfully, the result is rejected with
    `RequiredToolCallError`.
14. **Validates the value.** For a template capture, every output name must be
    present (`LLM did not return required key: <name>`). Each explicitly typed
    field (`[x: T]`), each typed raw invoke, and each claim is validated and
    converted with `TypeAdapter(T).validate_python`. Untyped fields (`[x]`)
    are kept as returned.
15. **Commits.** With history, the turn is committed with the adapter's
    continuation state only now, after the value was accepted. On any
    exception the turn is cancelled and the continuation state is discarded.

Steps 13 to 15 belong to Kedi; an adapter never commits history or validates
the final value itself.

## Threads and Loops

A template capture or claim runs as a job on Kedi's execution engine, so
`produce_sync` is usually called on a worker thread, possibly for several
blocks at once. Raw invokes run on the calling thread. In both cases:

- keep per-call state in `ContextVar`s or in the scopes Kedi opens, never in
  plain attributes;
- run async work with `kedi.agent_adapter.execution.run_sync`, which uses
  Kedi's shared event loop; see [Concurrency and Event
  Loops](concurrency.md).

## Recording the Sequence

This adapter records each method Kedi calls on it:

```python title="run_lifecycle.py"
import asyncio
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from chat_adapter import ChatAdapter


class TracingAdapter(ChatAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[str] = []

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        self.events.append("enter tool_scope")
        with super().tool_scope(tools):
            yield
        self.events.append("exit tool_scope")

    def register_tool(self, tool: Any) -> None:
        if tool.name == "population":
            self.events.append("register population")
        super().register_tool(tool)

    @contextmanager
    def profile_override(self, profile: Any) -> Generator[None, None, None]:
        self.events.append("enter profile_override")
        with super().profile_override(profile):
            yield
        self.events.append("exit profile_override")

    def produce_sync(self, **kwargs: Any) -> Any:
        self.events.append(f"produce_sync {sorted(kwargs)}")
        return super().produce_sync(**kwargs)


class Backend:
    async def complete(self, request: BackendRequest) -> Reply:
        if not any(m["role"] == "tool" for m in request.messages):
            return Reply(tool_calls=(ToolCall("c1", "population", {"city": "Paris"}),))
        return Reply(data={"count": 2_100_000})


SOURCE = """
@population(city: str) -> int:
  ###
  Return the population of a city.
  ###
  > tool:
    risk: read_only
  = `2_100_000`

> use: population
>> Paris has [count: int] inhabitants.
= `count`
"""

adapter = TracingAdapter(Backend(), model="m1")
runtime = compile_program(parse_program(SOURCE), adapter=adapter)
try:
    assert runtime.run_main() == 2_100_000
finally:
    asyncio.run(runtime.aclose())

assert adapter.events == [
    "enter tool_scope",
    "register population",
    "enter profile_override",
    "produce_sync ['instructions', 'output_schema', 'template']",
    "exit profile_override",
    "exit tool_scope",
], adapter.events
```

The scopes are strictly nested: tools are registered before the profile is
entered, both are open while the adapter runs, and both close in reverse
order after it returns.
