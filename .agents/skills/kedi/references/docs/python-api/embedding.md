# Embedding and Resource Ownership

Use `query` for a procedure body, `bind` for a complete file, and `interactive`
for incremental fragments. Low-level compilation is appropriate when the host
owns source loading, execution, and resource cleanup.

## A Complete Deterministic Host

```python
import asyncio

from kedi.lang import compile_program, parse_program

source = """
@total(values: list[int]) -> int:
    = `sum(values)`

= `total(values)`
"""

runtime = compile_program(
    parse_program(source, source_path="<totals>"),
    runtime_globals={"values": [2, 3, 5]},
)
try:
    assert runtime.run_main() == 10
finally:
    asyncio.run(runtime.aclose())
```

No model is called. In an async host, use `await runtime.aclose()` in its
existing event loop rather than nesting `asyncio.run`. Close caller-owned
adapters and external clients according to their integration's contract.

## Async Hosts With Reusable Clients

An async HTTP client can keep sockets bound to the event loop where it was
used. Do not reuse it across independent `asyncio.run()` calls. In an async
server or notebook host, keep model requests on the host's existing loop while
the synchronous Kedi program runs in a worker:

```python
import asyncio

from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.lang import compile_program, parse_program

async def run_program(source, adapter):
    with use_async_owner_loop():
        runtime = compile_program(parse_program(source), adapter=adapter)
        try:
            return await asyncio.to_thread(runtime.run_main)
        finally:
            await runtime.aclose()
```

This applies to both Pydantic and LangChain adapters. The context propagates
through Kedi's workers, preserves deadlines and cancellation, and does not
take ownership of the loop or client. Create and close caller-owned clients
on that same host loop. Calling synchronous Kedi code directly on the selected
loop is rejected instead of blocking the loop on itself. Outside this explicit
scope, the existing synchronous execution behavior is unchanged.

## Which Configuration Surface?

| Entry point | Owns | Does not provide |
| --- | --- | --- |
| `configure`, `context` | Default/temporary backend, profile and execution options | A running incremental session |
| `query`, `bind` | Callable-local overrides and optional response memoization | Executor or subagent-limit constructor parameters |
| `interactive` | Incremental runtime; executor, engine, cwd, subagent limits | Parallel or re-entrant fragment execution |
| `compile_program` | Parsed program and low-level runtime construction | High-level decorator configuration merging |
| `session` | Shared conversation/artifact lifetime across separate calls | Persistent Kedi variables or declaration frames |

Do not pass an executor or subagent limit through `configure(**adapter_kwargs)`
expecting it to configure the runtime. Those extra kwargs construct the adapter.
See [Public Parameters](public-parameters.md) for accepted names.

## Bound Agent Surfaces

`AgentSurface` binds a prepared agent, profile, tool specifications, and optional
conversation/artifact manager for a complete native run. Its constructor takes
`agent`, `profile`, `tool_specs`, `conversation_state`, optional
`artifact_manager`, and `close_conversation_state=False`.

Keep `run_scope()` open for the complete run or stream consumption. A surface
rejects overlapping runs, use after close, and close during an active run.
`close()`/`aclose()` are idempotent and close the conversation only when
explicitly requested. The caller still owns the adapter and runtime. This is
an advanced integration surface, not a replacement for the ordinary decorators.

## Error Ownership

A context manager restores configuration, not external side effects. A failed
fragment may leave earlier writes committed. Catch `KediExecutionError` from
`kedi.errors` to inspect Kedi frames and the original exception; never interpret
an execution failure as a valid empty result. Interactive persistence failures
use `SessionPersistenceError`, with `SessionDumpError` and `SessionLoadError`
identifying the operation.
