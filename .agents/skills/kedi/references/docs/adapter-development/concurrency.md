# Concurrency and Event Loops

One adapter instance serves a whole program. Kedi calls it from several
threads at once, runs children in background tasks, and nests calls inside
tool calls. This page states the rules that keep those calls apart and keep
provider clients working.

## Where Calls Come From

| Call | Thread | Concurrency |
|---|---|---|
| template captures and claims | an engine worker thread | independent blocks run at the same time |
| raw invokes (`<<`) | the thread running the program | one at a time per program thread |
| a child (`invoke`, `produce_subagent`) | a task on Kedi's event loop | several children, and background children, at once |
| a nested call from a tool | inside the outer call's tool execution | while the outer call waits for the tool |

Calls of one conversation are the exception: Kedi runs them strictly in turn
order (see [Conversations and History](conversations.md)).

## State

- Keep per-call state in `ContextVar`s or in objects Kedi passes (the
  conversation context, the child request). Kedi copies the context into
  worker threads and tasks, so a value set inside a scope is visible to
  everything that call does, and to nothing else.
- Never keep the active tools, profile, child request, or conversation in a
  plain attribute. A second block running at the same time would see them.
- Instance attributes are for configuration and for objects shared safely by
  all calls, such as an async provider client.

## One Event Loop

Kedi runs all model work that overlaps in time on one shared event loop. That
is what lets provider connection pools and background children work across
calls (`docs/EVENT_LOOPS.md` in the Kedi repository has the full design).
For an adapter:

1. **Run async work from sync entry points with `run_sync`**
   (`kedi.agent_adapter.execution.run_sync`). Never call `asyncio.run`, start
   a thread with its own loop, or keep a private loop for model work.
2. **Never block the loop.** Inside coroutines, use async clients. A blocking
   call stalls every concurrent call and child; if a backend only offers a
   blocking client, run it with `asyncio.to_thread`.
3. **Bind clients to the loop that uses them.** Create async clients lazily
   inside the running loop, or use clients that do so. A client created on one
   loop and used on another fails with errors such as "attached to a different
   loop" or "Event loop is closed".
4. **Hold locks only around state changes**, never across an `await` of a
   model request, an approval, or a tool.
5. **An async host keeps its loop.** When a host runs Kedi inside
   `use_async_owner_loop()`, Kedi uses the host's loop; `run_sync` follows,
   and the adapter must not close it.

## Resources

`runtime.aclose()` releases the runtime's sessions, children, and artifacts;
it does not close the adapter. Whoever created the adapter owns it. Give an
adapter that holds connections or processes `close()` and `aclose()` methods
and call them when the adapter is no longer needed. Resources that belong to
one conversation go in the conversation's `cleanup` instead.

## Running It

```python title="run_concurrency.py"
import asyncio
import threading

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter


class Slow:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.prompts: list[str] = []

    async def complete(self, request: BackendRequest) -> Reply:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.prompts.append(request.messages[-1]["content"])
        await asyncio.sleep(0.3)
        with self.lock:
            self.active -= 1
        field = "city" if "capital" in request.messages[-1]["content"] else "river"
        return Reply(data={field: "answer"})


SOURCE = """
>> The capital of France is [city].
>> The longest river in France is [river].
= `(city, river)`
"""

backend = Slow()
runtime = compile_program(parse_program(SOURCE), adapter=ChatAdapter(backend, model="m1"))
try:
    assert runtime.run_main() == ("answer", "answer")
finally:
    asyncio.run(runtime.aclose())

assert backend.peak == 2
assert sum("capital" in p for p in backend.prompts) == 1
assert sum("river" in p for p in backend.prompts) == 1
```

The two independent blocks reached the backend at the same time, each with
its own prompt and output schema, through one adapter instance.
