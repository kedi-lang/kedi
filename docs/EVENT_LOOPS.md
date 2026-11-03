# Event loops for synchronous Kedi work

This is a design decision record. Read it before changing
`src/kedi/agent_adapter/execution.py`, the operation scope in
`src/kedi/core/lifecycle.py`, the Codex App Server tool bridges, decision-model
cleanup in `src/kedi/agent_adapter/models.py`, or the loop native subagents run
on. The behavior described here is deliberate. A simpler-looking version
(`asyncio.run` per call, one loop per operation, a loop that never closes)
reintroduces bugs that are already measured and tested. The alternatives
section below lists them.

User-facing behavior is documented in
[Concurrency: Event Loops](https://docs.kedi-lang.org/runtime/concurrency/#event-loops)
(source: `kedi-docs/docs/runtime/concurrency.md`).

## The constraint

Provider clients bind to the event loop that uses them. Pydantic AI's HTTP
connection pool (httpx/httpcore) keeps connections opened on one loop.
`Provider._refresh_http_client` replaces its own client only when the client is
closed or the loop it last ran on is **not running**. So while two loops run at
the same time, both use one pool. A request on the second loop then picks up a
connection bound to the first, and fails with "bound to a different event
loop", "Event loop is closed", or hangs.

The failure needs a *second* request on a shared client while another loop is
live: provider retries, tool-loop steps, delegations, or concurrent calls.
Single-request calls usually pass, which is why the bug was hard to see.

So: **all synchronous Kedi work that overlaps in time must run on one loop.**

## Where work runs

| Work | Loop |
| --- | --- |
| Inside an async host's `use_async_owner_loop()` scope | The host's loop. Kedi never owns or closes it. |
| A synchronous operation (`run_main`, a procedure called from Python, test/eval runs, interactive fragments) on any thread | The shared loop, held from its first model request to its end |
| A direct `run_sync` call outside an operation (`adapter.invoke_sync()`, `produce_sync()`, hooks, approvals) | The shared loop, held for the call |
| Codex App Server tool awaitables (synchronous turns) | The shared loop. The client holds it from its first tool call until `close()`, so background delegations outlive the tool that started them. |
| Native subagents | The operation's loop (`_shared_execution_loop()`) |
| Synchronous Kedi code called from a coroutine running on the loop it would use (for example an async tool calling `runtime.run_main()`) | A separate loop: a nested `run_sync` gets its own loop in a worker thread; an operation gets a private `_SharedLoop` |
| A synchronous Codex turn started from a coroutine | The client's private tool loop |

The shared loop (`_SharedLoop`) is process-wide and reference-counted. It
starts on first hold and closes, like `asyncio.run`, when the last holder
releases it.

## Invariants

1. **Overlapping work shares one loop.** Do not start a loop of your own
   (`asyncio.run`, `OwnedAsyncLoop`, a new thread with a loop) for Kedi model
   work. Route it through `run_sync` or `_shared_execution_loop()`.
2. **Never wait synchronously on a blocked loop.** Synchronous code that runs
   on a loop's thread blocks that loop until it returns. `_BLOCKED_LOOPS`
   records such loops along the call chain, including work handed to other
   threads, and work is never routed to them. Without this, an async tool that
   runs a sync Kedi operation deadlocks. Stable did not, so this is a regression
   guard.
3. **No new shared loop while the previous one is closing.** `hold()` waits for
   the closing loop. The provider keeps its client while the loop it last ran
   on still runs, so a new loop started during the close reproduces the
   original bug.
4. **The loop closes when nothing holds it.** Holders are operations, direct
   calls and open Codex clients. Keep holds balanced, and never close the
   shared loop directly.
5. **Clients shared on a loop are cleaned up once, when nothing uses that
   loop.** Decision-model (TypeSafe) clients keep one client per loop. Closing
   it after each call breaks concurrent calls that still use it. Register the
   cleanup with `_defer_loop_cleanup`; on a host's loop it runs when no
   operation or call uses that loop (`_HostLoops`). The call closes the client
   itself only on a loop Kedi does not share.
6. **The async host's loop takes precedence** over the shared loop, and Kedi
   does not close clients or loops the host owns.
7. **Deterministic programs start no loop.** Operations hold the shared loop
   lazily.
8. **`run_sync` keeps `asyncio.run` semantics.** A cancelled coroutine raises
   `asyncio.CancelledError`, not `concurrent.futures.CancelledError`. An
   interrupted waiter cancels its task and gives it time to clean up.
9. **A forked child starts fresh.** The shared loop state resets after fork.
   Holds from before the fork are stale there and are ignored on release.

## Decisions and rejected alternatives

- **`asyncio.run` per call** (the original behavior). Every call gets a new
  loop, so concurrent calls, retries and tool loops share one provider client
  across live loops. This is the bug itself.
- **One loop per operation** (the first fix). It fixed concurrency within one
  run, but not:
  - operations started from your own threads,
  - direct adapter calls,
  - Codex App Server tools, which ran on the client's own tool loop.

  An operation started from a coroutine on its own loop also deadlocked.
- **A process loop that never closes.** Withdrawn after review. It lost
  `asyncio.run`-like cleanup at the end of work, and it conflicted with
  per-call decision-client cleanup and with code and tests that expect the loop
  to close.
- **Starting a new loop while the old one is still closing.** See invariant 3.
- **One loop per model or provider.** One operation uses several models, and
  tools and delegations call other models from inside a run. Their awaits
  would cross loops.
- **Running async user tools off the shared loop.** Their own model calls would
  then run on other loops, which brings the bug back for well-behaved async
  tools.
- **Rejecting synchronous Kedi calls from coroutines.** Stable supports them.
  They get a separate loop instead.

## Known trade-offs

- **A coroutine that blocks the shared loop stalls all concurrent Kedi work in
  the process,** not just its own operation. Offload blocking work with
  `asyncio.to_thread`. Kedi runs sync tools, hooks and approval handlers off
  the loop.
- **Synchronous Kedi code called from a coroutine runs on a separate loop.**
  This avoids deadlock, but sharing a provider client between that work and
  other concurrent work is not covered.
- **Two loops for one async host.** Kedi work outside an async host's
  `use_async_owner_loop()` scope runs on the shared loop while the host's loop
  also runs Kedi work.
- **Cost.** About 0.6 ms more per single-call run than one loop per operation,
  measured with `FunctionModel`. Parallel runs are faster than stable.

## Changing this code

- Write the failing test first, against the unchanged code. Then change the
  code. `tests/test_event_loop_sharing.py` covers:
  - overlapping-request invariants with a recording model,
  - the same scenarios over HTTP,
  - user threads, direct calls and Codex bridges,
  - async hosts,
  - decision-client cleanup,
  - inline sync calls in async tools,
  - interrupts, fork, and the last release on the loop itself.
- Timing-dependent tests must run repeatedly. Run the file at least 20 times;
  HTTP tests run under a deadline instead of hanging the suite.
- `make prod` must pass with 100% coverage.
- Keep this record and the user-facing "Event Loops" section current.
