# Concurrency

Kedi runs independent model templates concurrently by default on a bounded,
shared pool of eight workers. Dependent calls wait for the values they need.
No environment variable or special syntax is required.

## Concurrent Default

Independent `>>` or `<<` calls can overlap:

```kedi
>> The first independent result is [first: str].
>> The second independent result is [second: str].
```

Select sequential execution for an adapter that is not thread-safe or when
independent calls have external effects that must occur in source order.

## Configure Execution

Environment:

```console
$ KEDI_PARALLEL=4 kedi program.kedi
```

Unset or empty keeps concurrency enabled. Accepted truthy values are `1`,
`true`, `yes`, and `on`; `0`, `false`, `no`, or `off` selects sequential execution.
A positive integer other than `1` sets the worker count; `1` selects the normal
parallel default. A negative integer selects sequential execution.
Invalid values are rejected rather than guessed.

Python:

```python
import kedi

kedi.configure(max_workers=4)

with kedi.parallel(max_workers=4):
    run_workflow()
```

Use `kedi.configure(parallel=False)` or `kedi.context(parallel=False)` for
sequential execution. An explicit Python mode takes precedence over the
environment. `max_workers` must be a positive bound.

## Automatic Dependencies

There is no parallel operator. The runtime follows value dependencies:

```kedi
>> Incident <incident> affects [service: str].
>> The owner of <service> is [owner: str].

>> Incident <incident> occurred in [region: str].
>> The runbook for <region> is [runbook: str].
```

The service and region calls can start together. Each downstream call begins as
soon as its own input resolves, so the two chains pipeline independently.

Calls sharing a conversation scope, including an interactive session, retain
their ordered turns. Concurrency does not bypass conversation ordering.

## Promises and Forcing

Template outputs are represented internally as opaque promises until a value is
needed. Normal Kedi code does not observe them. A bare Python read or
`globals()["name"]` forces the value.

Advanced non-forcing operations such as `globals().get("name")`,
`globals().items()`, `globals().values()`, and `dict(globals())` can expose the
raw promise for forwarding. Resolve it with `kedi.force(value)`.

Using an unresolved promise as an ordinary value raises `KediPromiseLeak`
instead of silently stringifying, indexing, comparing, or serializing the wrong
object. Application logic should not catch and normalize this exception; it
indicates an interpreter/advanced-integration error.

## Snapshot Semantics

When a template is scheduled, Kedi snapshots its value environment by value. A
later assignment on the main thread cannot alter that call's inputs.

Dependencies and input snapshots are preserved in either mode. Stochastic
model outputs can differ between runs, and independent calls or map side
effects can finish out of order. Protect shared mutable state and select
sequential execution when external effects require source ordering.

## Failure Draining

Every scheduled template runs, including one whose output is never consumed.
Before a scope returns or propagates another failure, Kedi drains its work. The
first model failure is raised; additional concurrent failures are logged.

This prevents silent background exceptions and resource leaks. It also means a
fire-and-forget `>>` still has cost and can fail.

## Adapter Thread Safety

Parallel execution may call an adapter's synchronous production path from
several worker threads. Built-in adapters are designed for this contract. A
custom adapter must be thread-safe or serialize its own critical section.

Agent/tool calls must also keep invocation scopes isolated. Do not store
request-specific mutable state on a shared adapter without synchronization.

## Shared Pools

Thread pools are process-global and cached by worker count. The first request
for one size creates that pool; later runs using the same size reuse it. Select
a small bound based on provider rate limits and workload, not CPU count alone.

## Adaptive Job Manager

`JobManagerEngine` is an advanced opt-in engine that adds AIMD concurrency,
transient-error retries with exponential backoff and jitter, and a circuit
breaker. It is not selected by the public `parallel()` helper.

Construct it explicitly through `compile_program(engine=...)` when operating a
rate-limited backend and when retry semantics are acceptable. Do not enable
automatic retries for non-idempotent external effects without a deduplication
strategy.
