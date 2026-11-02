# Concurrency and Promises

## Configure Parallel Execution

Python API execution uses promise-pipelined concurrent template calls by
default. Adjust the shared worker bound globally:

```python
kedi.configure(max_workers=4)
```

or temporarily:

```python
with kedi.context(parallel=True, max_workers=4):
    result = run_pipeline("...")
```

Use `kedi.configure(parallel=False)` or `kedi.context(parallel=False)` to run
sequentially. Dependencies and input snapshots are preserved in either mode;
stochastic model outputs and the completion order of independent calls can
differ. Shared mutable state and externally ordered effects need explicit care.

## The `parallel()` Context

`kedi.parallel()` is shorthand for a context override:

```python
with kedi.parallel(max_workers=6):
    result = analyze("...")
```

Equivalent:

```python
with kedi.context(parallel=True, max_workers=6):
    result = analyze("...")
```

The previous configuration is restored on exit.

## Worker Limits

`max_workers` bounds the shared thread pool used for independent model calls.
Pools are process-wide and reused by worker count. Use a positive value.

The CLI/runtime environment also recognizes `KEDI_PARALLEL`:

- unset or empty: parallel with eight workers unless the API sets another bound;
- `0`, `false`, `no`, or `off`: sequential;
- `1`, `true`, `yes`, or `on`: parallel with eight workers;
- a positive integer: parallel with that worker count;
- a negative integer: sequential;
- another value: configuration error.

An explicit Python API `parallel` setting overrides the environment for
decorated runs and interactive sessions. An explicit session `engine` takes
precedence over both.

## `KediPromise`

In parallel mode, an output field is initially represented internally by a
`KediPromise` view over the template's shared future:

```python
from kedi import KediPromise
```

One model call with several output fields creates one future and one promise
view per field. Downstream templates receive dependency edges without blocking
the scheduling thread.

Normal substitutions, procedure calls, Python name access, and final query
returns resolve at genuine consumption points. Most application code should
never observe a promise.

## Force a Promise

Resolve an intentionally obtained low-level promise:

```python
from kedi import force

value = force(possible_promise)
```

`force()` calls `.resolve()` only for `KediPromise`; every other value passes
through unchanged. `promise.map(fn)` creates a derived promise using a
completion callback without consuming another worker slot.

## Promise Leaks

Promises are deliberately opaque. Using an unresolved promise through
`str()`, an f-string, comparison, indexing, iteration, arithmetic, attribute
access, hashing, or a call raises `KediPromiseLeak`.

This is not a normal “value is not ready” condition. It means a promise escaped
runtime deferral and reached a concrete-value operation. The loud failure
prevents silent serialization or subtly wrong results.

`repr(promise)` is safe and never resolves. Copy/deep-copy preserves identity;
pickling an unresolved promise raises.

## Dataflow Resolution

Kedi snapshots the value environment by value when a template is scheduled.
Later writes on the main thread do not change that job's inputs.

Inside embedded Python, ordinary bare-name lookup resolves a promise. Advanced
non-resolving dictionary operations such as `globals().get("name")`,
`globals().items()`, `globals().values()`, and `dict(globals())` can expose the
raw promise for forwarding. Call `kedi.force(...)` before concrete use.

At the outermost return, Kedi drains all scheduled jobs, including unconsumed
template calls. The first failure is raised; additional concurrent failures
are logged.

## Thread Safety

Parallel mode may call an adapter's synchronous production path concurrently
from multiple worker threads. Custom adapters must be thread-safe or serialize
internally.

Python tools can also run under adapter concurrency. Protect mutable shared
state explicitly. Do not rely on the GIL as an application-level consistency
guarantee.
