# CodeMode Sandbox and Recovery

CodeMode is bounded execution over hydrated tools, not general host Python.
The model must discover a tool and request its schema before using it unless
the application explicitly lists that exact exposed name in `preload_tools`.
See [Discovery and Execution](codemode.md#model-facing-tools) for that protocol.

## Supported Sandbox Subset

The shared CodeMode instruction teaches the verified Monty subset:

- scalar, list, tuple, dictionary, and string literals;
- indexing and read-only slicing;
- arithmetic, comparisons, boolean expressions, and f-strings;
- `if`/`elif`/`else`, `for`, `while`, `break`, and `continue`;
- `range`, `enumerate`, `zip`, comprehensions, sorting, filtering, grouping,
  joining, and aggregation;
- small helper functions;
- `asyncio.gather` for independent hydrated tool calls.

The pinned Monty 1.2 runtime supports both `mapping[key]` and mapping methods
such as `mapping.get(...)`.

CodeMode does not provide host filesystem, environment, process, unrestricted
network, or third-party package access. Monty's `eval` and `exec` remain inside
the same sandbox; they do not grant host access. Prefer direct expressions.
Generators are not supported. This is not general CPython execution.

## Lifecycle and Limits

Every agent run receives an isolated catalog, hydration set, Monty process
checkout, and variable state. Kedi closes the session and cancels active host
callbacks on normal completion, errors, cancellation, and early close.
After a worker crash or watchdog timeout, its session is discarded. A subsequent
cell starts with empty variables. Cancellation also discards the interrupted
session; it must not reuse partially executed state or automatically replay tools.
Preloaded schemas count toward the same hydration set. Restarting Monty clears
variables but preserves that run's hydrated allowlist; it never carries the
allowlist into another run.
Concurrent cells execute in order of lock acquisition. A cell with `restart=True`
waits for the active cell to finish before discarding its sandbox variables.

The runtime bounds search pages, discovery payload bytes, hydrated tools, code
characters, nested call count, nested concurrency, individual result bytes,
aggregate result bytes, captured output, and execution time. Invalid cursors,
unhydrated calls, non-JSON nested values, denials, and budget failures are
model-correctable errors rather than silent fallbacks.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `default_search_limit` | `10` | Tool names returned when `search_tools` omits `limit`. |
| `max_search_limit` | `50` | Maximum accepted search page size. |
| `max_hydrated_tools` | `64` | Exact schemas that may be hydrated in one run. |
| `max_discovery_result_bytes` | `256000` | Serialized bound for search and schema results. |
| `max_code_chars` | `20000` | Maximum source length for one snippet. |
| `max_nested_calls` | `64` | Host tool calls allowed in one execution. |
| `max_concurrent_calls` | `8` | Concurrent host tool calls allowed in one execution. |
| `max_tool_result_bytes` | `256000` | Serialized bound for one nested result. |
| `max_total_tool_result_bytes` | `1000000` | Aggregate nested-result bound per execution. |
| `max_print_bytes` | `256000` | Captured standard-output bound. |
| `request_timeout` | `60` | Worker-response watchdog in seconds; expiry discards the session. Host tool execution has separate application/run limits. |

CodeMode telemetry records payload-free `search tools`, `get tool schema`, and
`execute code` spans. It records counts, byte totals, restart state, duration,
and outcome without recording query text, code, arguments, or tool results.

## Recover Without Repeating Effects

A Monty-language error returns captured `output` and an `error` separately.
Variables assigned before the failure remain available for the next
`execute_code` call. Inspect that checkpoint before repeating host calls;
a write that completed before a later expression failed is still a real write.

`restart=True` resets sandbox variables, not external effects. Application-tool
failures, policy denials, and execution-limit errors remain failed tool calls.
Do not treat an empty printed output as evidence that nothing ran.

## Different From an Artifact Query

`execute_code` consumes hydrated tools. `run_artifact_code` consumes values
already stored as artifacts. Dynamic workflows consume child-agent functions.
These contracts share a sandbox implementation but not the same capabilities.
