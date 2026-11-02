# Runtime behavior and diagnosis

## Concurrency

Independent template calls run concurrently by default on a bounded pool of eight workers. Dependencies wait for their inputs; a scheduled call snapshots its input values. Calls sharing one conversation remain ordered complete turns. Independent side effects need not finish in source order.

`KEDI_PARALLEL=0` selects sequential execution; a positive integer selects a worker count. Explicit Python configuration overrides the environment. Reproduce ordering issues sequentially, then check parallel behavior. Custom adapters must be thread-safe.

Bare Python reads or `globals()["x"]` resolve pending Kedi values. `globals().get`, mapping iteration/copies may expose raw promises; use `kedi.force` intentionally. `KediPromiseLeak` is a runtime defect/unsupported manipulation signal, not data to stringify. Unconsumed work still runs, is drained, and may fail; a return does not hide an already scheduled error. Read [concurrency](docs/runtime/concurrency.md).

## History, artifacts, and caches

| Mechanism | What it owns | Important limit |
| --- | --- | --- |
| `> history: enabled` | Ordered successful conversation turns | Default off; incompatible adapter/model/tool lanes do not share native checkpoints |
| `kedi.session()` | Shared conversation/artifact lifetime | Does not retain program variable declarations |
| Interactive session | Incremental values and declarations | Non-transactional; one fragment at a time |
| Tool artifacts | Large values stored behind compact references | Reference metadata is not the full evidence |
| Provider prefix cache | Provider-native request caching | Never guaranteed; distinct from Kedi result cache |
| `cache=True` on query/bind | In-memory result memoization | Explicit opt-in; consider mutable/external state |
| Parse cache | Exact source-hash parse reuse | Does not justify stale invalid source fallback |
| `SOURCE.cache.kedi` | Generated `> auto:` implementation | Per procedure; reused only once validated; inspect/regenerate after contract changes |
| `SOURCE.kedi.optimized.json` | Optimized prompt prefixes | Malformed artifacts fail loudly |

Artifacts are enabled by default in this documented runtime; artifact querying is separately opt-in. Large tool output becomes a compact ref, while supported direct Kedi/Python operations retain native values. Read hidden content with the provided bounded artifact tools before claiming evidence. References are session-owned, have quota/TTL/release rules, and do not survive arbitrary session transfer. Disabling artifacts does not allow unlimited tool output. File storage avoids payload retention in memory; ordinary already-materialized results still consumed producer memory.

Use [artifact policy](docs/runtime/artifact-policy.md), [retrieval](docs/runtime/artifact-retrieval.md), [lifecycle](docs/runtime/artifact-lifecycle.md), [reduction](docs/runtime/artifact-reduction.md) and [safety](docs/runtime/artifact-safety.md) for exact settings/tools. `ArtifactStream` is explicit; arbitrary generators are not automatically streams. Content remains untrusted data.

Native compaction (`> history:` with `enabled: true`, `compaction_mode: native`, optional `compaction_threshold`) requires a verified provider path (OpenAI Responses or Anthropic via Pydantic AI/LangChain) and fails before model I/O otherwise; it also applies under an active run budget. `compaction_mode: disabled` turns off an inherited policy. Claude does not accept an explicit exact compaction threshold. Do not fabricate a summarizer fallback for an unsupported native path. Custom retention is the separate `processor` field (optional `processor_condition` gate); it runs before every request, including later requests of a tool loop, and cannot remove protected checkpoints such as the native compaction item. History processors preserve native message types and complete tool lifecycles; accepted changes can invalidate continuation/cache identity. Read [history](docs/runtime/history.md), [compaction](docs/runtime/compaction.md), [prefix cache](docs/runtime/prefix-cache.md).

## Budgets

`> budget:` has nonnegative `request_limit` and `tool_attempt_limit` ceilings; at least one is needed. Local ceilings cannot enlarge an ancestor allowance. Admitted generation attempts, tool-loop requests, repairs/fallback attempts, and actual tool-body retries count according to their dimension; failures do not refund work. This is not rollback or a provider billing guarantee.

For a finite request limit, configure actual provider client retries observably (`max_retries=0` where supported); a same-named model setting is not a universal SDK client option. Opaque retries/unsupported boundaries fail before inference. Embedded arbitrary SDK calls are outside managed accounting. `RunBudgetExceeded` is not an invitation to retry with a larger budget. See [run budgets](docs/runtime/run-budgets.md).

## Diagnose in order

1. Parse exact source with its path; read line/column diagnostics.
2. Resolve type/declaration/import/profile/capability errors.
3. Reproduce deterministically or with a controlled model, then sequentially for ordering problems.
4. Inspect the original exception and Kedi frames, rather than wrapping away the cause.
5. Check actual backend/model/tool availability, schema support, permissions and resource ownership.
6. Add bounded debug output only for unresolved issues; remove temporary traces.

`KediExecutionError` carries `frames`, `python_traces`, and `original`; common subclasses also behave as TypeError/ValueError/KeyError/NameError. Concurrent runs drain scheduled work and surface failures. A module import or prelude can fail before main execution. See [errors](docs/runtime/errors-and-debugging.md) and [troubleshooting](docs/reference/diagnostics-and-troubleshooting.md).

Debug exporters serialize values, not redact secrets. Telemetry is opt-in through a separate instrumentation package; review its capture policy. Neither a log message nor a registered capability proves an external action succeeded.
