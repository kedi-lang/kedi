# Public Parameters

This reference distinguishes profile configuration from runtime construction.
Names below are checked against the Python call signatures in the documentation
test suite. Omitted selection values inherit; omission is not always equivalent
to explicitly passing `None`.

## Profile Parameters

`AgentProfile.history_processing` accepts `HistoryProcessingSettings` from `kedi.agent_profile`.
Its `processor` is a sync/async native history callback or `None` to explicitly
disable an inherited callback. The default profile value, `None`, means inherit.
Its optional `condition` gates a processor; `inherit_processor=True` can apply
one to an outer or adapter-default callback. Pydantic and LangChain constructors
also accept `history_processor_condition=` with `history_processor=`.
This is an `AgentProfile` field, not a shared keyword of the functions below.
It is separate from the `history` conversation-continuity setting. See
[History Processing](../runtime/history.md#native-configuration).

`configure`, `context`, `query`, `bind`, and `interactive` share these parameters:

| Parameter | Contract |
| --- | --- |
| `model` | Model identifier or explicit `None`; omitted value inherits where applicable |
| `adapter`, `agent` | Mutually exclusive framework/harness name or matching instance |
| `system` | Model instructions; explicit text replaces inherited instructions |
| `effort` | Backend-supported reasoning level |
| `settings` | Backend settings mapping, shallow merged by key |
| `tools` | Callable sequence, merged by registered name |
| `env` | Native globals, later mappings override earlier values |
| `mcp_servers` | Server specifications, appended without deduplication |
| `approval` | Allow/deny policy or synchronous/asynchronous handler |
| `hooks` | Hook settings or event-to-handler mapping |
| `skills` | Boolean or `SkillsSettings` policy |
| `artifacts` | Boolean, mapping, or `ArtifactPolicy`; false disables inheritance |
| `conversation` | Explicit shared state or `None`; omission inherits |

`configure` rebuilds current-context defaults. `context` overlays and restores
them. `query`/`bind` override at the callable boundary; lexical Kedi directives
still take effect inside the program. Settings such as `tool_reason` and
`typesafe_threshold` belong inside `settings`, not arbitrary top-level kwargs.

## Execution Parameters

`configure`, `context`, and `interactive` also accept `parallel`, `max_workers`,
and `loop_iteration_limit`. `parallel=None` leaves mode selection inherited;
positive worker bounds control independent model calls, not subagent admission.
These entry points accept `**adapter_kwargs` for adapter construction.

`query` additionally accepts optional positional `fn` and `cache=False`.
`bind` requires keyword-only `file`, and adds `cache=False`, `reload=False`.
Both accept `requires`, a sequence of canonical adapter capability names checked
before the callable's model I/O. Neither accepts arbitrary adapter kwargs or
execution constructor parameters.

## Incremental Runtime Parameters

`interactive` additionally accepts:

| Parameter | Default and purpose |
| --- | --- |
| `executor` | `None`: default embedded-Python executor |
| `engine` | `None`: derive from execution configuration; explicit engine wins |
| `cwd` | `None`: current directory for synthetic source/import resolution |
| `subagent_max_depth` | `5`: child delegation depth |
| `subagent_max_concurrency` | `4`: concurrently admitted child runs |
| `subagent_timeout_seconds` | `120.0`: child deadline; `None` disables |
| `subagent_usage_limits` | `None`: default `SubagentUsageLimits` |
| `subagent_state_path` | `None`: optional coordinator persistence path |

See [Subagent Limits](../agentic-engineering/subagent-limits.md) for ceilings,
default request/tool budgets and timeout distinctions. These options do not
extend provider limits.

## Small Helpers

| Callable | Parameters and result |
| --- | --- |
| `parallel` | keyword-only `max_workers=8`; sync/async configuration context |
| `session` | `state=None`; sync/async conversation context, closed on exit |
| `reset_config` | No arguments; reset configuration, not type registry or caches |
| `cache_info` | No arguments; `CacheInfo(parse_entries, response_entries)` |
| `clear_cache` | No arguments; clear current-process caches and advance generation |
| `capture_decisions` | No arguments; synchronous evidence capture context |
| `decision_info` | `name`; active Kedi binding evidence or `None` |
| `force` | One value; resolve a `KediPromise`, otherwise return unchanged |

Types/tools, approvals, hooks, model factories, and persistence operations have
their own contracts in the [public export index](public-exports.md).
