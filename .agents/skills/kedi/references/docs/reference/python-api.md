# Python API Reference

All stable high-level names below are imported from `kedi`.

Use the [public export inventory](../python-api/public-exports.md) for all root
exports and the [parameter reference](../python-api/public-parameters.md) for
the exact configuration surfaces. Examples with `...` below are signature
sketches, not executable programs.

## Public Selection Types

- `FrameworkAdapterName` is
  `Literal["pydantic", "langchain"]`.
- `AgentName` is `Literal["claude", "codex", "acp", "a2a"]`.
- `AdapterLike` is `AgentAdapter[Any] | LazyAdapter`.
- `ApprovalMode` is `Literal["allow", "deny"]`.
- `ApprovalHandler` is a sync or async callable from `ApprovalRequest` to
  `ApprovalDecision`.

These aliases make extension signatures precise. Pass framework names through
`adapter=` and harness names through `agent=`.

## Query and Bind

```python
import kedi


@kedi.query(
    model=...,
    adapter=...,
    agent=...,
    system=...,
    effort=...,
    settings=...,
    requires=(),
    tools=(),
    env=...,
    mcp_servers=(),
    approval=...,
    hooks=...,
    skills=...,
    artifacts=...,
    conversation=...,
    cache=False,
)
def function(value: str) -> str: ...
```

The function docstring starts with `kedi` and contains a procedure **body**, not
a full top-level program. The Python stub body is never executed. A normal
function produces a sync wrapper; an async function produces an async wrapper
that offloads blocking adapter work.

```python
import kedi


@kedi.bind(
    file="workflow.kedi",
    reload=False,
    cache=False,
    # same backend/profile arguments as query
)
def function(value: str) -> str: ...
```

`bind` resolves the path relative to the decorated function's source file,
loads a complete Kedi program, passes call arguments as runtime globals, and
runs top-level main. `reload=True` reparses when invoked; `cache=True` enables
the response cache. The Python return annotation informs static typing but the
Kedi return remains the runtime authority.

## Configuration

`configure(...)` accepts `model`, mutually exclusive `adapter`/`agent`,
`system`, `effort`, `settings`, `tools`, `env`, `mcp_servers`, `approval`,
`hooks`, `skills`, `artifacts`, `conversation`, `parallel`, `max_workers`,
`loop_iteration_limit`, and
adapter-specific keyword arguments.
Each call rebuilds defaults; it does not merge with a previous `configure`.

`context(...)` accepts the same values and merges them over current
configuration for sync or async `with`, restoring the previous ContextVar state
on exit. `parallel(max_workers=8)` is shorthand for a parallel context.
`reset_config()` restores library defaults.

Effective environment precedence is configured/local tools, call arguments,
automatically registered types, configure environment, then progressively local
context/query/bind environment. Name collisions in tool arguments fail rather
than silently selecting one source.

## Tool and Type Decorators

```python
import kedi


@kedi.tool(
    name=None,
    description=None,
    retries=0,
    retry_on=None,
    risk="mutating",
)
def tool_name(value: str) -> object: ...
```

Risk is `read_only`, `mutating`, or `sensitive`. Retries must be nonnegative and
produce `retries + 1` total attempts. `retry_on` is `None` for all ordinary
`Exception` failures, or a sequence of eligible `Exception` classes; an empty
sequence disables matching. Sync and async functions are preserved. Only
callable-body failures are retried.

`@kedi.type` converts an ordinary class to a dataclass or preserves an existing
Pydantic/dataclass surface. `inject=True` registers it by name for query
annotations in the defining module; `inject=False` removes that implicit
registration while the class can still be supplied through `env`.

## Adapter Structured Output

Structured-output adapters accept ordinary prompt text through `template`; this
low-level argument does not require Kedi template syntax. Choose one output form:

```python
from typing import Annotated

from pydantic import BaseModel


class Review(BaseModel):
    accepted: bool
    reason: str


dynamic = await adapter.produce(
    template="Evaluate the proposal.",
    output_schema={
        "accepted": Annotated[bool, "Whether the proposal should proceed"],
        "reason": Annotated[str, "Short justification"],
    },
)
typed = await adapter.produce(
    template="Evaluate the proposal.",
    output_type=Review,
)
```

The first `Annotated` argument is the field type and the second string is its
model-facing description. `output_schema` dynamically defines named result
fields; `output_type` returns the supplied prebuilt type. If both are supplied,
the field schema takes precedence. Pydantic AI, LangChain, Claude, Codex,
and WebGPU support both forms. ACP does not currently support structured output
and raises `NotImplementedError` from `produce()`.

## Artifacts and Sessions

`ArtifactPolicy` validates memory/file storage, size threshold, TTL, bounded
preview/read limits, quota, record count, and cleanup interval. `artifacts=`
accepts a policy, mapping, boolean, or `None` on `configure`, `context`,
`query`, and `bind`.

`session(state=None)` is a synchronous and asynchronous context manager. It
creates or activates a `ConversationState` so calls can share portable model
history and artifact ownership. Kedi remains stateless by default.

Public artifact models are `ArtifactRef`, `ArtifactChunk`,
`ArtifactSearchResult`, `ArtifactReleaseResult`, and `ArtifactHandle`. See
[Artifacts and Sessions](../python-api/artifacts-and-sessions.md).

## Media

`kedi.media` provides `Media`, `Image`, `Audio`, `Video` and `Document` for
prompts with media. Construct them with `from_path`, `from_url`, `from_bytes`,
`from_data_uri` or `from_file_id`; `Media.from_path` picks the kind from the
media type, and `Image.coerce(value)` applies the validation typed parameters
use. See [Media Input](../core-language/media-input.md).

## Interactive Execution

`interactive(...) -> InteractiveSession` creates a synchronous, process-local
incremental runtime. `InteractiveSession.execute(source, *, source_name=None)`
parses and executes one complete fragment exactly once while retaining earlier
values, declarations, imports, conversation state, and artifacts. Interactive
fragments return `None`; use bindings to preserve values and `> show:` to
display them. Root `=` returns are rejected before execution, while procedure
returns retain their ordinary early-exit semantics. `close()` is idempotent;
the session is also a context manager.

`dump_session(session, path, *, format="portable")` atomically writes a strict,
pickle-free snapshot by default. `format="python"` requires `kedi[snapshots]`
and stores supported user Python graphs alongside session-owned Kedi state.
Unsupported state raises `SessionDumpError` without changing the destination.
`load_session(path, *, session_type=InteractiveSession, adapter=None,
executor=None, engine=None, conversation_state=None,
close_conversation_state=None, trusted=False)` restores saved logical state
without replaying previous executable fragments. Python snapshots require explicit
`trusted=True` and a matching environment; their deserialization can execute arbitrary
Python code and saved values may contain credentials.
`session_type` is keyword-only and may be an `InteractiveSession` subclass.
`InteractiveSession.dump()` and `InteractiveSession.load()` expose the same
format/trust options. Invalid, modified, incompatible or untrusted Python snapshots
raise `SessionLoadError`. Notebook/bridged executor state is not implicitly trusted
or converted to local Python state.

See [Interactive Execution](../runtime/interactive-execution.md) for supported
configuration, source identity, failure semantics, restrictions, and the
`kedi --idle` frontend.

## Approvals

`@kedi.approval` installs a sync or async handler in current configuration and
returns the same callable. Policies are `ApprovalPolicy.allow()`,
`ApprovalPolicy.deny()`, `ApprovalPolicy.dynamic(handler)`, or the strings
`"allow"`/`"deny"`.

`ApprovalRequest` fields are `tool_name`, immutable deep-copied `arguments`,
`risk`, `adapter_shortname`, `description`, immutable optional `metadata`,
`reason`, and `tool_reason_enabled`. The reason is a model-provided explanation,
not evidence that a call is safe; it is not passed to the underlying tool.
Handlers return:

```python
kedi.ApprovalDecision.allow(reason=None)
kedi.ApprovalDecision.deny(reason=None)
kedi.ApprovalDecision.edit(arguments, reason=None)
```

Only edit may carry replacement arguments. Edited calls are reclassified and
validated before execution.

## Hooks and Decision Evidence

`@kedi.on(event)` accepts an event name or a sequence of names. Handlers can
observe events or return the supported decision for that event; consult
[Hooks from Python](../python-api/hooks.md) for ordering and registration scope.

`capture_decisions()` retains decision evidence within a capture context.
`decision_info(binding)` inspects evidence associated with a binding rather than
re-evaluating it. See [Decision Evidence](../python-api/decisions.md) for lookup
arguments, record fields, lifetime, mutation invalidation and cache omissions.

## MCP

```python
kedi.McpServerSpec(
    transport="stdio",
    command="server",
    args=(),
    env=None,
    url=None,
    headers=None,
).normalized()
```

Canonical transports are `stdio`, `sse`, and `http`. Stdio requires `command`;
remote transports require `url`. Normalization removes fields inapplicable to
the selected transport. MCP lists append across configuration scopes.

## Cache and Concurrency

`cache_info()` returns frozen `CacheInfo(parse_entries, response_entries)`.
`clear_cache()` clears parse/completed-response caches and advances generation
so stale in-flight calls cannot repopulate them.

`KediPromise.resolve()` returns the concrete value; `.map(fn)` derives without a
worker slot. `force(value)` resolves only a Kedi promise. Concrete use of an
unresolved promise raises `KediPromiseLeak`; `repr()` is non-forcing.

## Low-Level Runtime

`KediRuntime` is the compiled container. Common operations are `run_main()`,
`set_initial_globals(mapping)`, `drain()`, procedure/main decorators,
`m(expressions)`, `invoke(expressions, capture=...)`, trace inspection, and
execution-error construction.

`i(name)`, `o(name, pytype=None)`, and `c(procedure, *args)` construct low-level
input, output, and call expressions for `KediRuntime.m()`.

`Executor` is the runtime-checkable embedded-Python protocol; `DefaultExecutor`
uses host `eval`/`exec` and is not sandboxed. `MarkdownDebugExporter` records
code and values but does not redact secrets.

See the [Python API guide](../python-api/index.md) for complete examples and
scope behavior.
