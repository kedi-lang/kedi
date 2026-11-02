# Python API

The Python API embeds Kedi programs in typed Python callables. It preserves the
same template, substitution, output, profile, tool, MCP, approval, skills,
artifact, conversation, and runtime semantics as `.kedi` files.

## Embed Kedi in Python

Use `@kedi.query` for a short program in a docstring:

```python
import kedi


@kedi.query
def summarize(text: str) -> str:
    """kedi
    >> A one-sentence summary of <text> is [summary: str].
    = `summary`
    """
    ...


kedi.configure(adapter="pydantic", model="openai:gpt-6-luna")
print(summarize("Kedi combines LLM templates with Python."))
```

Use `@kedi.bind` when the implementation belongs in a separate `.kedi` file:

```python
@kedi.bind(file="summarize.kedi")
def summarize(text: str) -> str:
    ...
```

In both forms, Python owns the callable signature and Kedi owns execution. The
stub body is never called.

## Decorator-Based Programs

The primary program/type/tool decorators are:

| Decorator | Role |
| --- | --- |
| `@kedi.query` | Compile a Kedi procedure body from the function docstring |
| `@kedi.bind(file=...)` | Run a complete file-backed Kedi program |
| `@kedi.type` | Register a Python class for Kedi type resolution |
| `@kedi.tool` | Add tool metadata and optional retry behavior to a callable |

`@kedi.approval` registers a default dynamic approval handler in the current
Python API context.

`@kedi.on(...)` registers lifecycle hooks. The introductory model-backed call
also requires the selected provider's dependencies and credentials.

## Global and Scoped Configuration

`kedi.configure(...)` replaces process-context defaults for subsequent calls:

```python
kedi.configure(
    adapter="pydantic",
    model="openai:gpt-6-luna",
    system="Answer with evidence.",
)
```

`kedi.context(...)` temporarily merges overrides:

```python
with kedi.context(model="openai:gpt-6-luna"):
    result = summarize("...")
```

Use `async with` in asynchronous code. Configuration is held in a
`ContextVar`, so scoped overrides follow async task context rather than a
single mutable process-global stack.

## Registered Types and Tools

```python
from pydantic import BaseModel


@kedi.type
class Finding(BaseModel):
    severity: str
    message: str


@kedi.tool(risk="read_only")
def search_docs(query: str) -> list[str]:
    """Search the local documentation index."""
    return []
```

Registering a tool makes it available to the runtime environment. The Kedi
program must still opt into it with `> use: search_docs`.

For a reusable group of tools and additive guidance, use a
[Capability definition](capabilities.md). Profiles continue to override agent
configuration; capabilities do not change that existing behavior.

## Runtime Control

Independent template calls run concurrently by default. Adjust the worker bound:

```python
with kedi.parallel(max_workers=4):
    result = summarize("...")
```

`kedi.cache_info()` and `kedi.clear_cache()` inspect and clear the in-memory
parse and response caches. `kedi.force(value)` explicitly resolves a low-level
`KediPromise`; ordinary query results are resolved before they return.

`kedi.session()` creates an explicit stateful boundary for model history and
artifact ownership. Artifact handling is enabled by default and can be
configured or disabled through `artifacts=` on contexts and decorators.

`kedi.interactive()` creates a persistent process-local runtime for executing
complete Kedi fragments without replaying earlier fragments:

```python
with kedi.interactive() as interactive_session:
    interactive_session.execute("[value: int] = `40`")
    interactive_session.execute("> show: `value + 2`")  # displays 42
```

See [Interactive Execution](../runtime/interactive-execution.md) for source
identity, lifecycle, failure behavior, and terminal REPL usage.

## Public API Map

Common imports come from the package root:

```python
from kedi import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    AgentMessageEvent,
    AgentRunStateEvent,
    AgentStreamEvent,
    ArtifactChunk,
    ArtifactHandle,
    ArtifactPolicy,
    ArtifactRef,
    ArtifactReleaseResult,
    ArtifactSearchResult,
    AsyncAgentEventQueue,
    CacheInfo,
    ConversationState,
    InteractiveSession,
    KediPromise,
    KediPromiseLeak,
    KediRuntime,
    McpServerSpec,
    bind,
    cache_info,
    clear_cache,
    configure,
    context,
    force,
    interactive,
    observe_agent_events,
    parallel,
    query,
    reset_config,
    session,
    tool,
    type,
)
```

`observe_agent_events(...)` provides adapter-neutral, completed commentary and
final messages as a non-authoritative side channel. See
[Stream Events](../agentic-engineering/stream-events.md) for callback and async
queue examples, event ordering, and failure semantics.

Compiler entry points are in `kedi.lang`:

```python
from kedi.lang import compile_program, parse_program
```

Executor protocols and the default implementation are also re-exported from
`kedi`; specialized engine and playground executor classes live in their
respective submodules.

## In This Section

**Calling Kedi**

- [Query Decorator](query.md)
- [Bind Decorator](bind.md)

**Configuration**

- [Configuration and Context](configuration-and-context.md)
- [Public Parameters](public-parameters.md)

**Integration**

- [Types and Tools](types-and-tools.md)
- [Permissions and External Tools](approvals-mcp-and-skills.md)
- [MCP](mcp.md)
- [Skills](skills.md)
- [Hooks](hooks.md)

**State and Execution**

- [Artifacts and Sessions](artifacts-and-sessions.md)
- [Decision Evidence](decisions.md)
- [Cache Control](cache-control.md)
- [Concurrency and Promises](concurrency-and-promises.md)
- [Runtime and Executors](caching-runtime-and-executors.md)
- [Embedding and Ownership](embedding.md)
- [Executors](executors.md)

**Pages**

- [API Reference](../reference/python-api.md)
- [Public Export Index](public-exports.md)
