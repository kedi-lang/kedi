# CodeMode

CodeMode lets a model discover tools and process intermediate results without
putting each intermediate result in its conversation. Instead of sending every application tool schema to the model,
Kedi exposes three stable control tools and keeps selected tool execution inside
a bounded Monty sandbox.

This can reduce context traffic for large catalogs or multi-tool computations.
It is not a universal cost or latency improvement: discovery adds calls, and a
small shell-centric toolset may gain little. Choose it for the dataflow, not
as a promise that every task becomes cheaper.

CodeMode is disabled by default and implemented for Pydantic AI, LangChain,
Claude Agent SDK, and Codex App Server. All four adapters expose the same three
controls and use the same catalog, Monty subset, limits, approval composition,
and artifact boundary. DSPy intentionally remains unsupported.

## Enable CodeMode

Enable it in the current lexical scope:

```kedi
> adapter: pydantic
> codemode: enabled
```

The directive is valid at top level, in a profile, and in a procedure. Disable
an inherited or adapter-constructor policy with `> codemode: disabled`.

The expanded form enables CodeMode and configures its limits. `enabled`
defaults to `true` when omitted:

```kedi
> codemode:
    preload_tools:
        lookup_release
        list_versions
    default_search_limit: 10
    max_search_limit: 50
    max_hydrated_tools: 64
    max_nested_calls: 64
    max_concurrent_calls: 8
    max_tool_result_bytes: 256000
    max_total_tool_result_bytes: 1000000
    request_timeout: 60
```

`> codemode:` is Kedi-owned and is never forwarded as provider model
configuration. `> settings:` does not accept a `codemode` field.

The equivalent Python API is available on every supported adapter:

```python
from kedi.agent_adapter import PydanticAdapter

adapter = PydanticAdapter("openai:gpt-6-luna", codemode=True)
```

Use `CodeModeSettings` when Python should preload known tools:

```python
from kedi.agent_adapter import CodeModeSettings, PydanticAdapter

adapter = PydanticAdapter(
    "openai:gpt-6-luna",
    codemode=CodeModeSettings(preload_tools=("lookup_release",)),
)
```

## Preload Known Tools

`preload_tools` accepts one exact exposed tool name or an indented list. Kedi
resolves the names and hydrates their schemas before the first model request,
so generated code may call those tools without `search_tools` or
`get_tool_schema`. Their schemas are included in the first CodeMode instruction.
Preloading does not execute a tool or expose a result.

Resolution is atomic and run-local. Unknown or ambiguous names, duplicate
catalog names, and `max_hydrated_tools` overflow fail before model I/O and leave
no partial hydration. Aliased tools must be named by the alias exposed to the
model. Preloaded tools count toward `max_hydrated_tools` and remain hydrated
after `execute_code(restart=True)` resets sandbox variables.

Static lists with more than five unique names produce an LSP warning by
default. VS Code users may change the lint-only
`kedi.codemode.preloadWarningThreshold` setting; it does not change the runtime
hydration limit.

## Model-Facing Tools

The model sees three CodeMode controls instead of the ordinary application tool
schemas.

For a small executable fixture, configure a model through the CLI or embedding
application and expose one read-only tool:

````kedi
```
from kedi import tool

@tool(risk="read_only")
def load_counts() -> list[int]:
    """Return the counts from the local test fixture."""
    return [2, 3, 5]
```

> use: load_counts
> codemode: enabled
[answer] << Sum load_counts through CodeMode. Return only the total.
= <answer>
````

The expected total is `10`. The controls below describe how a model reaches
that result; enabling CodeMode does not itself execute the tool.

### `search_tools`

```python
search_tools(
    *,
    query: str | None = None,
    limit: int | None = None,
    cursor: str | None = None,
)
```

The result contains only `tool_names` and `next_cursor`. Omitting `query`
returns the next deterministic alphabetical page. Supplying a query performs a
bounded name search; it does not return descriptions or schemas. The cursor is
opaque and bound to the current catalog snapshot, query, page size, and offset.
A changed tool catalog makes old cursors stale.

### `get_tool_schema`

```python
get_tool_schema(*, tool_names: list[str])
```

Names must exactly match values returned by `search_tools`. The result includes
the description, input JSON Schema, output schema when available, sequential
constraint, and the callable name used by Monty. Successfully resolved tools
are added to the current run's hydrated allowlist. Unknown, duplicate, or
ambiguous names fail explicitly.

### `execute_code`

```python
execute_code(*, code: str, restart: bool = False)
```

Only hydrated tools exist in the sandbox. Tool functions are async and accept
keyword arguments. The following snippets illustrate code generated after the
named tools have been registered, discovered, and hydrated; they are not host
Python programs with built-in `list_records` or `list_users` functions:

```python
records = await list_records(project="kedi")
active = [record for record in records if record["active"]]
len(active)
```

Independent calls may run concurrently:

```python
import asyncio

users, projects = await asyncio.gather(
    list_users(team="runtime"),
    list_projects(owner="kedi-lang"),
)
{"users": len(users), "projects": len(projects)}
```

A sequential tool runs exclusively relative to the other nested calls in that
snippet. Merely hydrating it does not serialize unrelated independent calls.
`restart=True` resets variables in the current run's sandbox without affecting
another run. Variables and successful tool results persist between
`execute_code` calls in the same run; reuse them instead of repeating completed
host calls.

When Monty code fails, `execute_code` returns the captured standard output and
traceback message separately:

```json
{
  "output": "loaded 3 rows",
  "error": "AttributeError: 'list' object has no attribute 'get'"
}
```

`output` contains only text printed before the failure, while `error` contains
only the sandbox error. For ordinary interpreter errors, variables assigned
before the failure remain available to the next cell, so the model can inspect
the checkpoint and continue without
repeating successful host calls. Application-tool failures, approval denials,
and Kedi execution limits remain failed tool calls rather than successful
sandbox results.

A worker crash or watchdog timeout is different: Monty has lost that session.
Kedi discards it, and the next cell starts with empty variables. Cancellation
also discards the interrupted session. No completed host tool is automatically replayed.

## Supported Sandbox Subset

CodeMode runs restricted Python in Monty, not CPython. See
[Sandbox and Recovery](codemode-sandbox.md#supported-sandbox-subset) for supported
constructs, unavailable host capabilities, and recovery rules.

## Tool Semantics

Nested calls use the adapter's fully assembled application tools through one
Kedi-owned invocation bridge. Kedi therefore retains:

- schema and callable validation;
- `argument_validator` canonicalization;
- static and dynamic risk resolution;
- approval allow, deny, and edit behavior;
- edit revalidation;
- `required_before_output` tracking;
- nested tool telemetry and cancellation;
- sequential constraints and local MCP session ownership.

CodeMode requires Kedi-owned inline approval. A deferred decision cannot safely
suspend and replay half of a snippet. Pydantic therefore rejects
`approval_resolution="external"` while CodeMode is active; other adapters apply
the same inline policy ownership to nested calls.

## MCP Tools

MCP support follows the adapter's interceptable local tool path:

| Adapter | CodeMode MCP behavior |
| --- | --- |
| Pydantic AI | Local `MCPToolset` tools join the catalog; provider-native MCP is rejected. |
| LangChain | `MultiServerMCPClient` tools join the catalog. |
| Claude Agent SDK | Kedi-declared stdio, SSE, and HTTP MCP tools are materialized locally and join the catalog. |
| Codex App Server | Kedi MCP declarations remain unsupported by the adapter. |

Kedi preserves each exact exposed MCP tool name and schema. Duplicate names
from multiple sources are configuration errors. A provider-native path is not
accepted when it would let the model call an application tool outside
`execute_code`.

Claude and Codex keep their harness-native filesystem, search, shell, and other
control-plane tools available. Those controls are not application tools and do
not enter the CodeMode catalog.

## Artifact Boundary

Tool results called inside Monty are not admitted to model history one by one.
They remain bounded, JSON-compatible sandbox values while code filters, joins,
or aggregates them. Only the final `execute_code` result passes through Kedi's
normal artifact policy:

- a compact derived result remains inline;
- a large final result becomes an `ArtifactRef`;
- oversized individual or aggregate nested results fail explicitly;
- artifact-store read helpers are not exposed inside `execute_code`.

Use `run_artifact_code` when the source values are already artifact references.
CodeMode and artifact code have separate responsibilities.

## Lifecycle and Limits

Catalogs, hydration, and variable state are isolated per run. See
[Lifecycle and Limits](codemode-sandbox.md#lifecycle-and-limits) for defaults,
cleanup, and bounded resource behavior.

## Native Pydantic Capability

Use the public capability directly for a single native Pydantic run:

```python
from kedi.agent_adapter import PydanticAdapter, PydanticCodeModeCapability

adapter = PydanticAdapter("openai:gpt-6-luna")
result = adapter.run_sync(
    "Use CodeMode for this task.",
    capabilities=[PydanticCodeModeCapability()],
)
```

The capability is outermost around the fully assembled Pydantic application
toolset. It is a native Pydantic convenience; LangChain, Claude Agent SDK, and
Codex use adapter projections over the same Kedi CodeMode core.
