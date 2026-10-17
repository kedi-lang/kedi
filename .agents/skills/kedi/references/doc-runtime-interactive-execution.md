> Offline source: `kedi-docs-main/docs/runtime/interactive-execution.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Python API](#python-api)
- [Results](#results)
- [Source Identity and Imports](#source-identity-and-imports)
- [Configuration](#configuration)
- [Durable Snapshots](#durable-snapshots)
- [Lifecycle and Failure Semantics](#lifecycle-and-failure-semantics)
- [Terminal REPL](#terminal-repl)
- [Explicit Multiline Input](#explicit-multiline-input)
- [Syntax Highlighting](#syntax-highlighting)
- [Inspecting Values](#inspecting-values)
- [Commands, History, and Exit](#commands-history-and-exit)
- [Choosing a Surface](#choosing-a-surface)

# Interactive Execution

Interactive execution evaluates complete Kedi fragments against one persistent,
process-local runtime. It is useful for terminal exploration, notebooks,
debuggers, and hosts that need to submit code incrementally without replaying
earlier cells.

Kedi exposes the same execution model through two surfaces:

- `kedi.interactive()` creates a session for Python hosts.
- `kedi --idle` starts a terminal REPL backed by that session.

Normal file execution is unchanged. Interactive execution is an additional
surface, not a replacement for `compile_program(...).run_main()`.


## Python API

Create sessions with `kedi.interactive()` and execute one complete fragment at
a time:

```python
import kedi


with kedi.interactive() as session:
    session.execute("[base: int] = `40`")
    session.execute(
        """
@add_two() -> int:
    = `base + 2`
""".strip()
    )
    session.execute("> show: `add_two()`")  # displays 42
```

The second and third fragments can read declarations and values created by the
first fragment. Values, procedures, types, imports, profiles, directives,
conversation state, and artifacts remain available until the session closes.
Earlier fragments are neither concatenated nor executed again.

### Results

`InteractiveSession.execute()` returns `None`. Use `> show:` to display a value
and bindings to retain it. Interactive-root returns, including those in root
branches or loops, are rejected before any fragment side effect. Procedures
defined or called in a session retain normal typed returns.

Display evaluates once and does not replay earlier cells or introduce model
calls. Resolving a pending capture keeps its normal computation. Existing
notebooks using root returns need explicit migration; user files are not
silently rewritten.

### Source Identity and Imports

Every fragment gets a unique source name such as `<interactive:1>`. That name
appears in parse diagnostics and Kedi stack traces. A host can supply a more
useful identity:

```python
with kedi.interactive(cwd="examples/cells") as session:
    session.execute(
        "> import: helpers\n> show: `answer`",
        source_name="answer.kedi",
    )
```

A real `source_name` becomes the base for relative imports. Synthetic names use
the session's `cwd`, which defaults to the process working directory. Source
names must be unique within a session so diagnostics cannot ambiguously refer
to two fragments.

### Configuration

`kedi.interactive()` accepts the same model, adapter or agent, system prompt,
effort, settings, tools, environment, MCP, approval, skills, artifacts,
conversation, and parallel-execution inputs as the other high-level Python API
surfaces. It also accepts an executor, execution engine, `cwd`, and subagent
limits.

Configuration is resolved when the session is created. The session owns the
runtime assembled from that configuration; later calls to `kedi.configure()` do
not rebuild an existing session.

### Durable Snapshots

`kedi.dump_session(session, path)` writes a pickle-free snapshot only when the
complete logical session can be restored without replaying executable cells.
Restore it with `kedi.load_session(path)`. The corresponding
`InteractiveSession.dump()` and `InteractiveSession.load()` methods expose the
same operations:

```python
from pathlib import Path

import kedi


snapshot = Path("work.kedi-state")
with kedi.interactive() as session:
    session.execute("[base: int] = `40`")
    session.execute(
        """
@add_two() -> int:
    = `base + 2`
""".strip()
    )
    kedi.dump_session(session, snapshot)

with kedi.load_session(snapshot) as session:
    session.execute("> show: `add_two()`")  # displays 42
```

The operation is strict and all-or-nothing. Kedi validates all fragments,
environment values, active profile state, conversation state, and artifacts
before publishing the file. `SessionDumpError` reports every unsupported
boundary found during preflight. It never silently removes a binding, and an
existing destination remains untouched when validation fails. Successful
writes use a mode-`0600` temporary file, `fsync`, and atomic replacement.

The snapshot value codec preserves scalar values, bytes, complex and decimal
numbers, UUIDs, dates and times, paths, regular expressions, standard
containers with typed keys, and source-backed Kedi type instances. Kedi stores
fragment sources and digests so procedures, types, profiles, tracebacks, and
source identities can be rebuilt before native values are installed. The
document integrity hash, format version, and Kedi version are checked on load.

Kedi rejects state that cannot be restored with equivalent semantics:

- arbitrary Python callables, generators, open resources, concurrency
  primitives, tasks/futures, classes, and unknown object instances;
- shared or cyclic mutable object graphs;
- dynamic approval handlers, non-importable lifecycle hook handlers,
  process-bound tool/profile bindings, active
  artifacts, conversation turns, or adapter-native continuation state;
- imports, inline Python preludes, executable type defaults, and runtime-scoped
  declarations that would require old code to run during load.

Lifecycle hooks are stored without pickle. A handler is restorable only when
its module and qualified name resolve back to that exact callable. Importable
top-level functions are supported; lambdas, local functions, closures, bound
methods, and handlers from `__main__` make the dump fail during preflight.
Load only session snapshots you trust. Restoring an import-addressable hook
imports its Python module, so that module's normal import-time code may run;
Kedi does not execute the hook handler itself during load.

Adapters and executors are not serialized. Supply them when the restored
session needs those infrastructure dependencies:

```python
session = kedi.load_session(
    "work.kedi-state",
    adapter=adapter,
    executor=executor,
)
```

`load_session()` accepts a keyword-only `session_type=` factory and defaults
to `InteractiveSession`. An `InteractiveSession` subclass can be supplied when
the restored object needs application-specific behavior.

Apart from resolving documented import-addressable hook handlers, loading
compiles source-backed declarations but never re-executes prior
initializations, assignments, templates, tools, LLM requests, filesystem
writes, or other top-level side effects.

### Lifecycle and Failure Semantics

Use the context manager form when possible. `close()` is idempotent and releases
session-owned resources. Executing after close, closing during execution, or
starting concurrent or re-entrant `execute()` calls raises an error.

The execution model is synchronous and non-transactional. If a fragment fails:

- it is not retried automatically;
- state committed before the failure remains visible;
- completed external side effects are not rolled back;
- the source-aware exception identifies the failed fragment.

Interactive fragments reject package metadata, export directives, and
`@test`/`@eval` suites. Those constructs describe complete files, packages, or
validation surfaces rather than one incremental cell.


## Terminal REPL

See [Terminal REPL](doc-tooling-repl.md).

### Explicit Multiline Input

See [Terminal REPL](doc-tooling-repl.md).

### Syntax Highlighting

See [Terminal REPL](doc-tooling-repl.md).

### Inspecting Values

See [Terminal REPL](doc-tooling-repl.md).

### Commands, History, and Exit

See [Terminal REPL](doc-tooling-repl.md).

## Choosing a Surface

Use `kedi --idle` for direct terminal exploration. Use `kedi.interactive()` when
an editor, notebook, debugger, or application owns input, output, source names,
and lifecycle. Use ordinary file execution when a complete program should be
repeatable from source as one unit.
