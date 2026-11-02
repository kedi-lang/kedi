# Context and Runtime { #runtime }

The runtime turns Kedi source into a validated program, compiles declarations
and scopes, executes deterministic statements, and crosses the model boundary
only at explicit template or raw-invoke statements.

## Parse, Compile, Execute

1. Source loading merges the main file and supported companion artifacts while
   retaining a source map.
2. Parsing builds a structured program and reports syntax diagnostics.
3. Structural validation resolves directives, modules, declarations, and
   profile graphs.
4. Compilation creates runtime types, procedures, environments, and adapter
   state.
5. Execution evaluates top-level statements and the resulting dataflow.

`kedi parse` stops after parsing/validation. Normal `kedi file.kedi` continues
through compilation and execution.

## Value Environments

The runtime separates imports, prelude names, top-level values, procedure
frames, and lexical closures while presenting one predictable lookup model to
Kedi expressions. Native Python values stay native until a rendered text
boundary converts them to strings.

## Adapter Invocation

`>>` with output fields produces a typed schema call. `[name] << ...` produces
raw response text. `>>` without output fields still invokes the adapter and
discards its response. Initializations, assignments, and Python expressions do
not call a model.

## Execution Engines

Bounded concurrency is the default, using a shared pool of eight workers.
Independent model calls can overlap; dependent calls wait for their inputs.
Select sequential execution when external effects require source ordering.
The value and type contracts remain the same, but stochastic outputs, completion
order, and the first observed concurrent failure need not be identical.

## Interactive Execution

`kedi.interactive()` and `kedi --idle` execute complete fragments against one
persistent, process-local runtime. Earlier fragments are not replayed, while
their values, declarations, imports, conversation state, and artifacts remain
available to later fragments. See
[Interactive Execution](interactive-execution.md) for lifecycle, result,
traceback, multiline-input, and terminal-command semantics.

## Caches and Artifacts

The Python API has in-memory parse and optional response caches. AI-generated
procedures use `.cache.kedi`; prompt optimization uses `.optimized.json`.
These are separate systems and are documented in [Caching](caching.md).

Tool artifacts are a separate runtime feature for keeping large generated
fields and tool results out of model context. Native Kedi/Python reads retain
the original typed value while model calls receive bounded references. See
[Tool Artifacts](tool-artifacts.md).

## Diagnostics

Parse errors include file, line, column, source snippet, and message. Runtime
errors carry Kedi procedure/statement frames and remapped embedded-Python
locations. Capability and adapter errors fail at the narrowest boundary that
can identify the unmet contract. See
[Errors, Frames, and Tracebacks](errors-and-debugging.md) for the complete
exception model and programmatic inspection API.

## Telemetry

Kedi is a no-op until its separate OpenTelemetry instrumentor is enabled. The
instrumentor exposes runtime, agent, and artifact spans and metrics while
keeping content and source capture disabled by default. See
[Telemetry](telemetry.md) for configuration, Logfire integration, privacy
controls, span hierarchy, and metric names.

## In This Section

**Execution**

- [Execution and Dataflow](execution-and-dataflow.md)
- [Concurrency](concurrency.md)
- [Run Budgets](run-budgets.md)
- [Incremental Execution](interactive-execution.md)

**Context Management**

- [Artifacts](tool-artifacts.md)
- [History and Caching](caching.md)
- [Conversation History](history.md)
- [Compaction](compaction.md)
- [Provider Prefix Caching](prefix-cache.md)
- [Artifact Admission](artifact-policy.md)
- [Artifact Retrieval](artifact-retrieval.md)
- [Artifact Reduction](artifact-reduction.md)
- [Artifact Lifetime](artifact-lifecycle.md)
- [Artifact Safety](artifact-safety.md)

**Observability**

- [Telemetry](telemetry.md)
- [Errors and Tracebacks](errors-and-debugging.md)
- [Troubleshooting](../reference/diagnostics-and-troubleshooting.md)
