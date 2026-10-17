> Offline source: `kedi-docs-main/docs/runtime/index.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Parse, Compile, Execute](#parse-compile-execute)
- [Value Environments](#value-environments)
- [Adapter Invocation](#adapter-invocation)
- [Execution Engines](#execution-engines)
- [Interactive Execution](#interactive-execution)
- [Caches and Artifacts](#caches-and-artifacts)
- [Diagnostics](#diagnostics)
- [Telemetry](#telemetry)
- [In This Section](#in-this-section)

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
[Interactive Execution](doc-runtime-interactive-execution.md) for lifecycle, result,
traceback, multiline-input, and terminal-command semantics.

## Caches and Artifacts

The Python API has in-memory parse and optional response caches. AI-generated
procedures use `.cache.kedi`; prompt optimization uses `.optimized.json`.
These are separate systems and are documented in [Caching](doc-runtime-caching.md).

Tool artifacts are a separate runtime feature for keeping large generated
fields and tool results out of model context. Native Kedi/Python reads retain
the original typed value while model calls receive bounded references. See
[Tool Artifacts](doc-runtime-tool-artifacts.md).

## Diagnostics

Parse errors include file, line, column, source snippet, and message. Runtime
errors carry Kedi procedure/statement frames and remapped embedded-Python
locations. Capability and adapter errors fail at the narrowest boundary that
can identify the unmet contract. See
[Errors, Frames, and Tracebacks](doc-runtime-errors-and-debugging.md) for the complete
exception model and programmatic inspection API.

## Telemetry

Kedi is a no-op until its separate OpenTelemetry instrumentor is enabled. The
instrumentor exposes runtime, agent, and artifact spans and metrics while
keeping content and source capture disabled by default. See
[Telemetry](doc-runtime-telemetry.md) for configuration, Logfire integration, privacy
controls, span hierarchy, and metric names.

## In This Section

**Execution**

- [Execution and Dataflow](doc-runtime-execution-and-dataflow.md)
- [Concurrency](doc-runtime-concurrency.md)
- [Run Budgets](doc-runtime-run-budgets.md)
- [Incremental Execution](doc-runtime-interactive-execution.md)

**Context Management**

- [Artifacts](doc-runtime-tool-artifacts.md)
- [History and Caching](doc-runtime-caching.md)
- [Conversation History](doc-runtime-history.md)
- [Compaction](doc-runtime-compaction.md)
- [Provider Prefix Caching](doc-runtime-prefix-cache.md)
- [Artifact Admission](doc-runtime-artifact-policy.md)
- [Artifact Retrieval](doc-runtime-artifact-retrieval.md)
- [Artifact Reduction](doc-runtime-artifact-reduction.md)
- [Artifact Lifetime](doc-runtime-artifact-lifecycle.md)
- [Artifact Safety](doc-runtime-artifact-safety.md)

**Observability**

- [Telemetry](doc-runtime-telemetry.md)
- [Errors and Tracebacks](doc-runtime-errors-and-debugging.md)
- [Troubleshooting](doc-reference-diagnostics-and-troubleshooting.md)
