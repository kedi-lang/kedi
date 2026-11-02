# Projects and Execution

## Single-File Programs

A `.kedi` file can contain top-level values, types, procedures, profiles, tests,
evals, and the final executable statements. Single-file programs are a good fit
when the public surface is small and all prompts share one lifecycle.

## Multi-File Projects

Split reusable procedures, types, values, and profiles into modules. Imports are
relative to the importing file first, then fall back to bundled modules and the
local Kedi package registry.

```text
project/
  main.kedi
  labels.kedi
```

`labels.kedi`:

```kedi
@label(title: str) -> str:
  = `"-".join(title.strip().casefold().split())`

> export:
  label
```

`main.kedi`:

```kedi
> import: labels:
  label

= `label(args.title)`
```

Only explicitly exported names are visible to importers.
This complete two-file example needs no model, package installation or credentials.

## Entry Points

The source passed to `kedi` is the root program:

```bash
kedi main.kedi --title "Release Notes"
```

Modules are initialized as the root program is compiled. A package entry point
is the `main.kedi` inside the source directory declared by `package.kedi`.

From `project/`, the command above prints `release-notes`. The import resolves
beside `main.kedi`, not relative to wherever the shell happens to be. For example,
`kedi project/main.kedi --title "Release Notes"` also works from its parent.

## Source Loading Order

Kedi loads the root source, imported modules, optional generated
`*.cache.kedi` implementations, and optimized prompt artifacts through a source
map. Imports are initialized at most once per root compilation, including
diamond import graphs.

Bindings still obey source order. A later declaration or import can replace a
name introduced earlier, even when the binding kind changes.

## Runtime Environment

The runtime environment contains Kedi values, procedure frames, prelude names,
registered Python tools and types, configured environment values, and the
reserved CLI argument object. Python blocks see existing Kedi variables as
globals.

Do not use the runtime environment as a general mutable global store. Prefer
procedure parameters and returns for explicit dataflow; reserve prelude and
configured `env` for shared dependencies or types.

## Backend and Model Selection

The CLI defaults to the Pydantic adapter and `openai/gpt-6-luna`, unless
overridden by environment configuration. The learning examples explicitly select
`openai:gpt-6-luna` instead. To configure a source file that has no model directive:

```bash
kedi main.kedi \
  --adapter pydantic \
  --adapter-model openai:gpt-6-luna
```

Source directives and profiles can override CLI defaults in lexical scope. A
single scope cannot select both a framework adapter and an agent harness.

The deterministic `main.kedi` above does not need that model or call it. Model
selection becomes relevant only at a model boundary; loading credentials does
not introduce one. See [Backend Selection](../agentic-engineering/backend-selection.md)
for profile precedence and capability checks.

## Generated and Cached Artifacts

Depending on the features used, Kedi may create:

- `source.cache.kedi` for AI-generated procedure implementations;
- optimized prompt JSON and score artifacts;
- optimizer checkpoint directories;
- in-memory parse and response cache entries;
- an optional owner-only subagent state file configured by Python callers.

Generated artifacts are implementation inputs, not source-of-truth replacements
for the original procedure signatures, tests, eval datasets, or prompt spans.

These are different mechanisms, not one interchangeable cache. `--no-cache`
controls generated procedure code, not provider prefix caching or artifact
retention. See [Cache Control](../python-api/cache-control.md) and
[Prefix Cache](../runtime/prefix-cache.md). Review generated code before trusting
it, and treat conversation/checkpoint files as potentially sensitive.

## Choose an Execution Surface

- Use the CLI for a complete file and a fresh execution.
- Use [Python embedding](../python-api/embedding.md) when the host owns adapters,
  injected tools and runtime cleanup.
- Use [IDLE](../tooling/repl.md) or [Notebook](../tooling/notebook.md) for a retained
  interactive session. Cells still need to run in dependency order; editing an
  earlier cell does not automatically recompute later results.

Kedi executes embedded Python with the process's permissions. Neither a module
boundary nor an approval handler makes untrusted source safe to run.
