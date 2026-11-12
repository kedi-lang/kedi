# Run and Parse Commands

## Run a Source File

```bash
kedi program.kedi
```

Kedi loads imports, optimized prompts, generates missing `> auto:` procedures,
compiles, and runs the top-level program.

## Run Inline Source

```bash
kedi -c '= Hello from Kedi'
```

Inline source supports direct run or `--parse`, but not `--test`, `--eval`, or
`--optimize`, because those workflows need file-backed artifacts.

## Pass Program Arguments

Unknown `--name value` options after the source become `args.name`:

```bash
kedi greet.kedi --name Ada --verbose
```

For that command, `greet.kedi` can be:

```kedi
[name: str] = `args.name or "world"`
> if: `args.verbose is True`:
  = Hello, <name>. Verbose output is enabled.
> else:
  = Hello, <name>.
```

The result is `Hello, Ada. Verbose output is enabled.` No model is involved.

Values are strings; flags are `True`; missing names read as `None`. Hyphens
normalize to underscores. First occurrence wins when an option repeats.
Non-option positional extras are ignored.

## Parse Without Execution

All are supported:

```bash
kedi -p program.kedi
kedi program.kedi --parse
kedi -p -c '= Hello from Kedi'
```

A successful parse prints `Parsed successfully.` and performs no model call or
program execution.

## Explain Model Calls Without Execution

```bash
kedi explain program.kedi
```

`explain` parses the source directly, without importing its modules, running
its prelude, evaluating dynamic model selectors, or invoking a model. It lists
each static model-call site with its source-order adapter/model selection,
visible tools, and capability requirements. Unknown dynamic values are labeled
`unresolved`; they are not guessed. A known unsupported requirement makes the
command exit nonzero. The report omits prompt and tool-result payloads.

For a Python `@kedi.query` or `@kedi.bind` function, use
`kedi.explain(function)` to obtain the structured `ExplainReport` without
calling the function.

```python
import kedi


@kedi.query
def classify(note: str) -> str:
    """kedi
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > requires: structured_output
    >> The category of <note> is [category: str].
    = <category>
    """


report = kedi.explain(classify)
call = report.calls[0]
assert call.adapter.value == "pydantic"
assert call.model.value == "openai:gpt-6-luna"
assert call.requirements[0].status == "satisfied"
assert not report.diagnostics
```

This inspects the function without calling `classify`, supplying `note`, or
authenticating with a provider. `call.source_location` identifies the call site;
`call.tools` contains statically known tool metadata. A satisfied declaration
means the adapter advertises a capability, not that credentials or the remote
model have been tested. `explain` is not a dry-run execution or call-count forecast:
branches, loops, retries, and dynamic task starts still depend on runtime inputs.

## `kedi parse`

```bash
kedi parse program.kedi
```

This is an alternate parse-only form. It requires one source path.

## Select an Adapter or Agent

The CLI option is historically named `--adapter` and accepts all built-ins:

```bash
kedi program.kedi --adapter pydantic
kedi program.kedi --adapter codex
```

Inside source, use the semantically strict `> adapter:` for Pydantic/
LangChain and `> agent:` for Claude/Codex/ACP/A2A.

## Select a Model

```bash
kedi program.kedi \
  --adapter pydantic \
  --adapter-model openai:gpt-6-luna
```

Environment equivalents are `KEDI_ADAPTER` and `KEDI_ADAPTER_MODEL`. Lexical
source directives and profiles can override CLI defaults.

A provider API key authenticates requests; it does not select a model. Set the
model identifier explicitly and install its provider dependency. Deterministic
programs such as the greeting above need neither credentials nor a model call.

## ACP Commands

ACP commands must be embedded explicitly in a `> agent: acp:` connection body,
or supplied through `ACPAdapter(command=...)` in Python. The CLI does
not accept a global ACP command because one process-level value cannot describe
multiple ACP profiles.

## Exit Codes and Rendered Errors

Successful runs return zero. Source read, parse, execution, validation,
codegen, optimizer, and package errors return nonzero. Parse diagnostics include
source location and excerpts; execution errors include hydrated Kedi/Python
trace context when available. The frame model and embedded-Python mapping are
documented in [Errors, Frames, and Tracebacks](../runtime/errors-and-debugging.md).
