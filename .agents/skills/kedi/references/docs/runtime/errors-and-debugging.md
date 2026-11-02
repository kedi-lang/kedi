# Errors, Frames, and Tracebacks

Kedi reports failures in the language that owns the failing operation. Parse
diagnostics point to Kedi syntax, runtime failures carry Kedi execution frames,
and embedded Python tracebacks are mapped back from generated Python to the
original `.kedi` source.

## Failure Phases

| Phase | Typical error | What the location means |
| --- | --- | --- |
| Source loading | file, import, or package error | the source or module that could not be loaded |
| Parsing | `KediParseError` | the token or construct that violates Kedi syntax |
| Compilation | declaration, type, profile, or capability error | the Kedi construct whose contract cannot be built |
| Execution | `KediExecutionError` | the active Kedi call and statement stack |
| Adapter call | provider, protocol, approval, or schema error | the model-facing statement that initiated the call |

Start with `kedi parse program.kedi`. It checks source loading and parsing
without executing deterministic code or making a model call. Compilation and
runtime-only errors appear when the program runs.

## Parse Diagnostics

Malformed syntax produces `KediParseError` with one or more diagnostics:

```text
Kedi parse error:

  File "workflow.kedi", line 4, column 3
  ...
  ^

Kedi error: ParseError
  unexpected source at top level
```

The CLI and LSP use the same source-aware diagnostics. A diagnostic includes
the source path, line, column, relevant source text, and a focused message when
the parser can identify the violated rule.

## Execution Frames

A runtime failure is represented by `KediExecutionError`. Its `frames` tuple
contains immutable `KediTraceFrame` values captured while the program unwinds.
A frame records:

- `kind`: the machine-facing operation category;
- `label`: the human-facing operation name;
- `span`: the source range and physical source path;
- `procedure_name`: the active procedure when applicable;
- `source_snippet`: the hydrated source line used by the renderer.

Frame kinds cover the boundaries that matter to a Kedi author: top-level
program, procedure, template, variable initialization, variable initialization
block, assignment, assignment block, return, return block, inline Python,
fenced Python, Python statement, prelude, and type expression.

Frames are nested execution context, not a copy of Python's internal call
stack. Kedi deliberately omits compiler and executor implementation frames so
the traceback answers three questions directly:

1. Which top-level or procedure call was active?
2. Which Kedi statement crossed the failing boundary?
3. If Python was involved, which embedded Python line failed?

For example:

```kedi
@divide(value: int) -> int:
  = `100 // value`

= `divide(0)`
```

renders as:

```text
Kedi stack trace:
  in top-level program

  File "math.kedi", line 4, in inline python
  ❱ 4 = `divide(0)`

  File "math.kedi", line 1, in procedure divide
  ❱ 1 @divide(value: int) -> int:

  File "math.kedi", line 2, in return block
  ❱ 2   = `100 // value`

  File "math.kedi", line 2, in inline python
    embedded Python line 1
    ❱ 1 100 // value

Inline Python error: ZeroDivisionError
  integer division or modulo by zero
```

Read the frames from top to bottom as the Kedi path to the failure. The final
heading and detail identify the original exception.

## Embedded Python Trace Mapping

Inline expressions, fenced blocks, preludes, and type expressions are compiled
with unique virtual Python filenames. Kedi records a `KediPythonOrigin` for each
generated unit and converts matching Python traceback entries into
`KediPythonTrace` values.

Each Python trace retains:

- the embedded Python code and its generated filename;
- the Python line and optional column;
- the corresponding Kedi span and physical source path;
- the mapped Kedi line and hydrated source snippet;
- the owning procedure when one exists.

This mapping is why a `SyntaxError`, `NameError`, or exception raised several
calls deep can still point to a line inside the original `.kedi` block.
Generated virtual filenames are an implementation detail and are omitted from
the normal CLI output.

Nested embedded-Python calls may produce more than one mapped trace. The
renderer labels intermediate entries as embedded Python calls and the final
entry as the embedded Python error. It limits displayed context around the
failing line instead of printing an entire large fenced block.

Do not catch and rethrow an exception merely to recover a source location.
Uncaught propagation preserves the original traceback and gives Kedi more
mapping information.

## Exception Classes and Original Causes

All execution failures are catchable as `KediExecutionError`. Common Python
categories also use specialized subclasses:

| Kedi exception | Also behaves as |
| --- | --- |
| `KediTypeExecutionError` | `TypeError` |
| `KediValueExecutionError` | `ValueError` |
| `KediKeyExecutionError` | `KeyError` |
| `KediNameError` | `NameError` |

The exception's `original` attribute retains the provider, Python, or runtime
exception that Kedi wrapped. This is useful for retry classification and
programmatic inspection without exposing internal frames in the CLI.

Python callers can inspect or render the error:

```python
from kedi.errors import KediExecutionError

try:
    runtime.run_main()
except KediExecutionError as exc:
    print(exc.render())
    print(exc.frames)
    print(exc.python_traces)
    if exc.original is not None:
        print(type(exc.original).__name__)
```

`exc.render()` is plain text by default. Pass `use_color=True` for ANSI terminal
colors. `str(exc)` and `repr(exc)` use the same plain renderer.

## Source Maps, Imports, and In-Memory Source

Source loading retains the physical documents that contribute to a program.
Frames from imported modules, generated companions, tests, evaluations, and
optimization code can therefore point to their own files instead of a merged
virtual line.

Low-level callers of `parse_program()` and `compile_program()` should preserve
the `source_path` and `source_map` returned by source loading. In-memory source
without a physical path is still traceable, but its label will be synthetic,
such as `<command>`.

## Structural, Type, and Capability Failures

After syntax, Kedi validates declarations, imports and exports, profile graphs,
directives, package manifests, type expressions, parameter ordering, and known
adapter capabilities. Runtime type boundaries cover procedure parameters,
returns, model outputs, assignments, custom fields, and edited tool arguments.
Values are not silently coerced merely to hide a violated Kedi contract.

Backend selection, credentials, model availability, unsupported schema
formats, tool or MCP capability gaps, protocol failures, approval denials, and
provider errors surface at the narrowest adapter boundary that can identify the
unmet contract. Kedi does not turn these failures into empty output values.

Differentiate a capability failure from a provider failure. A capability error
means the selected adapter cannot represent the requested operation; a
provider error means a representable request failed operationally.

## Concurrent Failures

Parallel execution drains already-scheduled calls. The first failure is raised
to the caller; additional concurrent failures are logged so they are not
silently lost. The trace stack is context-local, so frames from concurrent
tasks do not belong to one shared mutable stack.

Reproduce a concurrency issue sequentially for a simpler trace, but treat any
difference in final values or failure semantics as a runtime bug.

## Executor Debug Events

For interactive breakpoints, stepping and bounded read-only scopes in VS Code or
Zed, use the optional [Debugger and Inspector](../tooling/debugger.md). It does not
enable the exporter below or write a persistent transcript.

Tracebacks explain a failed execution path. Executor debug events instead
record operations and values, including successful ones. An executor can emit
`ExecutorDebugEvent` records with a UTC timestamp, executor, step, event name,
and payload fields such as code, inputs, environment, arguments, result,
output, and error.

Attach a `MarkdownDebugExporter` from Python:

```python
from kedi import MarkdownDebugExporter
from kedi.executors import DefaultExecutor

exporter = MarkdownDebugExporter("runtime-debug.md")
executor = DefaultExecutor(debug_exporter=exporter)
```

When no path is supplied, the default name is
`<normalized-name>_debug_export_<UTC timestamp>.md` in the current directory.

“Sanitized” means values are converted to representable forms; it does not mean
secret redaction. Debug exports may contain prompts, code, environment values,
tool arguments, and outputs. Store them securely, inspect them before sharing,
and delete them when no longer needed.

## Optimization Debugging

`KEDI_DEBUG_OPTIMIZED=1` prints optimized-prompt loading and injection details
to stderr. Use it to diagnose stale or missing `.optimized.json` artifacts, not
as a general runtime trace switch.

## Debugging Order

1. Run `kedi parse`.
2. Resolve LSP structural and capability diagnostics.
3. Run sequentially and read the Kedi frames from top to bottom.
4. Inspect the final original-exception heading and embedded Python mapping.
5. Verify the active profile, model, tools, approvals, and adapter capability.
6. Add a scoped executor debug exporter only when the traceback is insufficient.
7. Re-enable parallelism and verify equivalent behavior.
