# Parameters and Returns

Procedure signatures define the native contract between Kedi, Python, tools,
tests, and evals. An omitted annotation is not `Any`: it defaults to `str`.

## Positional and Typed Parameters

```kedi
@render_issue(id: int, title: str, tags: list[str]) -> str:
  = \#<`id`> <title> \[<`", ".join(tags)`>\]
```

Kedi validates each argument against its annotation. It does not coerce a
rendered `"42"` into `42`, and it does not accept a dict where a custom model is
required.

Direct type names and runtime Python type expressions are equivalent:

```kedi
@first(items: `list[str]`) -> `str`:
  = `items[0]`
```

Prefer direct annotations when possible because the LSP can resolve and explain
them statically. Use backtick annotations for a type supplied by Python state.

## Default Parameters

Defaults are single-line Python expressions:

```kedi
@format_count(count: int, label: str = `"items"`, compact: bool = `False`) -> str:
  = `f"{count}{label}" if compact else f"{count} {label}"`

= <format_count(`3`)>
```

Required parameters must precede defaulted parameters. Duplicate parameters and
fenced block defaults are parse errors.

Defaults are evaluated when the procedure is defined, not on each call:

````kedi
```
seed = 1
```

@value(current: int = `seed`) -> int:
  = `current`

`seed = 9`
= `value()`  # 1
````

Untyped defaults retain their native Python value, but the parameter itself
still follows the signature contract. Annotate non-string defaults explicitly
for clarity.

## Call Arguments

Kedi angle-call syntax is positional:

```kedi
= <format_count(`3`, records, `True`)>
```

Use single backticks to pass a native Python result. Plain text and mixed
template arguments render to `str`. Escape text commas as `\,`.

Named arguments are available through Python because compiled Kedi procedures
have real Python signatures:

```kedi
= `format_count(count=3, compact=True, label="records")`
```

Use angle calls for normal Kedi composition. Use Python keyword calls when
argument order would be unclear or values are already native.

## Declared Return Types

Return annotations are enforced:

```kedi
~Row(id: int, active: bool)

@active_ids(rows: list[Row]) -> list[int]:
  = `[row.id for row in rows if row.active]`
```

Without `-> ...`, a procedure has a `str` return contract. Returning a native
integer, list, or model from an untyped procedure is an error. Annotate every
procedure that intentionally returns a non-string.

Kedi validates rather than coerces return values. `= "5"` cannot satisfy
`-> int`; use ``= `5` `` or parse explicitly.

## Rendered and Native Returns

A rendered return starts with `=` and combines literal text, substitutions,
calls, and inline Python:

```kedi
@status(name: str, healthy: bool) -> str:
  = <name>: <`"healthy" if healthy else "degraded"`>
```

A sole backtick expression returns its native result:

```kedi
@total(values: list[int]) -> int:
  = `sum(values)`
```

A fenced block can return a value after several statements:

````kedi
@median(values: list[float]) -> float:
  = ```
  ordered = sorted(values)
  middle = len(ordered) // 2
  if len(ordered) % 2:
      return ordered[middle]
  return (ordered[middle - 1] + ordered[middle]) / 2
  ```
````

Choose a rendered return for human-facing text and a native return for values
that callers will calculate with, validate, serialize, or pass to tools.

## Multiline Returns

Use a trailing backslash to continue one rendered return across physical lines:

```kedi
@notice(name: str) -> str:
  = Hello <name>, \
    your report is ready.
```

Return continuation is a text feature. It is not how multiline model prompts
work; `>>` prompts use same-indentation continuation lines instead. Use `\\` for
a literal backslash.

Kedi trims line-end whitespace while preserving meaningful internal spacing.
Escaped whitespace (`\s`, `\t`, and `\n`) remains significant even at a
boundary.

## Early Return and No Return

The first evaluated `=` ends the enclosing procedure or regular program.
Different reachable branches may each return; this is not a one-return-per-file
restriction. Statements after a definitely terminal path are rejected before
execution. A Python `return` inside a fenced block still exits that Python
block only.

```kedi
@choose(value: int) -> int:
  > if: `value < 0`:
    = `0`
  = `value`

= `choose(-1)`
```

This returns `0`. Work after a taken return is not started. Already scheduled
work is drained normally; its failures are not hidden by return.

### Display Without Returning

```kedi
[city] = Ankara
> show: The selected city is <city>.
> show: `len(city)`
= <city>
```

`> show:` evaluates once, displays with a newline, and continues without changing
the return value. It supports literal text, substitutions and inline Python,
but cannot declare output captures. It does not implicitly generate model
output or append display text to model history.

Notebook and IDLE roots reject `=` before execution, even in nested control
flow. Use `> show:` there; procedure returns remain valid. IDLE also accepts
`:show` as an alias. When migrating old repeated returns, convert intended
displays only; do not automatically print discarded computations or sensitive
values.

An untyped procedure with no returned value produces `""`. Relying on that is
appropriate only for side-effect-oriented helpers. Public procedures should
return deliberately and declare the contract.

## Errors

Kedi reports missing and extra arguments, invalid default ordering, unresolved
types, parameter mismatches, and return mismatches. These errors occur at the
boundary where the contract is violated, before an invalid value silently
propagates into later model or tool calls.
