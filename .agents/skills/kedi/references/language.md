# Language contracts for writing Kedi

## Contents

- [Source and templates](#source-and-templates)
- [Native values and calls](#native-values-and-calls)
- [Types](#types)
- [Scope and Python](#scope-and-python)
- [Control flow](#control-flow)
- [Escaping](#escaping)

## Source and templates

Use spaces consistently. Declarations and executable statements follow source order. The first top-level Python fence, apart from comments/allowed metadata, is the prelude; it runs during compilation. Keep imports/constants/helpers there. Later Python blocks have fresh locals.

Begin executable model prompts with `>>`. Same-indent continuation rows form **one** request with one output schema. Outputs become available only after that entire call completes. Start a new `>>` to use an earlier capture. A fieldless `>>` calls the model and discards its reply. `[answer] << prompt` preserves the entire reply as `str`; `<<` cannot contain output fields or a non-string capture type.

Use `<input>` for reading/rendering, `[output: T]` for capturing. Use `Annotated[T, "field meaning"]` to specify output semantics without changing T. Do not assume the prompt itself reads a path, searches the web, or observes external state.

## Native values and calls

| Context | Safe form and consequence |
| --- | --- |
| Integer/list/model initialization | A sole backtick expression, e.g. ``[n: int] = `5` `` |
| String initialization | `[label] = Count: <n>` |
| Native procedure return | Explicit `-> T`, then a sole backtick expression |
| Rendered return | `= Count: <n>`; returns a string |
| Native procedure argument | Python call or sole backtick argument, e.g. `<double(`5`)>` |
| Literal angle-call argument | `<double(5)>` passes string `"5"`; fails an integer contract |
| Keyword arguments | Python call `` `format_count(count=3)` ``; angle calls are positional |
| Whole call RHS | `[n: int] = <double(`5`)>` preserves the procedure's native result |
| Whole variable substitution RHS | `[copy] = <n>` renders text; use ``[copy: int] = `n` `` for native data |
| Whole nested argument | `<wrapper(<producer()>)>` preserves the nested result; mixed text renders |

Give every non-string procedure an explicit return annotation. Untyped parameters, custom fields, and procedure returns default to `str`, not `Any`. Required arguments precede defaults. Defaults are single-line Python expressions evaluated at definition, not on each call. Kedi procedures validate types instead of converting numeric-looking strings or dictionaries into the required contracts.

The first executed `=` ends the procedure or ordinary main program. Multiple reachable branch returns are valid; statements after a definitely terminal path are rejected. Use `> show:` for output followed by more work. Interactive roots reject returns even inside a branch/loop. A Python `return` inside a fence exits only that Python block; prefix the fence with Kedi `=` to return its result from the procedure.

## Types

Use `~Name(field: T, optional: U = `default`)` for a Pydantic-compatible model. Required fields precede defaults, defaults require explicit field types, field names are unique, and mutable defaults are copied per instance. Construct with keyword arguments; `.model_dump()` and `.model_dump_json()` remain available.

Prefer direct `str`, `int`, `float`, `bool`, `list[T]`, `dict[K,V]`, unions, `Optional`, `Literal`, `Annotated`, and declared models. Backtick type expressions must resolve to types. The documented native type namespace rejects `tuple`, `Tuple`, `Sequence`, `Mapping`, `Iterable`, `object`, `bytearray`, `slice`, and `range` as contracts even though Python can use such values. Use lists/dicts/named models at public boundaries.

For constraints, import `Constraints` from `kedi` in the prelude; then use it in backtick `Annotated` metadata or define a complete Python alias. Descriptions guide a model; constraints validate a result; neither guarantees correct model reasoning. Check provider schema support: Codex rejects the documented `Regex`, `HttpUrl`, and `FileUrl` output formats. A descriptive string can be appropriate when structured format validation is unnecessary.

Read [types](docs/core-language/types.md) for optional/default behavior, runtime namespace and exact supported constraints. Read [parameters](docs/core-language/parameters-and-returns.md) before constructing unusual signatures.

## Scope and Python

`[x] = ...` declares/replaces **in the current scope**, shadowing an outer x. `[x] := ...` updates the nearest existing Kedi binding and validates against its original contract. Assignment cannot introduce a name, add an annotation, or write reserved names/parameters. Each procedure, selected branch, and loop iteration owns a value scope.

Align triple-backtick fences and the base Python body with their surrounding Kedi indentation; indent nested Python suites further. Do not put `python` after the opening fence in actual `.kedi` source. Kedi expressions use backticks, not `${...}`, f-string interpolation outside Python, or Python `if` syntax in ordinary Kedi text.

Bare Python names and `globals()["name"]` read Kedi values; `locals()` does not. Existing Kedi bindings can be written back with validation; new ordinary block locals disappear. Nested Python functions need `global x`, not `nonlocal x`, to rebind Kedi x. Export block results through an explicit Kedi assignment. Python objects can still be mutated through aliases; scope is not a memory sandbox.

Use `Path(__file__).with_name(...)` for resources beside a file-backed program/module. Imports receive their own `__file__`. Plain relative Python paths follow process CWD, which Kedi does not change. Synthetic cells have no inferred file path.

## Control flow

- Deterministic `> if: `expr`:` and `> loop: `expr`:` require an **exact bool** and the final colon. `bool(value)` is an explicit conversion if needed. There is no `elif`; nest another branch.
- Model claims omit the final colon and use text/substitutions, including angle-wrapped Python. They cannot capture outputs. Unknown/unestablished claims classify false; this is a model judgment, not deterministic fact checking.
- Conditional loops default to 10,000 completed iterations. Update an outer binding with `:=` or Python, not a newly shadowing declaration. Bound model-driven loops deliberately.
- `> loop [item]: `iterable`:` evaluates the iterable once and traverses it sequentially. Each binder is iteration-local; it does not overwrite an outer binding of the same name.
- One immediately following sibling `> map:` is a deferred stage over retained iteration scopes. It creates no automatic list. It may overlap continuations and joins before the next statement. Use preallocated indexed destinations or synchronization for shared results; `total += value` is not an atomic reduction.

Read [loops/map](docs/core-language/loops-and-map.md) for iterator cleanup and nested stages. Read [scope](docs/core-language/scopes-and-bindings.md) for shadowing and write-back.

## Escaping

`#` begins a comment unless escaped; standalone paired `###` delimit block comments. A leading procedure/profile `###` becomes a docstring. Escape literal delimiters where they conflict: `\<`, `\>`, `\[`, `\]`, `\(`, `\)`, `\,`, `\=`, `\@`, `\#`, `\~`, a backtick, and backslash. `\n`, `\t`, `\s` preserve whitespace. Unsupported escapes fail. Backslash line continuation applies to rendered returns, not multiline model templates.

Use the executable [core examples](examples.md), [source structure](docs/core-language/source-structure.md), [substitutions](docs/core-language/substitutions-and-calls.md), and [escaping](docs/core-language/multiline-and-escaping.md) when syntax is uncertain.
