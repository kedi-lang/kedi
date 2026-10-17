> Offline source: `kedi-docs-main/docs/reference/syntax.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

> SOURCE RECONCILIATION: named calls use Python syntax; native angle calls are positional. Read core-language/parameters-and-returns.md for the exact rule.

## Contents

- [Program Forms](#program-forms)
- [Templates and Values](#templates-and-values)
- [Procedures](#procedures)
- [Custom Types](#custom-types)
- [Python Forms](#python-forms)
- [Validation and Generation](#validation-and-generation)
- [Escapes](#escapes)
- [Names and Comments](#names-and-comments)

# Syntax Index

## Program Forms

| Form | Meaning |
| --- | --- |
| `@name(params) -> Type:` | Define a lexically scoped procedure |
| `~Name(fields)` | Define a Pydantic-compatible custom type |
| `[name: Type] = value` | Deterministic variable initialization |
| `[name] := value` | Assign to the nearest visible binding |
| `>> prompt [field: Type]` | Structured model call |
| `[name] << prompt` | Raw text model call and capture |
| `>> prompt` with no fields | Raw model call whose response is discarded |
| `= value` | Return text assembled from literals and substitutions |
| `= `expression`` | Return one native Python value |
| `= ```...```` | Execute Python statements and return their value |
| `` `statement` `` | Execute one Python statement for side effects |
| `````...````` | Execute a multiline Python block |
| `> name: ...` | Apply a directive |
| `> artifacts:` | Configure scoped artifact storage for large values |
| `# ...` / `### ... ###` | Line or block comment |

Declarations and executable statements retain source order. Indentation defines
scope; tabs compare as width four, but spaces are recommended.

## Templates and Values

| Syntax | Role |
| --- | --- |
| `<name>` | Read and render a variable |
| `<procedure(args)>` | Call a procedure and render its result |
| `<`expression`>` | Evaluate Python and render the result |
| `[field]` | Capture a model-produced string |
| `[field: Type]` | Capture and validate a typed model output |
| `[field: `TypeExpr`]` | Resolve the output type from Python |

Angle brackets are reads; square brackets are writes. Substitution always
renders into surrounding text. A sole Python expression in an initialization,
assignment, or native return preserves the object.

Adjacent continuation lines after one `>>` are newline-joined into one model
call. Outputs become visible after the whole block completes. A new `>>` starts
a new call and may read outputs from a previous call.

## Procedures

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
@format_issue(
  issue_id: int,
  title: str = `"Untitled"`
) -> str:
  = Issue \#<issue_id>: <title>
```

Parameters support positional and named calls, Python-expression defaults, and
direct or backtick type annotations. Required parameters precede defaults.
Unannotated parameters and returns default to `str`.

Calls may nest:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
= `format_issue(issue.id, title=issue.title)`
```

Commas, parentheses, and delimiters that are literal Kedi text must be escaped
when they would otherwise be parsed as call syntax.

## Custom Types

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
~Finding(
  path: Annotated[str, "Repository-relative path"],
  severity: Literal["low", "high"],
  tags: list[str] = `[]`
)
```

Field names are unique identifiers. Required fields precede defaults, and a
defaulted field needs an explicit type. Mutable defaults are deep-copied per
instance.

## Python Forms

- Backticks inside an initialization, assignment, return, argument,
  substitution, annotation, model, setting, or directive are Python
  expressions.
- A standalone backtick line is a Python statement.
- Triple-backtick blocks contain Python statements. Add `=` before the opening
  fence when the block should return a value.
- The initial top-level Python fence is the prelude and supplies globals to
  following Kedi declarations.

Kedi values are available as Python globals in the runtime frame. Python writes
to recognized Kedi globals are synchronized; lexical locals remain scoped.

## Validation and Generation

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
@test: procedure_name:
  > case: unique_case:
    `assert procedure_name() == "ok"`

@eval: procedure_name:
  > data: rows:
    = `[(input_value, {"field": "expected"})]`
  > test_data: rows:
    = `[(held_out_value, {"field": "expected"})]`
  > metric: score(rows):
    = `procedure_name(rows) == expected["field"]`
```

`> optimize:` and `> auto:` occur inside procedures. An optimize body contains
one template block starting with `>>`; bare prompt beginnings are invalid.
`> auto:` instead takes an indented code-generation specification, not an
ordinary executable template block.

## Escapes

| Escape | Literal result |
| --- | --- |
| `\<` `\>` | angle brackets |
| `\[` `\]` | square brackets |
| `\(` `\)` | parentheses |
| `\,` | comma |
| `\=` `\@` `\#` | equals, at, hash |
| `\~` or `~~` | tilde |
| ``\` `` | backtick |
| `\\` | backslash |
| `\t` `\n` `\s` | tab, newline, preserved space |

A trailing backslash continues a rendered `=` return onto the next physical
line. It is unrelated to template continuation.

## Names and Comments

Identifiers match Python-style names: a letter or underscore followed by
letters, digits, or underscores. `args` is runtime-owned and cannot be assigned.
`#` begins a Kedi comment; use `\#` for a visible hash. A first `###` block in a
procedure or profile becomes its docstring.
