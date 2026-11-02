# Substitutions and Calls

Substitutions are Kedi's read syntax. They insert an existing value, a
procedure result, or a Python expression into rendered text. They never declare
an output.

## Variable Substitutions

Use `<name>` to read a value from the current lexical environment:

```kedi
[project] = Atlas
[status] = ready

> show: Project <project> is <status>.
```

Substitution always renders the value as text at that position. It is therefore
appropriate for prompts, messages, paths, and final text. If the next operation
needs the original `int`, `list`, model, or other Python object, use the value by
bare name inside Python or pass it as a native call argument.

```kedi
[retries: int] = `3`

# Text rendering
> show: Retries: <retries>

# Native arithmetic
> show: `retries + 1`
```

An unknown name is an error. Kedi does not silently render a missing
substitution as an empty string.

## Procedure Calls

Call a Kedi procedure inside angle brackets:

```kedi
@display_name(first: str, last: str) -> str:
  = <first> <last>

> show: Owner: <display_name(Ada, Lovelace)>
```

Angle-call arguments are positional. Kedi does not provide a
`name=value` form in this syntax. When keyword calling is important, call the
compiled Kedi procedure from Python:

```kedi
@label(name: str, prefix: str = `"Issue"`) -> str:
  = <prefix>: <name>

> show: `label(name="Parser", prefix="Bug")`
```

Use the ordinary angle-call form for readable prompt composition. Use a Python
call when you specifically need keyword arguments or already have native Python
objects.

## Nested Calls

Calls can be nested directly:

```kedi
@trim(value: str) -> str:
  = `value.strip()`

@slug(value: str) -> str:
  = `value.lower().replace(" ", "-")`

= <slug(<trim(  Release Notes  )>)>
```

The inner call completes before its value becomes the outer argument. A nested
call written as a whole argument preserves the procedure's native return value;
text surrounding that call turns the argument into a rendered string.

```kedi
@count(items: list[str]) -> int:
  = `len(items)`

@double(value: int) -> int:
  = `value * 2`

= `double(count(["a", "b", "c"]))`
```

For native collections, the direct Python form above is usually clearer than
embedding several backtick expressions in angle-call syntax.

## Native and Rendered Arguments

A single-backtick argument passes the evaluated Python value without converting
it to text. Every ordinary literal or mini-template argument is rendered as a
string:

```kedi
@describe(count: int, label: str) -> str:
  = <label>: <`str(count)`>

= <describe(`2 + 3`, Open items)>
```

Here `2 + 3` arrives as the integer `5`; `Open items` arrives as a string.
Without backticks, `<describe(5, Open items)>` passes `"5"` and fails the `int`
parameter check. Kedi validates rather than coercing `"5"` to `5`.

These argument forms are distinct:

| Form | Value passed |
| --- | --- |
| `plain text` | Rendered `str` |
| `<name>` | The referenced native value when it is the complete argument |
| `<procedure()>` | The native procedure result when it is the complete argument |
| `` `expression` `` | Native Python result |
| `prefix <name>` | Rendered `str` |
| ``<`expression`>`` | Python result used as the complete nested segment |

Prefer a native argument when the parameter is typed as anything other than
`str`. Prefer rendered arguments for human-language content.

## Inline Python Substitutions

Evaluate one Python expression with ``<`expression`>``:

```kedi
[items: list[str]] = `["parser", "runtime", "lsp"]`
= Components: <`", ".join(items)`>; total: <`len(items)`>.
```

The expression can read the prelude, imports, custom types, procedures, global
values, and the current procedure environment. It must be a single expression,
not a statement suite. Use a fenced Python block for imports, loops, exception
handling, or several statements.

## Bare Inline Python Segments

Inside rendered text, a bare backtick segment also evaluates and inserts Python:

```kedi
[limit: int] = `4`
= Processing `limit * 2` records.
```

This is equivalent to using ``<`limit * 2`>`` for insertion. Bare segments can
be easier to read when an expression contains angle brackets or square brackets.
Use angle-wrapped Python when the model boundary should be visually explicit;
use bare segments sparingly in prose so code and literal text remain easy to
distinguish.

## Commas and Argument Boundaries

Commas separate call arguments. Escape a comma that belongs to rendered text:

```kedi
@echo(value: str) -> str:
  = <value>

= <echo(alpha\, beta\, gamma)>
```

The procedure receives one string, `"alpha, beta, gamma"`. A comma inside a
backtick Python expression follows Python syntax and does not need Kedi's text
escape.

## Evaluation Order

Kedi evaluates substitutions from left to right while rendering an expression.
Nested calls are evaluated before the containing call. If one procedure call is
referenced both while constructing a structured prompt and while rendering that
same prompt's result, Kedi memoizes it for that expression so it is not invoked
twice.

Do not rely on model calls for deterministic side-effect ordering. Keep
side-effecting work in explicit statements or Python blocks, and treat
substitutions as value-producing expressions.

## Missing and Invalid Substitutions

Kedi reports an error when:

- a variable or procedure is not visible in the lexical scope;
- a procedure receives too few or too many positional arguments;
- an argument fails its declared parameter type;
- an inline Python expression raises;
- delimiters are unbalanced or an unsupported escape is used.

The error retains the Kedi source location. Do not add fallback values merely to
hide missing names; declare a procedure default or compute an explicit fallback
instead.
