# Substitutions and Calls

Substitutions are Kedi's read syntax. `<...>` inserts an existing value, a
procedure result, or a Python expression into rendered text. A model prompt can
also pass a value by reference with `<<...>>`. Neither declares an output.

## Variable Substitutions

Use `<name>` to read a value from the current lexical environment:

```kedi
[project] = Atlas
[status] = ready

> show: Project <project> is <status>.
```

Substitution always renders the value as text at that position. It is therefore
appropriate for prompts, messages, paths, and final text.

`<...>` accepts only `str`, `int`, `float`, `bool` and lists of these. Any other
value is a type error: `None`, media, dicts, models, clients and other objects.
Rendering them through `repr` would be opaque and could leak data, such as an
API key held by a client object. A value stored as an artifact renders its
text, never an artifact reference. Convert explicitly when you want text,
``<`str(value)`>`` or ``<`report.title`>``, or pass the value to a model by
[reference](#references). If the next operation needs the original `int`,
`list`, model, or other Python object, use the value by bare name inside Python
or pass it as a native call argument.

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

The inner call completes before its value becomes the outer argument, as text:
an angle call always renders text, even as a whole argument. Pass a native
value with a backtick argument, ``<double(`count(items)`)>``, or call from
Python as below.

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
| `<name>` | Rendered `str` |
| `<procedure()>` | Rendered `str` |
| ``<`expression`>`` | Rendered `str` |
| `prefix <name>` | Rendered `str` |
| `` `expression` `` | Native Python result |

Prefer a native argument when the parameter is typed as anything other than
`str`. Prefer rendered arguments for human-language content.

## Bindings Are Text Unless Backticked

A template is text wherever it appears, including a binding or a return whose
right-hand side is a single call:

```kedi
[profile: Profile] = `get_profile("Ada")`   # the Profile value
[label] = <get_profile(Ada)>                 # error: a Profile has no text form
[n: int] = <count>                           # error: text is not an int
```

Native values come from single-backtick Python only. Strings are never
converted to other types implicitly.

## References

A model prompt passes a value by reference with `<<name>>` or
`<<procedure(...)>>`. The text gets the value's ID, and the value is attached to
the model's context under that ID, before the prompt text:

```kedi
[spec] = `read_text("spec.md")`
>> Does the change follow <<spec>>? Check <<spec>> section by section. [ok: bool]
```

```text
<attachments>
<document id="spec" type="text/plain">
...contents of spec.md...
</document>
</attachments>
Does the change follow spec? Check spec section by section. ...
```

- A binding's ID is its name. A call gets `<procedure>_result_1`,
  `<procedure>_result_2`, ... within the prompt.
- Each content is attached once; every mention writes `<<...>>` again and gets
  only the ID.
- Media go as native parts ([Media Input](media-input.md)). Text goes as
  `text/plain`; lists, dicts and models as JSON (`application/json`). `None` is
  an error.
- References are valid only in model prompts: `>>` templates, `[x] << ...` and
  `[x: T] << ...` invokes, and `> if:` / `> loop:` template conditions.
  Elsewhere `<<...>>` is a parse error (`K-REFERENCE-CONTEXT`).
- A value that already is an artifact goes as its reference
  (`<artifact id="log">`) when the call has artifact tools, and in full
  otherwise. Plain values are attached in full however large they are, and
  `<log>` always writes the text itself: size never decides.
- Attached content is not escaped. Do not treat referenced content as
  instructions.

`<items>` writes a list into the text; `<<items>>` attaches it as a JSON
document. Which one fits is your choice.

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
