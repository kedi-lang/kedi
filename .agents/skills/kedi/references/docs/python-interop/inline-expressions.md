# Inline Python Expressions

Inline Python is a single expression enclosed in one pair of backticks. It is
evaluated in the active Kedi runtime environment.

## Expression Syntax

```kedi
[radius: float] = `3.5`
[area: float] = `3.14159 * radius ** 2`
= area\=<`round(area, 2)`>
```

The content must be a Python expression. Assignments, `import`, `for`, `try`,
and statement `return` require a fenced block. Comprehensions, conditional
expressions, lambdas, and function calls are expressions and are allowed.

## Expressions in Templates

Wrap an expression in angle brackets or use a bare backtick segment:

```kedi
[services: list[str]] = `["api", "worker"]`
>> Assess <`len(services)`> services: <`", ".join(services)`>.
The deployment risk is [risk: str].
```

Both Python forms insert the expression's string representation into rendered
text. Prefer ``<`...`>`` when visually separating Kedi substitutions; a bare
segment can be clearer when the Python contains `<`, `>`, or brackets.

## Expressions in Variable Initializations

A sole expression preserves its native value:

```kedi
[threshold: float] = `0.8`
[labels: list[str]] = `["safe", "review"]`
```

Literal text around the expression changes the initialization to rendered `str`:

```kedi
[message] = threshold\=<`threshold`>
```

This rule prevents accidental conversion in typed pipelines. Use a native
initialization for data and a rendered initialization for presentation.

## Expressions in Returns

A sole expression can satisfy a native return annotation:

```kedi
@enabled(flags: dict[str, bool], name: str) -> bool:
  = `flags.get(name, False)`
```

Inside a rendered return, it contributes text:

```kedi
@summary(count: int) -> str:
  = Processed <`count`> <`"item" if count == 1 else "items"`>.
```

An unannotated procedure has a `str` return contract. Annotate it before
returning a native bool, number, collection, or custom model.

## Expressions in Call Arguments

A backtick call argument passes the native result:

```kedi
@limit(values: list[int], count: int) -> list[int]:
  = `values[:count]`

= `limit([1, 2, 3, 4], count=2)`
```

An angle call embedded in text renders its result. A sole procedure-call
segment on a return or initialization preserves its native result. Use the
following mixed-text form for presentation:

```kedi
@format_limit(values: list[int], count: int) -> str:
  = `", ".join(str(value) for value in values[:count])`

= First values: <format_limit(`[1, 2, 3, 4]`, `2`)>
```

Backticks on the arguments preserve the native list and integer passed into
`format_limit`; the surrounding angle call deliberately renders its `str`
return. Use the Python call form for keyword arguments.

## Expressions in Types and Defaults

Runtime type expressions and defaults use backticks:

````kedi
```
from typing import Literal
Mode = Literal["fast", "safe"]
```

@run(mode: `Mode` = `"safe"`) -> `Mode`:
  = `mode`
````

Defaults are evaluated at declaration time. Type expressions resolve when their
contract is built or used. Prefer ordinary direct annotations when a type is
already known to Kedi.

## Expressions in Directives

Directives accept Python expressions where their reference page explicitly
allows dynamic values:

````kedi
```
selected_model = "openai:gpt-6-luna"
```

> model: `selected_model`
> settings:
    temperature: `0.1`
    stop_sequences: `["END"]`
````

Plain settings already parse booleans, numbers, and `null`; use Python only for
lists, mappings, computed values, or objects.

## Available Names

Expressions can read visible Kedi variables, procedure parameters, outer
lexical values, imports, prelude helpers, compiled Kedi procedures, custom
types, and the runtime type namespace.

````kedi
```
from pathlib import Path

def basename(value: str) -> str:
    path = Path(value)
    return path.name
```

@show(path: str) -> str:
  = `basename(path)`
````

Names are resolved according to source and lexical scope. A declaration that
appears later is not available retroactively.

## Side Effects

Python technically allows side effects inside calls and assignment expressions,
but value expressions should remain referentially clear. Use a single-backtick
statement line or fenced block for intentional mutation:

```kedi
@increment() -> int:
  [count: int] = `0`
  `count += 1`
  = `count`
```

This makes execution order visible and keeps prompts, evals, and traces easier
to interpret.

## Error Mapping

Syntax errors, missing names, assertion failures, and exceptions are reported
with their originating Kedi source span. Kedi does not replace failed
expressions with empty text. Catch an exception only when the program has a
meaningful recovery policy; otherwise let the runtime preserve the failure.
