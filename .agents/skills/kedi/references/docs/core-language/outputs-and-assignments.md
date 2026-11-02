# Outputs, Initialization, and Assignment

Square brackets are write syntax in Kedi. In a `>>` template they declare
fields the model must produce; on the left of `=` they initialize variables
with deterministic values. Neither form is the same as `<name>`, which reads a
value.

## Output Fields as L-Values

An output field embedded in a template is an L-value:

```kedi
>> The incident severity is [severity: str]. The summary is [summary: str].
= <severity>: <summary>
```

Kedi builds a structured output schema, asks the active adapter to fill it, and
stores each field in the current environment. Output names must be valid Python
identifiers: they start with a letter or underscore and contain only letters,
digits, and underscores.

Use model outputs for semantic extraction or generation. Do not use them for
values that Python can determine exactly.

## Simple and Typed Outputs

An untyped output defaults to `str`:

```kedi
>> A concise headline is [headline].
```

A typed output asks the adapter for a native value and validates it:

```kedi
>> The priority is [priority: int]. The owners are [owners: list[str]]. It is [blocked: bool] that the work is blocked.
```

Kedi does not merely include the annotation in the prompt. It resolves the type,
creates the provider schema, and validates the returned value. Unsupported
provider schema formats fail before a model call where the adapter can detect
them.

## Inline Python Output Types

Wrap a type expression in backticks when the type comes from runtime Python
state:

````kedi
```
from typing import Literal
Priority = Literal["low", "medium", "high"]
```

>> The ticket priority is [priority: `Priority`].
````

For ordinary built-in or custom Kedi types, prefer the direct form
`[priority: Priority]`. Runtime expressions are useful for types created by a
prelude, import, or factory, but they make static tooling less certain.

## Field Description Metadata

Use `Annotated[T, "description"]` to explain a field to the model while keeping
`T` as its runtime type:

```kedi
>> The country code is [code: Annotated[str, "Uppercase ISO 3166-1 alpha-2 country code"]].
```

The description becomes JSON Schema metadata for adapters that expose schemas.
The native description form uses a single-line string literal. `Annotated[T]`
is accepted as `T` but triggers an editor warning because it carries no
description. Python type aliases can carry additional non-string validation or
decision metadata; Kedi preserves that metadata. See
[Validation Constraints](types.md#validation-constraints).

Use descriptions for constraints the base type cannot express clearly. Do not
repeat obvious information such as `Annotated[int, "An integer"]`.

## Multiple Outputs and Same-Scope Initialization

One template can fill several fields:

```kedi
>> Version [version: str] was released on [date: date].
It includes [changes: list[str]].
```

All continuation lines in that `>>` block belong to one model request and one
combined schema. A later output or `=` initialization with the same name
replaces the value in the current scope:

```kedi
[status] = draft
>> The document status after review is [status: str].
= <status>
```

This does not update an outer lexical binding. Prefer a new name such as
`reviewed_status` when both values matter.

## Variable Initialization

Initialization does not contact a model:

```kedi
[title] = Release <version>
[attempts: int] = `2 + 1`
[copy] = <title>
```

The right-hand side has two evaluation modes:

- A sole native Python segment preserves its native result.
- A sole procedure call preserves that procedure's native result.
- Mixed literal text and substitutions render to `str`.

This distinction matters:

```kedi
[native: int] = `40 + 2`
[rendered] = Answer: <`40 + 2`>
```

`native` is the integer `42`; `rendered` is the string `"Answer: 42"`.

`=` always declares in the current lexical scope. A declaration inside an
`if`, `else`, or loop iteration shadows a visible outer name and disappears
when that child scope ends.

## Assignment

Use `:=` to assign a new value to the nearest visible Kedi binding instead of
shadowing it:

```kedi
[attempts: int] = `0`

> if: `True`:
  [attempts] := `attempts + 1`

= `attempts`
```

The target must already exist. `:=` does not accept a type annotation because
the existing binding owns its type contract; Kedi validates the new value
against that contract. Unknown and reserved targets fail.

A fenced Python result can be assigned without changing the target type:

````kedi
[total: int] = `0`
[total] := ```
return sum([1, 2, 3])
```
````

## Typed Initialization

A typed initialization validates the resulting value without coercing it:

```kedi
[ports: list[int]] = `[8000, 8001]`
[enabled: bool] = `True`
```

`[count: int] = five` fails instead of converting text. If parsing user text is
required, do it explicitly in Python:

```kedi
[raw_count] = 5
[count: int] = `int(raw_count)`
```

## Initialization from a Python Block

Use a fenced block when computing a value requires statements:

````kedi
[durations: list[int]] = `[14, 8, 21]`
[p95: int] = ```
ordered = sorted(durations)
index = round(0.95 * (len(ordered) - 1))
return ordered[index]
```

= p95\=<p95>
````

The block must execute `return` to supply the initialization value. New helper names
inside the block remain local; the initialized result is the supported way to
surface one of them.

## Output Capture versus Raw Capture

Use `>>` output fields when you need typed or multiple structured values:

```kedi
>> The detected language is [language: str]. The confidence is [confidence: float].
```

Use raw capture when the complete model response should remain an unstructured
string:

```kedi
[explanation] << Explain why the build failed in one paragraph.
```

Raw capture accepts no embedded output fields and no type other than optional
`str`. A plain `>>` request with no output fields runs the model and discards
its response.

## Resolution and Validation Errors

Kedi rejects duplicate output names within an invalid schema, malformed
identifiers, unknown types, values that do not match their initialization or
assignment contracts, and adapter
schemas the active backend cannot represent. Type validation is strict: a
numeric-looking string is still a string.

Choose annotations that match the wire capability of the selected adapter. For
example, a descriptive `Annotated[str, "..."]` can be more portable than a
provider-specific URL or regex format.
