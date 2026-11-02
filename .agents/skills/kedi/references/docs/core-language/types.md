# Types

Kedi types are runtime contracts, provider schemas, and editor information.
They are used in output fields, variable initializations, procedure signatures,
custom fields, tool schemas, evals, and the Python API. Assignments reuse the
target binding's established type contract.

## Built-In Types

The type environment includes model-facing Python and typing primitives such as:

- `str`, `int`, `float`, `bool`, and `bytes`;
- `list[T]`, `dict[K, V]`, and `set[T]`;
- `Union`, `Optional`, `Literal`, and `Annotated`;
- `datetime`, `date`, `time`, and `timedelta`;
- `Regex`, `Email`, `HttpUrl`, and `FileUrl`.

```kedi
~Window(start: datetime, duration: timedelta)

@window(start: datetime, duration: timedelta) -> Window:
  = `Window(start=start, duration=duration)`
```

Native Kedi annotations reject `tuple`, `Tuple`, `Sequence`, `Mapping`,
`Iterable`, `object`, `bytearray`, `slice`, and `range`. They remain available
to ordinary Python expressions and blocks; only their use as Kedi type
contracts is rejected. Use `list`, `dict`, or a named custom type for
model-facing schemas.

Unannotated outputs, variable initializations, parameters, returns, and custom
fields default to `str`. Add an annotation whenever a value is intentionally
not text.

## Container, Union, and Literal Types

Types can be nested:

```kedi
>> The model scores are [scores: dict[str, list[float]]].
>> The current state is [state: Literal["open", "closed", "blocked"]].
>> The responsible owner is [owner: str | None].
```

Choose a shape that the model and adapter can reliably represent. Deeply nested
unions may be valid Python but poor model interfaces; a named custom type often
produces a clearer schema.

## Inline Python Type Expressions

Backtick-wrapped annotations are evaluated at runtime:

````kedi
```
from typing import Literal
Severity = Literal["low", "medium", "high"]
```

@triage(level: `Severity`) -> `Severity`:
  = `level`
````

They have the same validation behavior as direct annotations. Prefer direct
syntax for built-ins and Kedi custom types. Use runtime expressions only when
the type is computed or imported through Python.

## Define Custom Types

Declare a Pydantic-compatible model with `~Name(fields)`:

```kedi
~Owner(
  name: Annotated[str, "Human-readable display name"],
  email: Email
)

~Issue(
  id: int,
  title: str,
  owner: Owner | None,
  labels: list[str] = `[]`
)
```

Fields without annotations are `str`. Field names must be unique valid
identifiers. Required fields must come before fields with defaults, and a
defaulted custom field must have an explicit annotation.

Custom types are lexically scoped. A type declared inside a procedure is visible
to that procedure and nested scopes, but not after the call. A nested type may
shadow an outer type with the same name.

## Defaults

Field defaults are single-line Python expressions evaluated at definition time:

```kedi
~Job(name: str, retries: int = `3`, tags: list[str] = `[]`)
```

Kedi deep-copies mutable defaults for each model instance, so instances do not
share the same list or dict. Required-after-default and untyped-default fields
are parse errors.

## Field Description Metadata

`Annotated[T, "description"]` keeps `T` as the runtime type and adds the string
to the generated schema:

```kedi
~Finding(
  path: Annotated[str, "Repository-relative POSIX path"],
  confidence: Annotated[float, "Value from 0.0 through 1.0"]
)
```

Descriptions should state semantic constraints or interpretation. They are
forwarded to model-facing schemas. `Annotated[T]` works as `T` but lacks useful
metadata and is warned about. The first string metadata item supplies the field
description when it is the first metadata argument. Non-string metadata from
Python type aliases is preserved for validation and integrations; it is not
discarded. Descriptions guide the model but do not enforce a numeric range.

## Validation Constraints

Import `kedi.Constraints` when bounds must be enforced. In a native `Annotated`,
wrap the metadata constructor call in backticks:

````kedi
```
from kedi import Constraints
```

~Review(
  confidence: Annotated[
    float,
    "Confidence of the review",
    `Constraints(ge=0, le=1)`
  ]
)
[review: Review] = `Review(confidence=0.9)`
> show: `review.confidence`
````

This also works inside nested lists, dictionaries, optional and union types.
The metadata expression produces a value, while the first `Annotated` argument
must still be a type. `Constraints` is not an implicit global.

Alternatively, define the whole annotation as a Python alias:

````kedi
```
from typing import Annotated
from kedi import Constraints

Rating = Annotated[
    float,
    "Rating on a zero-to-two scale",
    Constraints(ge=0, le=2),
]
```

[rating: `Rating`] = `1.5`
= `rating`
````

The Python alias supplies actual validation metadata. Kedi's native annotation
grammar does not accept arbitrary constructor calls such as
`Annotated[float, Constraints(ge=0)]`; use a backtick metadata expression or
reference a Python alias through a backtick type expression. These constraints
validate values, not the truth of model judgements. See [Jev](../agent-adapters/typesafe.md) for decision
metadata and supported schemas.

## Pydantic-Compatible Models

Generated custom types subclass Pydantic `BaseModel`. Construct them with
positional fields in declaration order, keyword arguments, or a mixture:

```kedi
~Person(age: int, name: str, city: str)

[person: Person] = `Person(30, name="Ada", city="London")`
= `person.model_dump_json()`
```

Pydantic APIs such as `model_dump()`, `model_dump_json()`, and
`model_json_schema()` are available. Prefer keyword construction in public code
because it remains readable if field order changes.

## Custom Types in Model Outputs

Use a custom type when one output has a meaningful structured shape:

```kedi
~Decision(
  approved: bool,
  reason: str,
  conditions: list[str] = `[]`
)

>> The request review decision is [decision: Decision].
= `decision.model_dump_json()`
```

The adapter receives the nested JSON schema and Kedi validates the model
response. Use multiple primitive outputs when the fields are local and simple;
use a named type when the structure is reused, nested, returned, or exposed to a
tool.

## Custom Types in Procedures

Custom types can cross procedure boundaries natively:

```kedi
~Ticket(id: int, title: str)

@format_ticket(ticket: Ticket) -> str:
  = \#<`ticket.id`> <`ticket.title`>

= `format_ticket(Ticket(id=7, title="Parser error"))`
```

Passing rendered JSON text is not equivalent to passing a `Ticket`. Construct
or validate the model in Python when converting external data.

## Adapter Schema Compatibility

Provider support is narrower than Python's type system. Kedi validates known
schema limitations before contacting the model:

- Codex supports formats including `date`, `date-time`, `duration`, `email`,
  and `time`.
- Codex rejects `Regex`, `HttpUrl`, and `FileUrl` model-output schemas.
- Claude accepts the listed built-in formats.
- Framework adapters may impose their own provider and model restrictions.

When strict wire validation is not essential, use
`Annotated[str, "Exact HTTPS URL ..."]` instead of an unsupported format. This
keeps the schema portable while retaining model guidance.

## Resolution and Validation Errors

Unknown type names fail loudly; they do not fall back to `str`. Values are
validated without implicit string-to-number coercion. Errors also identify
duplicate fields, invalid default ordering, incompatible initialization,
assignment, or return values, and unsupported adapter schema formats.

Types declared later are not available to earlier prelude or runtime
expressions. Define or import a type before the statement that resolves it.
