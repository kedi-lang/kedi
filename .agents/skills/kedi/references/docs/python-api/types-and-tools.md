# Types and Tools

## Register Types with `@kedi.type`

`@kedi.type` makes Python classes usable in Kedi output annotations and Python
expressions:

```python
import kedi


@kedi.type
class Finding:
    severity: str
    message: str
```

Bare classes are converted to standard dataclasses. The decorated name is
rebound to the resulting class.

## Pydantic Models

Existing Pydantic models are retained:

```python
from pydantic import BaseModel, Field


@kedi.type
class Review(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    summary: str


@kedi.query
def review(text: str) -> Review:
    """kedi
    >> A review of <text> is [result: Review].
    = `result`
    """
    ...
```

The selected adapter receives the Pydantic schema. Validation is performed by
the adapter/model integration when producing the typed output.

## Validation Constraints

`kedi.Constraints` exposes validation-only metadata, usable in Python models and
Kedi computed type expressions. It delegates to Pydantic without exposing defaults,
aliases, or serialization configuration:

```python
from typing import Annotated
from kedi import Constraints
from pydantic import BaseModel

class Review(BaseModel):
    score: Annotated[
        float,
        Constraints(ge=0, le=2),
    ]
    title: Annotated[
        str,
        Constraints(min_length=1, max_length=100),
    ]
```

Supported keywords are `ge`, `gt`, `le`, `lt`, `multiple_of`, `min_length`,
`max_length`, `pattern`, `strict`, `allow_inf_nan`, `max_digits`, and `decimal_places`.
Omitted options keep the underlying type's defaults. Constraints must match the
annotated type; the same validation semantics as Pydantic `Field` apply.

In Kedi, import this helper in a Python prelude, define an `Annotated` alias there,
and reference it with a backtick type expression. It is not a new native DSL call
syntax and does not require the optional TypeSafe integration.

## Pydantic Dataclasses

Pydantic dataclasses are recognized without reconversion:

```python
from pydantic.dataclasses import dataclass


@kedi.type
@dataclass
class Entity:
    name: str
    confidence: float
```

Place `@kedi.type` above `@dataclass` so it receives the finished Pydantic
dataclass.

## Standard Dataclasses

Standard dataclasses are also retained:

```python
from dataclasses import dataclass


@kedi.type
@dataclass
class Coordinate:
    latitude: float
    longitude: float
```

Whether an adapter can produce a particular dataclass schema depends on that
adapter's structured-output support.

## Bare Class Conversion

For a simple record, no explicit dataclass decorator is required:

```python
@kedi.type
class Label:
    name: str
    confidence: float = 1.0
```

Kedi applies `dataclasses.dataclass`. Methods and supported dataclass defaults
remain available. Classes that require custom metaclass behavior should be
defined explicitly rather than relying on conversion.

## Automatic Type Injection

`inject=True` is the default. The type is injected only for query/bind
callables defined in the same Python module:

```python
@kedi.type
class Result:
    value: str
```

Disable implicit injection:

```python
@kedi.type(inject=False)
class InternalResult:
    value: str


@kedi.query(env={"InternalResult": InternalResult})
def extract(text: str) -> InternalResult:
    """kedi
    >> The structured representation of <text> is [result: InternalResult].
    = `result`
    """
    ...
```

Configured and local `env` values override auto-injected type names. This can
be useful for dynamic schemas but should be deliberate.

## Register Tools with `@kedi.tool`

The tool decorator preserves a callable while attaching adapter metadata:

```python
@kedi.tool
def lookup_order(order_id: str) -> dict[str, object]:
    """Return public status information for one order."""
    return {"id": order_id, "status": "queued"}
```

Register it through `configure`, `context`, `query`, or `bind`, then opt in
inside Kedi:

```python
@kedi.query(tools=[lookup_order])
def answer(question: str) -> str:
    """kedi
    > use: lookup_order
    >> Use the order lookup when needed. The answer to <question> is [answer: str].
    = `answer`
    """
    ...
```

Decoration alone does not make the tool globally available.

## Names, Descriptions, and Retries

Override metadata and retry transient callable errors:

```python
@kedi.tool(
    name="search_docs",
    description="Search approved project documentation.",
    retries=2,
    retry_on=(TimeoutError, ConnectionError),
    risk="read_only",
)
def search_index(query: str) -> list[str]:
    return index.search(query)
```

The registered name defaults to `__name__`; description defaults to the
docstring. `retries=2` means at most three total attempts. `retry_on` accepts a
sequence of `Exception` classes and matches subclasses. Omit it to make all
ordinary `Exception` failures eligible, or pass an empty tuple to match none.
The filter alone does not enable retries.

Retries cover only failures raised by the callable body, for both sync and
async tools. Argument validation, hooks, approval, result handoff, cancellation,
`KeyboardInterrupt`, and `SystemExit` are not retried. A tool managed by an
adapter still has one retry owner; the decorator and adapter projection do not
multiply attempts. Negative retry counts and non-`Exception` filters are
invalid.

## Tool Risk Metadata

Every tool is classified as:

- `read_only`;
- `mutating`;
- `sensitive`.

Custom tools default to `mutating`. Mark a tool `read_only` only if it cannot
change external or local state and cannot expose sensitive data:

```python
@kedi.tool(risk="sensitive")
def read_secret(name: str) -> str:
    ...
```

Risk participates in approval. Read-only calls are automatically allowed;
mutating and sensitive calls require an allow policy or dynamic decision.

## Tool Argument Shadowing

The runtime environment cannot safely contain a function argument and a tool
under the same registered name:

```python
@kedi.tool(name="search")
def search_docs(query: str) -> list[str]:
    return []


@kedi.query(tools=[search_docs])
def answer(search: str) -> str:
    """kedi
    = <search>
    """
    ...
```

Calling `answer(...)` raises `KediExecutionError`. Rename either the parameter
or the tool. Kedi rejects the collision rather than allowing input to shadow a
callable capability.
