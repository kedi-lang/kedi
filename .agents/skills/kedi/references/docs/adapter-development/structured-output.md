# Structured Output

`produce` and `produce_sync` return values, not text. Kedi says what the value
must look like in one of two ways, the adapter asks the backend for it, and
Kedi validates what comes back.

## The Two Contracts

| Argument | Sent for | Value |
|---|---|---|
| `output_schema` | template captures (`>> ... [a] ... [b: T] ...`) | `dict[str, ResolvedKediType]`: one entry per output name, in template order |
| `output_type` | typed raw invokes (`[x: T] << ...`) and claims | a Python type: `T`, or `Annotated[bool, Field(description=...)]` for a claim |

Exactly one of them is set. `ResolvedKediType` is Kedi's resolved form of a
field type, including `Annotated` descriptions, constraints, `Literal`
choices, custom `~Type`s, and generics. Do not interpret it yourself; turn it
into a JSON Schema with Kedi's helpers.

## Building the JSON Schema

```python
from pydantic import TypeAdapter
from kedi.agent_adapter.output_schema import build_output_model

if output_schema is not None:
    schema = build_output_model(output_schema).model_json_schema()
else:
    schema = TypeAdapter(output_type).json_schema()
```

`build_output_model` builds the Pydantic model Kedi's own adapters send:

- one field per output name, with the field's type, description, and
  constraints;
- names that Pydantic cannot use as attributes (a leading underscore,
  `model_...`, or a `BaseModel` member) become internal field names with the
  Kedi name as the alias, so the JSON property is still the Kedi name;
- fixed-length tuples are expressed with `items` instead of `prefixItems`,
  which most model APIs reject;
- for template captures in completion mode, a model description that tells
  the model to return only the replacement values.

Call it inside `produce`: the description depends on the active request.

## Non-Object Roots

A claim's schema is a bare boolean, and a typed raw invoke's schema is
whatever `T` is: `{"type": "integer"}`, an array, an enum. Many structured
output APIs accept only an object at the root. For such a backend, wrap the
schema in one property and unwrap the answer:

```python title="object_root.py"
from typing import Any

from pydantic import TypeAdapter

from chat_adapter import ChatAdapter
from kedi.agent_adapter.output_schema import build_output_model


class ObjectRootAdapter(ChatAdapter):
    async def produce(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: Any = None,
        instructions: str | None = None,
        **_: Any,
    ) -> Any:
        if output_schema is not None:
            schema = build_output_model(output_schema).model_json_schema()
            reply, _messages = await self._run(
                prompt=template, instructions=instructions, output_schema=schema
            )
            return reply.data
        value_schema = TypeAdapter(output_type).json_schema()
        wrapped = {
            "type": "object",
            "properties": {"result": value_schema},
            "required": ["result"],
            "additionalProperties": False,
        }
        reply, _messages = await self._run(
            prompt=template, instructions=instructions, output_schema=wrapped
        )
        return reply.data["result"]
```

Move `$defs` to the wrapper's root if your backend does not resolve
references below the root.

## Return Values

| Call | Return |
|---|---|
| `output_schema` | a mapping, or an object with one attribute per output name |
| `output_type` | the value |

Return the decoded JSON. Then Kedi:

1. reads every output name from the result; a missing one fails the call with
   `LLM did not return required key: <name>`;
2. validates and converts each **explicitly typed** field (`[x: T]`), each
   typed raw invoke, and each claim with `TypeAdapter(T).validate_python`, so
   a JSON object becomes the program's `~Type` instance and `"3"` becomes `3`
   for an `int`;
3. keeps **untyped** fields (`[x]`) exactly as returned. Their schema type is
   `string`; return a string.

Do not convert values into Kedi types inside the adapter, and do not return
provider message or response objects.

## JSON Schema Formats

`supported_json_schema_formats` declares which JSON Schema `format` values the
backend enforces, for example `frozenset({"date", "date-time"})`. The Kedi
language server uses it for built-in adapters to flag field types that would
produce an unsupported format (`K-ADAPTER-SCHEMA-FORMAT`). Leave it `None`
when the backend accepts any format; Kedi still validates the returned value
against the full type.

## Advertising Structured Output

Set `supports_structured_output=True` only when the backend constrains the
answer to the schema, including nested objects, lists, unions, and
constraints. An adapter that only asks for JSON in the prompt does not support
structured output and must not write such a prompt either (see [Prompts and
Instructions](prompt-contract.md)). A program can demand the capability with
`> requires: structured_output`, which fails before any model call when the
adapter does not advertise it.

## Running It

```python title="run_structured_output.py"
import asyncio
from collections.abc import Callable

from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter
from object_root import ObjectRootAdapter


class Recorder:
    def __init__(self, answer: Callable[[BackendRequest], Reply]) -> None:
        self.answer = answer
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return self.answer(request)


def run(source: str, answer: Callable[[BackendRequest], Reply], adapter_type=ChatAdapter):
    backend = Recorder(answer)
    runtime = compile_program(parse_program(source), adapter=adapter_type(backend, model="m1"))
    try:
        return runtime.run_main(), backend.requests
    finally:
        asyncio.run(runtime.aclose())


CAPTURE = """
~Airport(code: str, runways: int)

>> [city] has [airports: list[Airport]] and [population: int] inhabitants.
= `(city, airports, population)`
"""

value, [request] = run(
    CAPTURE,
    lambda _r: Reply(
        data={
            "city": "Paris",
            "airports": [{"code": "CDG", "runways": 4}],
            "population": "2100000",
        }
    ),
)
city, airports, population = value
assert city == "Paris"
assert airports[0].code == "CDG" and airports[0].runways == 4
assert population == 2_100_000
schema = request.output_schema
assert list(schema["properties"]) == ["city", "airports", "population"]
assert schema["properties"]["population"]["type"] == "integer"
assert "replacement values" in schema["description"]
```

A missing field fails the call:

```python title="run_structured_output.py"
try:
    run(CAPTURE, lambda _r: Reply(data={"city": "Paris", "airports": []}))
except KediExecutionError as error:
    assert "LLM did not return required key: population" in str(error)
else:
    raise AssertionError("a missing field must fail")
```

A claim through a backend that needs an object root:

```python title="run_structured_output.py"
CLAIM = """
[answer] = `'no'`
> if: Paris is in France
  [answer] := yes
= <answer>
"""

value, [request] = run(CLAIM, lambda _r: Reply(data={"result": True}), ObjectRootAdapter)
assert value == "yes"
assert request.output_schema["properties"]["result"]["type"] == "boolean"
assert "claim is established" in request.output_schema["properties"]["result"]["description"]
```
