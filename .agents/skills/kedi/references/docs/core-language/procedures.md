# Procedures

Procedures package Kedi statements into typed, reusable callables. They are also
the unit exposed to tools, tests, evals, optimization, generated code, and the
Python API.

## Define and Call a Procedure

Declare a procedure with `@name(parameters):` and indent its body:

```kedi
@greet(name: str) -> str:
  = Hello, <name>.

= <greet(Ada)>
```

The name must be a valid Python identifier. The body can contain variable
initializations, assignments, templates, Python blocks, directives, nested
types, nested procedures, and a return. Call it with `<greet(Ada)>` in rendered
Kedi text or `greet("Ada")` in a Python expression.

## Procedure Bodies and Returns

Statements execute in source order. A return statement starts with `=`:

```kedi
@summarize(text: str) -> str:
  >> A one-sentence summary of <text> is [summary: str].
  = <summary>
```

The first evaluated `=` ends the current procedure immediately. Statements
after an unconditional return are rejected as unreachable; returns in
different `if`/`else` branches are valid because only one branch is evaluated.
If an untyped procedure reaches the end without a value, it returns an empty
string. For non-string behavior, add an explicit return annotation and return
a native value.

Keep deterministic transformation in Python and model judgement in `>>`
templates. A procedure may combine both, but its name and return type should
still describe one coherent operation.

## Local Variables

Parameters and variable initializations belong to the procedure invocation:

```kedi
[prefix] = global

@format(value: str) -> str:
  [prefix] = local
  [result] = <prefix>: <value>
  = <result>

= <format(item)> / <prefix>
```

The procedure returns `"local: item"` while the top-level `prefix` remains
`"global"`. New local names do not leak to callers.

Parameters are read-only bindings for lexical assignment purposes. Reuse a new
local name when deriving from a parameter rather than treating the parameter as
shared mutable state.

## Calling Other Procedures

Procedures can form pipelines:

```kedi
@normalize(value: str) -> str:
  = `value.strip().lower()`

@classify(value: str) -> str:
  >> The category of <value> is [category: str].
  = <category>

@process(value: str) -> str:
  [normalized: str] = <normalize(<value>)>
  = <classify(<normalized>)>
```

A sole call on an initialization or assignment right-hand side preserves the
called procedure's native return type. Text around the call renders the value
as `str`.

## Nested Procedures

A procedure can declare helpers visible only during that invocation:

```kedi
@invoice_total(lines: list[float]) -> float:
  @subtotal(values: list[float]) -> float:
    = `sum(values)`

  = `round(subtotal(lines), 2)`
```

The nested `subtotal` is not visible after `invoice_total` returns. A nested
definition shadows an outer procedure with the same name only inside its
lexical scope.

Use nested procedures for helpers that are meaningful only to one operation or
that need to capture invocation state. Use a top-level procedure when several
features should call it or when it should be exported, tested, optimized, or
exposed as a tool independently.

## Lexical Closures

Nested procedures capture the surrounding invocation and observe the latest
outer value:

```kedi
@make_report(title: str) -> str:
  [state] = draft
  @render() -> str:
    = <title>: <state>
  [state] = approved
  = <render()>
```

Calling `make_report("Release")` returns `"Release: approved"`, not
`"Release: draft"`. Closure lookup walks
the current local scope, outer procedure scopes, imports, and top-level globals.
A nested parameter shadows a captured name.

An explicit `:=` assignment made through nested lexical scopes can update a
captured Kedi binding. This is different from Python block rules, where
rebinding from a nested Python function requires `global`.

## Recursion

Compiled procedures are ordinary callable values and can be called from Python,
including recursively:

```kedi
@factorial(n: int) -> int:
  = `1 if n <= 1 else n * factorial(n - 1)`

= `factorial(5)`
```

Use recursion only when it improves the domain model. Iteration in a Python
block is usually clearer for large collections and avoids Python's recursion
limit. Model-calling recursion can also multiply cost and latency; bound it
explicitly.

## Procedure Documentation

Place a `###` block first in the body to define the procedure docstring:

```kedi
@lookup_order(order_id: str) -> str:
  ###
  Return the canonical status for one order.

  Args:
    order_id: External order identifier.
  ###
  = pending
```

Kedi surfaces this text in editor hover, generated Python stubs, procedure JSON
schemas, and tool descriptions. A block comment after another statement remains
a comment and does not become the procedure docstring.

### Structured Docstrings and JSON Schema

Kedi follows the summary-and-body convention from PEP 257 and recommends
Google-style Python docstrings for structured `Args:` and `Returns:` sections.
PEP 257 defines general docstring conventions but does not prescribe those
section names; `Args:` and `Returns:` come from the Google-style convention.

When a procedure is exposed as a tool, Kedi parses the leading docstring and
adds its descriptions to the generated JSON Schemas:

```kedi
@quote_shipping(destination: str, weight_kg: float, express: bool = `False`) -> float:
  ###
  Calculate a shipping quote for one destination.

  Args:
      destination: Country or region receiving the shipment.
      weight_kg: Package weight in kilograms; must be greater than zero.
      express: Whether to use the faster express service.

  Returns:
      Estimated shipping price in US dollars.
  ###
  = `0.0`
```

The procedure signature and docstring produce this input schema:

```json
{
  "additionalProperties": false,
  "properties": {
    "destination": {
      "title": "Destination",
      "type": "string",
      "description": "Country or region receiving the shipment."
    },
    "weight_kg": {
      "title": "Weight Kg",
      "type": "number",
      "description": "Package weight in kilograms; must be greater than zero."
    },
    "express": {
      "default": false,
      "title": "Express",
      "type": "boolean",
      "description": "Whether to use the faster express service."
    }
  },
  "required": ["destination", "weight_kg"],
  "title": "quote_shipping",
  "type": "object",
  "description": "Calculate a shipping quote for one destination. Returns `float` instance."
}
```

Kedi emits the return contract separately:

```json
{
  "type": "number",
  "description": "Estimated shipping price in US dollars."
}
```

Types, required fields, and defaults come from the procedure signature.
Descriptions under `Args:` are matched to parameters by name, while the
`Returns:` text describes the return schema. Docstring text documents the
contract; it does not override its types. Kedi also accepts common
reStructuredText forms such as `:param name:` and `:return:`, but Google style
is the recommended form for new code.

Document externally visible procedures and every procedure exposed with
`> use:`. Include semantic constraints that types cannot express; do not
duplicate the signature verbatim.

## Failure Behavior

Calling a procedure fails when arguments are missing or excessive, a native
value violates a parameter annotation, the body raises, or the return violates
its annotation. Kedi retains source spans across nested calls so the reported
failure points back to the relevant `.kedi` statement.
