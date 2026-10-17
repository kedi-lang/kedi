> Offline source: `kedi-docs-main/docs/examples/structured-extraction.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Complete Program](#complete-program)
- [Native Return Versus Rendering](#native-return-versus-rendering)
- [Multiple Captures](#multiple-captures)
- [Dynamic Python Type](#dynamic-python-type)
- [Validate Deterministic Constraints in Python](#validate-deterministic-constraints-in-python)

# Structured Extraction

This example extracts one reusable native object rather than parsing JSON text
after a model call.

## Complete Program

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> adapter: pydantic
> model: openai:gpt-5.6-luna
> instructions: Extract only facts stated in the incident report.

~Owner(
  name: Annotated[str, "Person or team responsible for follow-up"],
  email: Email | None
)

~Incident(
  title: str,
  severity: Literal["low", "medium", "high", "critical"],
  owner: Owner | None,
  affected_services: list[str],
  customer_visible: bool
)

@extract_incident(report: str) -> Incident:
  >> Incident report:
  <report>
  The normalized incident is [incident: Incident].
  = `incident`

[report: str] = Payment retries failed in checkout. The Payments team owns the follow-up.
[incident: Incident] = `extract_incident(report)`

= `incident.model_dump_json(indent=2)`
```

The adjacent lines after `>>` are newline-joined into one model request. Do not
insert a blank line inside that block: it can terminate template continuation.
`[incident: Incident]` creates a structured output schema and validates the
response; it is not an instruction to return an arbitrary JSON string.

Run with the selected provider's credentials configured. The exact title and
severity may vary. The report names the Payments team but supplies no email:
the expected result preserves that team, uses `null` for its email, and does not
invent contact details. Runtime schema validation alone cannot enforce factual
support; evaluate that separately with source evidence.

## Native Return Versus Rendering

Inside `extract_incident`, this return preserves the Pydantic model:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
= `incident`
```

This alternative returns text:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
= <incident>
```

Use the native form when another procedure, Python caller, metric, or tool needs
to inspect fields. Render only at a presentation boundary.

## Multiple Captures

For a local result that is not reused as one domain object, separate fields are
often simpler:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
@classify(message: str) -> tuple[str, bool]:
  >> <message> is a [category: Literal["question", "request", "incident"]].
  It is [urgent: bool] that this message requires urgent attention.
  = `(category, urgent)`
```

All fields in one block are filled by one model call and become visible only
after that call completes. A continuation line in the same block cannot
substitute a field captured earlier in that block.

## Dynamic Python Type

Use a backtick annotation only when the type genuinely comes from Python:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
````kedi
```
from typing import Literal

CurrentRegion = Literal["eu", "us", "apac"]
```

@extract_region(text: str) -> `CurrentRegion`:
  >> The region mentioned in <text> is [region: `CurrentRegion`].
  = `region`
````

For Kedi custom types and built-ins, direct annotations are clearer. Provider
schema support can be narrower than Python's type system; consult the selected
[adapter](doc-agent-adapters-index.md) before using URL, regex, or complex union
types.

## Validate Deterministic Constraints in Python

When deliberate normalization is desired, use deterministic Python:

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
@bounded_confidence(raw: float) -> float:
  = `max(0.0, min(1.0, raw))`
```

Model schemas validate shape and type. Cross-field business invariants still
belong in deterministic code, tests, or a Pydantic validator supplied through
Python.

Clamping changes an out-of-range value; it does not reject one. If rejection is
the contract, use a constrained type instead. See the runnable
[Constraints example](doc-core-language-types.md) for preserved `Annotated`
metadata and validation failures. Do not describe a model's self-reported
confidence as a measured probability.
