# Typed Child Results

A parent receives a result envelope, not the child's local environment. Decide
whether the child should return prose or a validated value before choosing the
delegation interface.

## Declare the Contract Once

```kedi
~Evidence(source: str, passed: bool, failures: list[str])

> profile: investigator:
    ###
    Inspect supplied test evidence without inventing missing results.
    ###
    > adapter: pydantic
    > instructions: Return only facts supported by the task's evidence.
    > output: Evidence

> profile: coordinator:
    > adapter: pydantic
    > subagent: investigator
    > max_agents: 2

> use: coordinator
[answer] << Ask investigator to assess: unit tests passed; integration tests were not run. Explain the remaining uncertainty.
= <answer>
```

Configure a model through the CLI or embedding application. The parent can call
`delegate_task(subagent="investigator", task=...)`. Because the child declares
`output`, it need not repeat a schema on every call. The task must include the
evidence: the child does not inherit the parent's conversation.

An illustrative successful tool result is:

```json
{
  "run_id": "<generated-run-id>",
  "subagent": "investigator",
  "task_summary": "Unit tests passed; integration coverage is unknown.",
  "final_result": {
    "source": "supplied test evidence",
    "passed": true,
    "failures": []
  }
}
```

The schema establishes the value's structure, not whether the model's account
of the evidence is correct. In particular, a `passed` boolean needs a precise
task definition. Validation cannot turn an incomplete test run into proof of a
release's safety.

## Schema Precedence

| Call configuration | Result contract |
| --- | --- |
| Explicit `final_schema` | Use this schema, even if the profile declares another output type. |
| No explicit schema; profile has `> output: Type` | Convert the profile type to JSON Schema. |
| Neither | Return child text as `task_summary`; `final_result` is `null`. |

The same precedence applies to direct delegation and dynamic child functions.
`> output:` does not change unrelated template captures or make the envelope a
Python instance of the custom Kedi type. `final_result` crosses the delegation
boundary as a validated JSON-compatible value; dynamic workflows use dictionary
access such as `evidence["final_result"]["passed"]`.

## Per-Call Contracts

The model-facing `final_schema` parameter accepts JSON Schema, not a Python
class or a Kedi type name. For example, this schema requests an integer count:

```json
{"type": "integer", "minimum": 0}
```

Use the profile default for a stable reusable contract. Use an explicit schema
when a particular delegation intentionally needs a different result. A schema
override does not grant tools or widen permissions.

### Local Schema References

`final_schema`, profile `> output:` types, and task captures support document-local
JSON Pointer references, including Pydantic's generated `$defs`:

```json
{
  "$defs": {"Count": {"type": "integer", "minimum": 0}},
  "type": "object",
  "properties": {"completed": {"$ref": "#/$defs/Count"}},
  "required": ["completed"]
}
```

Kedi expands these references before wrapping or sending the result schema.
Nested models, lists, unions, and repeated references retain their constraints;
the original schema is not modified. `$ref` sibling constraints are conjunctive
in modern JSON Schema dialects, not overrides of the referenced constraints.

No external schema is fetched. External references, `$id`, dynamic references,
unknown pointer targets, and recursive expansion are rejected. The 32 KiB schema
size limit and maximum nesting depth of 16 also apply after expansion, preventing
small reference graphs from expanding into unbounded schemas. A data property
named `$ref`, or a `$ref` string inside `default`, `const`, or examples, is not
itself a schema reference.

## Failure Outcomes

- Invalid or unsupported schemas are rejected rather than treated as prose.
- A profile output type that cannot be represented as JSON Schema fails.
- Duplicate `output` members fail at parsing/compilation.
- A returned value that fails the selected schema does not become a successful
  `final_result`. Adapter repair may occur within its configured limits, but
  exhausted validation fails the child run.

Pydantic AI and LangChain children validate the complete JSON Schema inside the
model's output correction loop, including required fields, extra-property
restrictions, and the 1,000-character `task_summary` limit. An invalid result is
returned to the model as validation feedback. Repairs consume the same child's
request/token budget and configured output retries (Pydantic AI's output retries,
LangChain's `output_retries`; one by default); they do not restart the workflow
or completed siblings. When a native `> task` capture or the child profile's
`> output:` uses a Python type, its Pydantic validators run in that same loop. A
result is recorded as completed only after both checks pass. The native
`> await` value retains its type; delegation tools and persisted child history
use JSON values. Other adapters still reject invalid results before completion,
without a correction turn.

Native conversion uses strict JSON validation. A date's JSON string, a UUID's
string, an enum's declared JSON value, or a tuple's JSON array becomes the
corresponding Python value when the declared type allows it. Nested and optional
values follow the same rules. This does not enable general coercion: `"3"` does
not satisfy `int`, and `1` does not satisfy `bool`. Custom validators run before
completion; awaiting the same successful task again reuses its validated value.
The tool result and persisted history retain JSON representations of those values.

Provider strict mode is not a substitute for local validation. For a Pydantic
child, the model-facing envelope describes the 1,000-character summary limit
without the strict-incompatible `maxLength` keyword. Kedi checks the original
length constraint locally. Other unsupported constraints may still make a
provider request non-strict; Kedi does not force strict mode.

An application can enforce relationships that JSON Schema does not express,
such as `available >= requested` whenever `sufficient` is true, with a
Pydantic `model_validator` on its native task output type. Kedi returns that
validation error to the Pydantic child within the configured retry budget.

```python
from pydantic import BaseModel, model_validator


class StockDecision(BaseModel):
    available: int
    requested: int
    sufficient: bool

    @model_validator(mode="after")
    def check_quantities(self):
        if self.sufficient != (self.available >= self.requested):
            raise ValueError("Stock decision contradicts the quantities")
        return self
```

A native task can capture `[decision: StockDecision]` after the embedding
application makes the type available to the program. The same check applies
when a child profile declares `> output: StockDecision`.

For executable validation of the parent consuming a child result, see
[Reviewed Evidence Workflow](reviewed-evidence.md).
