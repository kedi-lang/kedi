> Offline source: `kedi-docs-main/docs/agentic-engineering/subagent-results.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

# Typed Child Results

A parent receives a result envelope, not the child's local environment. Decide
whether the child should return prose or a validated value before choosing the
delegation interface.

## Declare the Contract Once

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
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

## Failure Outcomes

- Invalid or unsupported schemas are rejected rather than treated as prose.
- A profile output type that cannot be represented as JSON Schema fails.
- Duplicate `output` members fail at parsing/compilation.
- A returned value that fails the selected schema does not become a successful
  `final_result`. Adapter repair may occur within its configured limits, but
  exhausted validation fails the child run.

For executable validation of the parent consuming a child result, see
[Reviewed Evidence Workflow](doc-agentic-engineering-reviewed-evidence.md).
