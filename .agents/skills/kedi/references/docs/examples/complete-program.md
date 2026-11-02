# Complete Program

This compact application combines typed extraction, deterministic Python,
procedure tools, profiles, command-line input, tests, and evals without hiding
the boundaries between them.

## `incident_report.kedi`

````kedi
```
INCIDENTS = {
    42: "Checkout requests time out. Payments owns mitigation.",
    77: "Search results are stale. Search owns reindexing.",
}
```

~Incident(
  id: int,
  title: str,
  owner: str,
  severity: Literal["low", "medium", "high", "critical"]
)

@lookup_incident(incident_id: int) -> str:
  ###
  Return the source report for one incident identifier.
  ###
  = `INCIDENTS[incident_id]`

@format_incident(incident: Incident) -> str:
  = `f"#{incident.id} [{incident.severity}] {incident.title} - {incident.owner}"`

> profile: analyst:
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > instructions: Look up the requested identifier. Infer severity from the report, not invented facts.
    > use: lookup_incident
    > approval: allow

@extract_incident(incident_id: int) -> Incident:
  > use: analyst
  >> According to lookup_incident, incident <incident_id> is [incident: Incident].
  = `incident`

@test: format_incident:
  > case: formats_all_fields:
    ```
    incident = Incident(id=1, title="Latency", owner="Platform", severity="high")
    assert format_incident(incident) == "#1 [high] Latency - Platform"
    ```

[incident_id: int] = `int(args.incident_id)`
[incident: Incident] = `extract_incident(incident_id)`

= `format_incident(incident)`

@eval: extract_incident:
  > data: reports:
    = ```
    return [
      (42, {"owner": "Payments"}),
      (77, {"owner": "Search"}),
    ]
    ```
  > metric: owner_accuracy(reports):
    = `extract_incident(reports).owner == expected["owner"]`
````

Run incident 42:

```bash
kedi incident_report.kedi --incident-id 42
```

Unknown CLI options become fields on the reserved `args` binding; hyphens are
normalized to underscores. The first repeated option wins.

## Why Each Boundary Is Explicit

- `lookup_incident` is deterministic and should not be a model call.
- `extract_incident` uses a typed capture because understanding prose requires
  a model.
- ``= `incident` `` returns the native custom type.
- `format_incident` renders only at the final presentation boundary.
- The profile captures a stable adapter, model, instructions, tool, and policy.
- The test checks deterministic formatting without spending model tokens.
- The eval scores semantic extraction with explicit expected data.

The top-level initialization passes native values through Python expressions.
Writing `<incident>` inside a prompt would intentionally serialize it for the
model instead.

The analyst is used by `extract_incident`, on both the CLI and eval paths.
Its `allow` policy is limited to this trusted, in-memory procedure tool;
procedure tools otherwise default to mutating risk. It is not a recommended
policy for arbitrary filesystem tools. Missing identifiers raise a lookup error,
and severity remains model judgment rather than a verified incident fact.

Configure the selected provider's credentials before normal execution or eval.
The documentation tests replace only the model with a local `FunctionModel`,
then assert that lookup really runs before a typed result reaches the formatter.
That verifies orchestration, not extraction accuracy with Gemini.

## Verification Commands

```bash
kedi incident_report.kedi --parse
kedi incident_report.kedi --test
kedi incident_report.kedi --eval
```

`--test` and `--eval` are validation modes; they do not continue into ordinary
top-level execution. Keep deterministic tests separate from model-backed evals
so failures identify the correct layer.
