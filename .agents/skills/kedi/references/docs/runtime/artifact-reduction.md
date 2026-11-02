# Reducing Artifact Data

This model-assisted example combines two tool results. It requires a tool-capable model; a requested strategy does not guarantee the model follows it.

## End-to-End Multi-Artifact Reduction

The following program exposes two large datasets as ordinary Kedi tools. The
agent receives compact references, joins the hidden payloads with Artifact Compute, and
returns only the aggregate needed by the template:

```kedi
> approval: allow
> artifacts:
    threshold: 1kb
    preview_chars: 120

@load_services() -> list:
    ###
    Return service ownership records.
    ###
    = ```
    return [
        {
            "service_id": f"svc-{index:04d}",
            "owner": f"team-{index % 20:02d}",
        }
        for index in range(5000)
    ]
    ```

@load_incidents() -> list:
    ###
    Return unresolved P1 incident records.
    ###
    = ```
    return [
        {
            "service_id": f"svc-{index % 5000:04d}",
            "severity": "P1",
            "unresolved": True,
        }
        for index in range(12000)
    ]
    ```

> use:
    load_services
    load_incidents

>> Call each data tool once. Join their artifact results by service_id with
run_artifact_code. The owner with the most unresolved P1 incidents is [owner],
with [count: int] incidents. Release both source artifacts after the aggregate
is known.

= <owner> owns <count> unresolved P1 incidents.
```

Neither large list is copied into model history. The source tool calls produce
compact `tool_call_result_N` references, `run_artifact_code` returns a small
owner/count reduction, and `release_artifact` frees both payloads after the
evidence has been consumed. If the reduction itself crosses `threshold`, it is
stored as an `artifact_code_result_N` and remains readable after its source
artifacts are released.
