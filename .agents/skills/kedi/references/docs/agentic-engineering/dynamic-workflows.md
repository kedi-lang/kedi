# Dynamic Workflows

`run_workflow` accepts only `code`. A child's optional `final_schema` belongs
inside the generated code, on that child call. Missing code or extra outer
arguments produce a correctable tool error before any child starts; they do
not grant a new delegation attempt or bypass the existing run budgets.

Delegate mode lets the parent choose one child call at a time. Dynamic mode
lets it write a bounded orchestration program over its declared children. This
is useful for dependency graphs, concurrent independent work, and typed result
composition; it does not grant access to arbitrary profiles.

## Enable the Mode

```kedi
> profile: evidence:
    > adapter: pydantic
    > instructions: Assess the supplied evidence and return an integer count of failures.
    > output: int

> profile: explanation:
    > adapter: pydantic
    > instructions: Explain the supplied findings concisely.

> profile: coordinator:
    > adapter: pydantic
    > subagent: evidence
    > subagent: explanation
    > max_agents: 4
    > workflow: dynamic
```

Apply `coordinator` before a model invocation. It receives `run_workflow(code)`
instead of the delegation/lifecycle tool set. At least one direct child is
required. Each exposed child is an async, keyword-only function accepting
`task` and optional `final_schema`.

A direct child may also come from a capability applied with `> use:`. With a
deferred capability, the model loads that capability before the child becomes
available to `run_workflow`; nested capabilities retain their individual load
boundaries. A profile whose resolved contributions contain no child is rejected
before inference.

## Dependent Work

This is **model-generated Monty code**, not a host Python script:

```python
evidence_result = await evidence(task="Tests: parser passed; compiler failed.")
count = evidence_result["final_result"]
explanation_result = await explanation(
    task=f"Explain what {count} failing test suite means for release readiness."
)
{"failures": count, "explanation": explanation_result["task_summary"]}
```

Each child returns the same `run_id`, `subagent`, `task_summary`, and
`final_result` envelope as delegate mode. A declared profile output type supplies
the default schema. The workflow's last expression becomes its result; printed
text is bounded diagnostic output, not an alternate structured return value.

## Independent Work

```python
import asyncio

parser, compiler = await asyncio.gather(
    evidence(task="Parser tests: 12 passed, 0 failed."),
    evidence(task="Compiler tests: 10 passed, 2 failed."),
)
{"failures": parser["final_result"] + compiler["final_result"]}
```

`asyncio.gather` expresses concurrency, not unlimited capacity. All child
starts still consume ancestor budgets and active slots. A plain `await` chain
preserves dependencies. The outer `run_workflow` tool is sequential so separate
workflow executions do not race the same parent orchestration state.

## Sandbox and Recovery

Monty is a restricted Python environment. It has no host filesystem, network,
environment variables, credentials, runtime objects, processes, clocks, or
arbitrary imports. Read returned dictionaries using `result["key"]`; do not
assume the entire CPython standard library or mapping API is available.

Syntax and type checking happen before children start. Child failures become
sanitized `RuntimeError` values that generated code may catch. Successful
identical child calls can be reused from a bounded retry-salvage table when a
workflow is corrected. This is recovery within the workflow machinery, not a
global cache for every task with similar wording.

Budget exhaustion is terminal. Cancellation cancels and joins the workflow's
owned children. Output is bounded and only JSON-safe values cross the boundary.
Dynamic workflows cannot nest, although a child can use ordinary delegation
within the existing limits.

## Not CodeMode

| Dynamic workflow | CodeMode |
| --- | --- |
| Functions represent permitted child agents. | Functions represent hydrated application tools. |
| Each child has a conversation and profile. | Each tool retains its schema, risk, and approval contract. |
| Composes child results and dependencies. | Filters/joins tool data before returning it to the model. |
| Enabled with `> workflow: dynamic` in a profile. | Enabled with `> codemode: enabled`. |

Both use Monty; neither grants unrestricted host Python execution. See
[CodeMode](codemode.md) and the [reviewed workflow](reviewed-evidence.md).
