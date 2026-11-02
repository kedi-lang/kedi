# Reviewed Evidence Workflow

This example connects the whole boundary: a parent delegates, a child reads a
local fixture, the child returns a typed result, and the parent writes a report
through an approval handler. The parent cannot read the fixture directly; the
child cannot write the report.

Download both files into one directory:
[reviewed_evidence.kedi](../assets/examples/reviewed_evidence.kedi) and
[reviewed_evidence.py](../assets/examples/reviewed_evidence.py).

## The Kedi Program

```kedi
~Review(failures: list[str], may_release: bool)

> profile: reviewer:
    ###
    Read the named test suite and report failures without inventing results.
    ###
    > adapter: pydantic
    > use: read_evidence
    > output: Review
    > instructions:
        Read evidence for the requested suite.
        Release only if no tests failed.

> profile: coordinator:
    > adapter: pydantic
    > subagent: reviewer
    > max_agents: 2
    > use: write_report
    > approval: `approve_report`
    > instructions:
        Delegate the review and use its validated result.
        Write failure names and the release decision to requested.txt.
        Use write_report and return the actual saved path.
        Never treat missing evidence as success.

> use: coordinator
[receipt] << Review suite parser and save its report.
= <receipt>
```

The Python companion supplies all three referenced callables through
`runtime_globals`. There are no hidden network tools: `read_evidence` returns
12 passing tests and one failure named `unicode_identifier` from a local fixture.

## Host Responsibilities

`run_review(adapter, output_dir)` builds the runtime and closes it in `finally`.
Its child tool calls `current_subagent_execution()` to record the child run,
profile, and conversation identity. It raises if invoked outside a child.

The report approval handler accepts only `write_report` and replaces its path
with the application's fixed `output_dir/review.txt`. The writer independently
checks that destination before performing I/O. Kedi revalidates edited arguments
before calling it. The original model-provided path is never the write target.

The function returns the model's `receipt` plus native observations, approval
requests, and actual writes. These make the effect inspectable without treating
the final model sentence as proof that it happened.

## Run With a Provider

Install Kedi with its Pydantic provider support and configure the provider's
credentials. From the directory containing the two downloaded files:

```python
from kedi.agent_adapter import PydanticAdapter
from reviewed_evidence import run_review

result = run_review(
    PydanticAdapter("openai:gpt-6-luna"),
    output_dir="reports",
)
print(result["receipt"])
print(result["writes"])
```

This makes real model calls. Tool selection and prose can vary. Expected behavior
is a report that names `unicode_identifier` and does not approve release. An
empty `writes` list is not success, regardless of what the model claims.

## Deterministic Verification

The documentation tests run these exact files with a local Pydantic
`FunctionModel`. They assert the parent delegates, the child's tools exclude the
writer, the parent's tools exclude the reader, the returned child schema is
used, run IDs agree with the child tool's context, and the parent consumes that
result before writing. They also check that approval changes the destination
without changing the report text.

This establishes the program/runtime contract, not a model-accuracy result.
Separate negative tests exercise unavailable children, invalid results, denied
writes, and lifecycle failures. No external model or credentials are needed
for the deterministic test suite.

## Adapter Boundary

The downloadable program selects Pydantic explicitly. To use LangChain, change
both profile declarations to `> adapter: langchain` and supply a configured
`LangChainAdapter` to the same embedding function. The tools, approval handler,
child result envelope, and limits remain Kedi contracts; do not retain an
explicit Pydantic profile and assume that injecting a different adapter changes
its backend selection. The deterministic tutorial checks use Pydantic, while
the shared adapter regression suite covers the corresponding LangChain surfaces.

## Dynamic Variant

Add `> workflow: dynamic` to the coordinator. Its child call then happens inside
`run_workflow`, for example:

```python
review = await reviewer(task="Inspect suite parser using read_evidence.")
review["final_result"]
```

The parent receives this structured workflow result and still calls its
approved writer afterward. Do not move writing into the Monty snippet: only
declared child functions exist there. The test suite runs both variants.
