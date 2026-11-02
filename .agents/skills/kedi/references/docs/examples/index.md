# Cookbook { #examples }

These examples combine Kedi features into reviewable programs. Read the focused
language pages first when you need a complete rule rather than a guided
scenario.

For checkout-ready programs with bundled input files, per-example guides, and
regression tests, use the
[repository example gallery](https://github.com/kedi-lang/kedi/tree/stable/examples).
Its [verification notes](https://github.com/kedi-lang/kedi/blob/stable/examples/VALIDATION.md)
separate live model observations from deterministic runtime checks.

## Choose an Example

| Example | Main concepts | Requires model calls |
| --- | --- | --- |
| [Structured Extraction](structured-extraction.md) | custom types, typed captures, native returns | yes |
| [Tools and Approvals](tools-and-approvals.md) | Kedi and Python tools, risk, argument editing | yes |
| [Agent Delegation](agent-delegation.md) | profiles, structured children, background lifecycle | yes |
| [Evaluation and Optimization](evaluation-and-optimization.md) | datasets, metrics, `> optimize:`, GEPA | yes |
| [Modules and Packaging](modules-and-packaging.md) | exports, selective imports, `package.kedi` | no |
| [Complete Program](complete-program.md) | a compact end-to-end application | yes |

## Follow a Workflow

These tutorials keep their full listings beside the authoritative feature
documentation. They are part of the cookbook, not additional APIs to learn.

| Task | Start here | What to inspect |
| --- | --- | --- |
| Transform nested collections | [Loops and Map](../core-language/loops-and-map.md) | collected native values, nested scope, conditional filtering |
| Route a generated draft using a scored judgment | [Jev Workflow](../agent-adapters/jev-workflow.md) | generative model versus evaluator, threshold, both routing outcomes |
| Delegate, review, and write a report | [Reviewed Evidence](../agentic-engineering/reviewed-evidence.md) | typed child envelope, actual tool effects, edited approval, hooks |
| Compose child work dynamically | [Dynamic variant](../agentic-engineering/reviewed-evidence.md#dynamic-variant) | declared children, returned payload, separate approved writer |
| Find evidence in a large tool result | [Artifact Retrieval](../runtime/artifact-retrieval.md) | opt-in query, bounded reads, what remains outside model context |
| Join several large results | [Artifact Reduction](../runtime/artifact-reduction.md) | executable reduction, bounded output, retained artifact handles |
| Embed Kedi in a Python application | [Python Embedding](../python-api/embedding.md) | native result, runtime lifetime, cleanup |
| Validate before optimizing | [Validation Workflow](../evals-and-optimization/validation-workflow.md) | exact outputs, failing cases, row errors versus suite errors |
| Develop across files | [Local Package](../modules-and-packaging/local-package.md) | explicit exports, isolated installation, receipt |
| Explore incrementally | [Notebook](../tooling/notebook.md) | manual cell order, retained state, rerunning a cell |

For deterministic and model-judged branch conditions, use
[Control Flow](../core-language/control-flow.md). A claim-driven branch is a
model boundary; a Python predicate is not. Do not call a model for a condition
your program can compute exactly.

## Runnable Conventions

Unless a page shows a directory tree, save its complete listing as
`program.kedi` and run:

```bash
kedi program.kedi
```

The model-backed listings in this section use `openai:gpt-6-luna`
to make backend selection explicit.
Replace it with a model configured for your environment. Provider credentials
are read by the selected adapter; never place API keys in a `.kedi` source file.

## Syntax Used in Examples

- `<value>` substitutes an existing value into prompt or return text.
- `<`python_expression`>` evaluates Python and renders its result as text.
- `[field: Type]` in a `>>` block asks the model for a typed output.
- `[name: Type] = expression` performs deterministic variable initialization.
- ``= `python_expression` `` returns the native Python value.
- `[text] << prompt` captures the raw model text; bare `<<` is not an operator.

These distinctions matter. Use a typed output when a model must infer a value,
Python when the answer is deterministic, and a native Python return when the
caller should receive an object rather than its string representation.

## Verification

Documentation CI parser-checks every `kedi` fence. Package manifests are parsed
as `package.kedi`; a fence with a different required filename can declare
`file=...`, and an intentionally non-parseable fragment must declare
`no-parse`. This proves syntax and AST construction only. Imports, compilation,
provider schemas, credentials, and adapter capabilities still require their
documented file context and runtime tests against the production backend.

The offline documentation suite also executes the cookbook's complete programs:
local package installation, real tool calls behind a local Pydantic
`FunctionModel`, typed child delegation, and deterministic eval fixtures.
The fixture supplies model responses; the runtime, tools, validators, and file
effects are real. This proves those execution contracts, not provider accuracy,
latency, or optimizer gains. The linked Jev tutorial checks routing with a test
adapter, not live Jev scoring; artifact examples are model-assisted recipes,
not measured retrieval-quality results.

## In This Section

**Pages**

- [Incident Triage Application](complete-program.md)
- [Structured Extraction](structured-extraction.md)
- [Tools and Approvals](tools-and-approvals.md)
- [Typed Agent Delegation](agent-delegation.md)
- [Evaluation and Optimization](evaluation-and-optimization.md)
- [Building a Kedi Package](modules-and-packaging.md)
