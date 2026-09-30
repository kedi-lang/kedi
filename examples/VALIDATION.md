# Example verification

Checked on **September 30, 2026**, using the source checkout, Python 3.12.10,
Pydantic AI 2.45.0, and the locally built Tree-sitter grammar.

This is an example-validation record, not a benchmark or an accuracy estimate.
Inputs are the fictional fixtures committed with each workflow.

## Catalog quality gates before the game addition

- Example and source-path regression suite: **66 passed**.
- Full repository suite: **5,659 passed, 1 skipped**; **100% statement and branch
  coverage**. The skipped Harbor test requires an unavailable optional
  `harbor.agents.installed.base` module.
- `make check`: passed lint, formatting, and source type checks. `ty` reported
  two existing unused-suppression warnings in optional Harbor imports;
  BasedPyright reported no diagnostics.
- Repository README and language specification: all **84 Kedi code blocks** parse.
- Published documentation: all **342 Kedi code blocks** parse.
- `make check-docs`: passed; editor copies of the language specification match.

## Flappy Kedi follow-up

The later [game example](flappy_kedi/README.md) uses a `.kedi` entrypoint,
a `.kedi` decision loop, and an imported Python graphics/physics package.
The game, catalog, type-system, and source-path regression suites pass together.
The full core suite was also rerun after fixing native string initialization
for union, literal, and annotated types, retaining 100% statement/branch coverage.

Live checks are recorded separately in the game guide: the unrestricted pilot
survived 80 decisions and five gates, while the subsequent 60% confidence policy
rejected eight flaps and the flight ended after 15 decisions with zero gates.
The latter run had no below-threshold flap applied. These are smoke checks,
not a comparison with statistical significance.

## Deterministic checks

`tests/test_curated_examples.py` runs the actual source files with controlled
model responses. The provider is stubbed, but parsing, compilation, local tool
execution, artifact storage/retrieval, and result validation are real.

- Every example source parses, and every workflow has an indexed guide.
- File-reading examples run from their own directory, the repository root,
  and an unrelated directory. CLI checks resolve sibling files through
  `__file__`; imported modules and concurrent programs retain their own paths
  without changing the process working directory.
- Meeting actions preserve nullable fields, the nested description, and the
  non-empty task constraint in the output schema.
- Release-note classification skips the internal change and preserves order.
- Support invokes the actual policy-reading procedure before returning a `Reply`.
- Parallel review uses a two-party rendezvous: neither response is released
  until both children have started. Both return nested typed models.
- Artifact investigation stores the real fixture, omits its hidden details from
  the preview, and retrieves evidence through the real `query_artifact` tool.
- Python API checks both decorated function calls and the `main.py` entrypoint.
- Jev routing checks both sides of the 0.85 threshold and metadata access after
  capture resolution, with two decisions in one provider request.
- Module imports produce the expected quote without a model. The actual CLI
  entrypoint and both colocated `@test` cases also pass.
- Knapsack independently verifies all target optima, rejects invalid selections,
  accepts ties, and exercises dataset/optimizer conversion.

Run the checks from the repository root:

```bash
python -m pytest tests/test_curated_examples.py tests/test_knapsack_example.py tests/test_source_file_binding.py --record-mode=none
```

These tests do not measure whether a real model will consistently choose the
right action or produce a factually correct explanation.

## Live model checks

The generative examples used **gpt-5.6-luna, medium effort**, through
**codex-auth-helper 1.8.0**, with WebSocket-first transport and HTTP fallback.
Jev routing used **jev-latest** through **kedi-typesafe 0.2.1** and
**typesafe-sdk 0.6.0**. The helper was installed in an isolated temporary target;
the checkout environment's dependencies were not replaced. Logfire was not enabled.

Request counts below are provider requests, including turns after tools. A
template is not necessarily one request when its agent uses tools or retries.

| Workflow | Requests | Observed behavior |
| --- | ---: | --- |
| Meeting actions | 1 | Three agreed actions; correct owners/deadline; missing values remain null; suggestion and completed work excluded |
| Release notes | 5 | Three classifications and two generated notes; internal fixture rename omitted; Linux and 100 MB restrictions retained |
| Support reply | 2 | One real policy tool call; asks for order/tracking details; does not claim a completed refund |
| Parallel review | 2 | One call per child, no coordinator call; identifies missing account authorization and exception-path cleanup |
| Artifact investigation | 6 | Source read, one artifact query, three bounded reads; correctly distinguishes failed code rollback from effective pool-size correction |
| Python API | 2 | Actual script prints a typed incident with `resolved=False` and a notice that does not claim recovery |
| Jev routing | 1 | Routes to billing and escalates; observed urgency probability 0.88 against threshold 0.85 |
| Knapsack demo | 1 | Returns `['C', 'D']`, weight 9 and value 18 |

The module example requires zero model requests and returns subtotal 49,500,
discount 4,950, and total 44,550 cents.

## What changed during verification

The release-note example initially returned an empty list. Merely rewording
the condition did not fix it. Explicitly scoping the procedure to **classifying
the supplied text, not verifying whether a deployment happened**, and quoting
that text in the claim produced the intended path. The final prompt was checked
again; this is prompt development on a tiny fixture, not held-out evidence.

The first artifact run used four sequential reads and no query. The final
example explicitly requests targeted retrieval before neighboring reads. It
then used one query and three reads. Both runs reached the core diagnosis;
this does **not** establish token savings versus an inline-output baseline.

Jev metadata is accessed only after the capture has resolved. The live checker
also needed provider-specific configuration: disable hidden SDK retries for a
request budget, avoid an unsupported per-request WebSocket timeout, and do not
attach a hard request budget to Jev's opaque retry policy. Those checker
configuration failures were not successful model runs.

The Python example uses a native dataclass registered by `@kedi.type`, not a
Pydantic `BaseModel`; its entrypoint prints that value directly. This distinction
is covered by the entrypoint regression, not only by calling its functions.

Live prose is still fallible. One reliability review overstated which exceptions
could leak a connection even though it found the real missing-cleanup issue.
Treat recommendations as drafts to review, not as proof of a complete code audit.
No latency speedup, retrieval savings, or model-accuracy claim is made here.

Knapsack's earlier row-level validation and the intentionally rejected answer
are documented in its [own guide](knapsack/README.md#verification-notes).
A live GEPA optimization search was not run as part of this gallery rewrite.
