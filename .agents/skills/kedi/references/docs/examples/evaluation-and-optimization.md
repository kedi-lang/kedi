# Evaluation and Optimization

This example exposes one extraction prompt to supervised optimization with a
metric that returns both a score and actionable feedback. Whether it improves
requires measurement; the listing is not evidence of a quality gain.

## Complete Program

````kedi
> adapter: pydantic
> model: openai:gpt-6-luna

@extract_owner(ticket: str) -> str:
  > optimize: owner_prompt:
    >> Read the support ticket below.
    Ticket: <ticket>
    The team that owns the next action is [owner]
  = `owner`

@eval: extract_owner:
  > data: tickets:
    = ```
    return [
      ("Payments: retries fail after authorization", {"owner": "Payments"}),
      ("Search indexing is twelve hours behind", {"owner": "Search"}),
      ("Identity reports invalid refresh tokens", {"owner": "Identity"}),
    ]
    ```

  > test_data: tickets:
    = ```
    return [
      ("Checkout cannot create payment intents", {"owner": "Payments"}),
      ("New documents do not appear in results", {"owner": "Search"}),
    ]
    ```

  > metric: exact_owner(tickets):
    = ```
    actual = extract_owner(tickets).strip().casefold()
    wanted = expected["owner"].casefold()
    if actual == wanted:
        return 1.0, None
    return 0.0, f"returned {actual!r}; expected {wanted!r}"
    ```

= `extract_owner("Identity: sessions expire immediately after login")`
````

The leading `>>` opens the template. Its continuation lines belong to the
same template, so this span makes one logical model call. Do not add `>>`
to each continuation line.

## Run Tests and Evals

```bash
kedi program.kedi --eval
```

Normal evaluation uses matching `test_data` when present; otherwise it falls
back to `data`. It reports one selected score, not separate training and test
scores.

## Run GEPA

Install the optional optimization dependencies and configure the selected
provider. See [GEPA prerequisites](../evals-and-optimization/gepa.md).
The default optimizer is `mock`, so select `gepa` explicitly for real search.

```bash
kedi program.kedi \
  --optimize \
  --optimizer gepa \
  --optimizer-max-metric-calls 30 \
  --eval
```

GEPA trains on `data`, validates against matching `test_data`, and stores only
an optimized prefix. The source prompt and its output schema remain
authoritative.

These two validation cases influence candidate selection. They are not an
untouched final test set. Freeze separate held-out cases before comparing the
baseline and optimized artifact, with the same model and repeated sampling.
An eval score below one does not by itself produce a nonzero CLI exit status.

Generated files are:

- `program.kedi.optimized.json` for selected prompt prefixes;
- `program.kedi.optimized_scores.json` for score metadata;
- `program.kedi.gepa/` for per-span checkpoints.

Use `--optimizer-fresh` to remove these optimization artifacts before a new
GEPA run. `--no-cache` instead controls generated `> auto:` code and does not
reset optimizer state.

## Dataset Shape

Explicit `(input, expected_dict)` rows are the least ambiguous format across
evaluation and optimization. For a procedure with multiple parameters, use
`((arg1, arg2), expected_dict)`. Metrics should be deterministic, stable, and
cheap compared with the model call; vague feedback gives the optimizer little
usable evidence.

For a no-network starting point, run the
[Validation Workflow](../evals-and-optimization/validation-workflow.md), including
its intentionally failing variant. Use the
[Reproducibility Protocol](../evals-and-optimization/reproducibility.md) before
interpreting an optimization result as an improvement.
