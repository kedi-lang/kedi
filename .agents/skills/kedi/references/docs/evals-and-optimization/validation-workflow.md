# A Complete Validation Workflow

This program needs no model or credentials. It distinguishes a regression test
from a dataset score, with explicit expected values and failure feedback.

## Normalize and Validate

Create `labels.kedi` with this complete program:

````kedi
@normalize_label(text: str) -> str:
  = `"-".join(text.strip().casefold().split())`

@test: normalize_label:
  > case: whitespace:
    `assert normalize_label("  Release   Notes  ") == "release-notes"`
  > case: empty:
    `assert normalize_label("   ") == ""`
  > case: idempotent:
    ```
    once = normalize_label("  Release Notes  ")
    assert normalize_label(once) == once
    ```

@eval: normalize_label:
  > data: labels:
    = `[("First Draft", {"label": "first-draft"})]`
  > test_data: labels:
    = `[("  Release   Notes ", {"label": "release-notes"}), ("", {"label": ""})]`
  > metric: exact(labels):
    = ```
    actual = normalize_label(labels)
    correct = actual == expected["label"]
    feedback = None if correct else f"Expected {expected['label']!r}; got {actual!r}"
    return correct, feedback
    ```
````

Run the two checks independently:

```sh
kedi labels.kedi --test
kedi labels.kedi --eval
```

All three test cases pass. The eval selects the two `test_data` rows and reports
`normalize_label::exact: 1.0000`. It does not also report the training score.
The metric calls the procedure explicitly; naming the suite alone does not.

## Detect a Regression

Change `split()` to `split(" ")`. Repeated spaces now produce repeated hyphens:
the whitespace test fails, and one of the two eval rows fails. The test command
exits nonzero, while the completed eval reports `0.5000` with feedback but exits
successfully. CI must enforce its own score threshold if using evals as gates.

## Interpret Errors Separately

````kedi
@reciprocal(value: int) -> float:
  = `1.0 / value`

@eval: reciprocal:
  > data: values:
    = `[2, 0, 4]`
  > metric: positive(values):
    = `reciprocal(values) > 0.0`
````

This eval scores two of three rows, approximately `0.6667`. The zero input adds
`error: ...` feedback; it does not stop the last row. This is an execution error,
not evidence that a model made a wrong judgment.

Dataset construction is different: returning `42` instead of `[2, 0, 4]`
fails the evaluation itself because the dataset must be iterable.

## Add a Model Deliberately

Replace a deterministic procedure with a typed template only when the task needs
model inference. Keep the dataset contract and metric, configure credentials,
and record the model and adapter. A stochastic model does not turn an invariant
test into an accuracy estimate: use repeated, matched evaluations for that.
