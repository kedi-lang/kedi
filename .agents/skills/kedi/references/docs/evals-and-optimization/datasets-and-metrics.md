# Datasets and Metrics

## Training Datasets with `> data:`

A training dataset is Python code that returns an iterable:

````kedi
@word_count(text: str) -> int:
  = `len(text.split())`

@eval: word_count:
  > data: samples:
    = ```
    return [
      ("one two", {"count": 2}),
      ("one two three", {"count": 3}),
    ]
    ```

  > metric: exact(samples):
    = `word_count(samples) == expected["count"]`
````

The name after `> data:` is local to the eval suite and is referenced by the
metric declaration.

## Test Datasets with `> test_data:`

Use the same dataset name for held-out rows:

```kedi
@word_count(text: str) -> int:
  = `len(text.split())`

@eval: word_count:
  > data: samples:
    = `[("training example", {"count": 2})]`
  > test_data: samples:
    = `[("held out example", {"count": 3})]`
  > metric: exact(samples):
    = `word_count(samples) == expected["count"]`
```

`--eval` prefers `test_data: samples` over `data: samples`. Prompt optimizers
use `data` for training and matching `test_data` for validation.

## Iterable Requirements

Dataset code may return a list, tuple, generator, set, dictionary, dictionary
view, or another non-string iterable. Kedi materializes it to a list before
evaluation. Returning `str`, `bytes`, or a non-iterable is an error.

Do not use unordered sets when repeatable row order matters. Dataset creation
can call prelude helpers and use the compiled runtime, but it should avoid
model calls and mutable global side effects.

## Raw Items

A non-tuple row is bound directly to the dataset variable:

````kedi
@nonempty(text: str) -> bool:
  = `bool(text.strip())`

@eval: nonempty:
  > data: values:
    = `["Kedi", "reference"]`
  > metric: valid(values):
    = `nonempty(values)`
````

No `expected` binding is created for raw rows.

## Input and Expected Tuples

The portable convention for supervised data is `(input, expected_dict)`:

````kedi
@join_words(words: list[str]) -> str:
  = `" ".join(words)`

@eval: join_words:
  > data: cases:
    = ```
    return [
      (["hello", "world"], {"text": "hello world"}),
      (["kedi"], {"text": "kedi"}),
    ]
    ```

  > metric: exact(cases):
    = `join_words(cases) == expected["text"]`
````

Using a dictionary as the expected value is the least ambiguous shape across
normal evals and optimization. `None` is useful for optimizer-side analytical
metrics, but normal `--eval` does not inject an `expected` variable when the
expected value is `None`.

Direct evaluation uses these shape rules, independently of procedure types:

| Two-item tuple | Dataset variable | `expected` |
| --- | --- | --- |
| `(input, dictionary)` | input | dictionary |
| `(input, None)` | input | not injected |
| `(tuple_input, scalar)` | tuple input | scalar, except `None` |
| Other pairs, including `("text", "label")` | entire pair | not injected |

Here scalar excludes lists, dictionaries and tuples. A two-element list is a
raw row, not a supervised tuple. Optimization splits every two-item tuple;
expected dictionaries avoid this difference between the two paths.

For a procedure with multiple parameters, make the input itself a tuple:

```kedi
= `[((2, 3), {"sum": 5}), ((5, 8), {"sum": 13})]`
```

The optimizer can unpack that input tuple into procedure parameter bindings.
The metric may also call the procedure explicitly with `add(*cases)`.

## Mapping Items

Returning a dictionary is equivalent to iterating its `.items()`:

```kedi
@split_words(text: str) -> list[str]:
  = `text.split()`

@eval: split_words:
  > data: cases:
    = `{"short": ["short"], "two words": ["two", "words"]}`
  > metric: exact(cases):
    = `split_words(cases[0]) == cases[1]`
```

Direct eval preserves ordinary `(key, value)` pairs as one raw dataset value
when the value is a list, tuple, or scalar. A dictionary value is recognized as
an expected-output record and split into `cases=key` plus `expected=value`.

Optimization normalizes every two-item tuple, including mapping items, as
`(input, expected)`. If the same suite will be optimized, prefer an explicit
list of `(input, expected-dict)` rows rather than relying on mapping-pair
heuristics.

## Metric Bindings

For each row, Kedi binds the metric's dataset parameter:

```kedi
@predict(text: str) -> str:
  = `text`

@eval: predict:
  > data: cases:
    = `[("Kedi", {"label": "Kedi"})]`
  > metric: accuracy(cases):
    = `predict(cases) == expected["label"]`
```

When the row is recognized as `(input, expected)`, `cases` receives the input
and `expected` receives the expected value. Raw rows bind only `cases`.

Optimization additionally binds a one-parameter procedure's parameter name to
the input. For multi-parameter procedures, tuple inputs are unpacked across
parameter names. Do not depend on those convenience bindings in a metric when
an explicit call through the dataset variable is clearer.

## Boolean and Float Scores

Booleans normalize to exact scores:

```kedi
= `prediction == expected`
```

Return a float for partial credit:

```kedi
= `matched_fields / total_fields`
```

The direct evaluator does not clamp scores to `[0, 1]`; the metric defines the
scale. Use finite values on a consistent `0.0` to `1.0` scale. The development
mock optimizer clamps its own scores, so out-of-range metrics are not portable
between evaluation and optimizer paths.

## Score and Feedback Results

Return a two-tuple to attach diagnostic feedback:

````kedi
@summarize(text: str) -> str:
  = `text`

@eval: summarize:
  > data: cases:
    = `[("Kedi is a language.", {"fact": "language"})]`
  > metric: quality(cases):
    = ```
    result = summarize(cases)
    score = 1.0 if expected["fact"] in result else 0.0
    feedback = None if score else f"Missing fact: {expected['fact']}"
    return score, feedback
    ```
````

The tuple must have exactly two elements. The first is converted to `float`;
the second becomes text unless it is `None`.

## Analytical Metrics

An analytical metric computes quality without a gold label:

````kedi
@compress(text: str) -> str:
  >> A shorter version of <text> is [summary]
  = `summary`

@eval: compress:
  > data: passages:
    = `[("Kedi combines prompts with Python procedures.", None)]`
  > metric: compression(passages):
    = ```
    summary = compress(passages)
    score = min(1.0, len(passages) / max(1, len(summary)))
    return score, f"{len(summary)} output characters"
    ```
````

Use `(input, None)` for optimization compatibility. Since no normal eval
`expected` binding is created for `None`, analytical metrics should not read it.
