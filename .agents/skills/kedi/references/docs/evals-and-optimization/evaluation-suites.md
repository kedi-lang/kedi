# Evaluation Suites

## Define `@eval:` Suites

An evaluation suite belongs at top level and names the procedure being scored:

````kedi
@answer(question: str) -> str:
  >> Answer precisely: <question>.
  The answer is [response: str].
  = <response>

@eval: answer:
  > data: questions:
    = ```
    return [
      ("What is 2 + 2?", {"answer": "4"}),
      ("What is the capital of France?", {"answer": "Paris"}),
    ]
    ```

  > metric: exact_match(questions):
    = `answer(questions).strip() == expected["answer"]`
````

The suite definition itself does not run during ordinary program execution.

## Match a Procedure

The suite name should match its target procedure. This is mandatory for prompt
optimization: every procedure containing `> optimize:` needs a same-named eval
suite. Evaluation metrics still call procedures explicitly; the suite name
does not insert an implicit procedure call.

## Training and Test Data

`> data:` defines optimization/training rows. `> test_data:` defines held-out
rows under the same dataset name:

````kedi
@sentiment(text: str) -> str:
  >> The sentiment of <text> is [label: Literal["positive", "negative"]].
  = `label`

@eval: sentiment:
  > data: examples:
    = `[("Loved it", {"label": "positive"}), ("Hated it", {"label": "negative"})]`
  > test_data: examples:
    = `[("A delightful surprise", {"label": "positive"})]`
  > metric: accuracy(examples):
    = `sentiment(examples).lower() == expected["label"]`
````

For `kedi --eval`, a test dataset replaces the same-named training dataset for
scoring. It does not print separate train and test scores. If no matching
`test_data` exists, evaluation falls back to `data`.

During optimization, `data` is always the training set. Matching `test_data`
is passed as validation data. That data participates in candidate selection;
keep a separate final dataset to measure generalization.

## One Metric per Suite

Each suite accepts one metric. Defining multiple `> metric:` blocks is a parse
error. A metric normally names its dataset:

```kedi
@predict(text: str) -> str:
  = `text`

@eval: predict:
  > data: examples:
    = `[("Kedi", {"text": "Kedi"})]`
  > metric: accuracy(examples):
    = `predict(examples) == expected["text"]`
```

The legacy dataset-free metric form can still execute once, but new programs
should use a named dataset so evaluation and optimization agree on inputs.

## Evaluation Execution

Run every eval suite in a file:

```bash
kedi program.kedi --eval
```

Select the runtime adapter and model as usual:

```bash
kedi program.kedi --eval \
  --adapter pydantic \
  --adapter-model openai:gpt-6-luna
```

Each row gets fresh metric bindings but shares the compiled runtime. This is
not process or mutable-object isolation. Exceptions raised while
loading data abort the eval. Exceptions raised by a metric row become a score
of `0.0` with `error: ...` feedback, allowing the remaining rows to run.

`--eval` reports scores without enforcing a minimum acceptable score. A completed
evaluation scoring `0.0`, including row errors, can exit with status `0`. Inspect
scores and feedback explicitly. Parse errors, data-loading errors and missing
dataset references are command failures.

## Train and Test Reporting

Scores are arithmetic means over the selected rows. An empty selected dataset
reports `0.0` with `no examples` feedback. CLI output has this form:

```text
Evals:
- sentiment::accuracy: 0.7500 | one label used extra punctuation
```

The normal eval command reports either matching test rows or fallback training
rows. GEPA separately calculates and stores a post-optimization training score
in `program.kedi.optimized_scores.json`.

## Feedback Results

A metric may return:

- `bool`, normalized to `1.0` or `0.0`;
- a numeric value coercible to `float`;
- `(score, feedback)`, where feedback is converted to text.

Feedback from multiple rows is joined with `; `. Return concise, actionable
feedback when optimizing: it becomes evidence the optimizer can use to improve
the prompt.
