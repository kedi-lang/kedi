# Prompt Optimization

## Mark an `> optimize:` Span

An optimize span is an executable template block inside a procedure:

```kedi
@extract_owner(ticket: str) -> str:
  > optimize: owner_extraction:
    >> Read the ticket below.
    Ticket: <ticket>
    The responsible person's name is [owner]
  = `owner`
```

The whole indented body becomes one logical adapter operation. Tool rounds or
retries may require multiple provider requests. Captures resolve when the
operation completes.

## Named Prompt Spans

The span name is a stable artifact key. In
`program.kedi.optimized.json`, prompts are stored as:

```json
{
  "extract_owner": {
    "owner_extraction": "Optimized instruction prefix..."
  }
}
```

Choose names that describe the prompt's job. Renaming a span disconnects it
from the old artifact entry.

<span id="explicit-and-legacy-block-forms"></span>

## Required Template Marker

Start new optimize templates with `>>`, just like ordinary template blocks:

```kedi
@extract(document: str) -> str:
  > optimize: fields:
    >> Read <document>.
    The document is titled [title] and was written by [author].
  = `<title> + " by " + <author>`
```

The first prompt line must start with `>>`; an optimize body with a bare prompt
is a parse error. Continuation lines at the same indentation belong to that
template and do not repeat the marker. Blank lines within the prompt preserve
paragraph breaks. Blank lines before or after the template are also allowed and
do not create another model call.

Substitutions and captures work exactly as they do in `>>` blocks:

- `<document>` reads and renders an existing value.
- `<helper(document)>` renders a procedure call result.
- `` <`expression`> `` renders a Python expression.
- `[title]` captures a string.
- `[items: list[str]]` requests and captures a typed value.

## Multiple Spans

A procedure may contain multiple independently named spans:

```kedi
@solve(problem: str) -> int:
  > optimize: parse:
    >> The problem <problem> combines [left: int] with [right: int]
    using [operation].
  > optimize: calculate:
    >> The result of <left> <operation> <right> is [answer: int]
  = `answer`
```

They execute in procedure order. Because they are separate model calls, the
second span can substitute fields produced by the first. GEPA optimizes spans
one at a time and stores one prefix per span.

## Required Eval Suites

Every optimized procedure needs:

1. a same-named `@eval:` suite;
2. at least one `> data:` dataset;
3. at least one metric;
4. a metric dataset name that refers to training data.

```kedi
@extract_owner(ticket: str) -> str:
  > optimize: owner:
    >> The owner identified in <ticket> is [owner]
  = `owner`

@eval: extract_owner:
  > data: tickets:
    = `[("Owner: Ada", {"owner": "Ada"})]`
  > metric: exact(tickets):
    = `extract_owner(tickets) == expected["owner"]`
```

Kedi validates these requirements before constructing model-backed optimizer
services, so configuration errors fail before expensive optimization begins.

## Training Data

The optimizer uses the first declared training dataset for a procedure. Use
explicit `(input, expected)` rows. For multi-parameter procedures, the input is
a tuple ordered like the procedure signature:

```kedi
= `[(("left text", "right text"), {"same": False})]`
```

Metrics should return stable scores and useful feedback. An optimizer can only
improve what the dataset and metric expose.

## Validation Data

A `> test_data:` block with the same name becomes the optimizer's validation
set. If GEPA has no test set, it uses the first quarter of training examples
when at least four training examples exist; otherwise it validates on the full
training set.

`--optimizer-max-validation-examples N` truncates an explicit test set for
validation throughout the GEPA run. It does not truncate the training set.

These validation rows participate in candidate selection. The fallback subset
is not removed from training. Neither provides an untouched final test set;
see [Reproducible Comparisons](reproducibility.md).

## Optimized Output Artifacts

Run optimization independently:

```bash
kedi program.kedi --optimize --optimizer gepa
```

Or optimize and then report eval results in the same command:

```bash
kedi program.kedi --optimize --optimizer gepa --eval
```

Kedi stores only an optimized **prefix**. It preserves the original source
template, including output fields and type annotations, and prepends the prefix
at runtime. This prevents an optimizer from becoming the owner of the output
schema.

File-backed execution, tests, and evals load `program.kedi.optimized.json`
automatically. A missing file means “use the source prompt.” A malformed file
is an execution error; Kedi never silently falls back from corrupt optimized
state.

## Fresh Optimization Runs

GEPA normally seeds from prior optimized prompts and resumes per-span
checkpoints. Start from the source template only with:

```bash
kedi program.kedi --optimize --optimizer gepa --optimizer-fresh
```

`--optimizer-fresh` deletes:

- `program.kedi.optimized.json`;
- `program.kedi.optimized_scores.json`;
- `program.kedi.gepa/`.

It is unrelated to `--no-cache`, which controls generated `> auto:`
implementations.
