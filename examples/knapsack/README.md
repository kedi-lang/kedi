# Knapsack: a prompt with an executable objective

Ask a model to pack a bag, then check the selection in code. This example puts
a typed procedure, an optimizable template, and its evaluation next to each
other in [main.kedi](main.kedi).

It is a small demonstration of Kedi's evaluation and optimization workflow,
not a replacement for a deterministic knapsack solver or a model benchmark.

## Run it

From the repository root, with Kedi installed in your active environment:

```sh
export OPENAI_API_KEY="..."
kedi examples/knapsack/main.kedi --adapter pydantic --adapter-model openai:gpt-6-luna
```

The program does not fix a model or enable telemetry; choose the model at the CLI.
Model calls require network access and may incur charges.

The demo offers four items, represented as `name: (weight, value)`, and a capacity
of 10. An optimal answer is `['C', 'D']`: weight 9, value 18. Item order
does not matter. The command returns the selected names, not a model's claim
about their total value.

## Follow the program

1. `@solve_knapsack(...) -> list[str]` defines a typed procedure. Python tuples
   hold the item data. The backtick-wrapped parameter annotation uses Python's
   `dict[str, tuple[int, int]]` type directly.
2. `> optimize: selection:` marks one template as an optimization target. Its
   indented body starts with `>>`. `<items>` and `<capacity>` supply inputs;
   the sentence ends with the typed capture `[selected: list[str]]`.
3. `` = `selected` `` returns the captured list. An optimization block also works
   during ordinary execution; it does not start an optimizer by itself.
4. `@eval: solve_knapsack:` defines the data and executable scoring rule.
   Python handles arithmetic and validation, not the model.

There is no separate reasoning capture, answer parser, logging setup, or helper
framework. Type validation checks the shape of the answer; the metric checks
whether that answer actually solves the problem.

## Evaluate the selection

```sh
kedi examples/knapsack/main.kedi --adapter pydantic --adapter-model openai:gpt-6-luna --eval
```

`--eval` uses the two `> test_data:` cases in preference to the four `> data:`
cases. The `optimality` score is the fraction of cases solved exactly:

- Duplicate or unknown item names score 0.
- An overweight selection scores 0, even if it has a high value.
- A feasible but suboptimal selection scores 0.
- Any feasible selection with the optimal value scores 1, including tied answers.

Feedback includes weight, value, and the target value. The optimum comes from
the dataset, not the model's self-assessment. Repository tests independently
check every target by enumerating all subsets. Model output can still be wrong;
a successful CLI invocation is not proof of an optimal answer.

## Optimize the template

Install the optional optimizer dependency in the same environment:

```sh
python -m pip install -e '.[dspy]'
kedi examples/knapsack/main.kedi \
  --adapter pydantic \
  --adapter-model openai:gpt-6-luna \
  --optimize --eval --optimizer gepa \
  --optimizer-model openai/gpt-6-luna \
  --optimizer-reflection-model openai/gpt-6.1-sol \
  --optimizer-max-metric-calls 30
```

GEPA uses the four training cases and the metric's feedback to revise the
`selection` template. The two `test_data` cases participate in validation and
candidate selection, so they are **not an untouched test set**. Use additional
held-out cases and repeated runs before claiming an improvement. These tiny
problems may already score perfectly without optimization.

The metric-call limit is an optimization budget, not a dollar cap or a limit on
all provider requests. Reflection and any final evaluation add model calls.

Kedi saves the candidate in `main.kedi.optimized.json` beside the source and
loads it on later runs. Keep that file with your experiment results; move it
out of the example directory to evaluate the original prompt again. Also keep
the model, settings, and dataset fixed when comparing runs.

## Check the example without an API key

From the repository's development environment:

```sh
python -m pytest tests/test_knapsack_example.py -q -o addopts= -p no:cov
```

These tests execute the actual Kedi example with controlled model responses,
verify the answer keys, and exercise invalid, suboptimal, and tied selections.
They test the example's behavior, not a live model's problem-solving ability.

### Verification notes

The example was checked on September 30, 2026, with Pydantic AI 2.45.0.
The offline checks also cover the GEPA signature and dataset conversion, and
the optimizer interface using a mock optimizer.

A live smoke run used `gpt-5.6-luna` at medium effort through
`codex-auth-helper` 1.8.0, with WebSocket-first transport and HTTP fallback.
The demo returned `['C', 'D']`; the final template scored 3/4 on training
cases and 2/2 on validation cases. The failed answer contained a serialized
list inside a single string element, which the metric correctly rejected.
It was not silently repaired into a successful answer.

These are single-run smoke results, not an accuracy estimate or evidence of
optimization gains. A live GEPA search was not run for this revision.
