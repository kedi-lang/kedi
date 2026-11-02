# Test, Eval, and Optimize Commands

## Run Tests

```bash
kedi program.kedi --test
```

Runs every `@test:` case and exits `1` if any fail.

## Run Evals

```bash
kedi program.kedi --eval
```

Runs each `@eval:` metric over matching `test_data` or fallback training data.

A completed score of `0.0` can still exit successfully. Enforce quality thresholds
separately and inspect error feedback. Use the executable
[validation workflow](../evals-and-optimization/validation-workflow.md) for a
complete fixture and [reproducible comparisons](../evals-and-optimization/reproducibility.md)
before interpreting optimization gains.

## Run Optimization

```bash
kedi program.kedi --optimize
```

Optimization is independent of `--eval`. Combine them to optimize first and
then report the resulting eval.

## Select an Optimizer

```bash
kedi program.kedi --optimize --optimizer gepa
```

Choices are `mock` and `gepa`; CLI default is `mock`. Use mock for deterministic
infrastructure tests, not production prompt improvement.

## Configure GEPA Models

```bash
kedi program.kedi --optimize --optimizer gepa \
  --optimizer-model openai/gpt-6-luna \
  --optimizer-reflection-model openai/gpt-6.1-sol
```

Each option has a matching `KEDI_OPTIMIZER_*` environment variable.

## Configure Budgets

```bash
kedi program.kedi --optimize --optimizer gepa \
  --optimizer-max-metric-calls 100 \
  --optimizer-model-max-tokens 8000 \
  --optimizer-reflection-model-max-tokens 32000 \
  --optimizer-reflection-minibatch-size 3 \
  --optimizer-max-validation-examples 20
```

## Start Fresh

```bash
kedi program.kedi --optimize --optimizer gepa --optimizer-fresh
```

Deletes optimized prompt/scores files and the GEPA checkpoint directory.

## Disable Codegen Cache

```bash
kedi program.kedi --test --no-cache
```

This controls `> auto:`'s `.cache.kedi`, not optimized prompt artifacts or the
Python API response cache.

## Quiet Mode

`--quiet` suppresses codegen and GEPA progress output. It does not suppress
errors or final test/eval results.

## Generated Artifacts

- `.cache.kedi`: AI-generated tests/implementations;
- `.kedi.optimized.json`: optimized prefixes;
- `.kedi.optimized_scores.json`: training scores;
- `.kedi.gepa/`: resume checkpoints.
