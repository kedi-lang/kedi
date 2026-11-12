# GEPA Optimization

## When to Use GEPA

GEPA reflectively evolves prompt instructions from execution traces, metric
scores, and feedback. Use it when:

- the procedure structure and output fields are already correct;
- representative training examples exist;
- a metric can distinguish better prompts from worse prompts;
- the extra model-call cost is justified.

Do not use GEPA to compensate for a wrong type schema, a nondeterministic
dataset, or a metric that rewards the wrong outcome.

## Run GEPA

GEPA is selected explicitly; the CLI's default optimizer is `mock`, which is
intended for deterministic development and tests:

```bash
kedi program.kedi --optimize --optimizer gepa
```

The mock optimizer is not a no-model-call mode: its metric can still invoke
model-backed procedures. Its output is not evidence of prompt quality gains.

`--optimize` works without `--eval`. Add `--eval` when the same command should
also score the resulting program.

## Main and Reflection Models

The main optimizer model drives DSPy execution. The reflection model analyzes
traces and proposes instruction changes:

```bash
kedi program.kedi --optimize --optimizer gepa \
  --optimizer-model openai/gpt-6-luna \
  --optimizer-reflection-model openai/gpt-6.1-sol
```

Environment equivalents are `KEDI_OPTIMIZER_MODEL` and
`KEDI_OPTIMIZER_REFLECTION_MODEL`. These are optimizer models, not necessarily
the normal runtime `--adapter-model`.

GEPA does not select a DSPy runtime adapter. Calls outside the optimized span,
including model-backed metrics, use `--adapter` and `--adapter-model`, just as
normal execution does. `--optimizer-model` continues to drive GEPA's own DSPy
module; reflection remains DSPy-backed. A metric procedure can use `> agent:`
or `> model:` to select its own backend or model.

Without an explicit reflection model, GEPA reuses the configured main DSPy LM,
including its provider and model settings. A main LM must therefore be configured.
To give reflection a separate model or token limit, set the reflection model
explicitly.

## Metric Call Budget

Bound optimization work with:

```bash
kedi program.kedi --optimize --optimizer gepa \
  --optimizer-max-metric-calls 100
```

The value is GEPA's evaluation-unit budget. DSPy may invoke the wrapped metric
more than once per unit, so it is not a strict count of model requests. More
rows, spans, and candidates increase total work.

## Token Budgets

Configure model output limits independently:

```bash
kedi program.kedi --optimize --optimizer gepa \
  --optimizer-model-max-tokens 8000 \
  --optimizer-reflection-model-max-tokens 32000
```

The defaults are 8,000 tokens for the main optimizer model and 32,000 for
reflection. Lower limits reduce cost but can truncate reasoning or generated
instructions.

## Reflection Minibatches

`--optimizer-reflection-minibatch-size` controls how many evaluated examples
are accumulated before reflection:

```bash
kedi program.kedi --optimize --optimizer gepa \
  --optimizer-reflection-minibatch-size 3
```

Small minibatches reflect more frequently. Larger minibatches provide broader
evidence per reflection step but consume more of the budget before adaptation.

## Validation Limits

When `> test_data:` is present, GEPA uses it as validation data:

```bash
kedi program.kedi --optimize --optimizer gepa \
  --optimizer-max-validation-examples 20
```

The limit slices the explicit validation set from the beginning. Leave it
unset to use all held-out rows. Without test data, GEPA derives validation rows
from training data.

## Checkpoints and Resume

Each span gets a checkpoint directory:

```text
program.kedi.gepa/
└── procedure__span/
    └── gepa_state.bin
```

Later runs resume from this state and seed from the saved optimized prefix. If
a resumed run ends immediately, its prior evaluation count may already meet
the current budget; increase `--optimizer-max-metric-calls`.

Resume does not guarantee zero model calls: GEPA can reevaluate the seed
validation set before loading the checkpoint, and Kedi computes a final training
score after selecting the candidate. These calls can occur even when the search
budget has already been spent. Output-format repairs can add further requests.

## Start Fresh

Discard prefixes, stored scores, and all checkpoints:

```bash
kedi program.kedi --optimize --optimizer gepa --optimizer-fresh
```

Use this after changing the metric, materially changing datasets, or replacing
the source prompt's objective. Keep resume state when only extending the
budget.

## DSPy Dependency

Install the optional dependency with `uv add 'kedi[dspy]'`.
GEPA is implemented through DSPy. If DSPy is unavailable, Kedi exits with an
installation error before optimization. The ordinary Pydantic, LangChain,
Claude, Codex, and ACP runtime adapters do not by themselves provide GEPA.

GEPA also requires valid credentials for the selected optimizer and reflection
models. Use `--quiet` to suppress progress reporting, not errors.

Checkpoints can contain example data, responses and feedback. Treat them as
trusted local state, not files to load from unknown authors. Quiet mode does
not redact sensitive data; review artifacts before sharing them.
