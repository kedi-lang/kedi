# Reproducible Comparisons

An optimizer producing an artifact is not evidence that the program improved.
Compare the complete executable procedure, not just the optimized text.

## Freeze the Evaluation Contract

Before comparing candidates, freeze the input rows, expected values, scoring
code, runtime revision, model identifier, adapter, tool implementations,
reasoning settings and retry limits. Record the source and artifact hashes.
If any changes, identify that run as a new condition rather than attributing
its effect to prompt optimization alone.

## Separate Three Datasets

1. Training rows drive candidate generation.
2. Validation rows select candidates. Kedi supplies matching `test_data` to
   GEPA for this purpose, despite the directive's name.
3. A final evaluation set stays outside optimization and manual prompt tuning.

Without explicit validation data, GEPA reuses training rows; the fallback is
not a disjoint holdout. Do not describe its score as unseen-task performance.
For a final measurement, use a separate eval file or frozen dataset loader
whose rows were never given to the optimizer.

## Compare Source and Optimized Prefixes

File-backed runs automatically load the adjacent `.optimized.json`. Use
separate clean directories for baseline and candidate runs, with identical
source files. Leave the baseline without that artifact and put the selected
artifact only beside the candidate source. Keep generated-code caches matched
when `> auto:` is involved.

`--no-cache` does not disable optimized prefixes. `--optimizer-fresh` deletes
optimization artifacts and checkpoints; archive them first if they are needed
for reproduction. Do not use it as an evaluation-only baseline switch.

## Report Quality and Work

Record per-case outcomes before aggregating. Report task success, execution
errors, schema errors, retry counts, tool/model requests, input/output/cached
tokens and latency separately. A mean score can conceal failures on one group.
Keep the denominator, number of repeats and aggregation rule explicit.

Run both conditions on the same cases with the same budgets. For stochastic
models, repeat samples and vary condition order to reduce warm-cache and
provider timing bias. Report uncertainty; one better response is not a
demonstrated improvement. Distinguish fresh and reused provider caches.

The built-in eval command reports aggregate metric scores and joined feedback;
it is not a complete experiment recorder. Additional per-case telemetry must
be collected by the surrounding experiment harness. Mock optimizer runs verify
wiring, not learning gains. Generated tests establish only their own assertions.

## Protect Evidence

Prompt artifacts, feedback, generated source and checkpoints may contain input
data or model responses. Keep secrets out of examples and inspect exported
records before sharing. Resume only trusted checkpoints. Record dependency
versions so a later run does not silently use a different optimizer.
