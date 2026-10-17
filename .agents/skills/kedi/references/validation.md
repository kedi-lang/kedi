# Validation, evaluation, optimization, and code generation

## Validation ladder

1. **Grammar:** Tree-sitter CST has no ERROR/missing nodes.
2. **Language structure:** `kedi parse` / `parse_program` validates the AST and placement rules.
3. **Deterministic runtime:** run complete no-model programs and check native types, results, imports, CLI options, and negative contracts.
4. **Controlled model integration:** real runtime with Pydantic AI TestModel/FunctionModel; assert prompts/schema/tool results and final outputs. Set `ALLOW_MODEL_REQUESTS=False` to prevent an accidental provider request.
5. **Actual integration:** selected provider/harness/MCP with credentials and supported capability contracts; record exactly what ran. Mocked semantic content cannot establish real-world quality.

The bundled `check_kedi.py` performs steps 1–2 by default. `--tree-only` skips AST validation, not just formatting. `verify_examples.py` performs 1–4 for its declared cases without credentials. It uses a disposable temporary directory for package installation, avoids touching the user's registry, and restores runtime state.

```bash
python "$KEDI_SKILL/scripts/check_kedi.py" "$KEDI_SKILL/assets/examples"
python "$KEDI_SKILL/scripts/verify_examples.py"
```

The offline manual preserves 154 supplied documentation pages. Its 354 Kedi snippets were checked individually for parseability using the documented source context (`package.kedi` for manifests). They are **not all complete executable programs**; some intentionally rely on definitions/inputs/models from adjacent prose. Copy full examples from assets for executable starting points. Do not treat parse-only status as execution evidence.

## Handwritten tests

Define top-level `@test: procedure:` with uniquely named `> case:` Python assertions. Call the target explicitly; its suite name does not call it for you. A single backtick line or a fenced block is valid. Shared fixtures belong in the prelude or case setup: normal top-level main initializations are not run as test setup. Cases share a compiled runtime, so avoid mutable/order-dependent fixtures.

```bash
kedi program.kedi --test
```

A failed case exits nonzero, but a procedure calling a real model remains a real model test. `--test` is not intrinsically offline. `--no-cache` controls generated code cache; it does not disable query response memoization or optimized-prefix loading.

## Evaluation

Use one metric per `@eval:` and give it a named dataset. Prefer rows `(input, {"expected_field": expected_value})` so direct evaluation and optimization agree on splitting. Multiple arguments use tuple input; unpack/call explicitly. Other pairs can remain raw rows under direct eval, so do not assume every `(x,y)` creates `expected`.

`> data:` supplies training/fallback rows. Matching `> test_data:` replaces training rows for ordinary `--eval`; it does not print both scores. Metrics return bool, numeric score, or `(score, feedback)`. Row exceptions become zero with error feedback, and an empty set scores zero. A completed zero-score eval can exit 0: inspect scores and feedback and apply the application's threshold.

During optimization, test_data participates in candidate selection and is not untouched final holdout data. Keep a separate final evaluation set. See [datasets](doc-evals-and-optimization-datasets-and-metrics.md), [eval suites](doc-evals-and-optimization-evaluation-suites.md), and [reproducibility](doc-evals-and-optimization-reproducibility.md).

## Prompt optimization

`> optimize: stable_name:` belongs in a procedure and contains one `>>` template block. Each optimized procedure needs a same-named eval with training data and a metric. Kedi stores a prefix while retaining source captures/types. Name changes detach old artifact keys.

`--optimize` defaults to the **mock** optimizer; explicitly select `--optimizer gepa` for GEPA and configure its separate model/budget. `--optimizer-fresh` removes optimized JSON/scores/checkpoints; it is unrelated to `--no-cache`. Inspect corrupted optimized state rather than hiding it. Use [prompt optimization](doc-evals-and-optimization-prompt-optimization.md) and [GEPA](doc-evals-and-optimization-gepa.md).

## Generated procedures

`> auto:` takes an indented natural-language implementation specification, without `>>`. Give a precise signature and independent handwritten boundary tests. The generator emits tests then a Python body, translates/merges/parses them and runs tests. Its escalating retry stages can make more requests than the numerical stage count suggests.

Generated source is cached by procedure name in `program.cache.kedi`; review/regenerate when changing a contract. `--codegen-agent mock` generates placeholders, not a correct implementation of an arbitrary specification. Passing generated tests is not independent evidence that the task was understood correctly. See [auto](doc-evals-and-optimization-ai-generated-procedures.md).

## Changing this skill

Keep source-document names/hashes in provenance. Recheck all added complete examples with both parser and asserted runtime behavior. Preserve expected failures as failures. When documentation and runtime disagree, write a version-specific note and a reproducer; do not silently rewrite normative semantics. Run the skill metadata validator as well as code checks. Test a realistic user task from the skill alone before saving a substantial change.
