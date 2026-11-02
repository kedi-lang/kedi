# Pitfalls, version gates, and documented exceptions

## Contents

- [Version gates](#version-gates)
- [Known runtime defects](#known-runtime-defects)
- [Frequent authoring mistakes](#frequent-authoring-mistakes)
- [Documentation reconciliation](#documentation-reconciliation)

## Version gates

Capability composition (`> capability:`) is marked **Unreleased** in the documentation: it needs a Kedi
runtime and `tree-sitter-kedi` grammar new enough to parse capability declarations. A version string alone
does not prove it; parse a capability snippet with the target environment before relying on it. The same
applies to any feature whose doc page carries an "Unreleased" note.

## Known runtime defects

None currently. When documented behavior and the runtime disagree, add a row here (contract, observed
behavior, workaround) and a check in `scripts/verify_examples.py` that asserts the defect still reproduces,
so the check fails once it is fixed.

## Frequent authoring mistakes

| Mistake | Correct approach |
| --- | --- |
| Bare ordinary prompt line | Start a template with `>>` |
| Read a capture in a continuation row of its own block | Start a second `>>` call |
| Use `= <model>` for an object result | Use a native backtick return and explicit procedure return type |
| Omit the return annotation for an int/list/model | Declare `-> int`, `-> list[T]`, or the named model |
| Pass `5` as text to an int parameter | Pass a native backtick expression or use a Python call |
| Use angle-call `name=value` | Angle calls are positional; use Python keyword syntax where supported |
| Put Python imports in an ordinary block and expect later visibility | Use the initial prelude or return a value into a Kedi binding |
| Rebind an outer variable with `[x] = ...` inside a branch/loop | Use `[x] := ...` |
| Add a type to `:=` or assign a missing name | Declare first; assignment preserves the owner contract |
| Omit the trailing colon on a deterministic condition | Use an exact-bool Python expression followed by `:` |
| Use bare backticks in a model-classified condition | Use angle-wrapped Python substitution; no output captures |
| Expect map to build a list or preserve completion order | Provide explicit storage; use disjoint indexed writes |
| Apply model/tools after defining a procedure and expect recapture | Configure before the definition or inside its body |
| Confuse a profile with an executing child | Start a task or a delegated invocation explicitly |
| Access native `result.final_result` after `> await` | Native result uses `.output`; delegated tool envelopes use `final_result` |
| Combine a child's output contract with task captures | Choose one source of task schema |
| Leave child work unawaited | Await every admitted task before the invocation ends |
| Use a named package manifest other than `package.kedi` | Preserve the exact basename |
| Assume package dependencies were pip-installed | Install Python dependencies separately |
| Treat `--eval` exit 0 as quality acceptance | Check scores and feedback against a threshold |
| Treat `--optimize` as GEPA by default | Explicitly choose `--optimizer gepa` |
| Assume mock codegen implements the requested procedure | Review generated code and independent tests; mock is a fixture |
| Treat `DefaultExecutor` as sandboxed | Ordinary Python has host authority |
| Install this multi-file skill using `kedi skills add` | That installer copies only SKILL.md; use a host supporting the complete skill directory |

## Documentation notes

The language specification is `docs/KEDI_DSL.md` in a Kedi checkout; the pages under
[docs/](docs/INDEX.md) mirror `kedi-docs` and give each feature's detailed contract. Where they disagree with
observed behavior, report the disagreement rather than redefining the language.

- The syntax reference says procedures support positional and named calls. Angle calls (`<f(a, b)>`) are
  positional; named arguments use Python keyword calls ([parameters](docs/core-language/parameters-and-returns.md),
  [substitutions](docs/core-language/substitutions-and-calls.md)).
- Backend selection follows source order: a later applied profile can override an earlier direct directive
  ([scoping](docs/agentic-engineering/scoping-and-capabilities.md), [profiles](docs/agentic-engineering/profiles.md)).
- `kedi.configure()` and the CLI load dotenv; low-level embedding entry points do not. Load required
  environment explicitly in an embedding host.
- The directive reference is not exhaustive for capabilities, run budgets, or every hook event; read the
  feature page.

When an application needs an unverified integration, preserve its documented schema and report that it has not been executed in that environment. Never silently remove type checks, approvals, child cleanup, or errors to make an example appear to work.
