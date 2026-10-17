# Pitfalls, version gates, and documented exceptions

## Contents

- [Compatibility gates](#compatibility-gates)
- [Observed snapshot discrepancies](#observed-snapshot-discrepancies)
- [Frequent authoring mistakes](#frequent-authoring-mistakes)
- [Documentation reconciliation](#documentation-reconciliation)

## Compatibility gates

The source snapshot reports Kedi 0.4.0. The supplied docs include explicitly **unreleased capability composition**. Treat that documentation as its intended contract, not a promise that every 0.4.0 install supports it.

The published `tree-sitter-kedi==0.4.1` parsed 343/354 original documentation snippets before correcting manifest filenames; the five capability declaration snippets failed. Grammar commit `6d36d33848708b099901795d60ca225a8647fa28` still reports 0.4.1 but includes capability declarations. With that commit and the attached Kedi source, all 354 snippets parse when the six manifests use `package.kedi`. Eager capability behavior is also exercised by the bundled runtime suite. This does not test LangChain or deferred capability loading.

A package version string alone cannot identify this grammar difference. Keep an exact source revision or wheel digest where reproducibility matters. Other submodules were not needed to validate core language behavior; do not clone/install all optional integrations by default.

## Observed snapshot discrepancies

These are local observations against the supplied source and the pinned parser, not corrections to the language specification. Preserve the intended contract and use the workaround until the target runtime is verified fixed.

| Intended documentation contract | Observed behavior | Workaround |
| --- | --- | --- |
| Python keyword calls are available for compiled Kedi procedures | Calling an **imported** procedure with a keyword can fail in `_make_generic_proc_wrapper` with unexpected keyword argument | Use positional calls to imported Kedi procedures in this snapshot; keep keyword contracts in Python host functions where tested |
| A nested Python function with `global count` can rebind an existing Kedi global and write back | Two calls incremented the block's Python global but the outer Kedi `count` remained 0 in the tested example | Have the helper return a value, then assign it to the Kedi binding directly in the enclosing Python block; see `python_scope.kedi` |
| An eval metric can use an inline native return; a program can have a main return after validation metadata | A final inline metric return followed by a top-level main return was absorbed into the metric CST and rejected | Use a fenced return block for the metric when a main return follows, or end the file after the suite |

The verification examples retain the intended results (a changed count, correctly imported procedure output, and a real eval score), using these workarounds. They were not changed to accept incorrect runtime behavior.

Minimal reproductions are preserved as data in [snapshot-discrepancies.json](../assets/validation/snapshot-discrepancies.json); they are intentionally not positive examples. Recheck before asserting the limitation still exists on another revision.

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

## Documentation reconciliation

The following reconciliation notes describe the supplied documentation snapshot. In a Kedi checkout, prefer its current language specification and dedicated documentation pages. Recheck these historical observations against the target revision; keep observed implementation facts separately labeled.

- The syntax index loosely says positional and named calls. [Parameters](doc-core-language-parameters-and-returns.md) and [substitutions](doc-core-language-substitutions-and-calls.md) specify positional angle calls and Python keyword calls. The imported-procedure issue above is an additional implementation discrepancy.
- A backend-selection priority list can suggest that direct directives always win. [Scoping](doc-agentic-engineering-scoping-and-capabilities.md) and [profiles](doc-agentic-engineering-profiles.md) explicitly specify source order; a later applied profile can override an earlier direct directive.
- Installation prose advises explicit dotenv loading for embedding. Supplied `api/configuration.py` also calls `dotenv.load_dotenv()` on the configure path. Do not generalize that to every low-level runtime entry point; explicitly load required environment in an embedding host.
- The directive index is not exhaustive of the capability page, run budgets, or all hook events. Read the detailed feature contract and current capability metadata.
- Some source-repository specification examples are illustrative or intentionally incorrect. The dedicated docs archive and verified assets are the primary authoring references here; the source specification was used to understand implementation context, not copied as an unqualified executable cookbook.

When an application needs an unverified integration, preserve its documented schema and report that it has not been executed in that environment. Never silently remove type checks, approvals, child cleanup, or errors to make an example appear to work.
