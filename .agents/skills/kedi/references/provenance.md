# Snapshot provenance and verification boundaries

## Authoritative inputs

The original skill validation used the supplied documentation for intended semantics and the source snapshot to install the runtime, inspect behavior, and verify examples. These records describe that snapshot. For work in a Kedi checkout, its current language specification and documentation take precedence over this bundled manual; implementation disagreements should still be reported explicitly.

| Input | Identity |
| --- | --- |
| `kedi-docs-main.zip` | SHA-256 `7218c0fee74aebee467e8b1245102e04fa0c98844896aaa7608fbcdb4f9c1f7d` |
| `kedi-stable (2)(1).zip` | SHA-256 `7fd07c8192931e8c8444da1e223284f9b2cc2e33e0f0fea0ad65c6109663706f` |
| Kedi source-reported version | `0.4.0`; the ZIP provides no original Git revision |
| Parser repository | `https://github.com/kedi-lang/tree-sitter-kedi` |
| Parser revision used | `6d36d33848708b099901795d60ca225a8647fa28` |
| Parser reported package version | `0.4.1` (also used by an older published wheel; see pitfalls) |
| Validation date | 2026-10-16 |

The [source-document manifest](../assets/validation/source-documents.json) records SHA-256 identities for all 154 original documentation Markdown pages, not hashes of the adapted local copies. Each local `doc-*.md` names its source. Adaptations include navigation, local link routing, explicit snippet status, and reconciliation notices; the installation and getting-started pages also reflect the current Python 3.11 minimum. Task guides supply the agent-facing workflow/rules.

The complete docs corpus is retained for exact APIs and uncommon use cases. It is not all inserted into context at once. Website fonts, branding, CSS, and JavaScript were not relevant to language authoring and are not bundled. The original reviewed-evidence Python/Kedi assets are included where referenced, with their original integration prerequisites; they are not asserted to have run.

## Verified environment

- CPython 3.12.14.
- Editable install of the supplied Kedi 0.4.0 source.
- Tree-sitter 0.25.2; tree-sitter-kedi 0.4.1 built from the exact revision above; tree-sitter-python 0.25.0.
- Pydantic 2.13.5 and pydantic-ai-slim 2.54.0.
- Exact installed distribution versions are recorded in [environment-versions.json](../assets/validation/environment-versions.json). These are evidence of this run, not permission to overwrite another project's lockfile.

`uv pip install --no-sources` was needed for the source ZIP's empty submodule directories. Building the parser used GCC because this interpreter's configured clang binary was unavailable. Both installation decisions are documented in [environment](environment.md).

## Checks and limits

The [example report](../assets/validation/example-verification.json) records 34 passing checks: both CST/AST parsing of complete example files, deterministic native values/scope/modules/CLI/packages/Python integration, real Kedi tests/eval, controlled structured/raw/model-claim/pipeline/map behavior, real local tool invocation, native child tasks/typed results, eager capabilities, budget failure, and negative syntax/type contracts.

The [documentation report](../assets/validation/documentation-parse.json) records **354/354** Kedi snippets parsed. Six require the literal `package.kedi` filename. Five capability-declaration snippets require the newer grammar revision. No blanket claim is made that all snippets are standalone or runtime-verified.

Live provider calls were disabled in the fixture suite. Provider inference quality, external authentication, live MCP/ACP/A2A services, Jev/Laya, deferred capabilities, GEPA optimization, generated `auto` implementations, editors, debugger, notebook, telemetry, and Harbor execution were not exercised. Those references preserve the supplied documented contracts and prerequisites.

The `auto.kedi` example is parse-only; `budget.kedi` intentionally fails on its second generation. `optimize.kedi` was run with its source prompt and controlled output, not optimized by a live service. Do not relabel these as end-to-end production validations.

Three implementation discrepancies have minimal reproducers in [snapshot-discrepancies.json](../assets/validation/snapshot-discrepancies.json) and documented workarounds in [pitfalls](pitfalls.md). Source/runtime version upgrades require rechecking those observations.

## Fresh-context authoring checks

Two separate authoring tasks were given only this skill and the existing interpreter, without the original source/doc archives:

- Inventory module/CLI: native typed report, source-relative JSON loading, empty-input/threshold/order cases; two files parsed, five handwritten Kedi tests passed, CLI and native host returned the expected report from another working directory.
- Support-ticket workflow: two concurrent typed child profiles each called a real local read-only policy tool; four fixture cases exercised missing/explicit owners, differing recommendations, and rejection of an invented owner. Repeated await reused the same run; hooks confirmed actual tool execution and unchanged policy data.

The second task's initial fixture incorrectly assumed the complete tool catalog contained only its application tool. The runtime also exposed documented artifact and parent-messaging tools. The fixture was corrected to assert required-tool presence and record the full inventory; the skill now explicitly explains this boundary. No application result or expected owner contract was weakened. The final four cases passed after that fixture correction.

Raw task programs, assertions, and observed summaries are preserved in [forward-tasks.json](../assets/validation/forward-tasks.json). These are additional authoring checks, separate from the 34 bundled checks.

## Maintaining the skill

Update source hashes and exact runtime/parser identities when replacing the docs snapshot. Reconcile all altered statements with their source page. Keep complete executable examples separate from fragments and interface sketches. Run both bundled scripts and a realistic fresh-context authoring task after a substantial change. Preserve negative tests and expected outputs; never change them merely to match a regression.
