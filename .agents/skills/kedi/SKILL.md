---
name: kedi
description: Write, run, test, debug, and explain Kedi (.kedi) programs and Python integrations using the documented language contracts. Use for Kedi syntax, typed LLM extraction, agent profiles, tools and MCP, subagents, workflows, imports and packages, evaluation and optimization, runtime behavior, or Kedi CLI/editor setup. Includes source-grounded references, executable examples, and parser checks.
---

# Kedi

Treat Kedi as an indentation-scoped, Python-integrated language for typed model calls and agent orchestration. Use Python for exact computation; use templates for model judgement or generation. A template does not fetch files, browse, or execute tools unless those capabilities are explicitly supplied.

## Work sequence

1. Read [environment and execution](references/environment.md) and [language rules](references/language.md) before creating a program. Inspect the target project's instructions, environment, Kedi version, parser version, and existing conventions.
2. In a Kedi checkout, `docs/KEDI_DSL.md` is the language specification and `kedi-docs/docs/` the feature documentation. [references/docs/](references/docs.md) mirrors those pages for offline use (kept in sync by CI); prefer the target checkout's copy when it is newer. Check [pitfalls](references/pitfalls.md) for version gates and known runtime defects, and record implementation disagreements rather than redefining the language.
3. Choose the smallest complete [example](references/examples.md) and the relevant topic below. Follow links on demand; do not load the whole documentation.
4. Define native input/output contracts, deterministic operations, model calls, and explicit tool/context boundaries. Configure the intended backend before defining procedures that capture it.
5. Parse with `python "$KEDI_SKILL/scripts/check_kedi.py" path/to/program.kedi` or `kedi parse path/to/program.kedi`. These checks do not execute source. Tree-sitter alone is only a grammar check; the Kedi AST validator also checks structure.
6. Execute deterministic behavior and handwritten tests, then exercise model paths with controlled fixtures. Test with the actual selected provider only when that environment is available and the task calls for it. Parse success or a mock response does not establish live model quality.
7. Report the command, observed result, and any untested integration. Preserve native return types and close owned runtimes. Do not hide failures behind empty strings or claim work ran from a prompt alone.

Set `KEDI_SKILL` to the absolute directory containing this file. Use the same Python environment for Kedi and the scripts. The skill does not install Kedi automatically.

## Essential syntax

```kedi
@total(values: list[int]) -> int:
  = `sum(values)`

[numbers: list[int]] = `[2, 3, 5]`
= `total(numbers)`
```

| Form | Meaning |
| --- | --- |
| `>> ... [field: Type]` | One structured model call; same-indent continuation rows share it |
| `<name>` / `<procedure(args)>` | Read/render a value or call into text |
| `[name] << ...` | Capture the complete raw model response as a string |
| `[name: Type] = ...` / `[name] := ...` | Declare locally / update an existing nearest binding |
| `@name(args) -> Type:` / `~Type(fields)` | Typed procedure / Pydantic-compatible data type |
| ``= `expression` `` | Return a native Python value and end this invocation |
| `= text <value>` / `> show: ...` | Return rendered text / display and continue |
| ``> if: `condition`:`` | Deterministic exact-`bool` branch; the final colon matters |
| `> if: claim <input>` | Model-classified claim; no final colon |
| ``> loop [item]: `items`:`` / `> map:` | Sequential traversal / attached deferred continuation stage |
| `> name: ...` | Directive; configuration bodies use plain keys, profiles use `>` members |

Use explicit annotations for non-string procedure parameters and returns; omitted annotations default to `str`. Keep native values in Python expressions instead of rendering then trying to coerce them. Align Python fence bodies with their surrounding Kedi indentation. Never read a capture earlier in its own template block; start another `>>`.

## Read by task

| Task | Reference |
| --- | --- |
| Install, select the interpreter, run files/CLI/REPL, check syntax | [Environment](references/environment.md) |
| Types, templates, native/rendered values, calls, scope, loops, Python | [Language](references/language.md) |
| Frameworks vs harnesses, profiles, tools, approvals, hooks, MCP, skills | [Agents and tools](references/agents.md) |
| Task/await, task groups, live input, dynamic workflows, child results | [Orchestration](references/orchestration.md) |
| Imports, exports, package manifests, filesystem and bundled modules | [Modules](references/modules.md) |
| Python query/bind, runtime ownership, sessions, executor integration | [Python embedding](references/python.md) |
| History, concurrency, artifacts, caching, budgets, diagnostics | [Runtime](references/runtime.md) |
| Unit tests, evaluation, prompt optimization, code generation | [Validation](references/validation.md) |
| Working recipes and their exact validation level | [Examples](references/examples.md) |
| Common mistakes, documentation inconsistencies, version gates | [Pitfalls](references/pitfalls.md) |
| Full documentation: exact APIs/settings, adapter-specific details, editors, notebook, telemetry, A2A, Jev/Laya, benchmarks | [Documentation index](references/docs.md) |

Search the documentation for exact names: `rg -n 'SubagentUsageLimits|processor_condition' "$KEDI_SKILL/references"`. Read the linked detailed page before using a less common API; do not invent directives or infer an adapter capability from another adapter.

## Included tools

- `scripts/check_kedi.py FILE_OR_DIR ...`: recursively validate `.kedi` files, preserving source paths and `package.kedi` identity; `--tree-only` checks only CST grammar, `--json` emits a report. Missing dependencies and parse failures exit nonzero.
- `scripts/verify_examples.py`: run the bundled examples, negative contract checks, a parse of every documentation snippet, and known-defect checks, without credentials or live provider calls. Uses real Kedi and Pydantic AI `FunctionModel`/`TestModel`; also checks actual tool execution and child results.
- `scripts/sync_docs.py [--check]`: regenerate (or verify) `references/docs/` from `kedi-docs/docs`.
- `assets/examples/`: copyable complete programs; see [examples](references/examples.md) for execution mode and prerequisites. Do not treat isolated documentation snippets as complete applications.

Read [validation](references/validation.md) before changing either checker. Never make a failing check pass by replacing the program's expected behavior with the observed wrong result.
