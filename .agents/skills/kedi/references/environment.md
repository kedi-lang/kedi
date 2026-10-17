# Environment and execution

## Choose an installation

Inspect the project's `AGENTS.md`, `pyproject.toml`, lockfile, `.gitmodules`, and existing virtual environment first. Use its interpreter for both Kedi and validation helpers. Current Kedi requires Python 3.11+, but optional integrations can have narrower requirements. The original validation used the supplied Kedi **0.4.0 source snapshot**; its historical results do not identify an arbitrary package with that version string. Prefer the target checkout's current documentation over the bundled snapshot.

For a new consumer project:

```bash
uv init my-kedi-project
cd my-kedi-project
uv add kedi
uv run kedi -c '= ready'
```

For a pip-managed environment, create/activate a venv then `python -m pip install kedi`. Pin the resolved version and dependencies after testing. Do not upgrade an existing project merely to use this skill.

For a full source checkout, initialize submodules at its recorded gitlinks before resolving local uv sources:

```bash
git submodule update --init --recursive
uv sync
uv run kedi -c '= ready'
```

A ZIP export has no Git history/gitlink revisions and may contain empty submodule directories. `uv pip install -e ./kedi-source` can fail because uv honors `[tool.uv.sources]`. For core execution with published dependencies, use:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --no-sources -e ./kedi-source
source .venv/bin/activate
kedi -c '= ready'
```

This bypasses local dependency overrides, not version constraints. It does not provide unreleased grammar. `pip install -e` also does not implement uv's local-source overrides. Do not edit the project's dependency constraints to make installation appear successful.

## Parser compatibility

Kedi uses `tree-sitter==0.25.2`, `tree-sitter-kedi==0.4.1` and a Python CST-to-AST validator in this snapshot. The published grammar 0.4.1 and repository commit `6d36d33848708b099901795d60ca225a8647fa28` both report 0.4.1, but only the latter parsed the unreleased capability declarations in the supplied manual during validation.

For that documented next-release feature, use a mutually compatible runtime/grammar pair; do not infer compatibility from the version string alone. The exact grammar used here can be built with:

```bash
git clone https://github.com/kedi-lang/tree-sitter-kedi tree-sitter-kedi
git -C tree-sitter-kedi checkout 6d36d33848708b099901795d60ca225a8647fa28
uv pip install --python .venv/bin/python --no-sources -e ./tree-sitter-kedi
```

A source build needs a C compiler and Python headers. If Python's configured compiler is absent but GCC is available on Linux, `CC=gcc CXX=g++` may be supplied to that install command. Do not substitute a different revision silently. See [provenance](provenance.md).

## Parse, then run

```bash
kedi parse program.kedi
kedi program.kedi --parse
kedi program.kedi
kedi program.kedi --test
kedi program.kedi --eval
```

`-c` supports direct source and `--parse`, not tests/evals/optimization. Shell backticks execute command substitution inside double quotes: use a file or single-quoted source, e.g. `kedi -c '= `2 + 3`'`. Parse-only does not run a prelude, import module code, or validate runtime types. Compilation can execute preludes and module initialization; do not describe compilation as a harmless static check.

Ordinary CLI execution prints the main return; diagnostics go to stderr. A successful `--eval` exit does **not** imply a passing score. Tests and evals do not run the normal top-level main statements as fixture setup.

## Supply a model deliberately

Deterministic execution and parsing need no provider credentials. For a real model select both backend and compatible model, using source directives, the CLI, or the host API. A model ID and credentials are separate choices. Install only its provider extra, for example `uv add 'pydantic-ai-slim[openai]'` for an OpenAI-backed Pydantic model; keep Kedi constraints in the resolver.

```bash
kedi program.kedi --adapter pydantic --adapter-model "$KEDI_ADAPTER_MODEL"
```

Set `KEDI_ADAPTER_MODEL` to a model available in the target environment. Avoid relying on the CLI's default Groq model. The manual's named models are snapshot examples, not a guarantee of current availability. The CLI loads dotenv; `kedi.configure()` also has a dotenv path, but direct low-level embedding should load required environment explicitly. Never print credentials during diagnosis.

Program options map into reserved `args`: `--incident-id 42` gives string `args.incident_id`; bare `--dry-run` gives `True`; missing attributes give `None`; duplicate options keep the first value. Convert numbers explicitly. Do not assign `args`.

## Interactive and optional surfaces

Use `kedi --idle`; use `> show:` at its root, not `=`. Sessions retain state and do not roll back earlier side effects after errors. `:dump`, `--record`, and `--load` use strict snapshots; unsupported values fail rather than being dropped.

Read the exact manual page before using an optional surface:

| Surface/submodule | Read |
| --- | --- |
| `tree-sitter-kedi` | This page and [syntax](doc-reference-syntax.md) |
| `kedi-vscode`, `kedi-zed`, `kedi-lsp` | [VS Code](doc-tooling-vscode.md), [Zed](doc-tooling-zed.md), [language server](doc-tooling-language-server.md) |
| `debugger` | [debugger](doc-tooling-debugger.md) |
| `notebook` | [notebook](doc-tooling-notebook.md), [runtimes](doc-tooling-notebook-runtimes.md), [files](doc-tooling-notebook-files.md) |
| `kedi-otel` | [telemetry](doc-runtime-telemetry.md) |
| `kedi-docs` | The offline [catalog](catalog.md) |
| Harbor integration | [Terminal-Bench](doc-tooling-terminal-bench.md); separate dependency environment |

The uploaded `.gitmodules` identifies these repositories, not installation requirements for every Kedi program. Only the parser checkout was fetched and built for this skill. Do not claim the editor, notebook, telemetry, or remote harness was tested.

Sources: [installation](doc-getting-started-installation.md), [CLI](doc-reference-cli.md), [run and parse](doc-tooling-run-and-parse.md), supplied `pyproject.toml`/`.gitmodules`, and local validation.
