# Environment and execution

## Choose an installation

Inspect the project's `AGENTS.md`, `pyproject.toml`, lockfile, `.gitmodules`, and existing virtual environment first. Use its interpreter for both Kedi and validation helpers. Current Kedi requires Python 3.11+, but optional integrations can have narrower requirements. Call that interpreter by path (`.venv/bin/python`, `uv run`): a shell alias such as `python` → a system Python can silently load an older Kedi or grammar and produce misleading parse errors.

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

## Parser compatibility

Kedi pins `tree-sitter`, `tree-sitter-kedi` and `tree-sitter-python` exactly in its `pyproject.toml`; in a
source checkout the grammar is the `tree-sitter-kedi` submodule, installed editable. Use the pinned pair.
Newer syntax (capabilities, conditional profile members, `> workflow:` blocks) needs a matching grammar;
check with `python -c "import importlib.metadata as m; print(m.version('kedi'), m.version('tree-sitter-kedi'))"`
and by parsing the relevant documentation snippet. Building the grammar from source needs a C compiler and
Python headers. Do not substitute a different grammar revision silently.

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

Set `KEDI_ADAPTER_MODEL` to a model available in the target environment. The CLI default (`openai/gpt-6-luna`, see [CLI](docs/reference/cli.md)) and the documentation's model names are examples, not a guarantee of availability in the target environment. The CLI loads dotenv; `kedi.configure()` also has a dotenv path, but direct low-level embedding should load required environment explicitly. Never print credentials during diagnosis.

Program options map into reserved `args`: `--incident-id 42` gives string `args.incident_id`; bare `--dry-run` gives `True`; missing attributes give `None`; duplicate options keep the first value. Convert numbers explicitly. Do not assign `args`.

## Interactive and optional surfaces

Use `kedi --idle`; use `> show:` at its root, not `=`. Sessions retain state and do not roll back earlier side effects after errors. `:dump`, `--record`, and `--load` use strict snapshots; unsupported values fail rather than being dropped.

Read the exact documentation page before using an optional surface:

| Surface/submodule | Read |
| --- | --- |
| `tree-sitter-kedi` | This page and [syntax](docs/reference/syntax.md) |
| `kedi-vscode`, `kedi-zed`, `kedi-lsp` | [VS Code](docs/tooling/vscode.md), [Zed](docs/tooling/zed.md), [language server](docs/tooling/language-server.md) |
| `debugger` | [debugger](docs/tooling/debugger.md) |
| `notebook` | [notebook](docs/tooling/notebook.md), [runtimes](docs/tooling/notebook-runtimes.md), [files](docs/tooling/notebook-files.md) |
| `kedi-otel` | [telemetry](docs/runtime/telemetry.md) |
| `kedi-docs` | Mirrored offline in [docs/](docs.md) |
| Harbor integration | [Terminal-Bench](docs/tooling/terminal-bench.md); separate dependency environment |

`.gitmodules` lists these repositories; they are not installation requirements for every Kedi program. Do not claim an editor, notebook, telemetry, or remote harness was tested unless you ran it.

Sources: [installation](docs/getting-started/installation.md), [CLI](docs/reference/cli.md), [run and parse](docs/tooling/run-and-parse.md).
