# Contributing

Kedi uses `uv` for dependency management and tool execution. Use the repo commands below
instead of ad hoc `pip`, `poetry`, or one-off local flows.

## Prerequisites

- Python `3.11+`
- `uv`
- `git`

## Clone And Install

```bash
git clone https://github.com/kedi-lang/kedi
cd kedi
uv sync --dev
```

`uv sync --dev` installs Kedi with the development tools used by CI: `ruff`, `ty`,
`basedpyright`, `pytest-cov`, and `pre-commit`.

To work on or run the Claude Agent SDK adapter, install its optional dependency too:

```bash
uv sync --extra claude
```

You usually do not need to activate `.venv` manually because the commands below use
`uv run` through the `Makefile`.

## Pre-commit

Install hooks with:

```bash
uv run --group dev pre-commit install
```

Run the hook manually with:

```bash
uv run --group dev pre-commit run --all-files
```

Current hook behavior:

- runs `make prod`
- `make prod` runs `make check` and `make check-coverage`
- `make check` runs `ruff check`, `ruff format --check`, `ty check src`, and `basedpyright src`
- `make check-coverage` runs the full test suite with 100% coverage enforcement

The hook is intentionally strict and can take longer than a lightweight formatter hook. If it
fails, fix the reported issue, re-stage any changed files, and commit again.

If you change `.pre-commit-config.yaml`, reinstall hooks with:

```bash
uv run --group dev pre-commit install
```

## Common Commands

Formatting and validation:

- `make format`: format Python code with `ruff format`
- `make check-formatted`: verify formatting with `ruff format --check`
- `make lint`: run `ruff check`
- `make ty`: run `ty check src`
- `make basedpyright`: run `basedpyright src`
- `make check`: run lint, format check, `ty`, and `basedpyright`

Tests and coverage:

- `make tests`: run the full test suite with `pytest`
- `make coverage`: run tests with coverage reporting
- `make check-coverage`: enforce 100% coverage (HTTP cassettes replay with `--record-mode=none`)

LLM integration tests in `tests/test_kedi_suite.py` use [cassetter](https://pypi.org/project/cassetter/).
CI replays committed cassettes under `tests/cassettes/` without network access. To refresh
them after changing `tests/files/*.kedi`:

```bash
export GROQ_API_KEY=...   # real key required when cassettes are missing
rm -rf tests/cassettes/test_kedi_suite   # optional: force a fresh record
uv run pytest tests/test_kedi_suite.py::test_kedi_program_scoping -xvs
```

Cassette redaction lives in `tests/cassette_redaction.py` (headers, response IDs, rate limits,
client fingerprint, and `recorded_at` normalization). Review that module before widening what gets
committed.

Higher-cost validation:

- `make check-matrix`: run lint and type checks across the configured Python matrix
- `make prod`: run `make check` and `make check-coverage`

## Recommended Local Workflow

For most changes:

1. Sync dependencies with `uv sync --dev`.
2. Make your change.
3. Run `make format`.
4. Run `make check`.
5. Run `make check-coverage`.
6. Run `uv run --group dev pre-commit run --all-files` if you want the exact commit gate before committing.

Run the stricter matrix gate when the change is release-sensitive:

```bash
make check-matrix
```

## Pull Requests

Before opening a pull request:

1. Make sure your branch is up to date with `stable`.
2. Ensure `make prod` passes locally.
3. Commit with a clean pre-commit result.
4. Push your branch and verify GitHub Actions passes.

## Notes

- Prefer `uv run ...` or `make ...` over calling tools directly.
- Do not add contributor instructions that depend on `pip`, `poetry`, `pdm`, or `pip-tools`.
- Keep generated caches, coverage reports, and local virtual environments out of commits.
