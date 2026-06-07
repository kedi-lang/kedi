# Kedi

[![coverage](https://img.shields.io/github/actions/workflow/status/kedi-lang/kedi/tests.yml?branch=stable&label=coverage)](https://github.com/kedi-lang/kedi/actions/workflows/tests.yml)

## A typed reasoning layer for LLM orchestration

Kedi is a compact DSL for writing typed LLM workflows. It combines prompt-like
template lines, Python interop, Pydantic-backed types, imports/exports, and test
or eval blocks in one readable source file.

### Quick Taste

Typed outputs and Python can live in the same procedure:

````kedi
~Review(decision: str, issues: list[str], summary: str)

@review_code(language: str, code: str) -> Review:
  [line_count: int] = ```
  return len([line for line in code.splitlines() if line.strip()])
  ```

  Review this <language> snippet with <`line_count`> non-empty lines.
  Return a structured review as [review: Review].

  = `review`
````

Kedi files can also behave like modules:

````kedi
# reviews.kedi
~Issue(line: int, severity: str, message: str)
~Review(decision: str, issues: list[Issue], summary: str)

> export:
  Issue
  Review
````

````kedi
# report.kedi
> import: reviews

@summarize_review(review: Review) -> str:
  Write a release-note style summary for <review>.
  Summary: [summary]
  = `summary`
````

### Install

Kedi is uv-first:

```bash
uv sync --dev
uv run kedi your_program.kedi
```

Pip fallback:

```bash
pip install -e .
kedi your_program.kedi
```

### Common Commands

```bash
kedi your_program.kedi
kedi your_program.kedi --test
kedi your_program.kedi --eval
kedi your_program.kedi --eval --optimize --optimizer gepa
kedi your_program.kedi --adapter-model groq:qwen/qwen3-32b
```

Extra CLI arguments are available inside Python expressions and blocks as `args`:

```bash
kedi report.kedi --country Turkey --dry-run
```

```kedi
@destination_label() -> str:
  = `f"{args.country}:{'dry-run' if args.dry_run else 'live'}"`
```

### Core Syntax

- Procedures: `@name(param[: type], ...) -> ret_type:`
- Types: `str`, `int`, `float`, `bool`, `list[T]`, `dict[K, V]`, custom `~Type`
- Untyped procedure and type fields default to `str`
- Template outputs: `Answer: [answer: int]`
- Substitutions: `<value>`, `<call(arg)>`, and inline Python with backticks
- Python blocks: triple-backtick blocks that return native Python values
- Modules: `> import: module_name`, `> export: name`, and `> export: *`
- Validation: `@test:` and `@eval:` blocks live next to the code they verify

### Developer Notes

- `make check`: ruff, format check, `ty`, and `basedpyright`
- `make check-coverage`: pytest with 100% coverage enforcement
- `make prod`: full local gate used by pre-commit

Kedi can also expose procedure metadata as JSON schema and Pydantic AI tools for
tool-registration workflows.

See [docs/KEDI_DSL.md](docs/KEDI_DSL.md) for the full language specification and
[CONTRIBUTING.md](CONTRIBUTING.md) for local development workflow.
