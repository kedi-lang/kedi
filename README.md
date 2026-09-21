# Kedi

[![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/kedi-lang/kedi/actions/workflows/tests.yml)

## A typed reasoning layer for LLM orchestration

Kedi is a compact DSL for writing typed LLM workflows. It combines prompt-like
template lines, Python interop, Pydantic-backed types, per-procedure model
routing, imports/exports, and test or eval blocks in one readable source file.

### Quick Taste

Typed outputs and Python can live in the same procedure:

````kedi
~Review(decision: str, issues: list[str], summary: str)

@review_code(language: str, code: str) -> Review:
  [line_count: int] = ```
  return len([line for line in code.splitlines() if line.strip()])
  ```

  >> Review this <language> snippet with <`line_count`> non-empty lines.
  The code: 
  <code>
  The review:
  [review: Review]

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
  Write a release-note style summary for the review: 
  <review>.
  Summary of the review above: [summary]

  = `summary`
````

### Install

Kedi is uv-first:

```bash
uv sync --dev
uv run kedi your_program.kedi
```

Enable the Claude Agent SDK harness in a source checkout with:

```bash
uv sync --extra claude
```

Pip fallback:

```bash
pip install -e .
kedi your_program.kedi
```

### Python API

Embed Kedi directly in Python when you want typed LLM calls to feel like normal
functions. `@kedi.query` reads Kedi from the function docstring, `@kedi.bind`
binds a `.kedi` file to a Python callable, and `@kedi.type` makes Python types
available to Kedi outputs.

```python
import kedi


@kedi.type
class Review:
    decision: str
    summary: str


@kedi.tool(description="Search local project notes.")
def search_docs(query: str) -> str:
    return "Kedi embeds typed prompt programs in Python."


kedi.configure(
    model="openrouter:google/gemini-3-flash-preview",
    adapter="pydantic",
    tools=[search_docs],
    env={"audience": "maintainers"},
)


@kedi.query(cache=True)
def review_snippet(language: str, code: str) -> Review:
    """kedi
    > use: search_docs
    >> Review this <language> snippet for <audience>.
    Review: [review: Review].
    = `review`
    """
    ...
```

Use `kedi.context(...)` for temporary profile/env overrides, `cache=True` for
call-result caching, `kedi.cache_info()` / `kedi.clear_cache()` for cache
inspection, and `@kedi.bind(file="program.kedi", reload=True)` when you want a
Python function backed by an editable Kedi source file.

Use `kedi.interactive()` when an editor, notebook, or terminal needs to execute
complete fragments incrementally without replaying earlier code. Definitions,
values, imports, profiles, conversation state, and artifacts persist for the
session:

```python
with kedi.interactive() as session:
    session.execute("[base: int] = `40`")
    session.execute("@add_two() -> int:\n    = `base + 2`")
    assert session.execute("= `add_two()`") == 42
```

Fragments return native values and are executed exactly once. The session is
process-local and non-transactional; completed state and side effects remain
visible after a later fragment fails.

Install the checked-out notebook extra and start the same runtime in one command:

```bash
uv run --extra notebook kedi notebook
```

The notebook is currently a local-build surface; the command uses this Kedi
checkout and its `notebook` submodule rather than a published package.

The notebook opens on localhost and uses browser-owned Pyodide by default.
Choose a discovered host Python from the runtime menu when cells should use
the host. Kedi creates a persistent, project-specific notebook virtualenv from
that interpreter and installs the checked-out Kedi runtime into it; the base
Python environment is not modified. Add an interpreter explicitly at serve
time with `kedi notebook --python /path/to/python`; the option is repeatable.
The package action beside the host runtime installs additional requirements
into this managed environment and reuses them on later notebook sessions.
Pyodide starts loading as soon as the page opens and remains alive across
cells. Notebook cells execute incrementally and are never replayed
automatically. A cell beginning with `!` runs as a terminal cell; browser mode
supports package installation with `!pip install` and `!uv add`, while host
mode runs commands in the notebook working directory. Terminal output streams
under the source while a command runs. Executed cells remain editable and
rerunnable, and execution does not create an empty cell automatically.
Core Pyodide, Python standard-library, Micropip, and Pydantic assets are shipped
with the notebook package; browser Python does not need a runtime CDN download.

Use `adapter=` for framework backends (`pydantic`, `dspy`, `langchain`) and
`agent=` for harness backends (`claude`, `codex`, `acp`). Both can be selected
globally with `kedi.configure`, temporarily with `kedi.context`, or for one
`@kedi.query` / `@kedi.bind` callable together with a local `model=` override.

### Agent Profiles

Route different procedures to different models, settings, system prompts, MCP
servers, and tool scopes without repeating adapter configuration. Profiles are
captured when a procedure is defined; later directives do not retroactively
change earlier procedures.

````kedi
> profile: fast:
    > model: groq:qwen/qwen3-32b
    > effort: minimal
    > settings:
        temperature: 0.2
> profile: quality:
    > agent: codex
    > model: openrouter/google/gemini-3-flash-preview
    > effort: high
    > system: Be precise and use the available tools when useful.
    > mcp:
        command: vsh
        args: `["run", "--mcp"]`

@draft(prompt: str) -> str:
  > use: fast
  >> Summarize: <prompt>
  Summary: [summary]
  = `summary`

@polish(text: str) -> str:
  > use: quality
  >> Improve the prompt: <text>
  Polished: [polished]
  = `polished`
````

- `> model: name` — set the model for subsequent procedure captures (plain name or `` `expression` ``)
- On Python 3.11+, `> adapter: pydantic` with `> model: codex/<model>` uses the optional
  `kedi[codex-model]` bridge and the user's existing `codex login` credentials.
- `> model: typesafe/<model>` selects the optional Jev bridge with either the Pydantic or
  LangChain adapter. Install `kedi[typesafe]` for Pydantic or `kedi[langchain,typesafe]` for
  LangChain. Jev supports typed classification and finite-choice outputs rather than free-form
  text.
- `> agent: name` — select an agent harness such as `codex` or `claude`; typed
  `> agent: acp:` and `> agent: a2a:` bodies bind local ACP processes or remote
  A2A cloud agents. Install `kedi[a2a]` for A2A client/server support.
- `> effort: minimal|low|medium|high|xhigh|max` — set reasoning effort
- `> system:` — set plain or multiline agent instructions with `<name>` and ``<`python`>`` substitutions
- `> settings:` — set adapter-filtered model settings; use backticks for Python values like `` `["END"]` ``
- `> mcp:` — load MCP tools; string fields can be plain strings or inline Python like `` `os.getenv("STDIO_COMMAND")` ``
- `> profile: name:` — define a reusable profile with nested `> model:`, `> effort:`, `> system:`, `> settings:`, `> mcp:`, and/or `> use:` members
- `> use: name` or `> use: \`name\`` — register a **procedure** as an agent tool when the name matches a procedure; otherwise merge a profile by **profile name** (not the importing module name)
- `> skills: enabled` — expose on-demand `list_skills` and `read_skill` tools.
  Discovery prefers the user Kedi registry, then project-local skills, then
  user-global skills; the expanded directive controls source merging and limits.
- Multiline `> use:` lists procedure tools explicitly:

````kedi
> use:
    web_browse
    `web_fetch`
````

Profiles and tool scopes can be shared across files with `> export:` / `> import:` like procedures and types.

### Common Commands

```bash
kedi your_program.kedi
kedi --idle
kedi -c "= done"
kedi -p -c "@broken("
kedi parse your_program.kedi
kedi your_program.kedi --parse
kedi your_program.kedi --test
kedi your_program.kedi --eval
kedi your_program.kedi --eval --optimize --optimizer gepa
kedi your_program.kedi --adapter pydantic --adapter-model groq:qwen/qwen3-32b
kedi skills add --path ~/my-skill
kedi skills add --repo owner/skill-repository
kedi-lsp   # language server for editors
```

`kedi --idle` opens the incremental terminal REPL. It uses `+++ ` for complete
fragments, `... ` for continuation lines, Tab for indentation, and
`:show <expression>` to inspect values. Use `:exit`, `Ctrl+C`, or `Ctrl+D` to
leave the session. These terminal commands do not add REPL-only syntax to Kedi
source files. See [Incremental Execution](docs/KEDI_DSL.md#incremental-execution)
for lifecycle, import, traceback, and concurrency semantics.

Extra CLI arguments are available inside Python expressions and blocks as `args`:

```bash
kedi report.kedi --country Turkiye --dry-run
```

```kedi
@destination_label() -> str:
  = `f"{args.country}:{'dry-run' if args.dry_run else 'live'}"`
```

### Core Syntax

- Procedures: `@name(param[: type], ...) -> ret_type:`
- Types: `str`, `int`, `float`, `bool`, `list[T]`, `dict[K, V]`, custom `~Type`
- Untyped procedure and type fields default to `str`
- Template blocks: `>>` opens an LLM prompt; continuation lines merge into one run
- Template outputs: `Answer: [answer: int]`
- Substitutions: `<value>`, `<call(arg)>`, and inline Python with backticks
- Python blocks: triple-backtick blocks that return native Python values
- Agent routing: `> model:`, `> profile:`, and `> use:` (see above)
- Modules: whole or selective `> import: package/module`, `> export: name`, and `> export: *`
- Validation: `@test:` and `@eval:` blocks live next to the code they verify
- Optimization: `> optimize: name:` marks template spans for `--optimize`

### Developer Notes

- `make check`: ruff, format check, `ty check src`, and `basedpyright src`
- `make check-coverage`: pytest with 100% coverage enforcement
- `make prod`: full local gate used by pre-commit
- Editor support: `kedi-lsp` (VS Code extension in `kedi-vscode/`, Zed extension in `kedi-zed/`)

Kedi can also expose procedure metadata as JSON schema and Pydantic AI tools for
tool-registration workflows.

See [docs/KEDI_DSL.md](docs/KEDI_DSL.md) for the full language specification and
[CONTRIBUTING.md](CONTRIBUTING.md) for local development workflow.
