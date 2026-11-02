# `@kedi.bind`

## File-Backed Programs

`bind` connects a Python signature to a complete `.kedi` program:

```python
import kedi


@kedi.bind(file="summarize.kedi")
def summarize(topic: str, audience: str = "developers") -> str:
    ...
```

`summarize.kedi`:

```kedi
>> A summary of <topic> for <audience> is [summary: str].
= `summary`
```

Unlike a query docstring, the file is a full top-level program. It may declare
types, procedures, profiles, imports, tests, and a main return. `bind` executes
that main program.

## Relative File Resolution

A relative `file=` path resolves from the Python source file that defines the
decorated function, not from the process working directory:

```text
project/
├── app.py
└── programs/
    └── summarize.kedi
```

```python
@kedi.bind(file="programs/summarize.kedi")
def summarize(topic: str) -> str:
    ...
```

Absolute paths are used as given. In interactive environments where Python
cannot identify the defining source file, a relative path falls back to the
current working directory.

## Signature Ownership

Python binds arguments and applies defaults before running the Kedi file:

```python
@kedi.bind(file="report.kedi")
def report(records: list[dict[str, object]], format: str = "markdown") -> str:
    ...
```

The file receives native `records` and `format` globals. Its top-level return
becomes the Python call result. As with `query`, the Python return annotation is
for static typing; the Kedi main return determines the actual value.

## Ignored Stub Bodies

The decorated Python body never executes. Use `...`:

```python
@kedi.bind(file="report.kedi")
def report(topic: str) -> str:
    ...
```

Decorating an `async def` produces an async wrapper that offloads Kedi's
blocking run to a thread.

## Reload on Source Changes

By default, Kedi reads and parses the file when the decorator is evaluated:

```python
@kedi.bind(file="report.kedi")
def report(topic: str) -> str:
    ...
```

Use `reload=True` during development or for intentionally dynamic source:

```python
@kedi.bind(file="report.kedi", reload=True)
def report(topic: str) -> str:
    ...
```

With reload enabled, every call rereads the file. Parsing is still cached by
the exact source hash, so unchanged content does not create duplicate parse
entries.

## Profile Overrides

`bind` accepts the same per-callable profile options as `query`:

```python
@kedi.bind(
    file="report.kedi",
    adapter="langchain",
    model="openai:gpt-6-luna",
    system="Cite the supplied records.",
    effort="medium",
    settings={"timeout": 120},
    requires=("structured_output", "history_replay"),
)
def report(records: list[str]) -> str:
    ...
```

Use `agent=` instead of `adapter=` for Claude, Codex, ACP, or A2A harnesses. The
two selection parameters are mutually exclusive. `requires` uses canonical
adapter capability names and fails before model I/O when the selected backend
cannot honor the file-backed callable's contract.

## Tools and Environment

```python
@kedi.bind(
    file="answer.kedi",
    tools=[search_docs],
    env={"audience": "operators"},
    skills=True,
)
def answer(question: str) -> str:
    ...
```

The file must include `> use: search_docs` where the model should see the tool.
Local `env` overrides function arguments and configured environment values.
Argument/tool name collisions fail before compilation.

`mcp_servers=` and `approval=` are also available and merge through the same
profile rules as `query`.

## Response Caching

`cache=True` caches the completed main result:

```python
@kedi.bind(file="report.kedi", cache=True, reload=True)
def report(topic: str) -> str:
    ...
```

With `reload=True`, the file's current source hash is part of the key, so a
source edit produces a cache miss. The key also includes arguments,
environment, backend/profile configuration, and tool names.

## Missing and Invalid Files

Without `reload=True`, missing, unreadable, or invalid source fails while the
defining module is imported and the decorator runs. With reload enabled, the
initial decoration still requires a valid file; later read or parse failures
surface on the call that observes the change.

The parser cache is not a stale-source fallback. Invalid new content raises
instead of reusing the last valid program.
