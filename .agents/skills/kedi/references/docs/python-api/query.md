# `@kedi.query`

## Docstring Programs

A query function's docstring must start with a standalone `kedi` line:

```python
import kedi


@kedi.query
def title_for(topic: str) -> str:
    """kedi
    >> A guide to <topic> could be titled [title: str].
    = `title`
    """
    ...
```

Kedi removes the header, dedents the docstring, and compiles the remaining text
as the body of an internal procedure. An absent docstring, a non-standalone
header, or an empty program fails when the decorator is evaluated.

The docstring therefore contains procedure-body statements, not another
top-level `@procedure` declaration. Procedure-valid directives such as
`> model:`, `> instructions:`, and `> use:` may appear in the body.

## Function Signatures

The Python signature defines call binding:

```python
@kedi.query
def explain(topic: str, audience: str = "developers") -> str:
    """kedi
    >> An explanation of <topic> for <audience> is [answer: str].
    = `answer`
    """
    ...
```

Python's normal positional, keyword, and default-argument rules apply before
Kedi starts. Every bound argument becomes a runtime value under its parameter
name.

## Ignored Python Bodies

The decorated body is metadata-only:

```python
@kedi.query
def classify(text: str) -> str:
    """kedi
    >> <text> is a [label: Literal["bug", "feature", "question"]].
    = `label`
    """
    raise AssertionError("never executed")
```

The assertion never runs. Prefer `...` to make this ownership obvious.

If the original callable is declared `async def`, Kedi returns an async wrapper
and runs blocking model work in a worker thread. The original async body still
does not run.

## Argument Binding

Arguments enter the environment as native Python values. Use:

- `<name>` to render a value into prompt text;
- `` <`name`> `` when explicit inline Python rendering is useful;
- bare `name` inside Python code to consume the native object.

```python
@kedi.query
def count_items(items: list[str]) -> int:
    """kedi
    = `len(items)`
    """
    ...
```

Configured `env` values have higher precedence than call arguments. Do not
reuse a parameter name in local `env` unless overriding caller input is
intentional.

## Native Return Values

The Kedi return controls the runtime value:

```python
from pydantic import BaseModel


@kedi.type
class Review(BaseModel):
    decision: str
    summary: str


@kedi.query
def review(text: str) -> Review:
    """kedi
    >> A review of <text> is [result: Review].
    = `result`
    """
    ...
```

``= `result` `` returns the native `Review`. `= <result>` renders it and
returns a string. The Python return annotation helps callers and type checkers,
but Kedi does not coerce a mismatched DSL result to that annotation.

## Dynamic Output Types

Type expressions can read Python arguments:

```python
from typing import TypeVar

T = TypeVar("T")


@kedi.query
def extract(text: str, output_type: type[T]) -> T:
    """kedi
    >> The structured representation of <text> is [result: `output_type`].
    = `result`
    """
    ...
```

`output_type` is a native Python class in the runtime environment. The adapter
must support that output schema.

## Per-Query Backend Overrides

Framework adapters use `adapter=`; process-backed harnesses use `agent=`:

```python
@kedi.query(
    adapter="pydantic",
    model="openai:gpt-6-luna",
    system="Return only the requested result.",
    effort="low",
    settings={"timeout": 120},
    requires=("structured_output",),
)
def extract_name(text: str) -> str:
    """kedi
    >> The person in <text> is named [name].
    = `name`
    """
    ...
```

Use `agent="codex"`, `agent="claude"`, or `agent="acp"` for harnesses. Passing
both `adapter` and `agent` is an error. Adapter instances are accepted when
they expose matching `kind` and `shortname` metadata.

`requires` declares adapter capabilities that this callable cannot operate
without. Names use the same canonical vocabulary as native `> requires:` and
are checked before model I/O. Unknown names fail during decoration; unsupported
capabilities fail when the callable resolves its adapter.

## Tools, Environment, and Skills

```python
@kedi.query(
    tools=[search_docs],
    env={"audience": "maintainers"},
    skills=True,
)
def answer(question: str) -> str:
    """kedi
    > use: search_docs
    >> Use the documentation when needed. The answer to <question> is [answer: str].
    = `answer`
    """
    ...
```

Local tools override configured tools with the same registered name. Skills
add the explicit `list_skills` and `read_skill` tools; they do not inject every
skill file into the prompt.

An argument may not have the same name as any registered tool in scope. Kedi
rejects that collision rather than silently making the tool uncallable.

## Approval and MCP Overrides

Queries accept `approval=` and `mcp_servers=`:

```python
from kedi import McpServerSpec


@kedi.query(
    approval="allow",
    mcp_servers=[
        McpServerSpec(
            transport="http",
            url="http://127.0.0.1:8000/mcp",
        )
    ],
)
def investigate(question: str) -> str:
    """kedi
    >> Based on evidence from available tools, the answer to <question> is [answer: str].
    = `answer`
    """
    ...
```

The selected adapter must support the requested capabilities. Local MCP
servers are appended to configured servers.

## Response Caching

`cache=True` opts this callable into the in-memory response cache:

```python
@kedi.query(cache=True)
def stable_summary(text: str) -> str:
    """kedi
    >> A summary of <text> is [summary: str].
    = `summary`
    """
    ...
```

The key includes source, callable identity, arguments, configured and local
environment, backend identity, model/profile settings, tools, MCP, approval,
and skills. Concurrent identical misses coalesce into one producer call.
Failures are not cached.

Cached results are deep-copied on storage and retrieval when possible. Values
that cannot be deep-copied fall back to by-reference behavior.
