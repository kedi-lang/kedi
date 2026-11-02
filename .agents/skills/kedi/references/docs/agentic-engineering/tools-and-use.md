# Tools and `> use:`

`> use:` exposes a callable as an agent tool, applies a profile, or composes a
[capability](capabilities.md). Its block form resolves each listed target in order.
Profiles override configuration; capabilities add owned tools and instructions.

## Expose a Procedure

````kedi
```
release_index = {"1.4.0": "Adds typed child results and fixes cancellation cleanup."}
```

@lookup_release(version: str) -> str:
  ###
  Return release notes for one exact version.
  ###
  > tool:
      name: lookup_release
      risk: read_only
  = `release_index[version]`

> use: lookup_release
> approval: allow

>> A summary of release 1.4.0 is [answer: str].
= <answer>
````

This fixture exposes one local lookup, marks it read-only, and explicitly
permits its invocation.

Kedi converts the procedure signature and docstring into a tool name,
description, JSON argument schema, and validated callable. Custom Kedi types
become nested schemas. Defaults remain optional arguments.

Write useful procedure docstrings before exposing a tool. The model must know
what the tool does, what its arguments mean, and what result it returns.

## Native Tool Metadata

A procedure may contain one `> tool:` declaration after its optional leading
docstring and before executable statements:

```kedi
@fetch_release(version: str) -> str:
  > tool:
      name: lookup_release
      description: Read notes for one exact release version.
      risk: read_only
      retries: 2
      retry_on:
          TimeoutError
          ConnectionError
  = `release_client.fetch(version)`
```

The fields are:

| Field | Contract |
| --- | --- |
| `name` | Model-facing alias; the source procedure keeps its original name |
| `description` | Model-facing description; omission uses the procedure docstring |
| `risk` | `read_only`, `mutating`, or `sensitive`; default `mutating` |
| `retries` | Nonnegative retry count; total attempts are at most `retries + 1` |
| `retry_on` | Visible `Exception` class names eligible for retry |

Retries wrap only procedure-body failures. Argument and result validation,
hooks, approval, cancellation, `KeyboardInterrupt`, and `SystemExit` are not
retried. Each native retry starts from a fresh copy of the validated arguments.
Omitting `retry_on` makes ordinary `Exception` failures eligible.
The exception classes are captured when the procedure is defined. Reassigning
an exception name later does not change an existing procedure's policy.

## Single-Line Resolution

For `> use: name`, Kedi resolves in this order:

1. a visible Kedi procedure;
2. a visible Python callable;
3. a profile.

A procedure or callable therefore wins over a profile with the same name.
Avoid collisions even though the resolution is deterministic.

Backtick names are accepted for syntax symmetry:

```kedi
> use: `lookup_release`
```

They still resolve a name; this is not an arbitrary tool expression.

## Multiline Tool Lists

The block form always lists tools and never applies profiles:

```kedi
> use:
    lookup_release
    current_time
```

Every entry must resolve to a Kedi procedure or Python callable. Use this form
when a scope intentionally exposes several tools.

## Python Callables

A callable introduced by the prelude, imports, Python API, or configured
environment can be registered:

````kedi
```
from datetime import datetime
from zoneinfo import ZoneInfo
import kedi

@kedi.tool(risk="read_only")
def current_time(*, timezone: str) -> str:
    """Return the current time for one IANA timezone."""
    return datetime.now(ZoneInfo(timezone)).isoformat()
```

> use: current_time
````

Python tool functions are invoked with keyword arguments. Complete annotations
and a docstring produce the best schema. Variadic or weakly typed signatures
reduce validation and model reliability.

## Tool Results

Tools return their native procedure or Python value. Framework adapters encode
that value into their tool-result protocol. A model-facing tool should return a
small, serializable object or text with enough context to interpret it.

Do not return open file handles, generators, process objects, or enormous
payloads. Use custom types or dictionaries for structured results and include
error context by raising a precise exception rather than returning an ambiguous
sentinel.

## Registration Scope

Tool frames are lexical:

```kedi
@outer_tool(query: str) -> str:
  = outer

> use: outer_tool

@inner() -> str:
  @outer_tool(query: str) -> str:
    = inner
  > use: outer_tool
  >> Based on the available tool's result, the answer is [answer: str].
  = <answer>
```

The inner registration shadows only inside `inner`; leaving the procedure
restores the outer binding.

Profiles imported from modules retain their private bound tool implementations.
An unbound later registration with the same name deliberately restores normal
caller-scope lookup.

## Risk and Approval

Custom Kedi procedures and Python tools default to `mutating`. Native
procedures use `> tool:` and Python callables use `@kedi.tool(...)` to declare
`read_only`, `mutating`, or `sensitive`. Adapter-owned tools may also attach an
argument-aware resolver that only elevates risk. Every risky invocation is
processed through the active approval policy before execution.

See [Approvals](approvals.md) for defaults and edited-argument validation.

The protected path is canonicalization, pre-tool hooks, risk/approval checks,
edit revalidation, execution, then a success/failure hook. Registration does
not execute a tool, and a model's decision to call one does not authorize it.
For model-provided justifications, see [Tool Reasons](tool-reasons.md).

## Adapter Support

Dynamic tool registration requires backend support. Framework adapters commonly
support Kedi tools; a harness may own a closed native tool surface and be unable
to accept external functions. The LSP reports a capability warning and the
adapter must not silently imply the tool is available.

Test each production profile with its actual backend. A syntactically valid
`> use:` is not evidence that every adapter can register it.
