# Backend Selection

Kedi separates framework adapters from agent harnesses because they expose
different execution and capability contracts.

## Frameworks with `> adapter:`

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna

>> A summary of <document> is [summary: str].
```

Built-in framework shortnames are `pydantic` and `langchain`.
Frameworks are appropriate when Kedi owns structured prompting and registers
typed tools through the framework's model interface.

## Harnesses with `> agent:`

```kedi
> agent: codex
> model: gpt-6-luna
> settings:
    cwd: .
    sandbox: workspace-write

>> The repository evidence suggests that [answer: str].
= <answer>
```

Built-in harness shortnames are `claude`, `codex`, `acp`, and `a2a`. Harnesses are
appropriate when the underlying agent owns its tool loop, repository context,
or protocol session.

## Literal and Dynamic Selection

Literal names are validated by the parser/LSP and runtime:

```kedi
> adapter: langchain
```

Use a backtick expression only when selection is genuinely dynamic:

````kedi
```
selected_backend = "pydantic"
```

> adapter: `selected_backend`
````

Dynamic selection postpones validation and reduces static diagnostics. Prefer a
literal or profile in production source.

## Required Capabilities

Make nonnegotiable backend behavior explicit with `> requires:`:

```kedi
> adapter: langchain
> requires:
    structured_output
    tool_registration
    history_replay
```

Literal selections receive source-located LSP errors when the adapter cannot
meet the contract. Dynamic selections are checked at runtime before model I/O.
Requirements accumulate through lexical scope and applied profiles; an inner
scope cannot erase an outer obligation. See
[Scoping and Capabilities](scoping-and-capabilities.md#explicit-requirements).

## ACP Commands

Select ACP using an explicit command:

```kedi
> agent: acp:
    command: `["uv", "run", "my-acp-agent"]`
```

The command can be plain text or a Python expression evaluating to a string or
sequence of strings. The connection body binds the command to the `acp` harness.

ACP commands are always explicit. Use `> agent: acp:` syntax or construct
`ACPAdapter(command=...)` in Python. Plain `> agent: acp`, CLI command options,
and environment command fallbacks are unsupported.

## A2A Endpoints

Bind a remote A2A harness to its endpoint and credential reference:

```kedi
> agent: a2a:
    endpoint: https://agents.example.com
    auth:
        scheme: bearer
        token_env: RESEARCH_AGENT_TOKEN
```

The remote process owns its model and tool loop, so local `> model:`, `> effort:`,
`> settings:`, `> mcp:`, and tool registration are not forwarded. Generic peers
support raw text; Kedi-served peers can negotiate typed captures. See
[A2A Cloud Agents](../agent-adapters/a2a.md).

## CLI and Environment Defaults

If source does not select a backend, Python configuration or CLI defaults can
provide one. The Python API loads `.env` and recognizes:

| Variable | Meaning |
| --- | --- |
| `KEDI_ADAPTER` | Framework adapter shortname |
| `KEDI_AGENT` | Harness shortname |
| `KEDI_ADAPTER_MODEL` | Model for either selected backend |

`KEDI_ADAPTER` and `KEDI_AGENT` are mutually exclusive.

## Selection Precedence

For a model call, selection priority is:

1. a direct directive in the current lexical scope;
2. a profile applied in that scope;
3. Python/CLI default agent profile;
4. environment-based default selection.

Nested scopes may choose another backend and restore the outer selection when
they exit. Sequential directives can also change backend kinds in the same
scope: the most recent selection applies to following calls. Python/CLI and
environment defaults do not prevent this. Procedures retain the selection
captured where they were defined.

When switching frameworks, Kedi preserves the explicit `> model:` or the initial
framework's string model ID if no model directive was given. The destination
must support that ID. Framework-specific Python model objects, credentials,
and provider clients are not converted to another framework.

An implicit framework model is not transferred to a harness. Explicit model
directives remain in scope; select a compatible model when switching kinds:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna
>> Summarize the repository requirements.

> agent: codex
> model: gpt-6-luna
>> Inspect the repository and summarize its test coverage.
```

Changing backends does not itself transfer conversation history. Configure
history explicitly when calls should share context.

## Invalid Combinations

A declarative `> profile:` cannot contain both `> adapter:` and `> agent:` members.
Unlike executable directives, these describe one backend, not a sequence.
Kedi also rejects
a framework name in `> agent:`, a harness name in `> adapter:`, conflicting
environment defaults, a dynamic value with the wrong type, and an adapter
instance whose declared kind does not match the selected API.

Define separate profiles when you need reusable configurations for both kinds.
