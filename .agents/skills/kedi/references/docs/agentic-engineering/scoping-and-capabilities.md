# Agent Scope and Adapter Requirements {#scoping-and-capabilities}

Agent state follows lexical source scope. Capability metadata determines whether
the selected adapter can honor the resulting contract.

Adapter capability metadata describes backend support. It is distinct from
[composable capabilities](capabilities.md), the `> capability:` language surface
for reusable instructions and owned tools.

## Lexical Agent State

```kedi
> adapter: pydantic
> model: default-model

@quick(text: str) -> str:
  > effort: low
  >> A summary of <text> is [summary: str].
  = <summary>
```

The effort override exists only inside `quick`; the outer adapter and model are
inherited. On return, the prior state is restored.

## Top-Level Capture

Top-level directives are captured by procedures declared after them. A later
top-level change does not retroactively change a previously declared procedure:

```kedi
> model: first-model

@first() -> str:
  >> A brief answer is [answer: str].
  = <answer>

> model: second-model

@second() -> str:
  >> A brief answer is [answer: str].
  = <answer>
```

This source-order rule also applies to tools, MCP, instructions, and profiles.

## Precedence

Start with the configured defaults and the captured lexical state. Then apply
profile applications and direct directives in source order in the current
scope. They are not separate priority tiers:

```kedi
> profile: concise:
    > instructions: Return one sentence.

> instructions: Return a paragraph.
> use: concise
```

Here the profile's one-sentence instruction wins because the profile is applied
last. Reversing those last two statements makes the paragraph instruction win.
Collection members follow the [profile merge rules](profiles.md#merge-rules).

Calling a previously defined procedure does not recapture the caller's model or
tools. To make a procedure use different configuration, declare the override in
its body or define it under that configuration.

## Tool Frames

Tool registration is block-scoped. Inner registrations can shadow an outer tool
with the same name; leaving the block restores the previous binding. A procedure
tool receives its schema from that procedure's signature and docstring.

The same profile used in two scopes does not make mutable tool state global.
Each invocation materializes its active tool surface.

## Configuration Scope and Value Scope

Agent configuration and variable bindings are related but separate. An `if` or
loop body has its own scope: newly initialized local values do not leak to the
outer environment. Mutation of an existing outer binding follows the language's
assignment rules; changing a model directive does not change those rules.

Likewise, a procedure captures agent configuration at definition time, not a
promise that all referenced mutable Python objects are deep-copied. Child-agent
conversation isolation does not sandbox arbitrary shared host objects. See
[Outputs and Assignment](../core-language/outputs-and-assignments.md) for
binding semantics and [Subagent Isolation](subagent-limits.md#isolated-child-configuration)
for the child boundary.

## Capability Metadata

Adapters advertise kind and capabilities such as:

- structured outputs;
- dynamic Kedi/Python tools;
- MCP servers;
- skills;
- subagent foreground or background execution;
- approval mediation.
- event-specific lifecycle hooks.

The parser and LSP combine literal backend selection with this metadata. Dynamic
backend expressions defer some checks to runtime.

## Explicit Requirements

Use `> requires:` when a capability is a hard contract rather than an inferred
feature warning:

```kedi
> adapter: pydantic
> requires:
    structured_output
    tool_registration
    stream_events
```

The scalar form accepts one name: `> requires: stream_events`. Requirements are
additive across lexical scopes and applied profiles, with stable
deduplication. They may appear at top level, inside a procedure, or inside a
profile.

Literal adapter selections are checked by the LSP. Dynamic selections defer
the decision until runtime, but Kedi still validates the contract before the
first model request. A missing or unadvertised capability is an error, not a
fallback. Python `query` and `bind` accept the same callable-boundary contract
through `requires=(...)`.

A2A can negotiate peer capabilities: an authenticated Agent Card lookup may
precede validation, but no remote task starts unless its requirements hold.
For example, `structured_output` requires the peer's Kedi structured-output
extension, even when the current call only requests a text response.

The canonical requirement vocabulary is:

```text
structured_output, tool_registration, mcp, profile_override, model_override,
effort, settings, codemode, native_approvals, native_approval_handler,
subagents, background_subagents, artifacts, native_artifacts,
stateful_history, history_replay, native_compaction, stream_events, hooks
```

## Errors versus Warnings

Structured output is central to `>>` output capture. If the selected adapter
does not support it, the LSP reports an error and execution raises rather than
falling back to manual JSON prompting.

Tool and MCP support is currently reported as a capability warning for adapters
that cannot register them, allowing the same source to become valid when the
adapter gains support. Do not ignore the warning in production: the intended
capability is not active.

Subagent delegation and other required runtime seams fail explicitly when the
adapter lacks child execution. Kedi never pretends that delegated work ran.

## Adapter Switching

Nested scopes may switch to another framework or harness:

```kedi
> adapter: pydantic

@repository_task(task: str) -> str:
  > agent: codex
  >> Complete this repository task: <task>. The result is [answer: str].
  = <answer>
```

The nested scope restores the outer selection when it exits. Within a single
executable scope, sequential `> adapter:` and `> agent:` directives select the
backend for subsequent calls. A declarative profile still specifies one kind.
See [Backend Selection](backend-selection.md#selection-precedence) for model
inheritance and framework-to-harness transitions.

## Production Guidance

Treat every capability warning as an unmet contract, test literal profiles with
their production adapter, and keep dynamic backend selection at application
boundaries. Static profiles give the LSP enough information to detect schema,
tool, and transport mismatches before a paid model call.
