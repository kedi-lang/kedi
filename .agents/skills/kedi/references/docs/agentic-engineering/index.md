# Agents and Orchestration { #agentic-engineering }

Kedi treats model execution as scoped agent configuration rather than a global
string. A scope can select a framework or harness, model, reasoning effort,
instructions, settings, tools, MCP servers, skills, approvals, and child agents.

Choose the language construct by who controls the next step:

| Construct | Who selects the next step? | What it provides |
| --- | --- | --- |
| Procedure | The Kedi program | Explicit statements, arguments, return types, and captured configuration. |
| Profile | No execution by itself | Reusable configuration applied to later calls or used by a child agent. |
| Tool-using model call | The model, within the exposed tool surface | A request that may call tools before returning its result. |
| Subagent | The parent requests work; the child owns its conversation | An isolated task with its own profile and a bounded lifecycle. |
| Dynamic workflow | The parent writes orchestration code | Sandboxed composition of the permitted child agents. |

Start with a procedure when the sequence is known. Add a tool-using call where
the model needs to choose an action. Use a child agent when a task needs a
separate conversation and capability boundary, not merely a different prompt.

## Framework Adapters and Agent Harnesses

Framework adapters embed Kedi's prompt and tool contracts in an agent framework:

- `pydantic`
- `langchain`

Agent harnesses drive a complete external coding/agent runtime:

- `claude`
- `codex`
- `acp`

Select the first group with `> adapter:` and the second with `> agent:`. These
are mutually exclusive backend kinds, not interchangeable aliases.

## Active Agent State

Directives update immutable agent state for following calls in the lexical
scope:

```kedi
> adapter: pydantic
> model: openai:gpt-6-luna
> effort: low
> instructions: Use tools only when they improve factual accuracy.
```

Top-level state is captured by following procedures. A procedure-body override
applies only to later calls in that invocation. When the scope exits, Kedi
restores the previous state.

## Profiles

Profiles name reusable state:

```kedi
> profile: reviewer:
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > instructions: Review evidence before making a claim.

> use: reviewer
```

Profiles can be exported, imported, merged with later directives, and exposed
as direct child subagents.

## Capability Surfaces

- Procedures become typed tools through `> use:`.
- `> mcp:` attaches external MCP tools.
- `> skills:` exposes scoped registry, project, and user skill discovery.
- `> codemode: enabled` progressively discloses tools and executes hydrated calls in
  bounded code.
- `> approval:` controls risky tool calls.
- `> hooks:` observes, edits, or denies agent lifecycle boundaries.
- `> subagent:` adds bounded delegation tools to a profile.
- semantic stream events expose completed commentary and final messages to
  terminals, UIs, and orchestration code without changing adapter results.

Each surface is capability-checked against the selected backend. Unsupported
structured output is an execution error. Tools and MCP currently produce
forward-compatible editor warnings where an adapter may gain support later.

## Design Rule

Define the narrowest agent surface that can perform the task. Give profiles
specific instructions, only required tools, bounded child agents, and the least
permissive approval and sandbox settings. A broader model is not a substitute
for a clear contract.

## In This Section

**Profiles and Configuration**

- [Instructions and Settings](instructions-and-settings.md)
- [Profiles and Composition](profiles.md)
- [Agent Scope and Capabilities](scoping-and-capabilities.md)

**Tools and Permissions**

- [Procedure and Python Tools](tools-and-use.md)
- [MCP Tools](mcp.md)
- [Skills](skills.md)
- [Approval Policies](approvals.md)
- [Tool Reasons and LLM Approval](tool-reasons.md)

**Lifecycle**

- [Lifecycle Hooks](hooks.md)
- [Stream Events](stream-events.md)

**Subagents and Workflows**

- [Subagents and Workflows](subagents.md)
- [Typed Child Results](subagent-results.md)
- [Foreground and Background Runs](subagent-lifecycle.md)
- [Limits, Isolation and Safety](subagent-limits.md)
- [Continuations and Persistence](subagent-continuations.md)
- [Dynamic Workflows](dynamic-workflows.md)
- [Python Embedding](subagent-python.md)
- [Reviewed Evidence Workflow](reviewed-evidence.md)

**CodeMode**

- [CodeMode](codemode.md)
- [Sandbox Limits and Recovery](codemode-sandbox.md)
