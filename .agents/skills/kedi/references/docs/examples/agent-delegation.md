# Agent Delegation

Subagents are explicit direct-child profiles. They are appropriate when a
coordinator needs an isolated conversation, tool set, model, or safety policy;
they are unnecessary for deterministic helper procedures.

## Foreground Structured Delegation

```kedi
~ResearchAnswer(claim: str, confidence: float, evidence: list[str])

> profile: researcher:
    ###
    Investigate one focused question and distinguish evidence from inference.
    ###
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > output: ResearchAnswer
    > instructions: Return concise evidence and identify uncertainty.

> profile: coordinator:
    ###
    Split a request only when independent research is useful.
    ###
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > subagent: researcher
    > max_agents: 2

> use: coordinator

>> After researcher reviews the evidence, its release recommendation is [answer: ResearchAnswer].
Pass this evidence to the child: parser tests passed, but the migration check failed.
Release requires both checks to pass. Ask the child to explain which check blocks release.
= `answer`
```

Kedi gives `coordinator` a `delegate_task` tool. The model supplies
`subagent="researcher"`, a self-contained `task`, and optionally a
`final_schema`. Here `> output: ResearchAnswer` fixes the child's result contract;
the parent does not need to invent that schema. The runtime validates structured
child output before returning it as `final_result`.

The child cannot see the parent's prompt or local Kedi values. Its task must
include the objective, necessary input, constraints, and expected evidence.

This is evidence analysis, not web research: neither profile has a search tool.
Expected behavior is to identify the migration failure as the release blocker.
The confidence field is a model-produced estimate, not a calibrated probability.
The parent's capture validates its own answer, not equality with the child's
payload. Use the child envelope directly when exact result preservation matters.

## Result Contract

A completed delegation returns:

| Field | Meaning |
| --- | --- |
| `run_id` | Stable handle for status, wait, cancellation, or continuation |
| `subagent` | Selected direct child |
| `task_summary` | Validated summary of the child turn |
| `final_result` | Schema-conforming payload, or `None` for text-only work |

Only children listed with `> subagent:` are selectable. Unknown children,
cycles, and budget exhaustion fail explicitly.

## Background Lifecycle

Pydantic, Claude, Codex, and LangChain coordinators can ask for
`background=True`. They then receive these additional tools:

```text
subagent_status(run_id)
wait_subagent(run_id, timeout=...)
cancel_subagent(run_id)
```

The parent must call `wait_subagent` before consuming the result or returning a
final answer. Merely checking status or cancelling does not count as successful
observation. Unobserved background work causes the parent to fail closed.

ACP does not expose the child
execution seam.

## Continue a Child Conversation

`continue_subagent(run_id, task, final_schema=None, background=False)` starts a
new run in the same bounded child conversation. Only the owner may continue it,
only the latest completed run is eligible, and a conversation is limited to
eight completed turns.

Use continuation when the next task depends on the child's prior context. Start
a fresh `delegate_task` for independent work; this avoids accidental context
coupling.

## Safety and Persistence

Children receive isolated profiles and may narrow, never widen, inherited
working-directory, sandbox, and approval ceilings. A declared
`> max_agents: N` bounds descendant starts for one parent invocation; the hard
runtime ceiling is 100.

Python embedders can persist terminal run state with
`compile_program(..., subagent_state_path=...)` or
`KediRuntime(..., subagent_state_path=...)`. Completed results and valid
continuations survive restart. In-flight records become failed after restart;
Kedi does not pretend to resume a request whose process ownership was lost.

## Executable Composition

For a complete host program that inspects the actual child envelope, use
[Reviewed Evidence](../agentic-engineering/reviewed-evidence.md). Its tests run
foreground and dynamic delegation, prove parent/child tool isolation, and reject
invalid child output before a report is written. Continue with
[lifecycle rules](../agentic-engineering/subagent-lifecycle.md) and
[Python embedding](../agentic-engineering/subagent-python.md) for background work
and persisted conversations.
