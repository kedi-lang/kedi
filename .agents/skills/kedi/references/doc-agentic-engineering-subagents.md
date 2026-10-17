> Offline source: `kedi-docs-main/docs/agentic-engineering/subagents.md`. Normative documentation snapshot; examples may be fragments and were not all executed. Use the verified assets for complete runnable recipes.

## Contents

- [Declare Direct Children](#declare-direct-children)
- [Choose an Orchestration Mode](#choose-an-orchestration-mode)
- [Start a Child in Kedi Code](#start-a-child-in-kedi-code)
- [Child Isolation](#child-isolation)
- [`delegate_task`](#delegate_task)
- [Budgets and Usage Limits](#budgets-and-usage-limits)
- [Foreground and Background](#foreground-and-background)
- [Fail-Closed Observation](#fail-closed-observation)
- [Continue a Conversation](#continue-a-conversation)
- [Safety Ceiling](#safety-ceiling)
- [Restart Persistence](#restart-persistence)
- [Adapter Support](#adapter-support)
- [Reading Path](#reading-path)

# Subagents

Subagents let a profile delegate bounded, isolated work to explicitly declared
children. The parent chooses a task; the child runs with its own configuration
and returns evidence or a typed result. It is not an implicit ability to spawn
any profile or inspect the parent's local variables.

## Declare Direct Children

<!-- Kedi snippet: parse-checked; execution/context/provider requirements are defined by the surrounding text. -->
```kedi
> profile: investigator:
    ###
    Inspect supplied test evidence and identify what remains unverified.
    ###
    > adapter: pydantic
    > instructions: Separate observed results from missing evidence.

> profile: coordinator:
    > adapter: pydantic
    > subagent: investigator
    > max_agents: 3

> use: coordinator
[answer] << Ask investigator to assess: unit tests passed; integration tests were not run. Summarize release uncertainty.
= <answer>
```

Supply a model through the CLI or Python embedding. The parent sees
`delegate_task` with a child-name argument restricted to `investigator`.
The main program actually invokes the coordinator, but the model still chooses
whether to use its tool; configuration alone is not proof of delegation.

Forward profile references are valid. Unknown children and cycles are rejected.
For a graph A -> B -> C, A may call B but cannot call C unless A also lists C
directly. The first block comment in each profile supplies its description.

An eager [capability](doc-agentic-engineering-capabilities.md) can contribute an existing child binding
through `> subagent:`. Its approval and hooks apply only to calls targeting that
bound child, not every child behind the shared `delegate_task` tool. The child's
own model and instructions remain independent. Deferred child bindings are not
supported in the initial capability implementation.

## Choose an Orchestration Mode

The default `> workflow: delegate` gives the model delegation and supported
lifecycle tools. `> workflow: dynamic` instead gives it `run_workflow(code)`
and exposes the direct children as async functions inside a sandbox.

Use delegate mode for one child result at a time. Use
[Dynamic Workflows](doc-agentic-engineering-dynamic-workflows.md) for generated dependencies and fan-out.
Both modes preserve the same child boundaries, budgets, and approval ceilings.

## Start a Child in Kedi Code

Use native `> task` when the program, rather than the parent model, decides to
delegate. The header names one direct child and opens exactly one `>>` template
block. It starts the child immediately; `> await` waits at its own statement.

Read [Native Tasks and Await](doc-agentic-engineering-subagent-tasks.md) for a complete program, typed
output access, and ownership rules. Use
[Concurrent Result Processing](doc-agentic-engineering-subagent-processing.md) to process each result
as soon as it is ready and join all processing before continuing.

## Child Isolation

A child receives its own profile, not the parent's conversation, local values,
or tools. Include the objective, relevant data, constraints, and expected result
in `task`. See [Limits, Isolation and Safety](doc-agentic-engineering-subagent-limits.md).

## `delegate_task`

The model-facing arguments are `subagent`, `task`, optional `final_schema`,
and `background` where supported. A successful foreground result contains
`run_id`, `subagent`, `task_summary`, and `final_result`.

A profile's `> output: Type` supplies the default result schema; an explicit
`final_schema` overrides it. With neither, `task_summary` carries text and
`final_result` is null. Read [Typed Child Results](doc-agentic-engineering-subagent-results.md) for
schemas, validation, and examples.

## Budgets and Usage Limits

Total descendant starts, concurrent children, depth, per-child usage, and
deadlines are separate limits. Failed admitted starts still consume budget.
[Limits and Safety](doc-agentic-engineering-subagent-limits.md#resource-limits) gives defaults, caps,
and the behavior of each boundary.

## Foreground and Background

Foreground calls wait for a completed result. Background calls return a handle
and require later observation. See
[Foreground and Background Runs](doc-agentic-engineering-subagent-lifecycle.md) for exact tool results,
caller wait timeouts, cancellation, and cleanup.

## Fail-Closed Observation

A background handle is not evidence that work succeeded. Under fail-closed
policy, unresolved delegated work prevents the parent from completing as
though it had succeeded. Status inspection is not successful result observation.
See [Observe Before Returning](doc-agentic-engineering-subagent-lifecycle.md#observe-before-returning).

## Continue a Conversation

`continue_subagent` starts a new run in the same bounded conversation. Only
the latest completed run can be continued, by its owning profile. See
[Continuations and Persistence](doc-agentic-engineering-subagent-continuations.md).

## Safety Ceiling

Child working directories, sandboxes, and approvals cannot widen inherited
host permissions. This is distinct from the child's isolated prompt/tool
configuration. See [Host Safety Ceilings](doc-agentic-engineering-subagent-limits.md#host-safety-ceilings).

## Restart Persistence

Persistence can restore completed results and bounded continuation state.
Interrupted live requests restore as failures, not resumed provider work.
See [Continuations and Persistence](doc-agentic-engineering-subagent-continuations.md#opt-into-restart-persistence).

## Adapter Support

Pydantic AI, LangChain, Claude Agent SDK, Codex App Server, A2A, and DSPy support
child execution. Background lifecycle requires asynchronous ownership and is
not available through DSPy's bridge. Unsupported capabilities fail explicitly;
consult the [capability matrix](doc-reference-capability-matrix.md).

## Reading Path

1. [Native Tasks and Await](doc-agentic-engineering-subagent-tasks.md): delegate from Kedi code.
2. [Typed Child Results](doc-agentic-engineering-subagent-results.md): define what the parent consumes.
3. [Foreground and Background Runs](doc-agentic-engineering-subagent-lifecycle.md): own and observe work.
4. [Concurrent Result Processing](doc-agentic-engineering-subagent-processing.md): process ready results.
5. [Limits, Isolation and Safety](doc-agentic-engineering-subagent-limits.md): bound resources and authority.
6. [Continuations and Persistence](doc-agentic-engineering-subagent-continuations.md): refine and restore.
7. [Dynamic Workflows](doc-agentic-engineering-dynamic-workflows.md): compose child calls in code.
8. [Python Embedding](doc-agentic-engineering-subagent-python.md): configure, inspect, and close a runtime.
9. [Reviewed Evidence Workflow](doc-agentic-engineering-reviewed-evidence.md): execute the complete path.

The model-facing tools and native task statements use the same coordinator.
Python callers can use `runtime.subagents(parent="coordinator")`; coordinator
internals are not required application APIs.
