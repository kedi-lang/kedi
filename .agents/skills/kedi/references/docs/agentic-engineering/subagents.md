# Subagents

Subagents let a profile delegate bounded, isolated work to explicitly declared
children. The parent chooses a task; the child runs with its own configuration
and returns evidence or a typed result. It is not an implicit ability to spawn
any profile or inspect the parent's local variables.

## Declare Direct Children

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

An eager [capability](capabilities.md) can contribute an existing child binding
through `> subagent:`. Its approval and hooks apply only to calls targeting that
bound child, not every child behind the shared `delegate_task` tool. The child's
own model and instructions remain independent. Deferred child bindings are not
supported in the initial capability implementation.

## Choose an Orchestration Mode

The default `> workflow: delegate` gives the model delegation and supported
lifecycle tools. `> workflow: blocking` keeps only a synchronous `delegate_task`:
no `background` argument, continuation, wait, cancel, status or live-input tools,
and its children get no channel back to it. `> workflow: dynamic` instead gives
it `run_workflow(code)` and exposes the direct children as async functions inside
a sandbox.

In delegate mode, a `> workflow:` block can also choose which control tools the
model sees. `delegate_task` is always offered:

```kedi
> profile: coordinator:
    > adapter: pydantic
    > subagent: researcher
    > workflow:
        > mode: delegate
        > tools:
            wait_subagent
            cancel_subagent
```

`tools` accepts `continue_subagent`, `wait_subagent`, `cancel_subagent`,
`subagent_status` and `send_subagent_input` (one per line, or one name on the
`> tools:` line); `tools: all` is the default and `mode` defaults to `delegate`.
`delegate_task` offers `background` only when `wait_subagent` is listed, and
`cancel_subagent` and `subagent_status` require it. Children can message the
parent only when `send_subagent_input` is offered. Using `tools` with
`mode: blocking` or `mode: dynamic` is an error because those modes offer no
control tools to choose from: remove `tools`, or use `mode: delegate`.

Use delegate mode when the model manages children over time, and blocking mode
for fan-out that needs none of those controls, which also keeps their schemas
out of every request. Use [Dynamic Workflows](dynamic-workflows.md) for generated
dependencies. All modes preserve the same child boundaries, budgets, and approval
ceilings.

Every profile also carries the artifact tools and their instructions on each
request by default, which costs on the order of a thousand input tokens per
request. Children that never see large results can turn that off with
`> artifacts: enabled: false`; leave it on for any child whose tools may return
large values, since a disabled profile sends them to the model in full.

## Start a Child in Kedi Code

Use native `> task` when the program, rather than the parent model, decides to
delegate. The header names one direct child and opens exactly one `>>` template
block. It starts the child immediately; `> await` waits at its own statement.

Read [Native Tasks and Await](subagent-tasks.md) for a complete program, typed
output access, and ownership rules. Use
[Concurrent Result Processing](subagent-processing.md) to process each result
as soon as it is ready and join all processing before continuing.

## Child Isolation

A child receives its own profile, not the parent's conversation, local values,
or tools. Include the objective, relevant data, constraints, and expected result
in `task`. See [Limits, Isolation and Safety](subagent-limits.md).

## `delegate_task`

The model-facing arguments are `subagent`, `task`, optional `final_schema`,
and `background` where supported. A successful foreground result contains
`run_id`, `subagent`, `task_summary`, and `final_result`.

A profile's `> output: Type` supplies the default result schema; an explicit
`final_schema` overrides it. With neither, `task_summary` carries text and
`final_result` is null. Read [Typed Child Results](subagent-results.md) for
schemas, validation, and examples.

## Budgets and Usage Limits

Total descendant starts, concurrent children, depth, per-child usage, and
deadlines are separate limits. Failed admitted starts still consume budget.
[Limits and Safety](subagent-limits.md#resource-limits) gives defaults, caps,
and the behavior of each boundary.

## Foreground and Background

Foreground calls wait for a completed result. Background calls return a handle
and require later observation. See
[Foreground and Background Runs](subagent-lifecycle.md) for exact tool results,
caller wait timeouts, cancellation, and cleanup.

## Fail-Closed Observation

A background handle is not evidence that work succeeded. Under fail-closed
policy, unresolved delegated work prevents the parent from completing as
though it had succeeded. Status inspection is not successful result observation.
See [Observe Before Returning](subagent-lifecycle.md#observe-before-returning).

## Continue a Conversation

`continue_subagent` starts a new run in the same bounded conversation. Only
the latest completed run can be continued, by its owning profile. See
[Continuations and Persistence](subagent-continuations.md).

## Safety Ceiling

Child working directories, sandboxes, and approvals cannot widen inherited
host permissions. This is distinct from the child's isolated prompt/tool
configuration. See [Host Safety Ceilings](subagent-limits.md#host-safety-ceilings).

## Restart Persistence

Persistence can restore completed results and bounded continuation state.
Interrupted live requests restore as failures, not resumed provider work.
See [Continuations and Persistence](subagent-continuations.md#opt-into-restart-persistence).

## Adapter Support

Pydantic AI, LangChain, Claude Agent SDK, Codex App Server, A2A, and DSPy support
child execution. Background lifecycle requires asynchronous ownership and is
not available through DSPy's bridge. Unsupported capabilities fail explicitly;
consult the [capability matrix](../reference/capability-matrix.md).

## Reading Path

1. [Native Tasks and Await](subagent-tasks.md): delegate from Kedi code.
2. [Typed Child Results](subagent-results.md): define what the parent consumes.
3. [Foreground and Background Runs](subagent-lifecycle.md): own and observe work.
4. [Concurrent Result Processing](subagent-processing.md): process ready results.
5. [Limits, Isolation and Safety](subagent-limits.md): bound resources and authority.
6. [Continuations and Persistence](subagent-continuations.md): refine and restore.
7. [Dynamic Workflows](dynamic-workflows.md): compose child calls in code.
8. [Python Embedding](subagent-python.md): configure, inspect, and close a runtime.
9. [Reviewed Evidence Workflow](reviewed-evidence.md): execute the complete path.

The model-facing tools and native task statements use the same coordinator.
Python callers can use `runtime.subagents(parent="coordinator")`; coordinator
internals are not required application APIs.
