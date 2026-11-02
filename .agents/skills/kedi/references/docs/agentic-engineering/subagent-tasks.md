# Native Tasks and Await

Use `> task` when the program decides which child to call. Unlike model-driven
`delegate_task`, this does not require a parent model request to choose a tool.
The selected parent profile must explicitly declare that direct child.

## Start Work and Read Typed Results

This program starts two independent reviews before waiting for either one:

```kedi
> profile: reviewer:
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > instructions: Review only the supplied change. Separate evidence from assumptions.

> profile: coordinator:
    > adapter: pydantic
    > subagent: reviewer
    > max_agents: 2

> use: coordinator
[change] = Add an export endpoint; unit tests pass, integration tests have not run.

> task [review_job]: reviewer:
    >> The main risk in <change> is [issue: str].
    The recommended next step is [recommendation: str].

> task [test_job]: reviewer:
    >> The missing verification for <change> is [verification: str].

> await [review]: review_job
> await [tests]: test_job
> show: `review.output.recommendation`
= `tests.output.verification`
```

Install the selected provider dependency and supply its credentials to run this
example. Only the child invokes the model; the coordinator needs no model for
these native statements. Each task starts at its declaration. Waiting for
`review_job` first does not prevent `test_job` from running.

## One Template, One Result

The task body contains exactly one `>>` template block. Continuation lines belong
to that block: `issue` and `recommendation` above are two fields of one child
result, not two child runs. A second `>>` block or arbitrary statements in the
task body are rejected.

`review` is a `SubagentResult`. Its `output` is a validated Pydantic model, not a
dictionary: use `review.output.recommendation`. The capture names do not become
parent bindings. `review.task_summary` contains the child's text summary;
`review.run_id` and `review.subagent` identify the run and child.

| Task template | Child profile | Result |
| --- | --- | --- |
| Contains captures | No `> output:` | `output` uses the template capture schema |
| No captures | Declares `> output: Type` | `output` is an instance of that type |
| No captures | No `> output:` | `output` is `None`; read `task_summary` |
| Contains captures | Declares `> output: Type` | Rejected: two competing output schemas |

The child receives only the rendered task and its own profile. Parent history,
tools, and local bindings are not implicitly shared. Substitute every input the
child needs, as with `<change>` above.

## Await and Ownership

Use `> await [result]: job` to bind the completed result, or `> await: job` to
observe completion and propagate errors without binding it. Awaiting the same
handle again observes the same run; it does not generate another answer.

A handle belongs to the invocation that started it. It cannot escape to another
invocation or survive an interactive session boundary as live work. Every started
task must be observed before its owner completes. Unawaited work is cancelled and
the invocation fails, even if a child happened to finish in the background.
Failure and cancellation clean up owned children; cancelling a wait is not a
guarantee that application-owned synchronous side effects can be undone.

## Capacity and Ready-First Processing

For additional evidence or a change of scope while a task is running, see
[Task Input and Redirection](subagent-messaging.md). `> send` preserves active
work; `> interrupt` replaces the current attempt without changing the handle.

`max_agents` limits admitted descendant starts. It does not increase concurrent
capacity. The separate `compile_program(subagent_max_concurrency=...)` limit rejects
excess starts rather than silently queueing them. See
[Limits, Isolation and Safety](subagent-limits.md) for configuration and defaults.
Root and child model requests also share any declared [run budget](../runtime/run-budgets.md).

Ordinary awaits execute in source order. To process whichever result becomes
ready first, use [Concurrent Result Processing](subagent-processing.md).
For Python-owned tasks and explicit cancellation, see
[Python Embedding](subagent-python.md).
