# Concurrent Result Processing

Start child tasks with `> task`, then handle their results as they finish.
Unlike sequential `> await` statements, a task group does not make a ready
result wait behind an earlier, slower child.

## Process Ready Results

```kedi
> profile: reviewer:
    > adapter: pydantic
    > model: openai:gpt-6-luna

> profile: coordinator:
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > subagent: reviewer
    > max_agents: 2

> use: coordinator
[change] = Add a retry policy to the upload endpoint.

> task [security_job]: reviewer:
    >> The security recommendation for <change> is [recommendation: str].
> task [reliability_job]: reviewer:
    >> The reliability recommendation for <change> is [recommendation: str].

> task_group:
    > await [security]: security_job
    > process:
        > show: <`security.output.recommendation`>
    > await [reliability]: reliability_job
    > process:
        >> The action for <`reliability.output.recommendation`> is [action: str].
        > show: <action>

> show: Both reviews and their processing have finished.
```

All arms register before processing begins. Each `> process:` attaches to the
immediately preceding `> await`. A bare await without processing is valid.
The engine bounds processing concurrency; sequential execution still consumes
ready results instead of enforcing the order in which awaits were written.
Waiting children do not reserve an engine worker. Nested groups support a
single-worker engine too.

## Scope and Configuration

An arm's result binder, local values, custom types, and nested procedures are
private to that arm. They are not available after the group or in sibling arms.
The process uses its parent's lexical agent configuration, not the child agent's
model or tools. Local overrides stay local, and parent budget ceilings still apply.

Ordinary executable Kedi statements, nested tasks, loops, groups, and Python
interop are available. Direct `=` returns are forbidden in a process, even
inside nested branches or loops. A procedure declared or called from a process
may return normally: it has its own return boundary. Use `> show:` for display.

## Completion and Failure

The group finishes only when all child results and processing work are joined,
including unconsumed model promises and tasks created by a process. Await those
descendants within the process; otherwise the group fails closed.

On failure, Kedi cancels owned children and queued work, joins already-running
work, and propagates the original error. It cannot undo Python side effects or
forcibly stop synchronous Python that is already executing.

Parallel processing does not impose a side-effect order or make shared mutations
atomic. Keep state local or use explicit synchronization for shared aggregates.

See [Native Tasks and Await](subagent-tasks.md),
[Lifecycle](subagent-lifecycle.md), and [Limits and Safety](subagent-limits.md).
