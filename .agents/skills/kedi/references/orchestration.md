# Subagents and workflow construction

## Choose the mechanism

| Need | Mechanism |
| --- | --- |
| Known child tasks started by the program | Native `> task` / `> await` |
| Let a model choose a permitted child | Profile child graph plus `delegate_task` |
| Process results as children finish | `> task_group:` with per-await `> process:` |
| Let a model write orchestration | `> workflow: dynamic` and sandboxed `run_workflow` |
| Host-controlled async orchestration | `async with runtime.subagents(parent=...)` |

Read [native tasks](doc-agentic-engineering-subagent-tasks.md), [results](doc-agentic-engineering-subagent-results.md), and [limits](doc-agentic-engineering-subagent-limits.md). The complete [parallel_review.kedi](../assets/examples/parallel_review.kedi) starts two children and combines typed results.

## Declare a graph

Define child profiles with instructions describing one responsibility. List them directly in the parent using `> subagent: child`; set a positive `> max_agents:`. Only direct children can be invoked. Forward child references work; unknown names/cycles fail. Child contexts do not implicitly receive parent variables/history/tools. Put all required context into each task and expose needed tools in the child profile.

`max_agents` limits descendant **starts per invocation**, not concurrent workers. Repeated, failed, or cancelled admitted calls count. Descendants debit all active ancestors; rejected admissions do not. The documented hard ceiling is 100 starts. Separate depth/concurrency/time/usage limits apply; the low-level defaults are depth 5, concurrency 4, timeout 120 seconds. Use public runtime parameters rather than assuming a `configure(**kwargs)` option reaches the coordinator.

## Start, wait, consume

A task body is exactly one `>>` block, with optional same-indent continuation rows. Inputs render once at start. A second block is invalid. A task handle is invocation-owned and is not the child's response.

`> await [result]: job` binds a `SubagentResult`; binderless `> await: job` still waits and propagates errors. Repeated awaits observe the same result. Access native results with `result.output`, `task_summary`, `run_id`, `subagent`.

| Schema source | Result access |
| --- | --- |
| Task captures `[review: Review]` | `result.output.review` |
| No task captures; child `> output: Review` | `result.output` is the validated Review |
| Neither schema | `result.output is None`; response in `task_summary` |
| Model delegation/dynamic workflow | Envelope uses `final_result`, not native-task `.output` |

Do not combine task captures with child `> output:`. Explicit delegated `final_schema` overrides the profile default. Validate shape before indexing a dynamic/delegated dictionary.

Unawaited work fails closed and is cancelled when the invocation ends. Await every admitted task. Failures/timeouts/schema errors raise at wait; catching an error does not refund admission. A cancelled waiter does not itself prove the child was cancelled or observed. DSPy lacks the asynchronous background contract.

## Process ready results

A task group registers all awaits first. Each optional process belongs to the immediately preceding await. Its binder/declarations are private to that arm; processing inherits parent configuration. Groups join children, process bodies and owned work before subsequent statements.

There are no direct enclosing `=` returns in a process, including nested branch/loop returns; procedures called from it retain their own returns. Use display or explicit synchronized outer result storage. Shared Python mutations are not transactional and can race. A failure cancels queued/owned work and drains running work; synchronous Python side effects cannot be rolled back.

## Send, interrupt, continue

`> send: job:` and `> interrupt: job:` each contain one fieldless `>>` block. They cannot change tools/profile/schema. Admission is not proof of delivery or understanding; finished tasks reject input. Interrupt waits for the old attempt to stop before replacement work starts. Use [messaging](doc-agentic-engineering-subagent-messaging.md) for supported backends and receipts.

Continuation creates a new admitted run in the same bounded completed child conversation, owned by the parent. It is distinct from sending input to a running task. Only the latest completed run can be continued; the documented bound is eight completed turns. Persistence stores terminal records/serializable state; in-flight work restores as interrupted failure, not a resumed provider request. Read [lifecycle](doc-agentic-engineering-subagent-lifecycle.md) and [continuations](doc-agentic-engineering-subagent-continuations.md).

## Dynamic orchestration and CodeMode

`> workflow: dynamic` exposes `run_workflow(code)` instead of delegation lifecycle tools and requires a resolved direct child. Its code runs in Monty, not host Python. Direct child names are async keyword-only functions taking a self-contained `task` and optional schema. The last expression becomes bounded JSON-safe output. It cannot import arbitrary modules or access host files/network/credentials. Dynamic workflows cannot nest; completed identical calls can be salvaged during correction. Read [dynamic workflows](doc-agentic-engineering-dynamic-workflows.md).

CodeMode is a separate tool-discovery/execution surface, not the task scheduler. Load exact tool schemas before use, respect its allowlist and limits, and let inner tool calls keep their approval checks. Read [CodeMode](doc-agentic-engineering-codemode.md) and [sandbox boundaries](doc-agentic-engineering-codemode-sandbox.md) rather than treating all embedded Python as sandboxed.
