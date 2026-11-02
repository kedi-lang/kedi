# Foreground and Background Runs

Delegation tools are model-facing tools added to the parent's profile. Native
`> task` / `> await` statements and `runtime.subagents(...)` are separate
program-controlled entry points to the same coordinator. The tool names below
describe the schema that the model sees.

## Native Task Ownership

`> task [job]: child:` starts a direct child as soon as the statement runs.
`> await [result]: job` blocks at that statement and returns a typed result;
`> await: job` waits without binding a name. Two tasks may be active before
the first await. Child failures, timeouts, and invalid results raise even for
the binderless form. A repeated await uses the same completed run.

If an invocation returns with a native task unawaited, Kedi cancels it, joins
it, and fails closed. Task handles do not survive the invocation; they are not
detached jobs. See [Subagents](subagents.md#start-a-child-in-kedi-code) for
syntax and [Python Embedding](subagent-python.md#delegate-directly-from-python)
for the async API.

## Foreground Result

`delegate_task(subagent, task, final_schema=None, background=False)` waits for
the selected direct child and returns `run_id`, `subagent`, `task_summary`, and
`final_result`. Use foreground work when the next parent step depends on it.

Every ordinary delegation starts a fresh conversation. A failure raises instead
of returning a success-shaped envelope with fabricated output.

## Background Handle

With `background=True`, a supported adapter starts the child and returns a
handle such as:

```json
{"run_id": "<generated-run-id>", "subagent": "investigator", "status": "running"}
```

This is an admission acknowledgment, not a result. The child may fail after the
handle is returned. Retain the exact `run_id`; do not construct one from a
profile name or assume it identifies every call to that profile.

| Tool | Returned data | Interpretation |
| --- | --- | --- |
| `subagent_status(run_id)` | `run_id`, `subagent`, `status` | Snapshot only; does not observe the result. |
| `wait_subagent(run_id, timeout=None)` | Completed result envelope | Successfully consumes the result. Repeated waits are allowed. |
| `wait_subagent(run_id, timeout=...)` while still running | Running handle | The caller stopped waiting; the child is still active. Wait again later. |
| `cancel_subagent(run_id)` | `run_id`, `cancel_requested`, `status` | Cancellation request, not proof that the child has stopped. |

Lifecycle states are `pending`, `running`, `completed`, `failed`, `cancelled`,
and `timed_out`. A cancellation response can still report `running` while the
child processes cancellation. Repeated cancellation is idempotent.

## Two Different Timeouts

The optional `wait_subagent` timeout bounds only that wait. It does not restart
or cancel the child. If the child is still running, the tool returns its handle
so the parent can do other work and wait again.

The runtime's `subagent_timeout_seconds` bounds the child itself, starting when
it is started, not on its first wait. Repeated short waits do not extend that
deadline. A child deadline produces a `timed_out` run and a failed wait, not a
running handle. See [Limits and Safety](subagent-limits.md).

## Observe Before Returning

Under fail-closed delegation policy, a parent cannot claim completion while
delegated work is unresolved. Background work must be successfully observed
through `wait_subagent`. Status inspection and cancellation alone are not
successful observation. A failed task may be recovered by a successful
replacement under the delegation recovery rules; ignoring it is not recovery.

Unknown or expired IDs, a failed/cancelled/timed-out child, invalid structured
results, and unresolved work are explicit failures. Do not convert these into
empty findings or an affirmative approval.

## Ownership and Cleanup

Live tasks belong to a runtime and an event loop. A handle is not a portable
job URL. Closing the runtime cancels and joins its owned children; it does not
leave them executing as detached host processes. Finished handles are retained
only within bounded retention.

Pydantic AI, LangChain, Claude and Codex support background lifecycle tools.
DSPy's bridge supports blocking delegation and continuation but not the
background lifecycle. See the [capability matrix](../reference/capability-matrix.md).

For durable *completed* results rather than live task migration, see
[Continuations and Persistence](subagent-continuations.md).
