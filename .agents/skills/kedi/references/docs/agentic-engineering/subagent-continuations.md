# Continuations and Persistence

Use a continuation when a completed child should refine earlier work. Use a
new delegation when the task should start without that conversation.

## Continue a Completed Run

The model-facing tool is:

```text
continue_subagent(run_id, task, final_schema=None, background=False)
```

It returns a new result envelope, or a new handle for supported background
execution. The new run has a new `run_id` but belongs to the same child
conversation. Keep the latest ID for the next continuation.

For example, first ask an investigator to list missing evidence, then continue
its completed run with the missing test log. The parent must supply that new
evidence in `task`; continuation does not provide access to parent variables.

## Admission Rules

Only the latest completed run can be continued. The calling profile must own
the conversation and still permit that child. A conversation cannot have two
active continuations. The maximum is eight completed turns, including the
initial run.

Stale IDs, failed/running runs, a different owning profile, a child no longer
allowed, and an exhausted conversation limit are rejected. A continuation is a
new child start and consumes the normal start and usage budgets.

Kedi uses native adapter resume state when available. Otherwise it supplies
bounded prior validated results. This fallback is not a promise to replay
every historical tool call or the child's entire original context.

## Opt Into Restart Persistence

Set `subagent_state_path` on `compile_program`, `KediRuntime`, or the Python
runtime context. The [embedding example](subagent-python.md) shows complete
configuration and cleanup.

The versioned JSON store contains run records, validated results, bounded
conversation turns, and serializable adapter resume state. Writes use an atomic
replacement with owner-only file permissions. It is application state, not a
credential vault: keep it outside version control and treat task/result content
as potentially sensitive.

| State at interruption | State after restore |
| --- | --- |
| Completed result and valid continuation state | Available again within retention and ownership rules. |
| Pending/running child | Failed with `InterruptedError`; no provider request is resumed. |
| Invalid store version, malformed record, or invalid serialized state | Explicit load/save failure, not silent state loss. |

Persistence is opt-in. Without it, closing/restarting the process loses the
in-memory run registry. A persisted ID is still subject to profile ownership;
loading the file does not grant arbitrary programs authority over its results.

## What Persistence Does Not Do

It does not checkpoint an in-flight model request, reconnect a dead event loop,
resume host processes started by tools, or undo effects that happened before a
crash. Make effectful tools idempotent when restart/retry is part of the
application's operating model.
