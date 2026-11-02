# Task Input and Redirection

Additional evidence and changes of scope do not require a new child task. Send
input to a running task, or interrupt its current attempt and continue the same
conversation. The task handle, result schema, delegation budget, and deadline
remain unchanged.

## Send From Kedi Code

```kedi
> profile: reviewer:
    > adapter: pydantic
    > model: openai:gpt-6-luna
    > instructions: Review the supplied evidence. State uncertainties explicitly.

> profile: coordinator:
    > adapter: pydantic
    > subagent: reviewer

> use: coordinator
[change] = Add an export endpoint. Integration tests have not run.
[migration_notes] = Existing access tokens retain their previous permissions.

> task [review_job]: reviewer:
    >> The security concerns in <change> are [concerns: list[str]].

> send: review_job:
    >> Include the migration evidence from <migration_notes>.

> await [review]: review_job
> show: <`review.output.concerns`>
```

`> send` does not cancel an active request or tool. The child receives the input
at the next model-input boundary. If it would otherwise finish, the accepted
input must be processed before the coordinator publishes the final result.
Do not assume that a very short task is still running: sending to a completed
task is rejected. Use [continuation](subagent-continuations.md) for finished work.

An input body accepts one `>>` template with continuation lines. Substitutions
are evaluated once at send time. Captures, nested directives, and additional
templates are rejected during parsing.

## Redirect the Current Attempt

```kedi
> interrupt: review_job:
    >> The user changed the scope. Review only authentication and authorization.
```

Interruption and the updated input are one operation. The old attempt cannot
publish a result for the new revision. Kedi waits for it to stop, cancels its
owned descendants, and resumes using the retained conversation. Sibling tasks
continue. Kedi retains completed tool results instead of automatically repeating
those calls. The model can still choose to call a tool again; interruption cannot
undo completed host effects. Use idempotency keys for operations that must not
be applied twice.

### Change One Task, Keep the Other Running

The review and release note are separate tasks. Redirecting the review does not
restart the release note or consume another delegation slot:

```kedi
> model: openai:gpt-6-luna

> profile: reviewer:
    > adapter: pydantic
    > instructions: Review the supplied change and identify concrete security concerns.

> profile: writer:
    > adapter: pydantic
    > instructions: Write a release note using only the supplied facts.

> profile: coordinator:
    > adapter: pydantic
    > subagent: reviewer
    > subagent: writer
    > max_agents: 2

> use: coordinator
[change] = Add an export endpoint. Existing access tokens retain their permissions.

> task [review_job]: reviewer:
    >> The concerns in <change> are [concerns: list[str]].

> task [note_job]: writer:
    >> The release note for <change> is [note: str].

> interrupt: review_job:
    >> The user changed the scope. Review only access-token authorization.

> await [review]: review_job
> await [release]: note_job
> show: <`review.output.concerns`>
> show: <`release.output.note`>
```

Another interrupt during cleanup is rejected as `busy`. Ordinary messages can
still queue for the replacement attempt. Cancellation, timeout, and owner
shutdown terminate the logical task instead of restarting it. A synchronous
tool running in a thread cannot be forcibly killed: if cleanup is not confirmed
within the bounded cleanup waits (five seconds per phase), the task fails instead
of starting overlapping work.

## Python Handles and Receipts

```python
async with runtime.subagents(parent="coordinator") as agents:
    job = await agents.start("reviewer", task="Review the export endpoint.")
    receipt = await job.send("Include the migration evidence.")
    if receipt.status == "rejected":
        print(receipt.reason)
    result = await job.wait()
    print(result.task_summary)
```

`job.task_id` aliases `job.run_id`. Repeated waits return the same final result,
including after redirection. Use `interrupt=True` with `send` to redirect work.

| Receipt field | Meaning |
| --- | --- |
| `message_id` | Runtime-assigned identifier; `None` on rejection |
| `target_id` | Logical task receiving the input |
| `revision` | Current task revision; increases on accepted interruption |
| `status` | `queued`, `interrupt_requested`, or `rejected` |
| `reason` | Stable rejection reason, or `None` |

Rejection reasons are `task_finished`, `not_authorized`, `unsupported`, `busy`,
`queue_full`, and `input_too_large`. Admission does not mean the model has read,
understood, or acted on the input. The queue holds at most 32 pending messages,
128 KiB total, and 32 KiB per message, measured as UTF-8. Accepted input is not
silently evicted or truncated.

Input events record admission, confirmed delivery, or `undelivered` on task
termination. An admitted message is not guaranteed delivery after cancellation,
timeout, or failure. Pending input is released when the task closes.
Initial task input is confirmed at the native request boundary, not merely when
Kedi assembles a prompt. A request rejected before that boundary, including one
rejected by native usage limits or Kedi's run budget, does not produce a delivery
event. Requests started before interruption remain charged; caller-supplied
cumulative usage is retained across continuations. Delivery does not imply a
successful response or model comprehension. History processors must preserve the
current request tail, including its task input.

If a Codex turn completes just before its control request is handled, Kedi keeps
the undelivered input for the next turn in the same conversation. It does not
mark a rejected steer as delivered. Other native queue or transport failures
remain visible; Kedi does not busy-retry an invalid enqueue. Interruption
pins the obsolete attempt's revision through nested runtime scopes, so that
attempt cannot start new tools or child tasks after its scope has been superseded.
If native stop confirmation fails or is cancelled, Kedi attempts bounded local
cleanup and reports the failure instead of starting replacement work. Already
completed host effects are not rolled back.

## Model-Driven Communication

Parents receive `send_subagent_input(task_id, message, interrupt=False)` in their
toolset. Children receive `send_parent_message(message)`. A child cannot select
an arbitrary recipient; it can address only its direct parent. A parent can
address only its own direct child tasks, not siblings or another invocation's
tasks. Input cannot alter tools, permissions, models, or output schemas.

A child message does not pause that child or guarantee a reply. It enters the
parent's next model request. A model-facing `wait_subagent` can return a running
handle with `input_available=true` when such input arrives. The parent must wait
again for the terminal child result. Native `> await` and Python `job.wait()`
are not changed into nonterminal waits.

A deterministic Python owner receives updates explicitly:

```python
messages = await agents.receive_messages()
for message in messages:
    print(message.sender_id, message.message)
```

This API does not create a parent model request. Messages and task handles are
invocation-owned and cannot be reused after the scope closes.

## Adapter Support

Pydantic AI uses native enqueue and cancellation; LangChain uses request-boundary
middleware, async cancellation, and invocation-local graph checkpoints. Codex
uses `turn/steer`, or interrupt followed by terminal acknowledgment and same-thread
continuation. Claude uses tool/stop hooks and a controllable SDK client; a pending
message blocks final stopping until the next input boundary.

Pydantic callers can still supply their own `CancellationToken`. Cancelling that
token cancels its registered runs; redirecting one Kedi task uses run-local
cancellation and does not cancel siblings that share the caller's token.

These adapters expose the same public tools, receipts, ownership checks, and
terminal waits. A2A, ACP, and DSPy reject live input as unsupported. Pydantic AI
versions without native enqueue support also reject it. No adapter promises
that interrupting work reverses already completed host effects.
