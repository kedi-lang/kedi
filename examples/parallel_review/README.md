# Two independent reviews of one patch

Ask a security reviewer and a reliability reviewer to inspect the same patch,
then combine their typed findings. Kedi starts the children directly: no parent
model must decide whether to delegate or generate orchestration code.

## Run

Complete the [shared setup](../README.md#setup), then:

```bash
kedi examples/parallel_review/main.kedi
```

The supplied [patch](change.diff) authenticates a user but never checks that the
user owns the requested account. It also closes a database connection only on
the success path. Look for an authorization check in the security recommendation
and guaranteed connection cleanup in the reliability recommendation.

## Task lifecycle

Each child profile has its own system instructions. The coordinator admits only
those two children, with `max_agents: 2`. Both inherit the selected model; a
profile can override it later without changing the orchestration.

Each `> task` starts one child with one template. The two handles are created
before either `> await`, allowing the work to overlap. Awaiting the first child
does not prevent the second from running. Both are joined before returning.

The capture named `review` belongs to the child's output model. Access it as
`security_result.output.review`, not as a parent variable or a dictionary.
The returned `ReleaseReview` combines those models without a third synthesis
call. Child failures propagate rather than silently producing a partial review.

The offline test uses a rendezvous to verify overlap, not a fragile stopwatch
speedup assertion. This is a workflow example, not a benchmark of model latency.
The patch is input text only and is never applied or executed.
