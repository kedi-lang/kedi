# Embedding Subagent Programs in Python

The language declares the profile graph. Python supplies the adapter, host
tools, global values, resource limits, and runtime lifetime. It does not need
to reach into the coordinator or call an invented `kedi.subagent()` function.

## Compile, Execute, Close

This complete embedding uses a profile-only program with a raw text result.
Configure provider credentials before running it:

```python
import asyncio

from kedi import SubagentUsageLimits
from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program

source = """
> profile: reviewer:
    > adapter: pydantic
    > instructions: Identify what the supplied evidence does not establish.

> profile: coordinator:
    > adapter: pydantic
    > subagent: reviewer
    > max_agents: 2

> use: coordinator
[answer] << Ask reviewer to assess: unit tests passed but integration tests were not run. Summarize uncertainty.
= <answer>
"""

limits = SubagentUsageLimits(
    request_limit=4,
    tool_calls_limit=6,
    input_tokens_limit=20000,
    output_tokens_limit=2000,
    total_tokens_limit=22000,
)
runtime = compile_program(
    parse_program(source),
    adapter=PydanticAdapter("openai:gpt-6-luna"),
    subagent_max_depth=2,
    subagent_max_concurrency=2,
    subagent_timeout_seconds=60,
    subagent_usage_limits=limits,
    subagent_state_path=".kedi-subagent-state.json",
)
try:
    print(runtime.run_main())
finally:
    asyncio.run(runtime.aclose())
```

This is a synchronous script. In an asynchronous application, keep execution
and cleanup under that application's lifetime; do not call `asyncio.run` inside
an already running event loop. The same subagent configuration is accepted by
`KediRuntime` and the public `kedi.context(...)` runtime context.

Input/total token budgets can require pre-request token counting from the model
integration. If the selected model cannot provide it, the child fails rather
than pretending that the limit was enforced. The local documentation fixture
implements token counting explicitly; a bare Pydantic `FunctionModel` does not.

The state path is optional. Exclude it from version control because task text,
results, and adapter continuation state may be sensitive. See
[Persistence](subagent-continuations.md).

## Delegate Directly From Python

When application code knows the child task, it can use the same runtime without
a parent model call. The scope owns every child it starts:

```python
async with runtime.subagents(parent="coordinator") as agents:
    job = await agents.start(
        "reviewer",
        task="Assess this evidence: unit tests passed; integration tests were not run.",
    )
    result = await job.wait()
    print(result.task_summary)
```

`start` admits the child and returns a `SubagentTask`; `wait` blocks for its
`SubagentResult`. The result exposes `output`, `task_summary`, `run_id`, and
`subagent`. If the child has no `> output:` declaration, `output` is `None`.
Repeated waits return the same result. The handle cannot be reused after its
scope closes or from another invocation. Leaving a task unawaited cancels it
and fails the scope; child failures and schema errors raise at `wait`.

A failure or child cancellation caught at `wait` counts as observed: closing
the scope does not raise that same outcome again. You can handle the failure
and start a replacement within the remaining budget. The failed or cancelled
invocation still consumes its `max_agents` slot. Cancelling only a waiter does
not cancel or observe the running child; await that child later or let scope
cleanup reject the unawaited work.

The `parent` must be the effective profile when this scope is called inside
Kedi execution. An independent embedding scope may select a compiled profile,
but it cannot use that choice to bypass a running child profile's permissions.
Use `await runtime.aclose()` when the application is done with the runtime.

## Inspect a Child From Its Tool

Running tasks also expose `await job.send(message, interrupt=False)`. For receipts,
redirection, and explicit owner message reception, see
[Task Input and Redirection](subagent-messaging.md).

`current_subagent_execution()` returns a `SubagentExecutionContext` inside a
child execution, or `None` outside it. Its public fields are `run_id`, `profile`,
and `conversation_id`. It is task-local, so concurrent children do not share one
global "current child" value.

```python
from kedi import current_subagent_execution, tool

@tool(risk="read_only")
def execution_identity() -> dict[str, str]:
    """Return the identity of the child currently inspecting evidence."""
    child = current_subagent_execution()
    if child is None:
        raise RuntimeError("This tool requires a child invocation")
    return {
        "run_id": child.run_id,
        "conversation_id": child.conversation_id,
        "profile": child.profile.name or "",
    }
```

Pass this callable in `runtime_globals` and expose it with `> use:` in the
child's profile. Registration does not invoke it. The
[reviewed evidence example](reviewed-evidence.md) demonstrates a real child tool
using this API and verifies its identity against the delegation result.

## Keep the Surfaces Separate

| Surface | Intended caller |
| --- | --- |
| Profile directives | Kedi source declares permissions and graph structure. |
| `delegate_task`, lifecycle tools, `run_workflow` | The parent model through its advertised tools. |
| `compile_program`, `KediRuntime`, `kedi.context` | The embedding application. |
| `runtime.subagents(...)`, `SubagentTask`, `SubagentResult` | Program-controlled child work in Python. |
| `SubagentUsageLimits`, execution context helper | Application configuration and child-aware tools. |

The coordinator's underscored runtime methods are implementation details, not
an application API. A parent task's final text is not a substitute for observing
its required child outcomes and host effects.
