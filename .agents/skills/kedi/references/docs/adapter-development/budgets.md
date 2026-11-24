# Budgets and Usage

Kedi limits what a run may spend in three ways: run budgets cap the number of
model requests and tool attempts, a size ceiling rejects requests that cannot
fit the context window, and subagent usage limits cap each child. All three
depend on the adapter telling Kedi what it does, at the moment it does it.

## Reporting Usage

After every completed backend request, report it:

```python
from kedi.agent_adapter.invocation import InvocationUsage, report_invocation

report_invocation(
    InvocationUsage(
        requests=1,
        tool_calls=len(reply.tool_calls),
        input_tokens=usage.input_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost,  # a Decimal, or None when unknown
    )
)
```

| Field | Meaning |
|---|---|
| `requests` | model requests completed (normally 1 per report) |
| `tool_calls` | tool calls the model made in that response |
| `input_tokens` | all input tokens, including cached ones |
| `cache_read_tokens`, `cache_write_tokens` | the cached part of the input, when the provider reports it |
| `output_tokens`, `total_tokens` | as reported by the provider |
| `cost_usd` | the provider's cost, when it reports one; leave `None` rather than estimating |

Kedi routes the report to everything observing the call: the request-size
reservation, model and agent telemetry, checkpoints, `kedi explain` usage,
and a child's usage limits. Report the figures the provider returns; a
figure it does not return stays at its default of 0, and is never estimated
by the adapter. A report outside a Kedi call is ignored.

## Run Budgets

```kedi
> budget:
    request_limit: 10
    tool_attempt_limit: 20
```

| Dimension | Counted | Who counts | Capability |
|---|---|---|---|
| requests | every actual model generation: tool-loop turns, output repairs, retries, fallback models | the adapter calls `admit_run_operation("request")` before each one | `supports_request_budget` |
| tool attempts | each start of a Kedi tool body, and each of its retries | Kedi, inside `tool.fn` | `supports_tool_attempt_budget` |

`kedi.run_budget.admit_run_operation(dimension)` charges the innermost budget
and every enclosing one, and raises `RunBudgetExceeded: Run request budget
exhausted (limit N, consumed N)` once a limit is reached. Call it immediately
before sending the request, after any local validation, so a refused request
is never sent. Let `RunBudgetExceeded` propagate; it is never a correctable
tool error.

Rules for advertising the capabilities:

- `supports_request_budget`: the adapter admits every physical request,
  including its own retries. The provider SDK must not retry on its own: with
  a finite request limit, disable SDK retries (`max_retries=0`) or reject the
  run. Kedi's helper `kedi.run_budget.assert_no_hidden_model_retries(model)`
  performs that check for SDK clients that expose `max_retries`.
- `supports_tool_attempt_budget`: every tool the backend can run is a Kedi
  tool called through `fn`. Kedi refuses tool-attempt budgets for calls with
  MCP servers or external tools, whose attempts it cannot see.

Without the capability, a budget fails before the call: `Adapter
'<shortname>' cannot enforce request run budget for this execution`.

## The Request Size Ceiling

Before each call, Kedi estimates the request (one token per byte of prompt,
instructions, tool schemas, output schema, media, and history) and rejects it
when the estimate exceeds 85 % of the context window minus a 16,384-token
output reserve. The window is 200,000 tokens unless the profile sets
`context_limit`. A stateful adapter can make the history part of the estimate
exact by implementing `SupportsHistoryTokenEstimate`:

```python
from collections.abc import Mapping
from typing import Any

def estimate_history_tokens(self, state: Mapping[str, Any]) -> int:
    """Estimate the model-visible content of a resume_state, without metadata."""
```

Usage reported during the call is checked against the reservation as it
arrives.

## Subagent Usage Limits

A child's request, tool-call, and token limits are enforced from the same
reports; see [Subagents](subagents.md#usage-limits).

## Running It

```python title="run_budgets.py"
import asyncio
import json

from kedi.lang import compile_program, parse_program
from kedi.run_budget import RunBudgetExceeded

from backend import BackendRequest, Reply, ToolCall
from chat_adapter import ChatAdapter


class Script:
    def __init__(self) -> None:
        self.requests = 0

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests += 1
        if not any(m["role"] == "tool" for m in request.messages):
            return Reply(tool_calls=(ToolCall("c1", "lookup", {"city": "Paris"}),))
        return Reply(text="done")


def run(budget: str) -> tuple[object, Script]:
    backend = Script()
    source = f"""
@lookup(city: str) -> str:
  ###
  Look up a city.
  ###
  > tool:
    risk: read_only
  = `city`

> budget:
    {budget}
> use: lookup
[answer] << Look up Paris.
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=ChatAdapter(backend, model="m1"))
    try:
        return runtime.run_main(), backend
    except Exception as error:
        return error, backend
    finally:
        asyncio.run(runtime.aclose())


value, backend = run("request_limit: 2")
assert value == "done" and backend.requests == 2

error, backend = run("request_limit: 1")
assert isinstance(error, RunBudgetExceeded) or isinstance(
    getattr(error, "original", None), RunBudgetExceeded
), error
assert "Run request budget exhausted (limit 1, consumed 1)" in str(error)
assert backend.requests == 1

error, _backend = run("tool_attempt_limit: 0")
assert "Run tool_attempt budget exhausted (limit 0, consumed 0)" in str(error), error
```

A second request was refused before it reached the backend, and a tool
attempt beyond the limit ended the run instead of reaching the model as a
tool error.
