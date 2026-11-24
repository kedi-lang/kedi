# Errors, Retries, and Cancellation

An adapter meets three kinds of failure: a tool call fails, a backend request
fails, or the run is cancelled. Kedi decides which failures the model may see
and correct, which ones are retried, and which ones end the program. The
adapter follows those decisions with three helpers.

## Tool Failures

When `tool.fn(**arguments)` raises, the adapter must pick one of three paths:

| Failure | Path | Example |
|---|---|---|
| `ToolApprovalDenied` (approval policy) or `HookDeniedError` (`pre_tool_use` hook) | return `exc.message` to the model as a failed tool result | `Tool 'delete_file' was denied: static deny policy` |
| `correctable_tool_error(tool, exc)` returns a message | return that message to the model | `Tool lookup failed: ValueError: unknown city. Correct the arguments if they caused this, or continue without this result.` |
| `correctable_tool_error(tool, exc)` returns `None` | re-raise; the program call fails | a `RuntimeError` in a procedure, `RunBudgetExceeded`, run cancellation |

`kedi.agent_adapter.tools.errors.correctable_tool_error(tool, exc)` is the
classification Kedi's built-in adapters use. It returns Kedi's model-facing
message for:

- application tools (Kedi procedures and Python tools) that raise
  `ValueError`, `LookupError`, or `OSError`: rejected input, a missing record,
  unreachable data;
- Kedi's own tool families: subagent delegation and lifecycle, dynamic
  workflows, artifact management, filesystem and terminal tools, CodeMode
  controls, and capability activation;

and `None` for budget exhaustion (`RunBudgetExceeded`), cancellation, failures
after a tool already returned, and every other exception. Do not write your
own classification or your own message; a message from this function is the
text the model should read.

Kedi's errors (`KediExecutionError` and its subclasses, including both denial
types) render a Kedi stack trace in `str(exc)`. Send `exc.message`, the plain
message, to the model.

The reference adapter's `_execute` implements exactly this table. A framework
with a tool-retry envelope (Pydantic AI's `ModelRetry`, LangChain's
`ToolMessage` with an error status) carries the same message through that
envelope instead of a JSON result.

## Request Failures

A backend request can fail transiently: a rate limit, an overloaded server, a
timeout. Retry only the request, never the whole run, and never a tool call:

```python
from kedi.agent_adapter.request_retry import wait_for_request_retry

attempt = 0
while True:
    try:
        return await backend.complete(request)
    except Exception as error:
        attempt += 1
        if not await wait_for_request_retry(error, attempt=attempt, retries=retries):
            raise
```

`wait_for_request_retry(error, attempt=, retries=)`:

- returns `False` when `attempt > retries` or the error is not transient;
- treats an error as transient when it has an integer `status_code` of 408,
  429, or 5xx, so keep the provider's exception (or give yours a
  `status_code`);
- waits for the provider's `Retry-After` (read from `error.response.headers`,
  following `__cause__`), or for exponential backoff from 0.5 s, and returns
  `True`;
- returns `False` instead of waiting longer than 30 seconds.

Kedi's built-in adapters default to two retries and read the
`request_retries` setting; `request_retry_limit(settings, default)` validates
that setting. Report usage only for requests that completed.

When the request finally fails, raise. Kedi wraps the exception in a
`KediExecutionError` with the Kedi source location and keeps the original as
`error.original`, so the retry policy of the job-manager engine can still
classify a `429` or a timeout. Never turn a failed request into an empty
result, and never fall back to a text request after a structured one failed.

## Cancellation

A run is cancelled when the program's caller cancels it, when a parent cancels
a child, when a CodeMode or workflow sandbox stops, or when an interrupt
supersedes a child attempt. Cancellation arrives as `asyncio.CancelledError`
in the adapter's coroutine, or as `RunCancelled` when synchronous work checks
`check_run_cancelled()`.

- Let both propagate. Do not catch `BaseException` or `asyncio.CancelledError`
  to return a value.
- Release what the call owns in `finally`: streams, sessions, subprocesses.
  `run_sync` gives the cancelled coroutine time to clean up.
- A cancelled call must not write continuation state (`result_state`) or
  report a result; Kedi discards the turn.
- A cancellation request is not proof that a remote side effect was undone.
  Do not report one as reversed.

## Error Messages

Raise errors that name the adapter and the failing part:

| Situation | Raise |
|---|---|
| a profile field the adapter does not support | `AgentProfileOverrideUnsupported` from `profile_override` |
| an MCP transport the backend cannot open | `ValueError` naming the transport, before the request |
| the backend returned no text or no structured value | `RuntimeError` naming the adapter |
| a protocol or stream error from the backend | the provider's exception, or one chained to it with `from` |

## Running It

```python title="run_errors.py"
import asyncio
import json
from collections.abc import Callable

from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from chat_adapter import ChatAdapter


class Scripted:
    def __init__(self, answer: Callable[[BackendRequest], Reply]) -> None:
        self.answer = answer
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return self.answer(request)


def run(source: str, backend) -> object:
    runtime = compile_program(parse_program(source), adapter=ChatAdapter(backend, model="m1"))
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def tool_program(body: str, approval: str = "allow", risk: str = "read_only") -> str:
    return f"""
@lookup(city: str) -> str:
  ###
  Look up a city.
  ###
  > tool:
    risk: {risk}
  = `{body}`

> approval: {approval}
> use: lookup
[answer] << Look up Atlantis.
= <answer>
"""


def call_then_echo(request: BackendRequest) -> Reply:
    results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
    if not results:
        return Reply(tool_calls=(ToolCall("c1", "lookup", {"city": "Atlantis"}),))
    return Reply(text=results[0]["error"])
```

A correctable failure reaches the model with Kedi's message:

```python title="run_errors.py"
answer = run(tool_program("(_ for _ in ()).throw(ValueError('unknown city'))"), Scripted(call_then_echo))
assert answer.startswith("Tool lookup failed: ValueError: unknown city."), answer
```

A denial reaches the model, and the run continues:

```python title="run_errors.py"
answer = run(tool_program("'ok'", approval="deny", risk="mutating"), Scripted(call_then_echo))
assert "was denied" in answer, answer
```

Any other failure ends the call:

```python title="run_errors.py"
try:
    run(tool_program("(_ for _ in ()).throw(RuntimeError('database is down'))"), Scripted(call_then_echo))
except KediExecutionError as error:
    assert "database is down" in str(error)
else:
    raise AssertionError("an unclassified tool failure must end the call")
```

A rate limit is retried; a bad request is not:

```python title="run_errors.py"
class ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = type("Response", (), {"headers": {"retry-after": "0"}})()


class Flaky:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.calls = 0

    async def complete(self, request: BackendRequest) -> Reply:
        self.calls += 1
        if self.calls == 1:
            raise ProviderError(self.status_code)
        return Reply(text="ok")


flaky = Flaky(429)
assert run("[answer] << Say ok.\n= <answer>\n", flaky) == "ok"
assert flaky.calls == 2

rejected = Flaky(400)
try:
    run("[answer] << Say ok.\n= <answer>\n", rejected)
except KediExecutionError as error:
    assert isinstance(error.original, ProviderError)
else:
    raise AssertionError("a 400 must not be retried")
assert rejected.calls == 1
```
