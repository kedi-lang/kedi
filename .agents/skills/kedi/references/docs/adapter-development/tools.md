# Tools

Kedi decides which tools a call may use and wraps each one with its safety
and bookkeeping. The adapter declares those tools to the backend, executes the
calls the model makes, and returns the results to the model. It implements two
methods.

## The Protocol

`kedi.agent_adapter.SupportsToolRegistration`:

```python
from collections.abc import Sequence
from contextlib import AbstractContextManager
from kedi.agent_adapter import ToolSpec

def tool_scope(self, tools: Sequence[object] = ()) -> AbstractContextManager[None]: ...

def register_tool(self, tool: ToolSpec) -> None: ...
```

Advertise `supports_tool_registration=True`. Without the protocol, a profile
with tools fails before the call with `AgentToolOverrideUnsupported: Active
adapter does not support tool registration (requested tools=...)`.

## Scopes

For every model call, Kedi enters `tool_scope()` and then calls
`register_tool` once per tool (see [Call Lifecycle](call-lifecycle.md)). For a
child run the adapter enters `tool_scope(request.tools)` itself (see
[Subagents](subagents.md)). A scope must:

- **start from exactly the given tools**, not from the enclosing scope's: an
  empty `tool_scope()` means no tools until Kedi registers them;
- **replace a tool registered again under the same name**;
- **restore the enclosing scope on exit**, also on error;
- **be isolated per task and thread**: parallel blocks, background children,
  and nested calls each have their own scope.

A `ContextVar` holding a tuple gives all four, as in the [reference
adapter](reference-adapter.md#tools). Kedi's own adapters use
`kedi.agent_adapter.tools.scope.ToolScope`, which implements the same rules
and can be used directly:

```python
from kedi.agent_adapter.tools.scope import ToolScope

self._scope = ToolScope()

def tool_scope(self, tools=()):
    return self._scope.tool_scope(tools)

def register_tool(self, tool):
    self._scope.register_tool(tool)

# the active tools: self._scope.scoped_tools
```

## What a `ToolSpec` Carries

| Field | Use in the adapter |
|---|---|
| `name` | the tool name the model sees and calls |
| `description` | the tool description, sent unchanged |
| `json_schema` | the JSON Schema of the arguments; `None` means an object without properties |
| `return_schema` | the JSON Schema of the result, for backends that accept output schemas on tools |
| `fn` | the callable to execute, already wrapped by Kedi; keyword arguments only |
| `sequential` | `True`: never run this call concurrently with another call of the same turn |
| `metadata` | `source` (`kedi`, `mcp`, `external`, ...) and `kind` (`subagent`, `artifact_management`, `filesystem`, ...) for routing and telemetry; `required_before_output` |
| `risk`, `risk_resolver`, `approval`, `capability_policy` | inputs to Kedi's approval; the adapter needs them only with native approvals (see [Approvals](approvals.md)) |
| `argument_validator` | used through `validate_tool_arguments`, below |
| `artifact_result`, `result_admitter`, `retry_policy`, `proc`, `deferred_capability` | Kedi-internal; applied inside `fn` |

Declare to the backend exactly `name`, `description`, and `json_schema`; do
not rename tools, merge descriptions, or simplify schemas. For a Kedi
procedure, Kedi builds the description from the procedure's `###` docstring
and its return type (``Double a number. Returns `int` instance.``), and the
schema from its typed parameters.

## Executing a Call

```python
arguments = validate_tool_arguments(tool, call.arguments)
result = tool.fn(**arguments)
if inspect.isawaitable(result):
    result = await result
```

1. **Validate.** `kedi.agent_adapter.tools.execution.validate_tool_arguments`
   applies the tool's `argument_validator`: it binds the arguments to the
   tool's signature, fills defaults, and checks each value against its declared
   type, building `~Type` instances from objects. Types are checked strictly:
   `"4"` is not accepted for an `int`. A rejected argument raises
   `ToolArgumentError`, a `ValueError`, which follows the correctable path
   below, so the model reads which parameter was wrong and can call the tool
   again.
2. **Call `fn` with keyword arguments.** Kedi has wrapped it, so the call:
   - asks the approval policy and may edit the arguments or raise
     `ToolApprovalDenied` (adapters without native approvals);
   - marks a `required_before_output` tool as attempted and returned;
   - runs the tool's own retries (`> tool: retries:`); never retry a tool
     call in the adapter;
   - admits a large result as an artifact and returns its reference instead
     (see [Artifacts](artifacts.md));
   - records the tool span and the run's tool history.
3. **Await** an awaitable result. Delegation, lifecycle, workflow, and many
   procedure tools are coroutines.
4. **Return the result to the model** as the backend's tool-result message,
   serialized with `pydantic_core.to_jsonable_python(result, fallback=str)`.
   Keep the tool-call id paired with its result.
5. **On failure**, follow [Errors, Retries, and
   Cancellation](errors.md#tool-failures): denials and correctable errors go
   back to the model, everything else propagates.

When the backend returns several tool calls in one turn, they may run
concurrently, except calls to `sequential` tools, which must run alone.

## Tools That Must Run Before the Answer

A tool with `metadata["required_before_output"] is True` must return
successfully before the model's final answer is accepted. Kedi checks it after
the adapter returns and rejects the result with `RequiredToolCallError:
Required tool <name> was not called; the adapter response cannot be accepted`.
Kedi's dynamic workflow tool, `run_workflow`, is such a tool.

To give the model a chance to recover instead, check the state before
accepting a final answer:

```python
from kedi.agent_adapter.tools.scope import current_required_tool_calls

state = current_required_tool_calls()
missing = set() if state is None else state.names - state.returned
```

When `missing` is not empty, keep the turn going and ask the model to call the
tools.

## Deferred and Capability Tools

A `> capability:` with `defer: true` registers tools whose
`deferred_capability` names a capability the model has not loaded yet. Only
adapters that advertise `supports_deferred_tools` receive them; Kedi refuses
the program for every other adapter. See [MCP, Skills, and Capability
Tools](mcp-and-skills.md).

## Running It

```python title="run_tools.py"
import asyncio
import json

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from chat_adapter import ChatAdapter

SOURCE = """
@double(n: int) -> int:
  ###
  Double a number.
  ###
  > tool:
    risk: read_only
  = `n * 2`

> use: double
[answer] << Double four.
= <answer>
"""


class Script:
    def __init__(self, arguments: dict) -> None:
        self.arguments = arguments
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
        if not results:
            return Reply(tool_calls=(ToolCall("c1", "double", self.arguments),))
        return Reply(text=json.dumps(results[0]))


def run(arguments: dict) -> tuple[dict, BackendRequest]:
    backend = Script(arguments)
    runtime = compile_program(parse_program(SOURCE), adapter=ChatAdapter(backend, model="m1"))
    try:
        return json.loads(runtime.run_main()), backend.requests[0]
    finally:
        asyncio.run(runtime.aclose())


result, first = run({"n": 4})
assert result == {"ok": True, "result": 8}
[declaration] = [tool for tool in first.tools if tool["name"] == "double"]
assert declaration["description"] == "Double a number. Returns `int` instance."
assert declaration["parameters"]["properties"]["n"]["type"] == "integer"
assert declaration["parameters"]["required"] == ["n"]

result, _first = run({"n": "four"})
assert result["ok"] is False
assert result["error"].startswith("Tool double failed: ToolArgumentError"), result
```

A valid call ran the procedure. An argument of the wrong type did not end the
run: the model received Kedi's correction and could call the tool again.
