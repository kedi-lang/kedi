# Approvals

Every tool call is approved before it runs. The policy comes from the program
(`> approval: allow`, `deny`, or a handler) and from the tool's risk. Who
applies it depends on one capability, `supports_native_approvals`.

## The Two Models

| | Kedi-wrapped approval (default) | Native approval |
|---|---|---|
| Capability | `supports_native_approvals` unset or `False` | `supports_native_approvals=True` |
| `tool.fn` | already applies the policy | runs the tool without asking |
| Children | Kedi applies every ancestor's policy to the child's tools | the adapter applies `request.approval_ceiling` |
| Backend-native tools (shell, file edits, provider tools) | not covered: the adapter must not expose them | the adapter's approval hook covers them |
| Use when | the backend only calls tools Kedi registered | the backend executes its own tools, or the framework has an approval step of its own |

An adapter without native approvals needs no approval code at all: calling
`tool.fn` asks the policy, and the [reference adapter](reference-adapter.md)
returns a denial to the model. Advertise native approvals only when the
adapter applies policies to every tool the backend can run.

## Requests and Decisions

`kedi.agent_adapter.approvals.policy`:

| Type | Fields |
|---|---|
| `ApprovalPolicy` | `mode` (`"allow"` / `"deny"`) or `handler` (a callable, sync or async); built with `ApprovalPolicy.allow()`, `.deny()`, `.dynamic(handler)` |
| `ApprovalRequest` | `tool_name`, `arguments` (validated, read-only), `risk` (`read_only`, `mutating`, `sensitive`), `adapter_shortname`, `description`, `metadata`, `reason`, `tool_reason_enabled` |
| `ApprovalDecision` | `decision` (`allow`, `deny`, `edit`), `edit_arguments`, `reason`; built with `.allow()`, `.deny(reason=)`, `.edit(arguments, reason=)` |

Resolution rules, the same for both models:

1. `read_only` tools are allowed without asking.
2. No policy (`None`) denies `mutating` and `sensitive` tools.
3. A static mode allows or denies.
4. A handler decides. An `edit` replaces the arguments; they are validated
   again, and an edit that raises the risk is asked again.
5. A denial raises `ToolApprovalDenied("Tool '<name>' was denied: <reason>")`.

## Native Approvals

With native approvals, Kedi registers unwrapped tools and the adapter applies
the policies for every call, in this order:

1. every policy in `approval_ceiling` (empty for a top-level call, the
   ancestors' policies inside `subagent_scope`), root first;
2. the tool's own policy: `tool_approval_policy(tool, profile.approval,
   arguments)` resolves `tool.approval`, a capability policy, or the profile's
   policy.

Each policy receives the arguments approved by the previous one:

```python title="native_approvals.py"
import json
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from kedi.agent_adapter import ToolSpec
from kedi.agent_adapter.approvals.policy import ApprovalPolicy, ToolApprovalDenied
from kedi.agent_adapter.tools.errors import correctable_tool_error
from kedi.agent_adapter.tools.execution import (
    resolve_tool_arguments_async,
    validate_tool_arguments,
)
from kedi.agent_adapter.tools.specs import tool_approval_policy

from backend import ToolCall
from chat_adapter import ChatAdapter


class NativeApprovalAdapter(ChatAdapter):
    capabilities = replace(ChatAdapter.capabilities, supports_native_approvals=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ceiling: ContextVar[tuple[ApprovalPolicy | None, ...]] = ContextVar(
            f"chat_ceiling_{id(self)}", default=()
        )

    @contextmanager
    def approval_ceiling(
        self, policies: tuple[ApprovalPolicy | None, ...]
    ) -> Generator[None, None, None]:
        token = self._ceiling.set(tuple(policies))
        try:
            yield
        finally:
            self._ceiling.reset(token)

    async def _approve(self, tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        local = tool_approval_policy(tool, self.active_profile.approval, arguments)
        approved = arguments
        for policy in (*self._ceiling.get(), local):
            approved = await resolve_tool_arguments_async(
                tool, approved, policy=policy, adapter_shortname=self.shortname
            )
        return approved

    async def _execute(self, tools: dict[str, ToolSpec], call: ToolCall) -> str:
        tool = tools.get(call.name)
        if tool is None:
            return await super()._execute(tools, call)
        try:
            arguments = await self._approve(tool, validate_tool_arguments(tool, call.arguments))
        except ToolApprovalDenied as exc:
            return json.dumps({"ok": False, "error": exc.message})
        except Exception as exc:
            message = correctable_tool_error(tool, exc)
            if message is None:
                raise
            return json.dumps({"ok": False, "error": message})
        return await super()._execute(tools, ToolCall(call.id, call.name, arguments))
```

The adapter enters `approval_ceiling(request.approval_ceiling)` in its
`subagent_scope`, next to the tool and profile scopes. Backend-native tools
need the same loop in the backend's approval callback, with an
`ApprovalRequest` built for the native operation.

Native adapters also own two details Kedi handles for wrapped tools:

- **Interactive handlers.** A handler may block for a person's answer. In
  async code, await `resolve_tool_arguments_async`; a handler that implements
  `SupportsAsyncApprovalResolution.resolve_async` is then asked without
  blocking the event loop. Advertise `supports_native_approval_handler` only
  when the backend's own approval prompt can drive such a handler.
- **Approval reasons.** With the `tool_reason` setting, the model supplies a
  reason with each call, and the reason reaches `ApprovalRequest.reason`.
  List `tool_reason` in `capabilities.settings` only when the adapter adds the
  reason argument to tool schemas and passes it to the request; see
  `kedi.agent_adapter.approvals.reason`.

## Running It

```python title="run_approvals.py"
import asyncio
import json

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from chat_adapter import ChatAdapter
from native_approvals import NativeApprovalAdapter

FENCE = "`" * 3
SOURCE = f"""
{FENCE}
from kedi import ApprovalDecision

def approve(request):
    if request.arguments["path"].startswith("reports/"):
        return ApprovalDecision.allow()
    return ApprovalDecision.edit({{"path": "reports/" + request.arguments["path"]}})
{FENCE}

@write_report(path: str) -> str:
  ###
  Write the report to a path.
  ###
  > tool:
    risk: mutating
  = `"wrote " + path`

> approval: `approve`
> use: write_report
[answer] << Write the report to notes.txt.
= <answer>
"""


class Script:
    async def complete(self, request: BackendRequest) -> Reply:
        results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
        if not results:
            return Reply(tool_calls=(ToolCall("c1", "write_report", {"path": "notes.txt"}),))
        return Reply(text=results[0].get("result") or results[0]["error"])


def run(source: str, adapter_type: type[ChatAdapter]) -> str:
    runtime = compile_program(parse_program(source), adapter=adapter_type(Script(), model="m1"))
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


for adapter_type in (ChatAdapter, NativeApprovalAdapter):
    assert run(SOURCE, adapter_type) == "wrote reports/notes.txt", adapter_type
    denied = run(SOURCE.replace("> approval: `approve`", "> approval: deny"), adapter_type)
    assert "was denied" in denied, (adapter_type, denied)
```

Both adapters apply the same policy: the handler's edit reaches the tool, and
`> approval: deny` returns a denial to the model.
