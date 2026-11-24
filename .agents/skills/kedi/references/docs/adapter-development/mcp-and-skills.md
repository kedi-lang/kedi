# MCP, Skills, and Capability Tools

Three profile features add tools beyond the program's own procedures. Two of
them, skills and eager capabilities, Kedi resolves completely; the adapter
sees ordinary tools and instructions. MCP servers are different: the adapter
connects to them, so it takes over the work Kedi's tool wrappers do for
registered tools.

## Skills

`> skills:` enables project-local skill discovery. Kedi adds the skill tools,
`read_only` and already wrapped like every registered tool, and the
`skills.instructions` part. The adapter needs no skill support beyond
[tool registration](tools.md); without it, Kedi refuses the profile with
`AgentToolOverrideUnsupported: Active adapter does not support skill tool
registration`.

## Capabilities

`> capability: name:` defines a bundle of instructions, tools, MCP servers,
approval, and tool hooks that a profile applies with `> use:`.

| Kind | What the adapter sees | Capability flag |
|---|---|---|
| eager (default) | the bundle's instructions as `capability:<identity>` instruction parts, its tools as registered tools whose `capability_policy` carries the bundle's approval and hooks, its MCP servers in `mcp_servers` | `supports_capabilities` |
| deferred (`defer: true`) | tools whose `deferred_capability` names a bundle that the model must activate first | `supports_deferred_tools` |

For eager bundles the adapter has nothing to implement: calling the wrapped
`fn` applies the bundle's approval, and `with_hook_lifecycle` (see
[Hooks](hooks.md)) dispatches its tool hooks. Advertise
`supports_capabilities=True` when the adapter registers tools and runs hooks
that way; applying a bundle adds an implicit `capabilities` requirement, and
Kedi refuses it for an adapter without the flag.

Deferred activation is a native protocol between Kedi and the framework's own
tool loading, implemented today by the Pydantic AI adapter only. A custom
adapter leaves `supports_deferred_tools` unset; Kedi then refuses deferred
bundles before inference.

## MCP Servers

`> mcp:` adds servers to `profile.mcp_servers`, a tuple of
`kedi.agent_profile.McpServerSpec`:

| Field | Meaning |
|---|---|
| `transport` | `stdio`, `sse`, or `http` |
| `command`, `args`, `env` | how to start a `stdio` server |
| `url`, `headers` | where to reach an `sse` or `http` server |

An adapter that advertises `supports_mcp=True` connects every server in the
active profile for the call, lists its tools, exposes them to the model, and
closes the connections when the call ends. It refuses, in `profile_override`,
any transport it cannot open; an adapter without MCP support refuses every
server (see [Profiles, Models, and Settings](profiles.md)).

MCP tools are not Kedi tools, so nothing wraps them. For each MCP call the
adapter must do what the wrapper does for a registered tool:

| Step | How |
|---|---|
| approve | describe the call as a `ToolSpec` with `metadata={"source": "mcp"}` and the default `risk="mutating"`, and apply the policies: `request.approval_ceiling` inside a child, then `tool_approval_policy(spec, profile.approval)` |
| hooks | `with_hook_lifecycle(spec, adapter_shortname=, origin="mcp", tool_call_id=)` |
| artifacts | `admit_current_artifact_result(result, tool_name=, call_id=)` before the result reaches the model, and advertise `supports_native_artifacts` |
| usage | MCP calls are invisible to Kedi's tool-attempt budget, which Kedi refuses for profiles with MCP servers |

A server's tool cannot declare itself read-only to Kedi: an MCP tool is
`mutating` unless the adapter has a trusted reason to classify it otherwise.

```python title="mcp_tools.py"
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from kedi.agent_adapter import ToolSpec
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.tools.execution import with_approval
from kedi.agent_adapter.tools.specs import tool_approval_policy


def mcp_tool_spec(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    call: Callable[[dict[str, Any]], Awaitable[Any]],
    profile_approval: ApprovalPolicy | None,
    approval_ceiling: Sequence[ApprovalPolicy | None],
    adapter_shortname: str,
) -> ToolSpec:
    """An MCP tool wrapped with the same approval chain as a registered tool."""

    async def run(**arguments: Any) -> Any:
        return await call(arguments)

    spec = ToolSpec(
        fn=run,
        name=name,
        description=description,
        json_schema=input_schema,
        metadata={"source": "mcp"},
    )
    local = tool_approval_policy(spec, profile_approval)
    for policy in reversed((*approval_ceiling, local)):
        spec = with_approval(spec, policy=policy, adapter_shortname=adapter_shortname)
    return spec
```

## Running It

```python title="run_mcp_and_skills.py"
import asyncio
from dataclasses import replace

from kedi.agent_adapter.approvals.policy import ApprovalPolicy, ToolApprovalDenied
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter
from mcp_tools import mcp_tool_spec


class CapabilityAdapter(ChatAdapter):
    capabilities = replace(ChatAdapter.capabilities, supports_capabilities=True)


class Recorder:
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return Reply(text="ok")


SOURCE = """
@lookup(city: str) -> str:
  ###
  Look up a city.
  ###
  > tool:
    risk: read_only
  = `city`

> capability: evidence:
    > instructions: Cite the supplied evidence.
    > use: lookup

> use: evidence
[answer] << Is Paris the capital of France?
= <answer>
"""

backend = Recorder()
runtime = compile_program(parse_program(SOURCE), adapter=CapabilityAdapter(backend, model="m1"))
try:
    assert runtime.run_main() == "ok"
finally:
    asyncio.run(runtime.aclose())
[request] = backend.requests
assert "Cite the supplied evidence." in (request.instructions or "")
assert "lookup" in {tool["name"] for tool in request.tools}


calls: list[dict] = []


async def remote_delete(arguments: dict) -> str:
    calls.append(arguments)
    return "deleted"


def delete_tool(profile_approval: ApprovalPolicy | None, ceiling=()):
    return mcp_tool_spec(
        name="delete_page",
        description="Delete a wiki page.",
        input_schema={"type": "object", "properties": {"page": {"type": "string"}}},
        call=remote_delete,
        profile_approval=profile_approval,
        approval_ceiling=ceiling,
        adapter_shortname="chat",
    )


async def main() -> None:
    assert await delete_tool(ApprovalPolicy.allow()).fn(page="Draft") == "deleted"
    for tool in (delete_tool(None), delete_tool(ApprovalPolicy.allow(), (ApprovalPolicy.deny(),))):
        try:
            await tool.fn(page="Home")
        except ToolApprovalDenied:
            pass
        else:
            raise AssertionError("a denied MCP call must not run")


asyncio.run(main())
assert calls == [{"page": "Draft"}]
```

The capability's instructions and tool reached the adapter like the program's
own; the MCP tool ran only when every policy allowed it, and a parent's
`deny` held even though the child allowed the call.
