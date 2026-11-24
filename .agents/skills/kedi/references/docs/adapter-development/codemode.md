# CodeMode

With `> codemode: enabled`, the model does not call the profile's tools one by
one. It gets three control tools and writes short Python programs that call
the application tools inside a sandbox (Monty). Kedi provides the sandbox, the
catalog, the limits, and the tool boundary; the adapter swaps the tools it
exposes for the controls.

## The Controls

| Tool | Purpose |
|---|---|
| `search_tools` | browse the catalog of application tools; returns names and a cursor |
| `get_tool_schema` | return exact schemas for chosen tools and make them callable in the sandbox |
| `execute_code` | run one bounded code cell; tools are awaited: `result = await lookup(city="Paris")` |

The application tools (procedures, Python tools, MCP tools the adapter
connected, artifact and skill tools) stay behind the controls. Each nested
call still goes through the tool's wrapped `fn`, so approval, hooks, artifact
admission, budgets, and telemetry apply per call.

## Implementing It

Kedi checks `supports_codemode` before the call: `CodeMode is not supported by
adapter '<shortname>'`. The adapter reads the policy from the active profile
and builds a bridge for the run:

| Step | Call |
|---|---|
| read the policy | `settings = normalize_codemode_settings(profile.codemode)` from `kedi.agent_profile`; `None` or `enabled=False` means no CodeMode |
| build the run's bridge | `CodeModeToolSpecBridge(tools, limits=CodeModeLimits(**settings.limit_values()), policies=..., adapter_shortname=..., preload_tools=settings.preload_tools)` from `kedi.agent_adapter.codemode` |
| expose the controls | declare `bridge.control_tools` to the backend instead of the application tools |
| add the instructions | `append_codemode_instructions(instructions, enabled=True, preloaded_tools=bridge.preloaded_tools)` |
| close | `await bridge.close()` when the run ends, also on error |

`policies` are the approval policies the bridge applies to each nested call.
Kedi's wrapped tools already apply theirs, so an adapter without native
approvals passes `()`; an adapter with native approvals passes the same chain
it applies to direct calls (see [Approvals](approvals.md)).

CodeMode needs inline approval: a handler that would defer a decision cannot
suspend a partially executed cell. The controls are `sequential` and never
produce artifacts; their results go to the model as they are.

```python title="codemode_adapter.py"
from dataclasses import replace
from typing import Any

from kedi.agent_adapter.codemode import (
    CodeModeLimits,
    CodeModeToolSpecBridge,
    append_codemode_instructions,
)
from kedi.agent_profile import normalize_codemode_settings

from backend import Reply
from chat_adapter import ChatAdapter


class CodeModeAdapter(ChatAdapter):
    capabilities = replace(ChatAdapter.capabilities, supports_codemode=True)

    async def _run(self, **kwargs: Any) -> tuple[Reply, list[dict[str, Any]]]:
        settings = normalize_codemode_settings(self.active_profile.codemode)
        if settings is None or not settings.enabled:
            return await super()._run(**kwargs)
        bridge = CodeModeToolSpecBridge(
            self._tools.get(),
            limits=CodeModeLimits(**settings.limit_values()),
            policies=(),
            adapter_shortname=self.shortname,
            preload_tools=settings.preload_tools,
        )
        kwargs["instructions"] = append_codemode_instructions(
            kwargs.get("instructions"), enabled=True, preloaded_tools=bridge.preloaded_tools
        )
        token = self._tools.set(bridge.control_tools)
        try:
            return await super()._run(**kwargs)
        finally:
            self._tools.reset(token)
            await bridge.close()
```

## Running It

```python title="run_codemode.py"
import asyncio
import json

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from codemode_adapter import CodeModeAdapter

SOURCE = """
@population(city: str) -> int:
  ###
  Return the population of a city.
  ###
  > tool:
    risk: read_only
  = `{"Paris": 2_100_000, "Lyon": 520_000}[city]`

> codemode: enabled
> use: population
[answer] << What is the combined population of Paris and Lyon?
= <answer>
"""

CODE = 'a = await population(city="Paris")\nb = await population(city="Lyon")\na + b'


class Script:
    def __init__(self) -> None:
        self.tools: list[list[str]] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.tools.append(sorted(tool["name"] for tool in request.tools))
        results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
        if not results:
            arguments = {"tool_names": ["population"]}
            return Reply(tool_calls=(ToolCall("c1", "get_tool_schema", arguments),))
        if len(results) == 1:
            return Reply(tool_calls=(ToolCall("c2", "execute_code", {"code": CODE}),))
        return Reply(text=str(results[1]["result"]))


backend = Script()
runtime = compile_program(parse_program(SOURCE), adapter=CodeModeAdapter(backend, model="m1"))
try:
    assert runtime.run_main() == "2620000"
finally:
    asyncio.run(runtime.aclose())

assert backend.tools[0] == ["execute_code", "get_tool_schema", "search_tools"]
```

The model saw only the three controls; one `execute_code` cell called the
procedure twice through its wrapped `fn` and combined the results.
