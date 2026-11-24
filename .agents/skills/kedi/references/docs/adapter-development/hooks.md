# Hooks

Hooks let a program or its host observe and change a run: edit or refuse the
prompt, edit or refuse a tool call, and observe tool results and failures.
Kedi owns the handlers, their order, timeouts, and decisions. The adapter
provides the points where they run and declares which events it supports.

## Events

| Event | When | Handler may | Request type |
|---|---|---|---|
| `user_prompt_submit` | before the request is assembled | continue, deny, or edit the prompt text | `UserPromptSubmitRequest` (`content`, read-only `media`) |
| `pre_tool_use` | before a tool runs, after argument validation | continue, deny, or edit the arguments | `PreToolUseRequest` (`tool_name`, `tool_call_id`, `origin`, `arguments`, ...) |
| `post_tool_use` | after a tool returned | observe | `PostToolUseEvent` (`result`) |
| `post_tool_use_failure` | after a tool raised | observe | `PostToolUseFailureEvent` (`error_type`, `error_message`, `interrupted`) |
| `post_tool_result_failure` | after a returned result could not be admitted | observe | `PostToolResultFailureEvent` |

Handlers come from three places, composed in this order: handlers registered
on the adapter instance (`adapter.on(...)`), the profile's `> hooks:`, and
hooks contributed by capabilities that own a tool.

## Declaring Support

Kedi refuses a program whose hooks the adapter cannot run, before any model
call:

- `capabilities.hooks` is `None`: `Agent hooks are not supported by adapter
  '<shortname>'`;
- an event is missing from it: `Agent hook events <names> are not supported by
  adapter '<shortname>'`.

`kedi.agent_adapter.HookCapabilities` lists each supported event with a
`HookEventCapability`:

| Field | Meaning |
|---|---|
| `bridge` | `"kedi"`: Kedi's dispatch functions run the handlers at the adapter's call sites. `"native"`: the backend's own hook system calls into Kedi. |
| `origins` | which tool sources the event covers: `kedi` (registered tools), `mcp`, `adapter_native`, `provider_builtin` |
| `supports_mutation` | whether an edit decision is applied |

`HookCapabilities.full(tool_bridge=, tool_origins=)` declares all five events
(`post_tool_result_failure` only for `kedi` tools); `HookCapabilities.prompt_only()`
declares only `user_prompt_submit`.

## Implementing Hooks

Three pieces:

1. **Registration.** Inherit `kedi.agent_adapter.HookRegistrationMixin` and
   call `self._init_hook_registration(hook_handler)` in `__init__`. The mixin
   provides `adapter.on(events)` and the snapshot Kedi reads. Kedi reads the
   profile's hooks from the adapter's `active_profile`.
2. **The prompt hook.** For an adapter based on the mixin, Kedi runs
   `user_prompt_submit` itself before assembling a program call, and binds the
   hook dispatcher for the whole call. Children and direct calls do not go
   through that path, so decorate the async entry points with
   `hooked_agent_run`: it runs the prompt hooks for such calls and does
   nothing when Kedi already ran them for the same run.
3. **Tool hooks.** Wrap every tool at its call site with
   `with_hook_lifecycle(tool, adapter_shortname=, tool_call_id=)` from
   `kedi.agent_adapter.hooks`. The wrapper dispatches `pre_tool_use`, then the
   tool, then `post_tool_use` or `post_tool_use_failure`. A `pre_tool_use`
   denial raises `HookDeniedError`, which the adapter returns to the model like
   an approval denial.

```python title="hooked_adapter.py"
from dataclasses import replace
from typing import Any

from kedi.agent_adapter import HookCapabilities, HookRegistrationMixin, ToolSpec
from kedi.agent_adapter.hooks import hooked_agent_run, with_hook_lifecycle

from backend import ToolCall
from chat_adapter import ChatAdapter


class HookedAdapter(HookRegistrationMixin, ChatAdapter):
    capabilities = replace(
        ChatAdapter.capabilities,
        hooks=HookCapabilities.full(tool_bridge="kedi", tool_origins=frozenset({"kedi"})),
    )

    def __init__(self, *args: Any, hook_handler: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_hook_registration(hook_handler)

    @hooked_agent_run("prompt")
    async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
        return await super().invoke(prompt=prompt, **kwargs)

    @hooked_agent_run("template")
    async def produce(self, *, template: str, **kwargs: Any) -> Any:
        return await super().produce(template=template, **kwargs)

    async def _execute(self, tools: dict[str, ToolSpec], call: ToolCall) -> str:
        tool = tools.get(call.name)
        if tool is not None:
            hooked = with_hook_lifecycle(
                tool, adapter_shortname=self.shortname, tool_call_id=call.id
            )
            tools = {**tools, call.name: hooked}
        return await super()._execute(tools, call)
```

`HookRegistrationMixin` comes first so that its attributes are set up before
the adapter's own; it has no `__init__` of its own.

## Errors

| Situation | Result |
|---|---|
| a `user_prompt_submit` handler denies | the call fails with `HookDeniedError: Kedi hook denied user_prompt_submit: <reason>` |
| a `pre_tool_use` handler denies | `HookDeniedError`; return `exc.message` to the model, the run continues |
| a handler raises or times out | `HookExecutionError` / `HookTimeoutError`; propagate, the run fails |
| an edit is not valid for the tool | the edited arguments are validated again and rejected like model arguments |

Never catch `HookExecutionError` to continue: a failing hook is a failing
safety or audit control.

## Running It

```python title="run_hooks.py"
import asyncio
import json

from kedi.agent_adapter.hooks import PreToolUseDecision, UserPromptSubmitDecision
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from hooked_adapter import HookedAdapter

SOURCE = """
@lookup(city: str) -> str:
  ###
  Look up a city.
  ###
  > tool:
    risk: read_only
  = `"found " + city`

> use: lookup
[answer] << Look up Lyon.
= <answer>
"""


class Script:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.prompts.append(request.messages[0]["content"])
        results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
        if not results:
            return Reply(tool_calls=(ToolCall("c1", "lookup", {"city": "Lyon"}),))
        return Reply(text=results[0].get("result") or results[0]["error"])


def run(adapter: HookedAdapter) -> str:
    runtime = compile_program(parse_program(SOURCE), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


observed: list[object] = []
backend = Script()
adapter = HookedAdapter(backend, model="m1")
adapter.on("user_prompt_submit")(
    lambda event: UserPromptSubmitDecision.edit(event.content + " Answer briefly.")
)
adapter.on("pre_tool_use")(lambda event: PreToolUseDecision.edit({"city": "Paris"}))
adapter.on("post_tool_use")(lambda event: observed.append(event.result))

assert run(adapter) == "found Paris"
assert backend.prompts[0] == "Look up Lyon. Answer briefly."
assert observed == ["found Paris"]

refusing = HookedAdapter(Script(), model="m1")
refusing.on("pre_tool_use")(lambda event: PreToolUseDecision.deny("lookups are off"))
assert run(refusing) == "Kedi hook denied pre_tool_use: lookups are off"
```
