# Subagents

A profile with `> subagent: name` lets its model delegate tasks to child
profiles. Kedi owns delegation: the delegation tools, budgets, run records,
validation of child results, continuation history, cancellation, and the
approval and sandbox limits that children inherit. The adapter owns one thing:
running a child task on its backend when Kedi asks.

This page adds that capability to the [reference adapter](reference-adapter.md).

## What Kedi Does and What the Adapter Does

| Step | Owner |
|---|---|
| Offer `delegate_task` and the other control tools to the parent model | Kedi, through `register_tool` |
| Check `max_agents`, depth, concurrency, and the parent's allowed children | Kedi |
| Evaluate the child profile and resolve the child's adapter | Kedi |
| Resolve the child's tools and instructions, apply inherited approval and cwd/sandbox limits | Kedi |
| Isolate the child's profile, tools, and conversation | **Adapter**, in `subagent_scope` |
| Run the child task and return text or a JSON object | **Adapter**, in `invoke` / `produce_subagent` |
| Report each backend request | **Adapter**, with `report_invocation` |
| Validate the child result and feed corrections back to the model | Kedi supplies the validator and the feedback text; the adapter runs the retry |
| Record the run, enforce usage limits, publish the result to the parent | Kedi |

## How Delegation Reaches the Parent

Delegation needs nothing from the parent adapter beyond tool registration.
When the active profile declares children, Kedi adds these tools to the call's
tool set and registers them with `register_tool`, like any other tool:

| Tool | Offered when | Purpose |
|---|---|---|
| `delegate_task` | always | start a child; `final_schema` requests a JSON result |
| `continue_subagent` | always | continue a completed child conversation |
| `wait_subagent`, `cancel_subagent`, `subagent_status` | the adapter advertises `supports_background_subagents` | manage children started with `background: true` |
| `send_subagent_input`, `send_parent_message` | the adapter advertises `supports_live_subagent_input` | message a running child or its parent |
| `run_workflow` | the profile uses `> workflow: dynamic` | run generated orchestration code; children still run through Kedi |

All of them are async `ToolSpec` functions with `risk: read_only`. Starting an
isolated child is not itself a mutation; the child's own tools carry the risk.
The adapter executes them like every other tool, which is why the reference
adapter awaits awaitable tool results.

Capability checks happen before the parent runs:

| Condition | Error |
|---|---|
| The parent's adapter does not advertise `supports_subagents` | `AgentSubagentUnsupported: Active adapter does not support Kedi subagent delegation` |
| The model passes `background: true` without `supports_background_subagents` | the tool call fails with `The active adapter does not support background subagents` |
| The child's adapter does not implement the protocol below | `KediExecutionError: Adapter '<shortname>' does not implement Kedi child execution` |

A child profile may select a different adapter (`> adapter:` or `> agent:`);
the child's adapter, not the parent's, runs the child.

## How Kedi Runs a Child

For each child run, Kedi:

1. evaluates the child profile and applies the inherited cwd/sandbox limits;
2. resolves the child's adapter and checks the protocol;
3. resolves the child's tools, already wrapped with the child's approval
   policy and every ancestor's policy (see [Approvals](#approvals-in-children));
4. renders the child's instructions;
5. calls `subagent_run_kwargs(limits)` with the remaining usage limits and
   adds `instructions` to the result;
6. enters `subagent_scope(request)`;
7. calls `await invoke(prompt=task, **run_kwargs)` for a text child, or
   `await produce_subagent(template=task, output_schema=schema, **run_kwargs)`
   when a result schema applies;
8. validates the result, records it, and returns it to the parent's tool call.

Kedi calls the async entry points. A child runs on Kedi's shared event loop,
never through `invoke_sync`.

## The Protocol

`kedi.agent_adapter.subagents.SupportsSubagentExecution`:

```python
class SupportsSubagentExecution(Protocol):
    shortname: str

    def telemetry_model_name(self) -> str | None: ...

    def subagent_scope(
        self, request: SubagentExecutionRequest
    ) -> AbstractContextManager[None]: ...

    def subagent_run_kwargs(self, limits: SubagentUsageLimits) -> Mapping[str, Any]: ...

    async def invoke(self, *, prompt: str, **kwargs: Any) -> str: ...

    async def produce_subagent(
        self, *, template: str, output_schema: dict[str, Any], **kwargs: Any
    ) -> Mapping[str, Any]: ...
```

Add `supports_subagents=True` to the capabilities. An optional class
attribute, `supports_native_subagent_resume`, changes how continuations are
sent; see [Continuations](#continuations).

### The Request

`subagent_scope` receives a `SubagentExecutionRequest`:

| Field | Type | What the adapter does with it |
|---|---|---|
| `profile` | `AgentProfile` | Apply it for the child's calls, as `profile_override` does: model, and any other field the adapter supports. |
| `tools` | `tuple[ToolSpec, ...]` | Replace the parent's tools with exactly these. They include the child's Kedi tools, its artifact and skill tools, and its own delegation tools when it has children. |
| `instructions` | `str \| None` | Also passed to `invoke` / `produce_subagent` as `instructions`; send it as the system message. |
| `approval_ceiling` | `tuple[ApprovalPolicy \| None, ...]` | Only for adapters with `supports_native_approvals`: apply every policy, in order, to native and Kedi tools. Otherwise Kedi has already applied them. |
| `usage_limits` | `SubagentUsageLimits` | The same limits passed to `subagent_run_kwargs`. |
| `output_state` | `SubagentOutputState \| None` | Pass to `accept_subagent_output` so a profile `> output:` type is validated natively. |
| `conversation` | `SubagentConversationContext` | Native continuation state: read `resume_state`, write `result_state`. |

## Implementing the Protocol

```python title="subagent_adapter.py"
from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from jsonschema.validators import validator_for

from kedi.agent_adapter.output_schema import OutputCorrection
from kedi.agent_adapter.subagents import (
    SubagentConversationContext,
    SubagentExecutionRequest,
    SubagentUsageLimits,
    accept_subagent_output,
)

from chat_adapter import ChatAdapter


class SubagentChatAdapter(ChatAdapter):
    capabilities = replace(
        ChatAdapter.capabilities,
        supports_subagents=True,
        supports_background_subagents=True,
    )
    supports_native_subagent_resume = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._child: ContextVar[SubagentExecutionRequest | None] = ContextVar(
            f"chat_child_{id(self)}", default=None
        )

    def telemetry_model_name(self) -> str | None:
        return self.active_profile.model or self._model
```

`telemetry_model_name` names the model in the child's run record and spans
when the child profile does not set `> model:`.

### Scoping a Child

```python title="subagent_adapter.py"
    @contextmanager
    def subagent_scope(self, request: SubagentExecutionRequest) -> Generator[None, None, None]:
        with self.tool_scope(request.tools), self.profile_override(request.profile):
            token = self._child.set(request)
            try:
                yield
            finally:
                self._child.reset(token)

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        # A Kedi model call made by one of the child's tools opens a new tool scope.
        # It is not the child's own turn, so it must not see the child request.
        token = self._child.set(None)
        try:
            with super().tool_scope(tools):
                yield
        finally:
            self._child.reset(token)
```

Two details matter:

- **Replace, do not extend, the tools.** `tool_scope(request.tools)` sets the
  child's tool set; the parent's tools, including its own `delegate_task`,
  must not be visible to the child.
- **Mark only the child's own turn.** The child request lives in a
  `ContextVar`, so it would be inherited by anything the child's tools do. A
  Kedi procedure used as a tool may run its own `>>` template, and Kedi enters
  `tool_scope()` before every program call. Clearing the child request in
  `tool_scope` keeps that nested call an ordinary call.

### Usage Limits

```python title="subagent_adapter.py"
    @staticmethod
    def subagent_run_kwargs(limits: SubagentUsageLimits) -> Mapping[str, Any]:
        return {"max_requests": limits.request_limit}
```

`SubagentUsageLimits` carries the remaining budget of the logical child task:
`request_limit` (default 8), `tool_calls_limit` (default 16), and optional
`input_tokens_limit`, `output_tokens_limit`, and `total_tokens_limit`. The
mapping you return is passed to `invoke` / `produce_subagent` as keyword
arguments, so translate the limits into whatever your backend understands.

Kedi enforces the limits from the usage you report with `report_invocation`:
a report that exceeds a limit raises `KediExecutionError: Subagent request
limit 8 exceeded: observed 9` inside `report_invocation`, and the child fails.
A continuation whose budget is already spent fails before the adapter is
called, with `Subagent request_limit exhausted; task was not restarted`.
Passing the limits to the backend stops the run earlier and cheaper, but
reporting usage is what makes the limits hold.

### Text Children

```python title="subagent_adapter.py"
    async def invoke(
        self,
        *,
        prompt: str,
        instructions: str | None = None,
        max_requests: int | None = None,
        **kwargs: Any,
    ) -> str:
        child = self._child.get()
        if child is None:
            return await super().invoke(prompt=prompt, instructions=instructions, **kwargs)
        reply, messages = await self._run(
            prompt=prompt,
            instructions=instructions,
            output_schema=None,
            history=resume_messages(child.conversation),
            max_requests=max_requests,
        )
        if reply.text is None:
            raise RuntimeError("The child returned no text")
        save_messages(child.conversation, messages)
        return reply.text
```

Without a result schema the child's text becomes the delegation's
`task_summary`, and `final_result` is `None`.

### Structured Children

A child returns structured data when the parent passes `final_schema` to
`delegate_task`, or when the child profile declares `> output: Type`. Kedi
wraps the schema before calling `produce_subagent`:

```json
{
  "type": "object",
  "title": "KediSubagentResult",
  "properties": {
    "task_summary": {
      "type": "string",
      "maxLength": 1000,
      "description": "Briefly state what you did, key decisions, and any limitations."
    },
    "final_result": {"...": "the requested schema"}
  },
  "required": ["task_summary", "final_result"],
  "additionalProperties": false
}
```

Send `output_schema` to the backend as given and return the decoded object.
Kedi validates it against the schema after `produce_subagent` returns; an
invalid result fails the child. To let the model correct itself instead, run
Kedi's validator in a retry loop:

```python title="subagent_adapter.py"
    async def produce_subagent(
        self,
        *,
        template: str,
        output_schema: dict[str, Any],
        instructions: str | None = None,
        max_requests: int | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        child = self._child.get()
        if child is None:
            raise RuntimeError("produce_subagent runs only inside subagent_scope")
        validator = validator_for(output_schema)(output_schema)
        history = resume_messages(child.conversation)
        prompt = template
        for _attempt in range(3):
            reply, messages = await self._run(
                prompt=prompt,
                instructions=instructions,
                output_schema=output_schema,
                history=history,
                max_requests=max_requests,
            )
            try:
                accepted = accept_subagent_output(reply.data, validator, child.output_state)
            except OutputCorrection as correction:
                history, prompt = messages, str(correction)
                continue
            save_messages(child.conversation, messages)
            return accepted
        raise RuntimeError("The child did not return a valid result after 3 attempts")
```

`accept_subagent_output(output, validator, output_state)`:

- checks the JSON Schema and raises `OutputCorrection` listing up to eight
  violations, ending with *"Return a corrected object with task_summary and
  final_result as root siblings."*;
- with an `output_state`, also validates `final_result` against the child
  profile's `> output:` type, and returns the value serialized back to JSON;
- returns the object to accept.

The correction message is Kedi's text. Send it to the model as the next user
turn; do not write your own. Each retry is a backend request and counts
against the child's limits.

### Continuations

`continue_subagent(run_id, task)` starts a new run in the same child
conversation. Kedi sends the earlier turns in one of two ways:

| `supports_native_subagent_resume` | What the adapter receives |
|---|---|
| absent or `False` | `prompt` begins with *"Continue the same child task using this verified prior conversation state:"*, followed by the earlier tasks and results as JSON, then the new task. The adapter needs no state. |
| `True` | `prompt` is the new task only. The adapter restores its own history from `conversation.resume_state`. |

With native resume, the adapter keeps its message history in the
conversation context:

```python title="subagent_adapter.py"
def resume_messages(conversation: SubagentConversationContext) -> list[dict[str, Any]]:
    return list(conversation.resume_state.get("messages", []))


def save_messages(
    conversation: SubagentConversationContext, messages: list[dict[str, Any]]
) -> None:
    conversation.result_state["messages"] = messages
```

- `resume_state` is the state your adapter wrote at the end of the previous
  successful run of this conversation; it is empty for the first run.
- Write `result_state` only once the run has succeeded. Kedi commits it after
  validating the result, and drops it when the run fails or is cancelled, so
  a failed attempt never becomes history.
- Both are plain dictionaries that Kedi persists with interactive sessions.
  Store JSON-compatible values only; keep live clients and SDK objects out.

## Running It

```python title="run_subagents.py"
import asyncio
import json
from collections.abc import Callable

from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from subagent_adapter import SubagentChatAdapter

PROFILES = """
~Status(order: str, state: str)

> profile: worker:
    ###
    Looks up the status of one order.
    ###
    > instructions: Report the status of the order you are given.
    {output}

> profile: coordinator:
    > subagent: worker

> use: coordinator
[answer] << Find out the status of order O-1.
= <answer>
"""


class ScriptedBackend:
    def __init__(self, answer: Callable[[BackendRequest], Reply]) -> None:
        self.answer = answer
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return self.answer(request)


def is_parent(request: BackendRequest) -> bool:
    return any(tool["name"] == "delegate_task" for tool in request.tools)


def tool_results(request: BackendRequest) -> list[dict]:
    return [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]


def run(source: str, answer: Callable[[BackendRequest], Reply]) -> tuple[object, ScriptedBackend]:
    backend = ScriptedBackend(answer)
    adapter = SubagentChatAdapter(backend, model="m1")
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main(), backend
    finally:
        asyncio.run(runtime.aclose())
```

A text child. The parent's model calls `delegate_task`; the child's model
answers with text, which comes back as `task_summary`:

```python title="run_subagents.py"
def text_child(request: BackendRequest) -> Reply:
    if not is_parent(request):
        return Reply(text="O-1 was delivered on Monday.")
    results = tool_results(request)
    if not results:
        return Reply(
            tool_calls=(
                ToolCall("p1", "delegate_task", {"subagent": "worker", "task": "Status of O-1?"}),
            )
        )
    return Reply(text=results[0]["result"]["task_summary"])


value, backend = run(PROFILES.format(output=""), text_child)
assert value == "O-1 was delivered on Monday."
child = next(r for r in backend.requests if not is_parent(r))
assert "Report the status" in (child.instructions or "")
assert child.tools == [] or all(t["name"] != "delegate_task" for t in child.tools)
```

A structured child from the profile's `> output:` type, whose first answer
forgets the wrapper and is corrected with Kedi's feedback:

```python title="run_subagents.py"
def structured_child(request: BackendRequest) -> Reply:
    if not is_parent(request):
        assert request.output_schema is not None
        assert request.output_schema["title"] == "KediSubagentResult"
        if len(request.messages) == 1:
            return Reply(data={"order": "O-1", "state": "delivered"})
        assert "task_summary and final_result" in request.messages[-1]["content"]
        return Reply(
            data={
                "task_summary": "Looked up O-1.",
                "final_result": {"order": "O-1", "state": "delivered"},
            }
        )
    results = tool_results(request)
    if not results:
        return Reply(
            tool_calls=(
                ToolCall("p1", "delegate_task", {"subagent": "worker", "task": "Status of O-1?"}),
            )
        )
    return Reply(text=results[0]["result"]["final_result"]["state"])


value, backend = run(PROFILES.format(output="> output: Status"), structured_child)
assert value == "delivered"
assert sum(not is_parent(r) for r in backend.requests) == 2
```

A continuation with native resume. The second child run starts from the
messages the first run saved:

```python title="run_subagents.py"
def continued_child(request: BackendRequest) -> Reply:
    if not is_parent(request):
        users = [m["content"] for m in request.messages if m["role"] == "user"]
        assert not any("verified prior conversation" in text for text in users)
        return Reply(text=f"answered {len(users)} task(s)")
    results = tool_results(request)
    if not results:
        return Reply(
            tool_calls=(
                ToolCall("p1", "delegate_task", {"subagent": "worker", "task": "Status of O-1?"}),
            )
        )
    if len(results) == 1:
        run_id = results[0]["result"]["run_id"]
        return Reply(
            tool_calls=(
                ToolCall("p2", "continue_subagent", {"run_id": run_id, "task": "And O-2?"}),
            )
        )
    return Reply(text=results[1]["result"]["task_summary"])


value, _backend = run(PROFILES.format(output=""), continued_child)
assert value == "answered 2 task(s)"
```

A background child. `delegate_task` returns a running handle at once;
`wait_subagent` returns the result:

```python title="run_subagents.py"
def background_child(request: BackendRequest) -> Reply:
    if not is_parent(request):
        return Reply(text="O-1 was delivered.")
    results = tool_results(request)
    if not results:
        arguments = {"subagent": "worker", "task": "Status of O-1?", "background": True}
        return Reply(tool_calls=(ToolCall("p1", "delegate_task", arguments),))
    if len(results) == 1:
        handle = results[0]["result"]
        assert handle["status"] in {"running", "completed"}
        return Reply(
            tool_calls=(ToolCall("p2", "wait_subagent", {"run_id": handle["run_id"]}),)
        )
    return Reply(text=results[1]["result"]["task_summary"])


value, _backend = run(PROFILES.format(output=""), background_child)
assert value == "O-1 was delivered."
```

A parent's `deny` binds a child whose own profile says `allow`:

```python title="run_subagents.py"
GUARDED = """
@cancel_order(order: str) -> str:
  ###
  Cancel an order.
  ###
  > tool:
    risk: mutating
  = `"cancelled " + order`

> profile: worker:
    ###
    Handles orders.
    ###
    > approval: allow
    > use: cancel_order

> profile: coordinator:
    > approval: deny
    > subagent: worker

> use: coordinator
[answer] << Cancel order O-1.
= <answer>
"""


def guarded_child(request: BackendRequest) -> Reply:
    results = tool_results(request)
    if not is_parent(request):
        if not results:
            return Reply(tool_calls=(ToolCall("c1", "cancel_order", {"order": "O-1"}),))
        return Reply(text=results[0].get("error") or results[0]["result"])
    if not results:
        arguments = {"subagent": "worker", "task": "Cancel O-1."}
        return Reply(tool_calls=(ToolCall("p1", "delegate_task", arguments),))
    return Reply(text=results[0]["result"]["task_summary"])


value, _backend = run(GUARDED, guarded_child)
assert "was denied" in value, value
```

## Background Children

Advertise `supports_background_subagents` only when a child can run while the
parent's call continues. Kedi starts a background child as a task on its
shared event loop and returns the handle to the parent's tool call. Two things
make that safe:

- the synchronous entry points use `run_sync`, so the parent call and the
  background child share Kedi's loop, and the child survives the tool call
  that started it;
- all per-run state lives in `ContextVar`s and in the request objects, so the
  parent's next model request and the child's requests do not see each
  other's tools, profile, or conversation.

## Approvals in Children

A child can never be more permissive than its ancestors. Every ancestor's
approval policy applies to the child's tools, in order from the root, followed
by the child's own policy; an ancestor without `> approval:` contributes
Kedi's default policy, which denies non-read-only tools.

- **Adapters without `supports_native_approvals`** need to do nothing. Kedi
  wraps the child's tools with all of these policies before they reach
  `subagent_scope`.
- **Adapters with `supports_native_approvals`** receive the ancestor policies
  as `request.approval_ceiling` and must apply them, then the child profile's
  policy, to every tool the child can call, including backend-native tools.

## Live Input and Interrupts

Two further capabilities let a parent talk to a running child:

| Capability | Effect |
|---|---|
| `supports_live_subagent_input` | Kedi offers `send_subagent_input` and `send_parent_message`. Input sent while a child runs is queued for that child. |
| `supports_subagent_interrupt` | `send_subagent_input` may set `interrupt: true`, which stops the child's current attempt so it restarts with the new input. |

Kedi's coordinator already handles input between attempts: when a child
finishes while input is queued, Kedi starts another attempt of the same run
with the input appended, after restoring `resume_state` from that attempt's
`result_state`. To deliver input during an attempt, or to stop one, wrap the
backend run in `controlled_attempt`:

```python
from kedi.agent_adapter.subagents.control import controlled_attempt, current_live_input

control = current_live_input()
if control is None:
    reply, messages = await run_child()
else:
    reply, messages = await controlled_attempt(
        control,
        run_child(),
        deliver=deliver_now,  # optional: inject AgentInput items into the running backend
        stop=stop_backend,    # optional: ask the backend to stop the current turn
    )
```

`controlled_attempt` raises `RunInterrupted` when the attempt is superseded;
let it propagate, because Kedi restarts the attempt. `deliver` receives
`AgentInput` items, whose `render()` text is what the model should see; return
`False` when they could not be delivered, and Kedi hands them to the next
attempt instead. Advertise
these capabilities only when the attempt really stops and restarts; a backend
that cannot be interrupted must leave `supports_subagent_interrupt` unset.

## Errors

| Situation | What happens |
|---|---|
| `invoke` / `produce_subagent` raises | The child run fails; the parent's tool call returns the failure to the parent model. |
| `produce_subagent` returns a non-object | `TypeError: Subagent structured output must be an object, got <type>` |
| The object violates the wrapped schema | `jsonschema.ValidationError`; the child fails. Use `accept_subagent_output` to correct instead. |
| A reported request exceeds the child's limits | `KediExecutionError: Subagent request limit N exceeded: observed M` |
| The parent cancels a child | The child's task receives `asyncio.CancelledError`; release backend resources and re-raise. |
