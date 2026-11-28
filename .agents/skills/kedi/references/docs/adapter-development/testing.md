# Testing an Adapter

An adapter is correct when real Kedi programs behave the same on it as on
Kedi's own adapters. Test it the way the pages in this section do: replace
the provider with a scripted backend, run programs through
`compile_program`, and assert on what the backend received and what the
program returned.

## Scripted Backends

A scripted backend answers each request with a function of that request and
records it:

```python
class ScriptedBackend:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return self.answer(request)
```

Decide the answer from the request, not from a counter: the tool results
already in `messages`, the tools offered, whether `output_schema` is set. A
script that keys on content keeps working when Kedi adds a correction turn or
runs blocks in a different order.

## What to Cover

Write one scenario per row that applies to your adapter:

| Area | Scenario | Page |
|---|---|---|
| text | `[x] << ...` returns the text | [Call Lifecycle](call-lifecycle.md) |
| structured output | a capture with a custom type and a typed field; a claim; a typed raw invoke | [Structured Output](structured-output.md) |
| prompt contract | `instructions` and the prompt arrive verbatim and once | [Prompts and Instructions](prompt-contract.md) |
| tools | a tool call, an invalid argument, an approval denial, a fatal tool error | [Tools](tools.md), [Errors](errors.md) |
| retries | a transient failure is retried; a 400 is not | [Errors](errors.md) |
| profiles | `> model:` and a supported setting reach the backend; unsupported fields are refused | [Profiles](profiles.md) |
| hooks | prompt edit, argument edit, denial | [Hooks](hooks.md) |
| artifacts | a large result becomes a reference and can be read | [Artifacts](artifacts.md) |
| children | text, structured with a correction, continuation, background, and an ancestor `deny` | [Subagents](subagents.md) |
| history | the second turn carries the first; a new lane starts empty | [Conversations](conversations.md) |
| budgets | a request limit stops the second request | [Budgets](budgets.md) |
| concurrency | two blocks run at once and stay apart | [Concurrency](concurrency.md) |

Every capability you advertise needs at least one scenario that fails when
the capability is broken. A capability with no scenario should not be
advertised.

## Parity With a Built-in Adapter

Run the same program through Kedi's Pydantic AI adapter, driven by a
`FunctionModel` that plays the same script, and compare what the program saw:

```python title="run_testing.py"
import asyncio
import json

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program

from backend import BackendRequest, Reply, ToolCall
from chat_adapter import ChatAdapter

SOURCE = """
@population(city: str) -> int:
  ###
  Return the population of a city.
  ###
  > tool:
    risk: read_only
  = `{"Paris": 2_100_000}[city]`

> instructions: Use the tool for numbers.
> use: population
>> Paris has [count: int] inhabitants.
= `count`
"""


def run(adapter: object) -> object:
    runtime = compile_program(parse_program(SOURCE), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


class Script:
    def __init__(self) -> None:
        self.instructions: list[str | None] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.instructions.append(request.instructions)
        results = [json.loads(m["content"]) for m in request.messages if m["role"] == "tool"]
        if not results:
            return Reply(tool_calls=(ToolCall("c1", "population", {"city": "Paris"}),))
        return Reply(data={"count": results[0]["result"]})


pydantic_instructions: list[str | None] = []


async def function_model(messages, info) -> ModelResponse:
    pydantic_instructions.append(info.instructions)
    returns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    if not returns:
        return ModelResponse(parts=[ToolCallPart("population", {"city": "Paris"}, tool_call_id="c1")])
    [output] = info.output_tools
    return ModelResponse(parts=[ToolCallPart(output.name, {"count": returns[0].content})])


script = Script()
assert run(ChatAdapter(script, model="m1")) == 2_100_000
assert run(PydanticAdapter(FunctionModel(function_model))) == 2_100_000
assert script.instructions[0] == pydantic_instructions[0], (script.instructions[0], pydantic_instructions[0])
```

Both adapters returned the same value, and the custom adapter sent the model
exactly the instruction text Kedi's own adapter sends.

## Conformance Check

`kedi adapter check MODULE:CLASS` runs Kedi programs through your adapter and
checks what reached the model against what Kedi rendered (see
[Rendered Request](rendered-request.md)). `CLASS.for_conformance(base_url)`
returns an adapter whose backend talks to the kit's local recording endpoint,
which speaks three protocols:

| Route | Protocol |
|---|---|
| `{base_url}/v1/chat/completions` | OpenAI Chat Completions |
| `{base_url}/v1/messages` | Anthropic Messages |
| `{base_url}/kedi/webgpu` | the WebGPU bridge payload |

Send requests with `stream` off; the endpoint refuses streamed ones. The kit
creates a fresh adapter for every run and closes it afterwards with `aclose()`
or `close()` when the adapter has one, so `for_conformance` may start
processes or open clients.

Each scenario runs one Kedi call in two variants with different text:

| Scenario | Program |
|---|---|
| `invoke` | a raw invoke with profile instructions |
| `produce` | a template capture of nested types with field descriptions |
| `claim` | a template condition |
| `tool-allow`, `tool-deny` | a tool the model calls once, with `approval: allow` or `deny` |
| `media` | an image prompt; runs only with `supports_media_input` |
| `request-budget` | `request_limit: 1` with a tool call, then with a failing first request; runs only with `supports_request_budget` |

Build the factory's client without SDK retries (`max_retries=0`). With retries
on, Kedi cannot count every request, refuses `request_limit`, and the
`request-budget` scenario fails.

Subagents are not covered yet; the report says so. For every recorded request
the kit checks that:

- each instruction part is in the system text exactly once, and the prompt is
  in the user turn exactly once;
- the output schema's descriptions and each tool's description arrive
  unchanged;
- the program returns the scripted value, and a tool's result reaches the model
  only when approval allows the call;
- a request the budget refuses never reaches the endpoint: the second request of
  the tool call fails with `RunBudgetExceeded`, and a failed first request is
  not retried behind the budget's back;
- text Kedi did not write is the same in both variants. A fixed harness prompt
  passes and the report lists it; text that depends on the request, such as
  "Fill these fields: …", fails.

`--json` prints the report as JSON. The command exits 1 when a check fails.
For a JSON-RPC bridge, `kedi adapter check --bridge COMMAND` also runs the
protocol's conformance fixtures; see
[JSON-RPC Bridges](json-rpc-bridges.md#conformance).
`PydanticAdapter`, `LangChainAdapter` and `WebGPUAdapter` have
`for_conformance`; the Claude and Codex harness adapters reach their models
through their CLIs, so this endpoint cannot record them:

```bash
kedi adapter check kedi.agent_adapter:PydanticAdapter
```

From Python, `check_adapter` takes the factory directly. The backend below
gives the [Reference Adapter](reference-adapter.md) an OpenAI-compatible
transport, and the reference adapter passes:

```python title="http_backend.py"
from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any

from backend import BackendRequest, Reply, ToolCall


class HttpBackend:
    """The scripted backend's contract over an OpenAI-compatible endpoint."""

    def __init__(self, base_url: str) -> None:
        self.url = f"{base_url}/v1/chat/completions"

    async def complete(self, request: BackendRequest) -> Reply:
        messages: list[dict[str, Any]] = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        for message in request.messages:
            if "tool_calls" in message:
                calls = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                    }
                    for call in message["tool_calls"]
                ]
                messages.append({"role": "assistant", "content": None, "tool_calls": calls})
            else:
                messages.append(message)
        body: dict[str, Any] = {"model": request.model, "messages": messages}
        if request.tools:
            body["tools"] = [{"type": "function", "function": tool} for tool in request.tools]
        if request.output_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.output_schema},
            }
        response = await asyncio.to_thread(self._post, body)
        message = response["choices"][0]["message"]
        calls = tuple(
            ToolCall(call["id"], call["function"]["name"], json.loads(call["function"]["arguments"]))
            for call in message.get("tool_calls") or ()
        )
        if calls or request.output_schema is None:
            return Reply(text=message.get("content"), tool_calls=calls)
        return Reply(data=json.loads(message["content"]))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        post = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(post, timeout=30) as response:
            return json.load(response)
```

```python title="run_conformance.py"
from kedi.agent_adapter.conformance import check_adapter

from chat_adapter import ChatAdapter
from http_backend import HttpBackend
from multimodal_adapter import MultimodalAdapter

for adapter in (ChatAdapter, MultimodalAdapter):
    report = check_adapter(lambda base_url: adapter(HttpBackend(base_url), model="m1"))
    assert report.passed, report.render()
    statuses = {result.name: result.status for result in report.results}
    assert statuses["media"] == ("passed" if adapter is MultimodalAdapter else "skipped")
```

## Running the Section's Examples

The scenarios on these pages are executable. `tests/test_adapter_development_examples.py`
in the documentation repository assembles every `python title="..."` block
into its module and runs each `run_*.py` scenario in a fresh interpreter.
Copying the modules into your test suite is a working starting point.
