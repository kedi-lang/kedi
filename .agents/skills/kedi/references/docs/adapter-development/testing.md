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

## Running the Section's Examples

The scenarios on these pages are executable. `tests/test_adapter_development_examples.py`
in the documentation repository assembles every `python title="..."` block
into its module and runs each `run_*.py` scenario in a fresh interpreter.
Copying the modules into your test suite is a working starting point.
