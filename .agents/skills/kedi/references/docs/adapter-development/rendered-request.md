# Rendered Request

Kedi renders every piece of text a model reads; the adapter transports it.
`current_request().render()` returns that text for the call in progress as one
value. It is the supported way for a custom adapter to build its backend
request.

## Reading the Request

```python
from kedi.agent_adapter import current_request

rendered = current_request().render()
```

`current_request()` is valid inside `invoke`, `produce` and their `_sync`
forms when Kedi makes the call. It raises `RuntimeError` outside a Kedi call. It
also raises in subagent entry points (`produce_subagent`, `invoke_subagent`):
the child runs inside its parent's request, so there use the `template` and
`output_schema` arguments.

| Field | Contents |
|---|---|
| `system` | the instruction parts joined with a blank line, verbatim; `None` when there are none |
| `instruction_parts` | the same parts, each with `identity`, `source` and `content` |
| `user` | the rendered prompt: the `prompt` or `template` argument |
| `media` | the prompt's attachments (see [Multimodal Input](multimodal-input.md)) |
| `output_json_schema` | the output schema Kedi built, with the model description and field descriptions; `None` for text |
| `output_tool_name` | `final_result` when there is an output schema, for a backend that takes the output as a tool call |
| `tools` | each tool's name, Kedi description, input and return schema, and `required_before_output` |
| `prompt_kind` | `raw`, `template` or `claim` |
| `call_kind` | `invoke` or `produce` |

Send `system` as the system or developer text, `user` with its `media` as the
user turn, the output schema as the backend's structured-output contract and
`tools` as its tool definitions. Without protocol parts, `system` is the same
text as the `instructions` argument.

## Backends Without Native Structured Output or Tool Calling

Some backends cannot take an output schema or tool definitions natively; a
small local model that only answers in text, for example. Declare it in the
capabilities:

```python
AdapterCapabilities(
    supports_structured_output=True,
    supports_tool_registration=True,
    supports_native_structured_output=False,
    supports_native_tool_calling=False,
)
```

`supports_structured_output` stays `True`: Kedi still asks the adapter for
structured values. The two new flags tell Kedi to render the contracts as text.
`render()` then appends Kedi-owned instruction parts after the others:

| Identity | Present when | Contents |
|---|---|---|
| `protocol.output` | `supports_native_structured_output=False` and the call has an output schema | `<OutputProtocol>`: answer with a JSON value matching the schema, and the schema itself |
| `protocol.tools` | `supports_native_tool_calling=False` and the call has tools | `<ToolProtocol>`: the call format, the final format, and `<Tools>`, each tool's description and schemas |

The model answers with one JSON object per turn:

- `{"action": "call_tool", "callId": "c1", "name": "lookup", "arguments": {...}}` to call a tool;
- `{"action": "final", "data": ...}` with the output value, or `{"action": "final", "text": "..."}` for a text call.

The adapter parses that object and runs tools as usual. It never writes this
text itself; the wording is versioned (`PROTOCOL_VERSION` in
`kedi.agent_adapter.request.prompts`). The protocol parts are not part of the
`instructions` argument, so hooks and telemetry see the instructions they
always saw.

An adapter that adds tools during the run, from MCP servers for example, passes
its final tool set so the catalog names them too:

```python
from kedi.agent_adapter.request import tool_definition

rendered = current_request().render(tools=tuple(tool_definition(spec) for spec in specs))
```

## Corrections

Corrections are Kedi's text too. `kedi.agent_adapter.request.prompts` has one
function for each:

| Function | Send it as a user turn when |
|---|---|
| `output_correction(error)` | the answer fails the output schema; `error` is the validation error |
| `required_tools_correction(missing, attempted=...)` | the model answers before a tool with `required_before_output` completed; `attempted` is whether it tried |
| `protocol_correction(detail)` | the answer is not one protocol action; `detail` is your parse error |

Take required tools from `ToolDefinition.required_before_output`. Do not infer
them from the prompt.

## Scenario

The adapter below forwards the rendered request to a text-only backend and
parses its JSON action.

```python title="rendered_adapter.py"
from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter

from kedi.agent_adapter import AdapterCapabilities, AgentAdapterKind, current_request
from kedi.agent_adapter.request.prompts import output_correction


class PromptedAdapter:
    """A backend that reads text and answers with one JSON action."""

    kind: AgentAdapterKind = "agent-framework"
    shortname = "prompted"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_native_structured_output=False,
        supports_native_tool_calling=False,
    )

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.sent: list[dict[str, Any]] = []

    def invoke_sync(self, *, prompt: str, **_: Any) -> str:
        return str(self._answer(None))

    def produce_sync(self, *, template: str, output_type: Any = None, **_: Any) -> Any:
        return self._answer(TypeAdapter(output_type))

    def _answer(self, validator: TypeAdapter[Any] | None) -> Any:
        rendered = current_request().render()
        messages = [{"role": "user", "content": rendered.user}]
        for answer in self.answers:
            self.sent.append({"system": rendered.system, "messages": list(messages)})
            action = json.loads(answer)
            if validator is None:
                return action["text"]
            try:
                return validator.validate_python(action["data"])
            except ValueError as error:
                messages += [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": output_correction(error)},
                ]
        raise RuntimeError("no valid answer")
```

```python title="run_rendered_request.py"
from kedi.lang import compile_program, parse_program

from rendered_adapter import PromptedAdapter

adapter = PromptedAdapter(['{"action": "final", "data": "three"}', '{"action": "final", "data": 3}'])
runtime = compile_program(
    parse_program("[count: int] << How many sides has a triangle?\n= `count`\n"), adapter=adapter
)
assert runtime.run_main() == 3

first, second = adapter.sent
assert first["system"].startswith("<OutputProtocol>\n")
assert '<OutputSchema>{"type": "integer"}</OutputSchema>' in first["system"]
assert first["messages"] == [{"role": "user", "content": "How many sides has a triangle?"}]
correction = second["messages"][-1]["content"]
assert correction.startswith("The structured output does not match the required schema:")
```

The first answer fails the schema, so the adapter sends Kedi's correction and the
second answer is accepted.
