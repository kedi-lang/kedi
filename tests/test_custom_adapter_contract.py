"""A custom adapter written only from the documented contract (docs: Custom Adapters).

The adapter below mirrors the documentation sample. Each scenario runs a real Kedi
program through ``compile_program`` and checks what the adapter received.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from kedi.agent_adapter import AdapterCapabilities, AgentAdapterKind, ToolSpec
from kedi.lang import compile_program, parse_program


@dataclass
class Call:
    method: str
    prompt: str
    instructions: str | None
    output_schema: dict[str, Any] | None = None
    output_type: Any = None
    tools: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


Reply = Callable[[Call, Sequence[ToolSpec]], Any]


class DocAdapter:
    """The documented minimal adapter: text, structured output, and Kedi tools."""

    kind: AgentAdapterKind = "agent-framework"
    shortname = "doc-adapter"
    capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_artifacts=True,
    )

    def __init__(self, reply: Reply) -> None:
        self._reply = reply
        self.calls: list[Call] = []
        self._tools: ContextVar[tuple[ToolSpec, ...]] = ContextVar(
            f"doc_adapter_tools_{id(self)}", default=()
        )

    @contextmanager
    def tool_scope(self, tools: Sequence[object] = ()) -> Generator[None, None, None]:
        token = self._tools.set(tuple(tool for tool in tools if isinstance(tool, ToolSpec)))
        try:
            yield
        finally:
            self._tools.reset(token)

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.set((*self._tools.get(), tool))

    def invoke_sync(self, *, prompt: str, instructions: str | None = None, **kwargs: Any) -> str:
        call = Call("invoke", prompt, instructions, kwargs=kwargs)
        return str(self._record(call))

    async def invoke(self, *, prompt: str, instructions: str | None = None, **kwargs: Any) -> str:
        return self.invoke_sync(prompt=prompt, instructions=instructions, **kwargs)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        output_type: Any = None,
        instructions: str | None = None,
        **kwargs: Any,
    ) -> Any:
        call = Call("produce", template, instructions, output_schema, output_type, kwargs=kwargs)
        return self._record(call)

    async def produce(self, **kwargs: Any) -> Any:
        return self.produce_sync(**kwargs)

    def _record(self, call: Call) -> Any:
        tools = self._tools.get()
        call.tools = tuple(tool.name for tool in tools)
        self.calls.append(call)
        return self._reply(call, tools)


def run(source: str, reply: Reply) -> tuple[Any, DocAdapter]:
    adapter = DocAdapter(reply)
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main(), adapter
    finally:
        asyncio.run(runtime.aclose())


def call_tool(tool: ToolSpec, **arguments: Any) -> Any:
    result = tool.fn(**arguments)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def test_template_capture_accepts_a_plain_mapping_and_receives_kedi_instructions() -> None:
    source = (
        "> instructions: Answer as a geographer.\n>> The capital of France is [city].\n= <city>\n"
    )

    value, adapter = run(source, lambda call, _tools: {"city": "Paris"})

    assert value == "Paris"
    [call] = adapter.calls
    assert call.method == "produce"
    assert set(call.output_schema or {}) == {"city"}
    assert call.output_type is None
    # The template argument is the prompt Kedi already rendered, not the raw template.
    assert call.prompt.startswith("Fields to be substituted")
    assert "The capital of France is [city]." in call.prompt
    assert call.instructions is not None
    assert "Answer as a geographer." in call.instructions
    assert "<TemplateInstructions>" in call.instructions


def test_claims_arrive_with_output_type_and_no_output_schema() -> None:
    source = (
        "[answer] = `'unknown'`\n"
        "> if: Paris is the capital of France\n"
        "  [answer] := yes\n"
        "> else:\n"
        "  [answer] := no\n"
        "= <answer>\n"
    )

    value, adapter = run(source, lambda _call, _tools: True)

    assert value == "yes"
    [call] = adapter.calls
    assert call.method == "produce"
    assert call.output_schema is None
    assert call.output_type is not None
    assert call.prompt == "Claim: Paris is the capital of France"
    assert call.instructions is not None and "Determine whether" in call.instructions


def test_typed_raw_invoke_arrives_with_output_type() -> None:
    source = "[count: int] << How many moons does Mars have?\n= `count`\n"

    value, adapter = run(source, lambda _call, _tools: 2)

    assert value == 2
    [call] = adapter.calls
    assert call.method == "produce"
    assert call.output_schema is None
    assert call.output_type is int
    assert call.prompt == "How many moons does Mars have?"


def test_untyped_raw_invoke_uses_invoke() -> None:
    value, adapter = run("[answer] << Say hi.\n= <answer>\n", lambda _call, _tools: "hi")

    assert value == "hi"
    [call] = adapter.calls
    assert call.method == "invoke"
    assert call.prompt == "Say hi."


def test_registered_tools_run_through_kedi_wrappers() -> None:
    source = (
        "@population(city: str) -> str:\n"
        "  ###\n"
        "  Return the population of a city.\n"
        "  ###\n"
        "  > tool:\n"
        "    risk: read_only\n"
        '  = `f"{city}: 2.1M"`\n'
        "\n"
        "> approval: allow\n"
        "> use: population\n"
        ">> Paris has [count] inhabitants.\n"
        "= <count>\n"
    )

    def reply(_call: Call, tools: Sequence[ToolSpec]) -> Any:
        tool = next(tool for tool in tools if tool.name == "population")
        return {"count": call_tool(tool, city="Paris").split(": ")[1]}

    value, adapter = run(source, reply)

    assert value == "2.1M"
    # Kedi registers its artifact management tools next to the program's tools.
    assert "population" in adapter.calls[0].tools
    assert "read_artifact" in adapter.calls[0].tools


def test_denied_tools_raise_inside_the_adapter() -> None:
    source = (
        "@delete_file(path: str) -> str:\n"
        "  ###\n"
        "  Delete a file.\n"
        "  ###\n"
        "  > tool:\n"
        "    risk: mutating\n"
        "  = `'deleted'`\n"
        "\n"
        "> approval: deny\n"
        "> use: delete_file\n"
        ">> The result is [result].\n"
        "= <result>\n"
    )
    seen: list[str] = []

    def reply(_call: Call, tools: Sequence[ToolSpec]) -> Any:
        tool = next(tool for tool in tools if tool.name == "delete_file")
        try:
            call_tool(tool, path="notes.txt")
        except Exception as exc:  # the adapter reports the failure to its model
            seen.append(type(exc).__name__)
        return {"result": "not deleted"}

    value, _adapter = run(source, reply)

    assert value == "not deleted"
    assert seen == ["ToolApprovalDenied"]


def documented_json_schema(call: Call) -> dict[str, Any]:
    """The JSON Schema recipe from the documentation."""
    from pydantic import TypeAdapter

    from kedi.agent_adapter.output_schema import build_output_model

    if call.output_schema is not None:
        return build_output_model(call.output_schema).model_json_schema()
    return TypeAdapter(call.output_type).json_schema()


def test_documented_json_schema_recipe_covers_captures_and_claims() -> None:
    schemas: list[dict[str, Any]] = []

    def reply(call: Call, _tools: Sequence[ToolSpec]) -> Any:
        schemas.append(documented_json_schema(call))
        return {"city": "Paris", "count": 3} if call.output_schema is not None else True

    source = (
        "[answer] = `'unknown'`\n"
        "> if: Paris is the capital of France\n"
        "  [answer] := yes\n"
        ">> The capital is [city] and it has [count: int] airports.\n"
        "= <city>\n"
    )
    value, _adapter = run(source, reply)

    assert value == "Paris"
    claim_schema, capture_schema = schemas
    assert claim_schema["type"] == "boolean"
    assert set(capture_schema["properties"]) == {"city", "count"}
    assert capture_schema["properties"]["count"]["type"] == "integer"
    # Kedi's output-model description reaches the backend with the schema.
    assert "replacement values" in capture_schema["description"]
