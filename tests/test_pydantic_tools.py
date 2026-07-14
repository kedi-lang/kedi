from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter import (
    kedi_procedure_to_pydantic_tool,
    kedi_program_to_pydantic_tools,
)
from kedi.agent_adapter.utils import _coerce_tool_arguments
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProcedureDef


def test_kedi_procedure_to_pydantic_tool_uses_schema_and_defaults() -> None:
    src = (
        '@translate(text: str, style: str = `"brief"`) -> str:\n'
        "  ###\n"
        "  Translate the provided text.\n"
        "  ###\n"
        '  = `f"{style}: {text}"`\n'
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    tool = kedi_procedure_to_pydantic_tool(rt, prog.procedures[0])

    assert tool.name == "translate"
    assert tool.description == "Translate the provided text. Returns `str` instance."
    assert (
        tool.tool_def.parameters_json_schema["description"]
        == "Translate the provided text. Returns `str` instance."
    )
    assert tool.tool_def.parameters_json_schema["properties"]["text"]["type"] == "string"
    assert tool.tool_def.parameters_json_schema["properties"]["style"]["default"] == "brief"
    assert tool.metadata is not None
    assert tool.metadata["kedi_return_schema"] == {"type": "string"}
    assert (
        tool.metadata["kedi_schema"]["description"]
        == "Translate the provided text. Returns `str` instance."
    )

    result = asyncio.run(tool.function_schema.call({"text": "Merhaba"}, cast(Any, None)))
    assert result == "brief: Merhaba"


def test_kedi_procedure_to_pydantic_tool_uses_parsed_docstring_description() -> None:
    src = (
        '@translate(text: str, style: str = `"brief"`) -> str:\n'
        "  ###\n"
        "  Translate the provided text.\n"
        "\n"
        "  Args:\n"
        "      text: Source text to translate.\n"
        "      style: Translation style.\n"
        "  ###\n"
        '  = `f"{style}: {text}"`\n'
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    tool = kedi_procedure_to_pydantic_tool(rt, prog.procedures[0])

    assert tool.description == "Translate the provided text. Returns `str` instance."
    assert (
        tool.tool_def.parameters_json_schema["properties"]["text"]["description"]
        == "Source text to translate."
    )
    assert (
        tool.tool_def.parameters_json_schema["properties"]["style"]["description"]
        == "Translation style."
    )
    assert tool.metadata is not None
    assert (
        tool.metadata["kedi_schema"]["description"]
        == "Translate the provided text. Returns `str` instance."
    )


def test_kedi_tool_description_includes_return_description_when_present() -> None:
    src = (
        "@summarize(text: str) -> str:\n"
        "  ###\n"
        "  Summarize the input text.\n"
        "\n"
        "  Returns:\n"
        "      Final summary text.\n"
        "  ###\n"
        "  = <text>\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    tool = kedi_procedure_to_pydantic_tool(rt, prog.procedures[0])

    assert tool.description == "Summarize the input text. Returns `str` instance."


def test_kedi_procedure_to_pydantic_tool_handles_custom_kedi_types() -> None:
    src = (
        "~Person(name: str, salary: int = `0`)\n"
        "@describe(person: Person) -> str:\n"
        '  = `f"{person.name} earns {person.salary}"`\n'
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    tool = kedi_procedure_to_pydantic_tool(rt, prog.procedures[0])

    schema = tool.tool_def.parameters_json_schema
    assert schema["properties"]["person"]["$ref"] == "#/$defs/Person"
    assert "Person" in schema["$defs"]

    result = asyncio.run(
        tool.function_schema.call({"person": {"name": "Ada", "salary": 5}}, cast(Any, None))
    )
    assert result == "Ada earns 5"


def test_kedi_tool_description_uses_simple_custom_return_contract() -> None:
    src = (
        "~Person(name: str, salary: int = `0`)\n"
        "@hire(name: str) -> Person:\n"
        "  = `Person(name=name, salary=10)`\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    tool = kedi_procedure_to_pydantic_tool(rt, prog.procedures[0])

    assert tool.description == "Returns `Person` instance."


def test_kedi_program_to_pydantic_tools_can_include_nested() -> None:
    src = (
        "@outer(text: str) -> str:\n"
        "  @inner(value: int = `1`) -> int:\n"
        "    = `value + 1`\n"
        "  = <text>\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    top_level = kedi_program_to_pydantic_tools(rt, prog)
    nested = kedi_program_to_pydantic_tools(rt, prog, include_nested=True)

    assert [tool.name for tool in top_level] == ["outer"]
    assert {tool.name for tool in nested} == {"outer", "inner"}
    inner = next(tool for tool in nested if tool.name == "inner")
    assert inner.tool_def.parameters_json_schema["properties"]["value"]["default"] == 1


def test_kedi_tool_works_with_pydantic_ai_agent() -> None:
    src = (
        "~SearchRequest(queries: list[str], limit: int = `5`)\n"
        "@search_docs(request: SearchRequest, include_snippets: bool = `True`) -> str:\n"
        "  ###\n"
        "  Search indexed sources using a typed request payload.\n"
        "  ###\n"
        "  [query_count: int] = ```\n"
        "return len(request.queries)\n"
        "```\n"
        '  = `f"queries={request.queries}|limit={request.limit}|include={include_snippets}|count={query_count}"`\n'
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    model = TestModel(call_tools=["search_docs"])
    tool = kedi_procedure_to_pydantic_tool(rt, prog.procedures[0])
    agent = Agent(
        model,
        tools=[tool],
        output_type=str,
    )

    result = agent.run_sync("Use the search_docs tool.")
    output = json.loads(result.output)
    assert model.last_model_request_parameters is not None
    function_tools = model.last_model_request_parameters.function_tools
    assert len(function_tools) == 1
    assert function_tools[0].description == (
        "Search indexed sources using a typed request payload. Returns `str` instance."
    )
    assert function_tools[0].parameters_json_schema["description"] == (
        "Search indexed sources using a typed request payload. Returns `str` instance."
    )

    assert list(output) == ["search_docs"]
    assert "limit=5" in output["search_docs"]
    assert "include=True" in output["search_docs"]
    assert "count=1" in output["search_docs"]


def test_kedi_tool_argument_coercion_covers_missing_none_model_and_invalid_paths() -> None:
    src = (
        '~Person(name: str)\n@describe(person: Person, raw) -> str:\n  = `f"{person.name}:{raw}"`\n'
    )
    prog = parse_program(src)
    rt = compile_program(prog)
    proc = prog.procedures[0]

    kwargs: dict[str, Any] = {"person": {"name": "Ada"}, "raw": "untouched"}
    _coerce_tool_arguments(rt, proc, kwargs)

    person: Any = kwargs["person"]
    assert person.name == "Ada"
    assert kwargs["raw"] == "untouched"

    _coerce_tool_arguments(rt, ProcedureDef(name="plain", params=["x"]), {"x": "value"})

    typed = ProcedureDef(name="typed", params=["x"], param_type_annotations={"x": "int"})
    with pytest.raises(TypeError, match="Missing parameter: x"):
        _coerce_tool_arguments(rt, typed, {})
    with pytest.raises(TypeError, match="Parameter 'x' does not match annotated type"):
        _coerce_tool_arguments(rt, typed, {"x": "not-an-int"})
