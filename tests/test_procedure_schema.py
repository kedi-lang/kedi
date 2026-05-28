from __future__ import annotations

from kedi.lang import (
    compile_program,
    parse_program,
    procedure_json_schema,
    procedure_return_json_schema,
    program_procedure_json_schemas,
)


def test_procedure_json_schema_includes_docstring_defaults_and_return_type() -> None:
    src = (
        '@translate(text: str, style: str = `"brief"`) -> str:\n'
        "  ###\n"
        "  Translate the provided text.\n"
        "  Keep the answer concise.\n"
        "  ###\n"
        "  = <text>\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    schema = procedure_json_schema(rt, prog.procedures[0])

    assert schema["title"] == "translate"
    assert schema["type"] == "object"
    assert schema["description"] == (
        "Translate the provided text.\nKeep the answer concise. Returns `str` instance."
    )
    assert schema["required"] == ["text"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["style"]["type"] == "string"
    assert schema["properties"]["style"]["default"] == "brief"
    assert procedure_return_json_schema(rt, prog.procedures[0]) == {"type": "string"}


def test_procedure_json_schema_parses_rest_param_and_return_descriptions() -> None:
    src = (
        "@search(text: str, limit: int = `5`) -> str:\n"
        "  ###\n"
        "  Search indexed sources.\n"
        "\n"
        "  :param text: Search query text.\n"
        "  :param limit: Maximum number of matches.\n"
        "  :return: Final formatted answer.\n"
        "  ###\n"
        "  = <text>\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    schema = procedure_json_schema(rt, prog.procedures[0])
    returns = procedure_return_json_schema(rt, prog.procedures[0])

    assert schema["description"] == "Search indexed sources. Returns `str` instance."
    assert schema["properties"]["text"]["description"] == "Search query text."
    assert schema["properties"]["limit"]["description"] == "Maximum number of matches."
    assert returns["description"] == "Final formatted answer."


def test_procedure_json_schema_uses_custom_kedi_types() -> None:
    src = (
        "~Person(name: str, salary: int = `0`)\n"
        "@hire(person: Person) -> Person:\n"
        "  = `person`\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    schema = procedure_json_schema(rt, prog.procedures[0])

    assert schema["properties"]["person"]["$ref"] == "#/$defs/Person"
    assert "Person" in schema["$defs"]
    assert schema["$defs"]["Person"]["properties"]["name"]["type"] == "string"
    assert schema["$defs"]["Person"]["properties"]["salary"]["type"] == "integer"
    assert schema["$defs"]["Person"]["properties"]["salary"]["default"] == 0

    returns = procedure_return_json_schema(rt, prog.procedures[0])
    assert returns["title"] == "Person"
    assert returns["type"] == "object"
    assert returns["properties"]["name"]["type"] == "string"
    assert returns["properties"]["salary"]["default"] == 0
    assert "description" not in returns


def test_procedure_json_schema_leaves_untyped_params_open() -> None:
    src = "@echo(value, count = `1`):\n  = <value>\n"
    prog = parse_program(src)
    rt = compile_program(prog)

    schema = procedure_json_schema(rt, prog.procedures[0])

    assert schema["required"] == ["value"]
    assert "type" not in schema["properties"]["value"]
    assert schema["properties"]["count"]["default"] == 1
    assert "type" not in schema["properties"]["count"]
    assert procedure_return_json_schema(rt, prog.procedures[0]) == {"type": "string"}


def test_program_procedure_json_schemas_can_include_nested_procedures() -> None:
    src = (
        "@outer(text: str) -> str:\n"
        "  @inner(value: int) -> int:\n"
        "    = `value + 1`\n"
        "  = <text>\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    top_level = program_procedure_json_schemas(rt, prog)
    nested = program_procedure_json_schemas(rt, prog, include_nested=True)

    assert list(top_level) == ["outer"]
    assert set(nested) == {"outer", "inner"}
    assert nested["inner"]["properties"]["value"]["type"] == "integer"


def test_procedure_json_schema_parses_google_style_docstrings() -> None:
    src = (
        '@summarize(text: str, style: str = `"brief"`) -> str:\n'
        "  ###\n"
        "  Summarize text.\n"
        "\n"
        "  Args:\n"
        "      text: Source text to summarize.\n"
        "      style: Response style.\n"
        "\n"
        "  Returns:\n"
        "      Final summary string.\n"
        "  ###\n"
        "  = <text>\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    schema = procedure_json_schema(rt, prog.procedures[0])
    returns = procedure_return_json_schema(rt, prog.procedures[0])

    assert schema["description"] == "Summarize text. Returns `str` instance."
    assert schema["properties"]["text"]["description"] == "Source text to summarize."
    assert schema["properties"]["style"]["description"] == "Response style."
    assert returns["description"] == "Final summary string."
