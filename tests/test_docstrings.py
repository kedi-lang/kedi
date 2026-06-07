from __future__ import annotations

from typing import Any

from kedi.lang import compile_program, parse_program
from kedi.lang.docstrings import _combine_description, parse_procedure_docstring
from kedi.lang.procedure_schema import (
    _apply_parameter_descriptions,
    _return_type_name,
    procedure_json_schema,
    procedure_return_json_schema,
)


def test_parse_procedure_docstring_empty_whitespace_and_parser_failure(
    monkeypatch,
) -> None:
    assert parse_procedure_docstring(None).description is None
    assert parse_procedure_docstring("   \n\t  ").description is None

    def raise_parse(_docstring: str):
        raise RuntimeError("bad docstring")

    monkeypatch.setattr("kedi.lang.docstrings.parse_docstring", raise_parse)

    parsed = parse_procedure_docstring("Raw fallback doc.")

    assert parsed.description == "Raw fallback doc."
    assert parsed.parameter_descriptions == {}
    assert parsed.return_description is None


def test_parse_procedure_docstring_handles_param_only_and_combines_description() -> None:
    parsed = parse_procedure_docstring(":param text: Source text.")

    assert parsed.description is None
    assert parsed.parameter_descriptions == {"text": "Source text."}
    assert _combine_description(None, None) is None
    assert _combine_description(None, " Long details. ") == "Long details."


def test_procedure_schema_for_unannotated_return_can_use_docstring_return_description() -> None:
    src = (
        "@summarize(text: str):\n"
        "  ###\n"
        "  Summarize text.\n"
        "\n"
        "  :return: Plain summary text.\n"
        "  ###\n"
        "  = <text>\n"
    )
    prog = parse_program(src)
    rt = compile_program(prog)

    schema = procedure_return_json_schema(rt, prog.procedures[0])

    assert schema == {"type": "string", "description": "Plain summary text."}


def test_procedure_schema_rejects_unknown_parameter_and_return_types() -> None:
    param_prog = parse_program("@bad(value: MissingType) -> str:\n  = <value>\n")
    param_rt = compile_program(param_prog)

    try:
        procedure_json_schema(param_rt, param_prog.procedures[0])
    except TypeError as exc:
        assert "Unknown parameter type annotation for 'value'" in str(exc)
    else:
        raise AssertionError("unknown parameter type should fail")

    return_prog = parse_program("@bad() -> MissingType:\n  = nope\n")
    return_rt = compile_program(return_prog)

    try:
        procedure_return_json_schema(return_rt, return_prog.procedures[0])
    except TypeError as exc:
        assert "Unknown return type annotation in procedure 'bad'" in str(exc)
    else:
        raise AssertionError("unknown return type should fail")


def test_procedure_schema_description_helpers_keep_existing_schema_descriptions() -> None:
    schema: dict[str, Any] = {
        "properties": {"text": {"description": "Existing."}, "limit": {}, "skip": []}
    }

    _apply_parameter_descriptions(
        schema,
        {"text": "New text.", "limit": "Maximum results.", "skip": "Ignored."},
    )
    _apply_parameter_descriptions({"properties": []}, {"text": "Ignored."})

    assert schema["properties"]["text"]["description"] == "Existing."
    assert schema["properties"]["limit"]["description"] == "Maximum results."
    assert schema["properties"]["skip"] == []

    def named_callable() -> None:
        return None

    assert _return_type_name(named_callable) == "named_callable"
