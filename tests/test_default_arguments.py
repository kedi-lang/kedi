from __future__ import annotations

import inspect

import pytest

from kedi.lang import compile_program, parse_program
from tests.utils_llm import run_without_llm


def test_parser_records_procedure_and_type_defaults() -> None:
    src = (
        '@proc(id: int = `0`, label = `"draft"`):\n'
        "  = <`str(id) + label`>\n\n"
        "~Person(name: str, salary: int = `0`)\n"
        "= ok\n"
    )

    program = parse_program(src)

    assert program.procedures[0].param_defaults == {
        "id": "0",
        "label": '"draft"',
    }
    assert [
        (field.name, field.default_expr) for field in program.type_defs[0].fields
    ] == [
        ("name", None),
        ("salary", "0"),
    ]


def test_parser_rejects_required_arguments_after_defaults() -> None:
    with pytest.raises(Exception, match="required parameter"):
        parse_program("@bad(a = `1`, b):\n  = ok\n")

    with pytest.raises(Exception, match="required field"):
        parse_program("~Bad(a: int = `1`, b: int)\n= ok\n")


def test_parser_rejects_duplicate_and_fenced_defaults() -> None:
    with pytest.raises(Exception, match="duplicate parameter"):
        parse_program("@bad(a, a):\n  = ok\n")

    with pytest.raises(Exception):
        parse_program("@bad(a = ```\nreturn 1\n```):\n  = ok\n")


def test_parser_rejects_untyped_type_field_default() -> None:
    with pytest.raises(Exception, match="requires a type annotation"):
        parse_program("~Bad(a = `1`)\n= ok\n")


def test_procedure_defaults_are_used_and_untyped_defaults_stay_native() -> None:
    src = (
        '@combine(x: int, step: int = `1`, label = `"v"`) -> str:\n'
        "  = `label + str(x + step)`\n\n"
        "= <combine(`5`)>\n"
    )

    assert run_without_llm(src) == "v6"


def test_procedure_defaults_are_definition_time_and_validated() -> None:
    src = (
        "```\n"
        "seed = 1\n"
        "```\n"
        "@value(x: int = `seed`) -> int:\n"
        "  = `x`\n\n"
        "`seed = 5`\n"
        "= <value()>\n"
    )

    assert run_without_llm(src) == "1"

    with pytest.raises(TypeError, match="Parameter 'x'"):
        run_without_llm('@bad(x: int = `"not-int"`) -> int:\n  = `x`\n\n= <bad()>\n')


def test_type_defaults_use_basemodel_and_do_not_share_mutable_values() -> None:
    src = (
        "```\n"
        "default_items = []\n"
        "```\n"
        "~Bag(name: str, count: int = `0`, items: list[int] = `default_items`)\n"
        "`default_items.append(99)`\n"
        "= ```\n"
        "a = Bag('a')\n"
        "b = Bag('b', count=2)\n"
        "a.items.append(1)\n"
        "return f\"{b.items}|{b.count}|{a.model_dump_json()}|{isinstance(a, __import__('pydantic').BaseModel)}\"\n"
        "```\n"
    )

    assert run_without_llm(src) == '[]|2|{"name":"a","count":0,"items":[1]}|True'


def test_python_interop_stub_exposes_defaulted_signature_and_source() -> None:
    rt = compile_program(
        parse_program("@kedi_proc(x: int, y: int = `1`) -> int:\n  = `x + y`\n\n= ok\n")
    )

    signature = rt.executor.evaluate_inline(
        rt, "str(inspect.signature(kedi_proc))", {"inspect": inspect}
    )
    source = rt.executor.evaluate_inline(
        rt, "inspect.getsource(kedi_proc)", {"inspect": inspect}
    )

    assert signature == "(x: int, y: int = 1) -> int"
    assert source == (
        "def kedi_proc(x: int, y: int = 1) -> int:\n"
        "    return __kedi_call__(x, y)\n"
    )
