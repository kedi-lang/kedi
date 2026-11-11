from __future__ import annotations

import inspect

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features.diagnostics import compute_diagnostics


@pytest.mark.parametrize(
    "source, code",
    [
        ("@describe(class):\n  = ok\n", "K-PARAM-NAME"),
        ("@class():\n  = ok\n", "K-PROC-NAME"),
        ("~class(value: int)\n", "K-TYPE-NAME"),
        ("~Lesson(class: int)\n", "K-TYPE-FIELD-NAME"),
        ("~Account(_id: int, owner: str)\n", "K-TYPE-FIELD-NAME"),
        ("~Account(__id: int)\n", "K-TYPE-FIELD-NAME"),
    ],
)
def test_unsupported_python_declaration_names_have_source_diagnostics(source, code) -> None:
    with pytest.raises(KediParseError) as caught:
        parse_program(source, source_path="names.kedi")
    assert any(diagnostic.code == code for diagnostic in caught.value.diagnostics)
    assert any(diagnostic.code == code for diagnostic in compute_diagnostics(source, "names.kedi"))


@pytest.mark.parametrize("name", ["_body", "_body_", "__kedi_call__", "_impl", "self"])
@pytest.mark.parametrize("python_call", [False, True])
def test_procedure_parameters_do_not_shadow_generated_helpers(name, python_call) -> None:
    call = "`echo(7)`" if python_call else "<echo(`7`)>"
    runtime = compile_program(
        parse_program(f"@echo({name}: int) -> int:\n  = `{name}`\n= {call}\n")
    )
    assert runtime.run_main() == (7 if python_call else "7")


def test_multiple_internal_parameter_collisions_remain_usable() -> None:
    source = """@add(_body: int, _body_: int, __kedi_call__: int, __kedi_call___: int) -> int:
  = `_body + _body_ + __kedi_call__ + __kedi_call___`
= `add(1, 2, 3, 4)`
"""
    assert compile_program(parse_program(source)).run_main() == 10


def test_type_fields_can_be_named_self_and_instance() -> None:
    source = """~Item(self: int, self_: int, instance: str)
= `Item(1, 2, 'item')`
"""
    item = compile_program(parse_program(source)).run_main()
    assert item.model_dump() == {"self": 1, "self_": 2, "instance": "item"}
    assert list(inspect.signature(type(item)).parameters) == ["self", "self_", "instance"]


def test_keyword_names_remain_legal_for_kedi_only_bindings() -> None:
    assert compile_program(parse_program("[class] = lesson\n= <class>\n")).run_main() == "lesson"


def test_procedure_name_does_not_shadow_its_own_python_bridge() -> None:
    source = "@__kedi_call__(value: int) -> int:\n  = `value`\n= `__kedi_call__(7)`\n"
    assert compile_program(parse_program(source)).run_main() == 7


def test_python_block_wrapper_does_not_overwrite_existing_bindings() -> None:
    source = """[class] = lesson
[__kedi_wrapper__: int] = `7`
[result: int] = ```
return __kedi_wrapper__
```
= `result`
"""
    assert compile_program(parse_program(source)).run_main() == 7
