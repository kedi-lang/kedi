from __future__ import annotations

import inspect
import warnings

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.adapters.webgpu import _output_adapter
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
        ("~Record(model_dump: int)\n", "K-TYPE-FIELD-NAME"),
        ("~Record(model_config: int)\n", "K-TYPE-FIELD-NAME"),
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


def test_model_prefixed_type_fields_that_are_not_model_members_remain_legal() -> None:
    source = "~Car(model_name: str)\n= `Car('roadster').model_name`\n"
    assert compile_program(parse_program(source)).run_main() == "roadster"


_RESERVED_OUTPUTS = ["_x", "__name__", "__x", "model_config", "model_dump", "model_name", "copy"]


def _answer_every_property(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool = info.output_tools[0]
    properties = tool.parameters_json_schema["properties"]
    return ModelResponse(parts=[ToolCallPart(tool.name, {name: 7 for name in properties})])


@pytest.mark.parametrize("name", _RESERVED_OUTPUTS)
def test_output_names_reserved_by_pydantic_reach_the_model_under_their_kedi_name(name) -> None:
    source = (
        f">> Give a number: [{name}: int] and [kedi_output_0: int]\n= `{name} + kedi_output_0`\n"
    )
    adapter = PydanticAdapter(FunctionModel(_answer_every_property))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert compile_program(parse_program(source), adapter=adapter).run_main() == 14


def test_webgpu_output_models_alias_reserved_names() -> None:
    adapter = _output_adapter({"_x": int, "copy": int}, None)
    assert set(adapter.json_schema()["properties"]) == {"_x", "copy"}
    value = adapter.validate_python({"_x": 1, "copy": 2})
    assert type(value).model_validate(value.model_dump(by_alias=True)) == value
