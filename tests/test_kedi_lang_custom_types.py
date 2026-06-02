from __future__ import annotations

from typing import Any

import pytest

from kedi.lang import compile_program, parse_program
from tests.utils_llm import run_without_llm


def _run(src: str) -> Any:
    return run_without_llm(src)


def test_basic_type_usage():
    src = (
        "~Person(age:int, name, city)\n"
        "@make(name, city):\n"
        "  = ```\n"
        "  p = Person(age=30, name=name, city=city)\n"
        "  return p.model_dump_json()\n"
        "  ```\n\n"
        "= <`make('Ada', 'London')`>\n"
    )
    out = _run(src)
    assert '"name":"Ada"' in out


def test_type_parameter_annotation_call():
    src = (
        "~Person(age:int, name, city)\n"
        "@make() -> Person:\n"
        "  = ```\n"
        "  return Person(age=40, name='Bob', city='NY')\n"
        "  ```\n\n"
        "@describe(p: Person):\n"
        "  = <`p.name + ':' + p.city`>\n\n"
        "= <describe(<`make()`>)>\n"
    )
    out = _run(src)
    assert out == "Bob:NY"


def test_assignment_with_custom_type():
    src = (
        "~Widget(id:int, label)\n"
        "[item: Widget] = ```\n"
        "return Widget(id=7, label='gizmo')\n"
        "```\n"
        "= <`str(item.id) + '/' + item.label`>\n"
    )
    out = _run(src)
    assert out == "7/gizmo"


def test_custom_type_in_return_annotation():
    src = (
        "~Thing(value:int)\n"
        "@make() -> Thing:\n"
        "  = ```\n"
        "  return Thing(value=3)\n"
        "  ```\n\n"
        "= <`make().value + 4`>\n"
    )
    out = _run(src)
    assert out == "7"


def test_custom_type_in_list_annotation():
    src = (
        "~Person(age:int, name, city)\n"
        "[people: list[Person]] = ```\n"
        "return [Person(age=20, name='A', city='X'), Person(age=30, name='B', city='Y')]\n"
        "```\n"
        "= <`str(len(people)) + ':' + people[1].name`>\n"
    )
    out = _run(src)
    assert out == "2:B"


def test_custom_type_referencing_other():
    src = (
        "~Address(city, country)\n"
        "~Person(name, addr: Address)\n"
        "= <`Person(name='Ada', addr=Address(city='London', country='UK')).addr.country`>\n"
    )
    out = _run(src)
    assert out == "UK"


def test_inline_python_sees_type():
    src = "~Box(size:int)\n= <`Box(size=9).model_dump()['size']`>\n"
    out = _run(src)
    assert out == "9"


def test_type_definition_with_inline_comment():
    src = "~Person(age:int, name)  # comment\n= <`Person(age=5, name='kid').age`>\n"
    out = _run(src)
    assert out == "5"


def test_tilde_escape_in_text():
    src = "= literal ~~Person string\n"
    out = _run(src)
    assert out == "literal ~Person string"


def test_unescaped_tilde_in_text_errors():
    with pytest.raises(Exception):
        parse_program("= bad ~ text\n")


def test_duplicate_field_raises():
    with pytest.raises(Exception):
        parse_program("~Item(name, name)\n= ok\n")


def test_invalid_field_name_raises():
    with pytest.raises(Exception):
        parse_program("~Item(123name)\n= ok\n")


def test_missing_paren_in_type_raises():
    with pytest.raises(Exception):
        parse_program("~Item(name\n= ok\n")


def test_unknown_field_type_raises():
    with pytest.raises(TypeError):
        _run("~Foo(child: Unknown)\n= ok\n")


def test_field_without_annotation_defaults_to_str():
    src = (
        "~User(name, age:int)\n"
        "@make() -> User:\n"
        "  = ```\n"
        "  return User(name='eve', age=22)\n"
        "  ```\n\n"
        "= <`make().model_dump()['name'].upper()`>\n"
    )
    out = _run(src)
    assert out == "EVE"


def test_type_definition_in_block_comment_ignored():
    src = "###\n~Ghost(name)\n###\n~Real(name)\n= <`Real(name='visible').name`>\n"
    out = _run(src)
    assert out == "visible"


def test_pre_python_block_cannot_use_future_type():
    with pytest.raises(Exception):
        compile_program(
            parse_program("```\nvalue = Item(name='x', qty=4)\n```\n~Item(name, qty:int)\n= ok\n")
        )


def test_nested_type_scope_does_not_leak():
    src = (
        "@outer():\n"
        "  ~Inner(name)\n"
        "  = <`Inner(name='scoped').name`>\n\n"
        "= <outer()>\n"
        "= <`Inner(name='oops')`>\n"
    )
    with pytest.raises(Exception):
        _run(src)


def test_type_inside_procedure_available_in_body():
    src = (
        "@outer():\n"
        "  ~Inner(name)\n"
        "  [item: Inner] = ```\n"
        "  return Inner(name='value')\n"
        "  ```\n"
        "  = <`item.name`>\n\n"
        "= <outer()>\n"
    )
    out = _run(src)
    assert out == "value"


def test_procedure_parameters_use_scoped_type():
    src = (
        "@outer():\n"
        "  ~Inner(name)\n"
        "  @inner(x: Inner):\n"
        "    = <`x.name`>\n"
        "  = <inner(<`Inner(name='scoped')`>)>\n\n"
        "= <outer()>\n"
    )
    out = _run(src)
    assert out == "scoped"


def test_type_redefinition_shadowing():
    src = (
        "~Label(value)\n"
        "@outer():\n"
        "  ~Label(value:int)\n"
        "  = <`Label(value=5).value`>\n\n"
        "[text_label: Label] = ```\n"
        "return Label(value='text')\n"
        "```\n"
        "= <`str(outer()) + '/' + text_label.value`>\n"
    )
    out = _run(src)
    assert out == "5/text"


def test_type_used_in_assignment_block_with_annotation():
    src = "~Item(name)\n[value: Item] = ```\nreturn Item(name='a')\n```\n= <`value.name`>\n"
    out = _run(src)
    assert out == "a"


def test_positional_args_construction_inside_python():
    src = (
        "~City(name, history, economy, cuisine)\n"
        "@make() -> City:\n"
        "  = ```\n"
        "  return City('Name', 'Hist', 'Eco', 'Food')\n"
        "  ```\n\n"
        "= <`make().cuisine`>\n"
    )
    out = _run(src)
    assert out == "Food"


def test_mixed_positional_and_keyword_args():
    src = (
        "~Person(age:int, name, city)\n"
        "[p: Person] = ```\n"
        "return Person(30, name='Ada', city='London')\n"
        "```\n"
        "= <`p.age`>\n"
    )
    out = _run(src)
    assert out == "30"
