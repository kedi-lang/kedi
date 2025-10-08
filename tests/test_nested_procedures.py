from __future__ import annotations

from typing import Any

import pytest

from marker_pattern.lang import parse_program, compile_program
import marker_pattern.core as core


def _run(src: str) -> str:
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    return out


def test_nested_simple_definition(monkeypatch: Any) -> None:
    # Define @bar inside @foo and call it
    src = (
        "@foo():\n"
        "  @bar():\n"
        "    = hi\n"
        "  = <bar()>\n\n"
        "= <foo()>\n"
    )
    out = _run(src)
    assert out == "hi"


def test_nested_shadowing_and_scope(monkeypatch: Any) -> None:
    # Inner bar should shadow outer bar
    src = (
        "@bar():\n  = outer\n\n"
        "@foo():\n"
        "  @bar():\n"
        "    = inner\n"
        "  = <bar()>\n\n"
        "= <foo()>\n"
    )
    out = _run(src)
    assert out == "inner"


def test_nested_call_chain(monkeypatch: Any) -> None:
    src = (
        "@foo(x:int):\n"
        "  @twice(n:int) -> int:\n"
        "    = `int(n)*2`\n"
        "  @add1(n:int) -> int:\n"
        "    = `int(n)+1`\n"
        "  = <add1(`twice(x)`)>\n\n"
        "= <foo(`5`)>\n"
    )
    out = _run(src)
    assert out == "11"


def test_nested_defined_then_used_in_python(monkeypatch: Any) -> None:
    src = (
        "@foo():\n"
        "  @hello():\n"
        "    = hi\n"
        "  = <`hello()`>\n\n"
        "= <foo()>\n"
    )
    out = _run(src)
    assert out == "hi"


def test_multiple_nested_procedures(monkeypatch: Any) -> None:
    src = (
        "@foo():\n"
        "  @a():\n"
        "    = A\n"
        "  @b():\n"
        "    = B\n"
        "  = <a()><b()>\n\n"
        "= <foo()>\n"
    )
    out = _run(src)
    assert out == "AB"


def test_inner_cannot_see_outer_param_without_pass(monkeypatch: Any) -> None:
    # Inner references outer x without receiving it -> NameError in inline python
    src = (
        "@foo(x):\n"
        "  @inner():\n"
        "    = <`int(x)`>\n"
        "  = <inner()>\n\n"
        "= <foo(7)>\n"
    )
    with pytest.raises(Exception):
        _run(src)


def test_inner_cannot_see_outer_assigned_var(monkeypatch: Any) -> None:
    src = (
        "@foo():\n"
        "  [y] = hello\n"
        "  @inner():\n"
        "    = <y>\n"
        "  = <inner()>\n\n"
        "= <foo()>\n"
    )
    with pytest.raises(Exception):
        _run(src)


def test_shadow_param_names(monkeypatch: Any) -> None:
    src = (
        "@foo(x):\n"
        "  @tripler(x):\n"
        "    = <`int(x)*3`>\n"
        "  = <tripler(4)>\n\n"
        "= <foo(10)>\n"
    )
    out = _run(src)
    assert out == "12"


def test_outer_cannot_use_inner_param_name(monkeypatch: Any) -> None:
    src = (
        "@foo():\n"
        "  @bar(n):\n"
        "    = <n>\n"
        "  = <`int(n)`>\n\n"
        "= <foo()>\n"
    )
    with pytest.raises(Exception):
        _run(src)


def test_two_level_nested_params(monkeypatch: Any) -> None:
    src = (
        "@foo(a:int):\n"
        "  @f1(n:int):\n"
        "    @f2(m:int):\n"
        "      = <`int(m)+1`>\n"
        "    = <f2(`int(n)*2`)>\n"
        "  = <f1(`a`)>\n\n"
        "= <foo(`5`)>\n"
    )
    out = _run(src)
    assert out == "11"


