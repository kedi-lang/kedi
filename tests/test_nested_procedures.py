from __future__ import annotations

from typing import Any

import pytest

from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


def _run(src: str) -> str:
    rt = compile_program(parse_program(src), adapter=MockAdapter())
    out = rt.run_main()

    assert isinstance(out, str)
    return out


def test_nested_simple_definition(monkeypatch: Any) -> None:
    # Define @bar inside @foo and call it
    src = "@foo():\n  @bar():\n    = hi\n  = <bar()>\n\n= <foo()>\n"
    out = _run(src)
    assert out == "hi"


def test_nested_shadowing_and_scope(monkeypatch: Any) -> None:
    # Inner bar should shadow outer bar
    src = "@bar():\n  = outer\n\n@foo():\n  @bar():\n    = inner\n  = <bar()>\n\n= <foo()>\n"
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
    src = "@foo():\n  @hello():\n    = hi\n  = <`hello()`>\n\n= <foo()>\n"
    out = _run(src)
    assert out == "hi"


def test_multiple_nested_procedures(monkeypatch: Any) -> None:
    src = "@foo():\n  @a():\n    = A\n  @b():\n    = B\n  = <a()><b()>\n\n= <foo()>\n"
    out = _run(src)
    assert out == "AB"


def test_shadow_param_names(monkeypatch: Any) -> None:
    src = "@foo(x):\n  @tripler(x):\n    = <`int(x)*3`>\n  = <tripler(4)>\n\n= <foo(10)>\n"
    out = _run(src)
    assert out == "12"


def test_outer_cannot_see_inner_param(monkeypatch: Any) -> None:
    src = "@foo(x):\n  @inner(n):\n    = <n>\n  = <`int(n)`>\n\n= <foo(7)>\n"
    with pytest.raises(Exception):
        _run(src)


def test_outer_cannot_see_inner_assignment(monkeypatch: Any) -> None:
    src = (
        "@foo():\n  @inner():\n    [secret] = value\n    = <secret>\n  = <`secret`>\n\n= <foo()>\n"
    )
    with pytest.raises(Exception):
        _run(src)


def test_outer_cannot_use_inner_param_name(monkeypatch: Any) -> None:
    src = "@foo():\n  @bar(n):\n    = <n>\n  = <`int(n)`>\n\n= <foo()>\n"
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


def test_lexical_closure_reads_assignment(monkeypatch: Any) -> None:
    src = "@goo():\n  [x:int] = `2`\n  @bar() -> int:\n    = `x`\n  = <`bar()`>\n\n= <goo()>\n"
    out = _run(src)
    assert out == "2"


def test_lexical_closure_reads_latest_outer_assignment(monkeypatch: Any) -> None:
    src = (
        "@outer():\n"
        "  [x] = before\n"
        "  @inner():\n"
        "    = <x>\n"
        "  [x] = after\n"
        "  = <inner()>\n\n"
        "= <outer()>\n"
    )
    out = _run(src)
    assert out == "after"


def test_python_rebinding_updates_nearest_existing_kedi_owner() -> None:
    src = '[value] = before\n@change():\n  `value = "after"`\n\n`change()`\n= <value>\n'

    assert _run(src) == "after"


def test_kedi_initialization_in_procedure_shadows_outer_binding() -> None:
    src = (
        "[value] = outer\n"
        "@change():\n"
        "  [value] = inner\n"
        '  `value = "updated-inner"`\n\n'
        "`change()`\n"
        "= <value>\n"
    )

    assert _run(src) == "outer"


def test_lexical_closure_updates_outer_assignment(monkeypatch: Any) -> None:
    src = (
        "@outer():\n"
        "  [val:int] = `1`\n"
        "  @mid():\n"
        "    [val] = bump\n"
        "    @inner():\n"
        "      = <val>\n"
        "    = <inner()>\n"
        "  = <mid()>\n\n"
        "= <outer()>\n"
    )
    out = _run(src)
    assert out == "bump"


def test_closure_chain_across_multiple_levels(monkeypatch: Any) -> None:
    src = (
        "@top():\n"
        "  [a:int] = `1`\n"
        "  @level1():\n"
        "    [b:int] = `a + 1`\n"
        "    @level2():\n"
        "      [c:int] = `b + 1`\n"
        "      @level3():\n"
        "        = <`c + a`>\n"
        "      = <level3()>\n"
        "    = <level2()>\n"
        "  = <level1()>\n\n"
        "= <top()>\n"
    )
    out = _run(src)
    assert out == "4"


def test_closure_preserves_runtime_inputs(monkeypatch: Any) -> None:
    src = (
        "@outer(base):\n"
        "  @adder(delta):\n"
        "    @inner():\n"
        "      = <`int(base) + int(delta)`>\n"
        "    = <inner()>\n"
        "  = <adder(3)>\n\n"
        "= <outer(7)>\n"
    )
    out = _run(src)
    assert out == "10"


def test_closure_allows_python_block_mutation(monkeypatch: Any) -> None:
    src = (
        "@outer():\n"
        "  [counter:int] = `0`\n"
        "  ```\n"
        "  counter += 5\n"
        "  ```\n"
        "  @inner():\n"
        "    ```\n"
        "    counter += 7\n"
        "    ```\n"
        "    = <`counter`>\n"
        "  = <inner()>\n\n"
        "= <outer()>\n"
    )
    out = _run(src)
    assert out == "12"


def test_closure_shadowing_parameter(monkeypatch: Any) -> None:
    src = (
        "@outer(x:int):\n"
        "  [x:int] = `x + 1`\n"
        "  @inner():\n"
        "    [x:int] = `x + 1`\n"
        "    @deep():\n"
        "      = <`x`>\n"
        "    = <deep()>\n"
        "  = <inner()>\n\n"
        "= <outer(`1`)>\n"
    )
    out = _run(src)
    assert out == "3"
