from __future__ import annotations

import pytest

from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


def _run(src: str) -> str:
    rt = compile_program(parse_program(src), adapter=MockAdapter())
    out = rt.run_main()
    assert isinstance(out, str)
    return out


def test_assignment_uses_prelude_type() -> None:
    src = (
        "```\n"
        "class Foo:\n"
        "    def __init__(self, value: int):\n"
        "        self.value = value\n"
        "```\n"
        "[foo: Foo] = `Foo(21)`\n"
        "= <`foo.value`>\n"
    )
    out = _run(src)
    assert out == "21"


def test_procedure_params_and_returns_use_prelude_type() -> None:
    src = (
        "```\n"
        "class Foo:\n"
        "    def __init__(self, val: int):\n"
        "        self.val = val\n"
        "```\n"
        "@factory(val: int) -> Foo:\n"
        "  = `Foo(val)`\n"
        "@consume(item: Foo):\n"
        "  = <`item.val`>\n"
        "[obj: Foo] = `factory(7)`\n"
        "= <consume(`obj`)>\n"
    )
    out = _run(src)
    assert out == "7"


def test_base_prelude_imports() -> None:
    src = (
        "```\n"
        "import math\n"
        "PI = math.pi\n"
        "def triple(x):\n"
        "    return x * 3\n"
        "```\n"
        "= <`round(triple(math.sqrt(4)) + PI, 2)`>\n"
    )
    out = _run(src)
    assert out == "9.14"


def test_prelude_overrides_global_for_blocks() -> None:
    src = (
        "```\n"
        "GLOBAL = 10\n"
        "def add_two(n):\n"
        "    return n + 2\n"
        "```\n"
        "[x:int] = ```\n"
        "return add_two(GLOBAL)\n"
        "```\n"
        "= <`x + GLOBAL`>\n"
    )
    out = _run(src)
    assert out == "22"


def test_only_first_block_is_prelude() -> None:
    src = "```\nVALUE = 5\n```\n```\nVALUE = 7\n```\n= <`VALUE`>\n"
    out = _run(src)
    assert out == "5"


def test_prelude_math_imports_success() -> None:
    src = "```\nimport math\n```\n= <`int(math.sqrt(9))`>\n"
    out = _run(src)
    assert out == "3"


def test_prelude_math_imports_failure() -> None:
    src = "```\n# dummy\n```\n\n```\nimport math\n```\n\n= <`math.sqrt(9)`>\n"
    with pytest.raises(Exception):
        _run(src)
