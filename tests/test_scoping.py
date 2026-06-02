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


def test_top_level_assignment_visible_in_python_block(monkeypatch: Any) -> None:
    src = (
        "[x:int] = `7`\n@p():\n  [y: int] = ```\n  return x * 2\n  ```\n  = <`str(y)`>\n\n= <p()>\n"
    )
    out = _run(src)
    assert out == "14"


def test_python_block_modifies_existing_kedi_var_only(monkeypatch: Any) -> None:
    src = (
        "@p():\n"
        "  [x] = base\n"
        "  ```\n"
        "  x = x + '-mod'\n"
        "  y = 'local'\n"
        "  ```\n"
        "  = <x>-<`y`>\n\n"
        "= <p()>\n"
    )
    with pytest.raises(RuntimeError):
        _run(src)


def test_new_python_names_do_not_leak_to_kedi(monkeypatch: Any) -> None:
    src = "@p():\n  ```\n  z = 'new'\n  ```\n  = <z>\n\n= <p()>\n"
    with pytest.raises(Exception):
        _run(src)


def test_inline_python_reads_globals_and_locals(monkeypatch: Any) -> None:
    src = "[g: int] = `2`\n@p(x):\n  = <`str(g + int(x))`>\n\n= <p(5)>\n"
    out = _run(src)
    assert out == "7"


def test_no_cross_block_persistence(monkeypatch: Any) -> None:
    src = (
        "@p():\n"
        "  ```\n"
        "  a = 10\n"
        "  ```\n"
        "  ```\n"
        "  try:\n"
        "      _ = a\n"
        "      raise AssertionError('should not persist across blocks')\n"
        "  except NameError:\n"
        "      pass\n"
        "  ```\n"
        "  = ok\n\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out == "ok"


def test_nested_procedure_can_see_globals(monkeypatch: Any) -> None:
    src = (
        "[seed] = start\n"
        "@inner():\n"
        "  [out] = ```\n"
        "  return seed + '-x'\n"
        "  ```\n"
        "  = <out>\n\n"
        "@outer():\n"
        "  = <inner()>\n\n"
        "= <outer()>\n"
    )
    out = _run(src)
    assert out == "start-x"


def test_inline_name_error_yields_empty(monkeypatch: Any) -> None:
    src = "= A<`unknown_name`>B\n"
    with pytest.raises(Exception):
        _run(src)


def test_assign_block_type_coercion(monkeypatch: Any) -> None:
    src = "@p():\n  [n: int] = ```\n  return '5'\n  ```\n  = <`str(n*2)`>\n\n= <p()>\n"
    with pytest.raises(TypeError):  # type coercion is not allowed, should raise TypeError
        _run(src)


def test_nested_call_reads_top_level_assignment(monkeypatch: Any) -> None:
    src = "[seed] = outer\n@inner():\n  = <seed>\n\n@outer():\n  = <inner()>\n\n= <outer()>\n"
    out = _run(src)
    assert out == "outer"


def test_inline_substitution_reads_globals(monkeypatch: Any) -> None:
    src = "[x] = value\n= <x>\n"
    out = _run(src)
    assert out == "value"
