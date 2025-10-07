from __future__ import annotations

from typing import Any

import pytest

import marker_pattern.core as core
from marker_pattern.lang import parse_program, compile_program


class _SingleFakeAgent(core.Agent):  # type: ignore[misc]
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:  # type: ignore[no-untyped-def]
        self.output_type = output_type

    def run_sync(self, prompt: str):  # type: ignore[override]
        # Very simple agent: return all requested fields as deterministic placeholders
        from typing import List, Dict

        data: Dict[str, Any] = {}
        # discover requested keys like [x, y]
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys: List[str] = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
        class R:
            def __init__(self, d: Dict[str, Any]):
                self.output = d
        for k in keys:
            data[k] = f"{k}-value"
        return R(data)


def _run(src: str) -> str:
    rt = compile_program(parse_program(src))
    rt.Agent = _SingleFakeAgent  # type: ignore[attr-defined]
    out = rt.run_main()
    assert isinstance(out, str)
    return out


def test_top_level_assignment_visible_in_python_block(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = (
        "[x:int] = `7`\n"
        "@p():\n"
        "  [y: int] = ```\n"
        "  return x * 2\n"
        "  ```\n"
        "  = <`str(y)`>\n\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out == "14"


def test_python_block_modifies_existing_marker_var_only(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
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
    out = _run(src)
    assert out == "base-mod-"


def test_new_python_names_do_not_leak_to_marker(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = (
        "@p():\n"
        "  ```\n"
        "  z = 'new'\n"
        "  ```\n"
        "  = <z>\n\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out == ""


def test_inline_python_reads_globals_and_locals(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = (
        "[g: int] = `2`\n"
        "@p(x):\n"
        "  = <`str(g + int(x))`>\n\n"
        "= <p(5)>\n"
    )
    out = _run(src)
    assert out == "7"


def test_no_cross_block_persistence(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
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
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
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
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = (
        "= A<`unknown_name`>B\n"
    )
    out = _run(src)
    assert out == "AB"


def test_assign_block_type_coercion(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _SingleFakeAgent, raising=True)
    src = (
        "@p():\n"
        "  [n: int] = ```\n"
        "  return '5'\n"
        "  ```\n"
        "  = <`str(n*2)`>\n\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out == "10"
