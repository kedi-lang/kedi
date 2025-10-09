from __future__ import annotations

from typing import Any

import pytest

from marker_pattern.lang import parse_program, compile_program
import marker_pattern.core as core


class _SingleFakeAgent:  # type: ignore[misc]
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
    core.Agent = _SingleFakeAgent  # type: ignore[assignment]
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    return out


def test_basic_multiline_return(monkeypatch: Any) -> None:
    src = (
        "= Start \\\n"
        "  middle \\\n"
        "  end\n"
    )
    out = _run(src)
    assert out == "Start middle end"


def test_multiline_with_substitutions(monkeypatch: Any) -> None:
    src = (
        "@p(x,y):\n"
        "  = Left <x> \\     \n"
        "    mid <`y.upper()`> \\\n"
        "    right\n\n"
        "= <p(alpha, beta)>\n"
    )
    out = _run(src)
    assert out == "Left alpha mid BETA right"


def test_multiline_escaped_backslash(monkeypatch: Any) -> None:
    src = (
        "= path: C:\\\\Users\\\\name \\\n"
        "  tail\n"
    )
    out = _run(src)
    assert "C:\\Users\\name" in out and out.endswith(" tail")


def test_multiline_inside_template_line(monkeypatch: Any) -> None:
    src = (
        "@p():\n"
        "  prefix \\\n"
        "    [x] \\\n"
        "    suffix\n"
        "  = <x>\n\n"
        "= <p()>\n"
    )
    out = _run(src)
    assert out.endswith("-value")


def test_multiline_with_nested_calls(monkeypatch: Any) -> None:
    src = (
        "@id(x):\n  = <x>\n\n"
        "= [[ [[ <id(A)>    \\   \n" # ignore whitespaces after \, but not before
        "      and <id(B)>    ]] ]]\n"
    )
    out = _run(src)
    assert out == "[ [ A and B    ] ]"


