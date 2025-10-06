from __future__ import annotations

from typing import Any, Dict

import marker_pattern.core as core
from marker_pattern.lang import parse_program, compile_program


class _FakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _FakeAgent:
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _FakeRunResult:
        # deterministically fill outputs
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
        data: Dict[str, Any] = {k: f"{k}-value" for k in keys}
        return _FakeRunResult(output=data)


def _compile(src: str):
    return compile_program(parse_program(src))


def test_assign_basic(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@p(x):\n"
        "  [y] = <x>\n"
        "  = <y>\n\n"
        "= <p(alpha)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "alpha"


def test_assign_with_call(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@id(x):\n"
        "  = <x>\n\n"
        "@p():\n"
        "  [v] = <id(beta)>\n"
        "  = <v>\n\n"
        "= <p()>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "beta"


def test_assign_then_llm(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@p(city):\n"
        "  [c] = <city>\n"
        "  <c> is located in [country] \n"
        "  = <country>\n\n"
        "= <p(Paris)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "country-value"


def test_assign_overwrite(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@p():\n"
        "  [x] = alpha\n"
        "  [x] = beta\n"
        "  = <x>\n\n"
        "= <p()>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "beta"


def test_assign_inline_python_rhs(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@p(a,b):\n"
        "  [s] = <`str(a)+str(b)`>\n"
        "  = <s>\n\n"
        "= <p(1, 2)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "12"


def test_assign_works_before_return(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@p():\n"
        "  [x] = hello\n"
        "  = world <x>\n\n"
        "= <p()>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "world hello"


def test_assign_interacts_with_outputs(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@p():\n"
        "  gain [a]\n"
        "  [b] = <a>\n"
        "  = <b>\n\n"
        "= <p()>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "a-value"


