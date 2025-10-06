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
        # deterministic fill
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
        data: Dict[str, Any] = {k: f"{k}-value" for k in keys}
        return _FakeRunResult(output=data)


def _compile(src: str):
    return compile_program(parse_program(src))


def test_python_block_sets_variables(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@calc(a,b):\n"
        "  ```\n"
        "  s = int(a) + int(b)\n"
        "  t = int(a) * int(b)\n"
        "  ```\n"
        "  = sum:<`s`> prod:<`t`>\n\n"
        "= <calc(6, 7)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "sum:13 prod:42"


def test_python_block_can_call_procedures(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@times2(n):\n"
        "  = <`int(n)*2`>\n\n"
        "@use(a):\n"
        "  ```\n"
        "  doubled = times2(a)\n"
        "  tripled = int(a) * 3\n"
        "  ```\n"
        "  = <`str(doubled)+'-'+str(tripled)`>\n\n"
        "= <use(5)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "10-15"


def test_python_block_with_llm_outputs(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@pipeline(x):\n"
        "  from <x> we ask for [y]\n"
        "  ```\n"
        "  z = 'prefix-' + y\n"
        "  ```\n"
        "  final [out] uses <`z`>\n"
        "  = <out>\n\n"
        "= <pipeline(alpha)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out.startswith("out-value") or out.startswith("prefix-")


