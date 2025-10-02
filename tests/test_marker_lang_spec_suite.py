from __future__ import annotations

from typing import Any, Dict

import marker_pattern.core as core
import pytest
from marker_pattern.lang import parse_program, compile_program


class _FakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _FakeAgent:
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _FakeRunResult:
        description = getattr(self.output_type, "description", "")
        data: Dict[str, Any] = {}
        # Generic fill
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys = []
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
        for k in keys:
            data.setdefault(k, f"{k}-value")
        return _FakeRunResult(output=data)


def _run(monkeypatch: Any, src: str) -> str:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    return out


@pytest.mark.parametrize(
    "src",
    [
        # Multiple outputs in one line
        (
            "@both(x):\n"
            "  <x> to [a] [[comma]] [b]\n"
            "  = <a> + <b>\n\n"
            "= <both(z)>\n"
        ),
        # Escaped commas in args
        (
            "@bar(s):\n"
            "  = <s>\n\n"
            "@foo():\n"
            "  = <bar(alpha,, beta,, gamma)>\n\n"
            "= <foo()>\n"
        ),
        # Nested call with literal and var
        (
            "@inner(x):\n"
            "  = <x>\n\n"
            "@outer(a,b):\n"
            "  = <a>-<b>\n\n"
            "@t(x):\n"
            "  = <outer(alpha, <inner(<x>)>)>\n\n"
            "= <t(z)>\n"
        ),
        # Only returns
        (
            "= Paris\n"
        ),
        # Interleaved spacing
        (
            "@r(a,b):\n"
            "  =  a:<a>   b:<b>   \n\n"
            "= <r(aa, bb)>\n"
        ),
    ],
)
def test_suite_various_valid(monkeypatch: Any, src: str) -> None:
    out = _run(monkeypatch, src)
    assert isinstance(out, str)
    assert len(out) > 0


