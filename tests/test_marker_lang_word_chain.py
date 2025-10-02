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
        description = getattr(self.output_type, "description", "")
        data: Dict[str, Any] = {}
        # naive heuristic: set 'word' to last letter + 'a'
        # e.g. previous 'apple' -> 'a' starts-with 'e'? Adjust simply to pass chaining deterministically
        # We'll just toggle between 'alpha' and 'atom' to simulate chaining
        if "[word]" in description or " is a word that starts with the last letter of " in description:
            data["word"] = "alpha"
        return _FakeRunResult(output=data)


def test_word_chain_from_file(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    with open("tests/markers/word_chain.marker", "r", encoding="utf-8") as f:
        src = f.read()
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    assert len(out) > 0


