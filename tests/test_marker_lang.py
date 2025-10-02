from __future__ import annotations

from typing import Any, Dict

import marker_pattern.core as core
from marker_pattern.core import MarkerRuntime
from marker_pattern.lang import parse_program, compile_program


class _FakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _FakeAgent:
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _FakeRunResult:
        # Simple heuristic for tests
        description = getattr(self.output_type, "description", "")
        data: Dict[str, Any] = {}

        # Heuristic mapping for the example
        if "is located in" in description and "Paris" in description:
            data["country"] = "France"

        # Parse required keys from the question prompt like "Fill in the fields: [country]."
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
            for k in keys:
                data.setdefault(k, f"{k}-value")

        return _FakeRunResult(output=data)


def _compile(src: str) -> MarkerRuntime:
    prog = parse_program(src)
    return compile_program(prog)


def test_parse_and_run_paris(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)

    src = (
        "@get_country(city):\n"
        "  <city> is located in [country].\n"
        "  = <country>\n\n"
        "= <get_country(Paris)>\n"
    )

    rt = _compile(src)
    out = rt.run_main()
    assert isinstance(out, str)
    assert out.lower() == "france"


