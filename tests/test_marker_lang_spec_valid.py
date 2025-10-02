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

        # Basic heuristics for some examples
        if "is located in" in description and "Paris" in description:
            data["country"] = "France"
        if "primary language" in description and "France" in description:
            data["language"] = "French"
        if "'Hello' in" in description and "French" in description:
            data["greeting"] = "Bonjour"

        # Fill required keys from prompt
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        if start != -1 and end != -1:
            keys = [k.strip() for k in prompt[start + 1 : end].split(",") if k.strip()]
            for k in keys:
                data.setdefault(k, f"{k}-value")

        return _FakeRunResult(output=data)


def _compile(src: str):
    return compile_program(parse_program(src))


def test_valid_basic_paris(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@get_country(city):\n"
        "  <city> is located in [country].\n"
        "  = <country>\n\n"
        "= <get_country(Paris)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out.lower() == "france"


def test_valid_nested_calls(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@get_country_info(city):\n"
        "  <city> is located in [country].\n"
        "  = <country>\n\n"
        "@get_language(country):\n"
        "  The primary language spoken in <country> is [language].\n"
        "  = <language>\n\n"
        "@translate_greeting(language):\n"
        "  'Hello' in <language> is [greeting]\n"
        "  = <greeting>\n\n"
        "= <translate_greeting(<get_language(<get_country_info(Paris)>)>)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "Bonjour"


def test_valid_escaping_and_literals(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@echo():\n"
        "  Please type <<hello>> and avoid [[brackets]].\n"
        "  = done\n\n"
        "= <echo()>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out == "done"


def test_valid_no_internal_collapse(monkeypatch: Any) -> None:
    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)
    src = (
        "@render(a, b):\n"
        "  =    Leading     <a>    and    <b>     combined     \n\n"
        "= <render(alpha, beta)>\n"
    )
    rt = _compile(src)
    out = rt.run_main()
    assert out.startswith("Leading     alpha    and    beta     combined")


