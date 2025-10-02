from __future__ import annotations

from typing import Any, Dict

import marker_pattern
from marker_pattern import i, o, c
from marker_pattern.core import MarkerRuntime

class _FakeRunResult:
    def __init__(self, output: Dict[str, Any]):
        self.output = output


class _FakeAgent:
    def __init__(self, model: str | None = None, *, output_type: Any = None, **_: Any) -> None:
        self.output_type = output_type

    def run_sync(self, prompt: str) -> _FakeRunResult:
        # derive required keys from prompt e.g. "Solve [country, population]."
        start = prompt.find("[")
        end = prompt.find("]", start + 1)
        keys = [k.strip() for k in prompt[start + 1 : end].split(",")]

        # try to read description to infer inputs (city/country/language)
        description = getattr(self.output_type, "description", "")

        data: Dict[str, Any] = {}

        # very small heuristic program for tests only
        if "Sivas" in description:
            if "located in" in description:
                data.update({"country": "Turkey"})

        if "Paris" in description:
            if "located in" in description:
                data.update({"country": "France"})

        if "primary language" in description:
            country = "France" if "France" in description else "Unknown"
            data.update({"language": "French" if country == "France" else "Unknown"})

        if "'Hello' in" in description:
            language = "French" if "French" in description else "Unknown"
            data.update({"greeting": "Bonjour" if language == "French" else "Hello"})

        # populate any remaining required keys with placeholders
        for k in keys:
            data.setdefault(k, f"{k}-value")

        return _FakeRunResult(output=data)


def test_city_country(monkeypatch: Any) -> None:
    # Monkeypatch the Agent used by the runtime
    import marker_pattern.core as core

    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)

    rt = MarkerRuntime()

    @rt.procedure
    def foo(city: str):
        rt.m([rt.i("city"), " is located in ", rt.o("country"), ". "])
        return rt.i("country")

    @rt.main
    def my_proc():
        return rt.c("foo", "Sivas")

    result = rt.run_main()
    assert result == "Turkey"


def test_nested_calls(monkeypatch: Any) -> None:
    import marker_pattern.core as core

    monkeypatch.setattr(core, "Agent", _FakeAgent, raising=True)

    rt = MarkerRuntime()

    @rt.procedure
    def get_country_info(city: str):
        rt.m([rt.i("city"), " is located in ", rt.o("country")])
        return rt.i("country")

    @rt.procedure
    def get_language(country: str):
        rt.m(["The primary language spoken in ", rt.i("country"), " is ", rt.o("language")])
        return rt.i("language")

    @rt.procedure
    def translate_greeting(language: str):
        rt.m(["'Hello' in ", rt.i("language"), " is ", rt.o("greeting")])
        return rt.i("greeting")

    @rt.main
    def greeting_chain():
        return rt.c("translate_greeting", rt.c("get_language", rt.c("get_country_info", "Paris")))

    result = rt.run_main()
    assert result == "Bonjour"

