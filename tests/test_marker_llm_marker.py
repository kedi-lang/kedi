from __future__ import annotations

import os
import pytest

from marker_pattern.lang import parse_program, compile_program


requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; real LLM tests are skipped",
)


@requires_groq
@pytest.mark.llm
def test_marker_llm_paris_country() -> None:
    src = (
        "@get_country(city):\n"
        "  <city> is located in [country].\n"
        "  = <country>\n\n"
        "= <get_country(Paris)>\n"
    )
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    assert out.strip().lower() == "france"


@requires_groq
@pytest.mark.llm
def test_marker_llm_language_chain_bonjour() -> None:
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
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    # tolerant assertion for greeting
    assert out.strip().lower().startswith("bonjour") or out.strip().lower() in {"bonjour", "bon jour"}


@requires_groq
@pytest.mark.llm
def test_marker_llm_word_chain_marker_file() -> None:
    with open("tests/markers/word_chain.marker", "r", encoding="utf-8") as f:
        src = f.read()
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    assert len(out.strip()) > 0
    assert out.strip().lower() != "apple"


