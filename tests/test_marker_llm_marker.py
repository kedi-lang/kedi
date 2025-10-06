from __future__ import annotations

import os
import pytest

from marker_pattern.lang import parse_program, compile_program
from utils_llm import llm_retry


requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; real LLM tests are skipped",
)


@requires_groq
@pytest.mark.llm
@llm_retry(tries=3, delay_s=0.5)
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
@llm_retry(tries=3, delay_s=0.5)
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
    low = out.strip().lower()
    assert low.startswith("bonjour") or low in {"bonjour", "bon jour"}


# Word-chain marker file is covered by the unified markers suite


