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
def test_marker_llm_escaping_and_literals() -> None:
    src = (
        "@show():\n"
        "  Please type <<hello>> to continue and avoid [[bad]] == examples.\n"
        "  = done\n\n"
        "= <show()>\n"
    )
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    assert out.strip().lower().startswith("done")


@requires_groq
@pytest.mark.llm
@llm_retry(tries=3, delay_s=0.5)
def test_marker_llm_no_internal_space_collapse() -> None:
    # Ensure only ends are trimmed; internal spacing preserved
    src = (
        "@render(a, b, c):\n"
        "  =    Leading     <a>    and    <b>     combined     \n\n"
        "= <render(alpha, beta, gamma)>\n"
    )
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    assert out.startswith("Leading     alpha    and    beta     combined")


@requires_groq
@pytest.mark.llm
@llm_retry(tries=3, delay_s=0.5)
def test_marker_llm_nested_calls_and_args() -> None:
    # Mix of literals and variables in nested calls
    src = (
        "@title(name):\n"
        "  = Dr. <name>\n\n"
        "@report(name):\n"
        "  =   Report:   <title(<name>)>   -    <date>   \n\n"
        "= <report(Alice)>\n"
    )
    rt = compile_program(parse_program(src))
    out = rt.run_main()
    assert isinstance(out, str)
    assert out.strip().startswith("Report: ")


