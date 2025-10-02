from __future__ import annotations

import os
import pytest

from marker_pattern.core import MarkerRuntime


requires_openai = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set; real LLM tests are skipped",
)


@requires_openai
@pytest.mark.llm
def test_llm_echo_ok() -> None:
    rt = MarkerRuntime()

    @rt.procedure
    def echo_ok():  # type: ignore[no-redef]
        rt.m(["Return exactly ", "ok", " as ", rt.o("answer"), "."])  # type: ignore[name-defined]
        return rt.i("answer")  # type: ignore[name-defined]

    @rt.main
    def run():  # type: ignore[no-redef]
        return rt.c("echo_ok")  # type: ignore[name-defined]

    out = rt.run_main()
    assert isinstance(out, str)
    assert out.strip().lower().startswith("ok")


@requires_openai
@pytest.mark.llm
def test_llm_paris_country() -> None:
    rt = MarkerRuntime()

    @rt.procedure
    def get_country(city: str):  # type: ignore[no-redef]
        rt.m([rt.i("city"), " is located in ", rt.o("country"), "."])  # type: ignore[name-defined]
        return rt.i("country")  # type: ignore[name-defined]

    @rt.main
    def main():  # type: ignore[no-redef]
        return rt.c("get_country", "Paris")  # type: ignore[name-defined]

    out = rt.run_main()
    assert isinstance(out, str)
    assert out.strip().lower() == "france"


@requires_openai
@pytest.mark.llm
def test_llm_word_chain() -> None:
    rt = MarkerRuntime()

    @rt.procedure
    def word_chain(word: str):  # type: ignore[no-redef]
        # Generate 5 words where each starts with the last letter of the previous word
        words = []
        for _ in range(5):
            print(f"word: {rt._current_env.get()['word']}")
            rt.m(
                [
                    rt.o("word"),
                    " is a word that starts with the last letter of ",
                    rt.i("word"),
                    ". ",
                ]
            )  # type: ignore[name-defined]
            
        # assertions:
        for idx, word in enumerate(words):
            if idx > 0:
                assert word.startswith(words[idx - 1][-1])

        return rt.i("word")  # type: ignore[name-defined]

    @rt.main
    def run():  # type: ignore[no-redef]∂
        return rt.c("word_chain", "apple")  # type: ignore[name-defined]

    out = rt.run_main()
