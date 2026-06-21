from __future__ import annotations

# ruff: noqa: E402
import asyncio
import logging
import os
from pathlib import Path
from typing import TypeVar

os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "dspy"))
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

from pydantic import BaseModel

import kedi

T = TypeVar("T")


@kedi.type
class Review:
    decision: str
    summary: str


@kedi.type(inject=False)
class Person(BaseModel):
    name: str
    age: int
    job: str


@kedi.tool(
    name="search_local_docs",
    description="Search the local Kedi notes bundled with this example.",
    retries=1,
)
def search_docs(query: str) -> str:
    docs = {
        "python api": "Kedi Python API embeds Kedi programs in Python docstrings.",
        "tools": "Use @kedi.tool to expose Python callables to Kedi prompts.",
        "types": "Backtick type annotations can resolve runtime Python types.",
    }
    needle = query.lower()
    matches = [text for key, text in docs.items() if key in needle or needle in key]
    return "\n".join(matches or docs.values())


kedi.configure(
    system="Use tools when they are relevant. Keep answers compact.",
    tools=[search_docs],
    env={
        "audience": "Python developers",
        "Person": Person,
    },
)


@kedi.query(cache=True, settings={"temperature": 0.2})
def summarize(topic: str) -> str:
    """kedi
    > use: search_local_docs
    >> Use the search_local_docs tool to summarize <topic>. [summary]
    = <summary>
    """
    ...


@kedi.query(cache=True)
def review_snippet(language: str, code: str) -> Review:
    """kedi
    >> Review this <language> snippet for <audience>.
    Return [review: Review].
    = `review`
    """
    ...


@kedi.query
def extract_person(*, text: str, output_type: type[T]) -> T:
    """kedi
    >> Extract [output: `output_type`] from <text>.
    = `output`
    """
    ...


@kedi.bind(file=Path(__file__).with_name("bound_summary.kedi"), cache=True, reload=True)
def summarize_from_file(topic: str) -> str:
    """Summarize with the file-backed Kedi program."""
    ...


@kedi.query(system="Answer in one sentence.")
async def answer_async(question: str) -> str:
    """kedi
    >> Answer <question>. [answer]
    = <answer>
    """
    ...


def main() -> None:
    print(summarize("python api tools"))
    print(
        review_snippet(
            language="python",
            code="def add(a: int, b: int) -> int:\n    return a + b",
        )
    )
    print(extract_person(text="Ada is a 36 year old programmer.", output_type=Person))
    print(summarize_from_file("file-backed Kedi programs"))

    with kedi.context(system="Prefer a practical example."):
        print(summarize("runtime Python types"))

    print(asyncio.run(answer_async("What is Kedi?")))
    print(kedi.cache_info())


if __name__ == "__main__":
    main()
