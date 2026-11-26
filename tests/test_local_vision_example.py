"""The local vision example (#198, P1): reading receipt images through Kedi with Cactus Mirage.

cactus-compute is optional and not installed in CI, so Mirage's text reader is faked. The
image prompt ``<<photo>>``, the LangChain adapter's image path, the program's parsing and its
add-up check are real.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_adapter import LangChainAdapter
from kedi.lang import compile_program, parse_program
from kedi.media import Image

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "local_vision"
PROGRAM = EXAMPLE / "receipts.kedi"


def regions(lines: list[list[str]]) -> list[dict[str, Any]]:
    """Text regions as Mirage returns them, right column first to exercise reading order."""

    found = []
    for row, words in enumerate(lines):
        for column, word in reversed(list(enumerate(words))):
            top = 100 + row * 80
            found.append(
                {"string": word, "bbox": [50 + column * 400, top, 300 + column * 400, top + 50]}
            )
    return found


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    sys.modules.pop("vision_model", None)
    import vision_model

    seen: list[bytes] = []

    class FakeMirage:
        def describe(self, image: bytes, sections: list[str]) -> dict[str, Any]:
            assert sections == ["text"]
            seen.append(image)
            return {"text": regions(vision_model.lines)}

    monkeypatch.setattr(vision_model, "_mirage", FakeMirage)
    vision_model.seen = seen
    return vision_model


def run(reader: Any, key: str) -> Any:
    photo = Image.from_bytes((EXAMPLE / "receipts" / f"{key}.png").read_bytes(), "image/png")
    runtime = compile_program(
        parse_program(PROGRAM.read_text(), source_path=str(PROGRAM)),
        adapter=LangChainAdapter(reader.MirageReader()),
        runtime_globals={"photo": photo},
    )
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def test_the_image_reaches_the_reader_and_the_receipt_is_parsed(reader: Any) -> None:
    reader.lines = [["CORNER CAFE"], ["Latte", "4.50"], ["Croissant", "3.20"], ["TOTAL", "7.70"]]
    result = run(reader, "cafe")
    assert result["writing"] == "CORNER CAFE\nLatte 4.50\nCroissant 3.20\nTOTAL 7.70"
    assert result["receipt"] == {
        "merchant": "CORNER CAFE",
        "items": [{"name": "Latte", "price": 4.5}, {"name": "Croissant", "price": 3.2}],
        "total": 7.7,
        "adds_up": True,
    }
    (image,) = reader.seen
    assert image == (EXAMPLE / "receipts" / "cafe.png").read_bytes()


def test_a_total_that_does_not_add_up_fails_the_check(reader: Any) -> None:
    reader.lines = [
        ["SUNRISE BAKERY"],
        ["Sourdough", "5.80"],
        ["Cinnamon roll", "2.90"],
        ["TOTAL", "9.70"],
    ]
    assert run(reader, "bakery")["receipt"]["adds_up"] is False


def test_the_ground_truth_matches_the_images() -> None:
    truth = json.loads((EXAMPLE / "receipts" / "truth.json").read_text())
    assert set(truth) == {path.stem for path in (EXAMPLE / "receipts").glob("*.png")}
    assert [key for key, receipt in truth.items() if not receipt["adds_up"]] == ["bakery"]


@pytest.mark.parametrize(
    "source",
    [
        "> import: media\n~Total(total: float)\n[t: Total] << Read <<photo>>.\n= `t`\n",
        "> import: media\n@shout(text: str) -> str:\n  > tool:\n      risk: read_only\n"
        "  = `text.upper()`\n> use: shout\n[w] << Read <<photo>>.\n= <w>\n",
    ],
    ids=["typed capture", "program tool"],
)
def test_the_reader_refuses_typed_output_and_tools(reader: Any, source: str) -> None:
    photo = Image.from_bytes((EXAMPLE / "receipts" / "cafe.png").read_bytes(), "image/png")
    runtime = compile_program(
        parse_program(source),
        adapter=LangChainAdapter(reader.MirageReader()),
        runtime_globals={"photo": photo},
    )
    try:
        with pytest.raises(Exception, match="no tools and no typed output"):
            runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
    assert reader.seen == []
