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


RECORD = EXAMPLE / "receipt_record.kedi"


def record(key: str, answer: dict[str, Any]) -> tuple[Any, list[Any]]:
    """Run receipt_record.kedi on a scripted vision model; return the result and its requests."""

    from pydantic_ai.messages import BinaryContent, ModelResponse, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter import PydanticAdapter

    seen: list[Any] = []

    def answer_with_the_receipt(messages: list[Any], info: AgentInfo) -> ModelResponse:
        images = [
            item
            for message in messages
            for part in getattr(message, "parts", ())
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content
            if isinstance(item, BinaryContent) and item.is_image
        ]
        seen.append({"images": images, "tools": [tool.name for tool in info.function_tools]})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    photo = Image.from_bytes((EXAMPLE / "receipts" / f"{key}.png").read_bytes(), "image/png")
    runtime = compile_program(
        parse_program(RECORD.read_text(), source_path=str(RECORD)),
        adapter=PydanticAdapter(FunctionModel(answer_with_the_receipt)),
        runtime_globals={"photo": photo},
    )
    try:
        return runtime.run_main(), seen
    finally:
        asyncio.run(runtime.aclose())


def test_the_shared_program_sends_the_image_and_types_the_receipt() -> None:
    truth = json.loads((EXAMPLE / "receipts" / "truth.json").read_text())["cafe"]
    answer = {key: truth[key] for key in ("merchant", "items", "total")}
    result, (request,) = record("cafe", answer)
    assert result["receipt"].model_dump() == answer and result["adds_up"] is True
    (image,) = request["images"]
    assert image.data == (EXAMPLE / "receipts" / "cafe.png").read_bytes()
    # The program turns the artifact contract off: no tools for a small model to misuse.
    assert request["tools"] == []


def test_the_shared_program_flags_a_receipt_that_does_not_add_up() -> None:
    truth = json.loads((EXAMPLE / "receipts" / "truth.json").read_text())["bakery"]
    answer = {key: truth[key] for key in ("merchant", "items", "total")}
    result, _ = record("bakery", answer)
    assert result["adds_up"] is False


PAIRS = EXAMPLE / "same_purchase.kedi"


def compare_pair(first: str, second: str, answer: dict[str, bool]) -> tuple[Any, dict[str, Any]]:
    """Run same_purchase.kedi on a scripted model; return the result and what it received."""

    from pydantic_ai.messages import BinaryContent, ModelResponse, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter import PydanticAdapter

    seen: dict[str, Any] = {}

    def answer_the_comparison(messages: list[Any], info: AgentInfo) -> ModelResponse:
        (part,) = [
            part
            for message in messages
            for part in getattr(message, "parts", ())
            if isinstance(part, UserPromptPart)
        ]
        seen["images"] = [item.data for item in part.content if isinstance(item, BinaryContent)]
        seen["text"] = "".join(item for item in part.content if isinstance(item, str))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    def photo(key: str) -> Image:
        return Image.from_bytes((EXAMPLE / "receipts" / f"{key}.png").read_bytes(), "image/png")

    runtime = compile_program(
        parse_program(PAIRS.read_text(), source_path=str(PAIRS)),
        adapter=PydanticAdapter(FunctionModel(answer_the_comparison)),
        runtime_globals={"first": photo(first), "second": photo(second)},
    )
    try:
        return runtime.run_main(), seen
    finally:
        asyncio.run(runtime.aclose())


def test_two_receipts_reach_the_model_under_their_own_names() -> None:
    result, seen = compare_pair("cafe", "books", {"same_shop": False, "same_total": False})
    assert result["duplicate"] is False
    assert seen["images"] == [
        (EXAMPLE / "receipts" / name).read_bytes() for name in ("cafe.png", "books.png")
    ]
    assert "Do the receipts first and second show" in seen["text"]


def test_the_same_receipt_twice_is_attached_once_under_its_first_name() -> None:
    result, seen = compare_pair("cafe", "cafe", {"same_shop": True, "same_total": True})
    assert result["duplicate"] is True
    assert seen["images"] == [(EXAMPLE / "receipts" / "cafe.png").read_bytes()]
    assert "Do the receipts first and first show" in seen["text"]
