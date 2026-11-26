"""The local extraction example (#198, P0): typed extraction through Kedi with Cactus Needle.

cactus-compute is optional and not installed in CI, so Needle's extraction is faked. The
Kedi program, the typed raw invoke, Pydantic AI's output tool and validation, and the
example's deterministic routing are all real.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "local_extraction"
PROGRAM = EXAMPLE / "refunds.kedi"


@pytest.fixture
def needle(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(EXAMPLE))
    sys.modules.pop("needle_model", None)
    import needle_model

    calls: list[dict[str, Any]] = []
    answers: list[Any] = []

    def fake(prompt: str, schema: dict[str, Any], system: str, max_new_tokens: int) -> Any:
        calls.append({"prompt": prompt, "schema": schema, "system": system})
        return answers.pop(0)

    monkeypatch.setattr(needle_model, "_extract", fake)
    needle_model.calls, needle_model.answers = calls, answers
    return needle_model


def run(needle: Any, message: str) -> Any:
    runtime = compile_program(
        parse_program(PROGRAM.read_text(), source_path=str(PROGRAM)),
        adapter=PydanticAdapter(needle.NeedleModel()),
        runtime_globals={"message": message},
    )
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


SOFA = "My name is Daniel Okafor. The sofa from order D-0081 is torn. It cost 1,299.00 euros."


def test_the_output_tool_schema_reaches_needle_and_the_value_is_typed(needle: Any) -> None:
    needle.answers.append(
        {"customer": "Daniel Okafor", "order_id": "D-0081", "amount": 1299.0, "reason": "damaged"}
    )
    result = run(needle, SOFA)
    assert result["refund"].amount == 1299.0 and result["route"] == "review"
    (call,) = needle.calls
    assert call["prompt"] == f"Refund request: {SOFA}"
    properties = call["schema"]["parameters"]["properties"]
    assert properties["reason"]["enum"] == ["damaged", "late", "wrong_item", "other"]
    assert "Full name of the customer" in properties["customer"]["description"]


def test_an_amount_missing_from_the_message_goes_to_a_person(needle: Any) -> None:
    # Needle misreads "1,299.00" as 1.299: small and damaged, but not in the message.
    needle.answers.append(
        {"customer": "Daniel Okafor", "order_id": "D-0081", "amount": 1.299, "reason": "damaged"}
    )
    assert run(needle, SOFA)["route"] == "review"
    needle.answers.append(
        {"customer": "Lena Fischer", "order_id": "F-2290", "amount": 120.0, "reason": "damaged"}
    )
    assert (
        run(needle, "Order F-2290 arrived smashed. I paid 120 euros. Lena Fischer")["route"]
        == "approve"
    )


def test_a_retry_extracts_from_kedis_prompt_not_the_retry_text(needle: Any) -> None:
    needle.answers += [
        None,
        {"customer": "Tom Becker", "order_id": "B-1002", "amount": 89.0, "reason": "late"},
    ]
    assert run(needle, "Order B-1002 came late, refund 89 euros. Tom Becker")["route"] == "review"
    first, second = needle.calls
    assert first["prompt"] == second["prompt"]


def test_needle_answers_only_with_structured_output(needle: Any) -> None:
    runtime = compile_program(
        parse_program("[reply] << Say hello.\n= <reply>\n"),
        adapter=PydanticAdapter(needle.NeedleModel()),
    )
    try:
        with pytest.raises(Exception, match="answers only with structured output"):
            runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
