from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
from kedi_typesafe import TypeSafeEvaluator, TypeSafeModel
from kedi_typesafe.integrations.langchain import TypeSafeChatModel
from pydantic import BaseModel
from typesafe_sdk import Choice, ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage

from kedi import PreToolUseDecision, PreToolUseRequest
from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.tools.specs import ToolSpec


class Supported(BaseModel):
    """Assess whether the supplied evidence establishes the claim."""

    supported: bool


class Client:
    def __init__(self) -> None:
        self.calls = 0

    async def system_one(self, state: Any, questions: Any, *, model: Any = None) -> Any:
        self.calls += 1
        assert self.calls <= 2, "Tool routing must terminate"
        answers = {}
        for key, question in questions.items():
            if isinstance(question, Choice):
                selected = (
                    "lookup"
                    if self.calls == 1
                    else next(option for option in question.criteria if option != "lookup")
                )
                answers[key] = ChoiceAnswer(
                    choice=selected,
                    confidence=1,
                    probabilities={
                        option: float(option == selected) for option in question.criteria
                    },
                )
            else:
                answers[key] = NoulAnswer(noul=0.95)
        return SystemOneResponse(
            model="test", answers=answers, usage=Usage(input_tokens=10, output_tokens=2)
        )

    async def aclose(self) -> None:
        raise AssertionError("Borrowed client")


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
@pytest.mark.parametrize("deny", [False, True])
def test_jev_tool_routing_obeys_kedi_hooks(adapter_name: str, deny: bool) -> None:
    client = Client()
    adapter = (
        PydanticAdapter(TypeSafeModel(client=client))
        if adapter_name == "pydantic"
        else LangChainAdapter(TypeSafeChatModel(client=client))
    )
    executed: list[bool] = []
    events: list[str] = []

    def lookup() -> str:
        """Read the supporting evidence."""
        executed.append(True)
        return "Evidence establishes the claim"

    def hook(event: Any) -> Any:
        events.append(event.event)
        if event.event == "pre_tool_use" and deny:
            return PreToolUseDecision.deny("Blocked by review")

    adapter.on(["pre_tool_use", "post_tool_use"])(hook)
    with adapter.tool_scope(
        [
            ToolSpec(
                fn=lookup, name="lookup", description="Read supporting evidence", risk="read_only"
            )
        ]
    ):
        result = adapter.produce_sync("Check the claim", output_type=Supported)
        assert result.supported is True
    assert executed == ([] if deny else [True])
    assert events == (["pre_tool_use"] if deny else ["pre_tool_use", "post_tool_use"])
    assert client.calls == 2


def test_risk_review_example_is_explicit_and_records_its_own_usage() -> None:
    import asyncio

    asyncio.run(_check_risk_review())


async def _check_risk_review() -> None:
    example = Path(__file__).parent / "fixtures" / "typesafe" / "risk_review.py"
    review_type = runpy.run_path(str(example))["RiskReview"]
    client = Client()
    evaluator = TypeSafeEvaluator(client=client)
    review = review_type(evaluator, frozenset({"lookup"}))
    event = PreToolUseRequest(
        run_id="test",
        sequence=1,
        adapter_shortname="pydantic",
        tool_call_id="1",
        tool_name="delete",
        origin="kedi",
        arguments={},
    )
    assert (await review(event)).decision == "deny"
    assert client.calls == 0
    allowed = PreToolUseRequest(
        run_id="test",
        sequence=2,
        adapter_shortname="pydantic",
        tool_call_id="2",
        tool_name="lookup",
        origin="kedi",
        arguments={},
    )
    assert (await review(allowed)).decision == "continue"
    assert client.calls == 1
    assert review.records[0].input_tokens == 10
    await evaluator.aclose()
