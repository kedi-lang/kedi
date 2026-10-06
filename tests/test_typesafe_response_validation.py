from typing import Annotated

import pytest
from kedi_typesafe import TypeSafeModel
from kedi_typesafe.core import TypeSafeEvaluator
from kedi_typesafe.integrations.langchain import TypeSafeChatModel
from pydantic import BaseModel
from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.decisions import collect_decision_evidence


class Ready(BaseModel):
    answer: Annotated[bool, "Does the supplied record explicitly say its status is ready?"]


class FreeText(BaseModel):
    answer: str


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("mode", ["missing-answer", "malformed-answer", "unsupported"])
async def test_invalid_decision_response_never_publishes_success_evidence(
    monkeypatch, backend, mode
):
    calls = []

    class Client:
        async def system_one(self, state, questions, **kwargs):
            calls.append(questions)
            response = SystemOneResponse(
                model="test-model",
                answers={name: NoulAnswer(noul=0.96) for name in questions},
                usage=Usage(input_tokens=10, output_tokens=1),
            ).model_dump(mode="json")
            if mode == "missing-answer":
                response["answers"] = {}
            else:
                next(iter(response["answers"].values()))["noul"] = "not-a-probability"
            return SystemOneResponse.model_validate(response)

    client = Client()
    monkeypatch.setattr(TypeSafeEvaluator, "async_client", lambda self: client)
    native = (TypeSafeModel if backend == "pydantic" else TypeSafeChatModel)(
        model_name="test-model", api_key="test-only", base_url="https://example.invalid"
    )
    adapter = (PydanticAdapter if backend == "pydantic" else LangChainAdapter)(native)
    marker = {
        "missing-answer": "answer keys",
        "malformed-answer": "validation",
        "unsupported": "string",
    }[mode]
    try:
        with collect_decision_evidence() as evidence:
            with pytest.raises(Exception, match=marker):
                await adapter.produce(
                    "Record Q-17 has status ready. Report whether it is ready.",
                    output_type=FreeText if mode == "unsupported" else Ready,
                )
        assert evidence == []
        assert len(calls) == (0 if mode == "unsupported" else 1)
    finally:
        await native.aclose_current()
