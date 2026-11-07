from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain_openai import ChatOpenAI

from kedi import RunBudget
from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.invocation import InvocationUsage, observe_invocations
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program

TEMPLATE = ">> The code is [code: str].\n= `code`\n"
DELEGATION = """
> profile: analyst:
  > instructions: Return the requested field.
> profile: coordinator:
  > subagent: analyst
  > max_agents: 2
> use: coordinator
> task [job]: analyst:
  >> The investigation status is [investigation: str].
> await [result]: job
= `result.output.investigation`
"""
GOOD = {"task_summary": "Read evidence", "final_result": {"investigation": "checked"}}
MISNESTED = {"final_result": {"investigation": "checked", "task_summary": "Read evidence"}}
OVER_LIMIT = {**GOOD, "task_summary": "a" * 1001}


def _respond(
    outputs: list[Any], requests: list[dict[str, Any]], *, status: str = "completed"
) -> Any:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        index = len(requests)
        output = outputs[min(index, len(outputs)) - 1]
        text = output if isinstance(output, str) else json.dumps(output)
        return httpx.Response(
            200,
            json={
                "id": f"resp_{index}",
                "object": "response",
                "created_at": 0,
                "status": status,
                "incomplete_details": (
                    {"reason": "max_output_tokens"} if status == "incomplete" else None
                ),
                "model": "gpt-5.2",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{index}",
                        "role": "assistant",
                        "status": status,
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
        )

    return respond


async def _run(
    source: str,
    outputs: list[Any],
    requests: list[dict[str, Any]],
    usage: list[InvocationUsage],
    *,
    output_retries: int = 1,
    status: str = "completed",
    run_budget: RunBudget | None = None,
) -> Any:
    transport = httpx.MockTransport(_respond(outputs, requests, status=status))
    async with httpx.AsyncClient(transport=transport) as http:
        model = ChatOpenAI(
            model="gpt-5.2",
            api_key="offline-fixture",
            use_responses_api=True,
            max_retries=0,
            http_async_client=http,
        )
        with use_async_owner_loop(), observe_invocations(usage.append):
            runtime = compile_program(
                parse_program(source),
                adapter=LangChainAdapter(model, output_retries=output_retries),
                subagent_max_depth=1,
                subagent_max_concurrency=2,
                run_budget=run_budget,
            )
            try:
                return await asyncio.to_thread(runtime.run_main)
            finally:
                await runtime.aclose()


def _correction(request: dict[str, Any], failed_output: Any) -> str:
    """Check that a corrected request carries the failed output, and return its feedback."""
    *earlier, feedback = request["input"]
    assert feedback["role"] == "user"
    if request.get("previous_response_id") is not None:
        # Responses continuation: the failed output is the stored previous response.
        assert request["previous_response_id"] == "resp_1"
    else:
        failed = earlier[-1]
        assert failed["role"] == "assistant"
        text = failed["content"][0]["text"]
        assert text == (
            failed_output if isinstance(failed_output, str) else json.dumps(failed_output)
        )
    return feedback["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial", "feedback"),
    [({"wrong": "x"}, "code: Field required"), ("The code is X7.", "expected valid JSON")],
    ids=["missing-field", "not-json"],
)
async def test_native_typed_output_gets_one_counted_correction_turn(
    initial: Any, feedback: str
) -> None:
    requests: list[dict[str, Any]] = []
    usage: list[InvocationUsage] = []

    assert await _run(TEMPLATE, [initial, {"code": "X7"}], requests, usage) == "X7"

    assert requests[0]["text"]["format"]["type"] == "json_schema"
    assert len(requests) == 2
    assert sum(item.requests for item in usage) == 2
    assert sum(item.input_tokens for item in usage) == 6
    assert feedback in _correction(requests[1], initial)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial", [GOOD, MISNESTED, OVER_LIMIT], ids=["good", "misnested", "over-limit"]
)
async def test_subagent_output_is_corrected_against_its_json_schema(
    initial: dict[str, Any],
) -> None:
    requests: list[dict[str, Any]] = []
    usage: list[InvocationUsage] = []

    assert await _run(DELEGATION, [initial, GOOD], requests, usage) == "checked"

    assert len(requests) == (1 if initial is GOOD else 2)
    assert sum(item.requests for item in usage) == len(requests)
    if initial is GOOD:
        return
    feedback = _correction(requests[1], initial)
    assert feedback.startswith("Subagent output does not match its JSON Schema:")
    if initial is MISNESTED:
        assert "'task_summary' is a required property" in feedback
        assert "'task_summary' was unexpected" in feedback
    else:
        assert "length 1001 exceeds maxLength 1000" in feedback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "outputs", "output_retries", "error"),
    [
        (TEMPLATE, [{"wrong": "x"}, {"wrong": "y"}], 1, "Failed to parse structured output"),
        (TEMPLATE, [{"wrong": "x"}, {"code": "X7"}], 0, "Failed to parse structured output"),
        (DELEGATION, [MISNESTED], 1, "'task_summary' was unexpected"),
        (DELEGATION, [MISNESTED, GOOD], 0, "'task_summary' was unexpected"),
    ],
    ids=[
        "typed-invalid-twice",
        "typed-uncorrected",
        "subagent-invalid-twice",
        "subagent-uncorrected",
    ],
)
async def test_exhausted_corrections_report_the_original_validation_error(
    source: str, outputs: list[Any], output_retries: int, error: str
) -> None:
    requests: list[dict[str, Any]] = []

    with pytest.raises(KediExecutionError, match=error):
        await _run(source, outputs, requests, [], output_retries=output_retries)

    assert len(requests) == output_retries + 1


@pytest.mark.asyncio
async def test_correction_is_a_request_the_run_budget_admits() -> None:
    requests: list[dict[str, Any]] = []

    with pytest.raises(KediExecutionError, match="Run request budget exhausted"):
        await _run(
            TEMPLATE,
            [{"wrong": "x"}, {"code": "X7"}],
            requests,
            [],
            run_budget=RunBudget(request_limit=1),
        )

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_truncated_output_is_reported_as_incomplete_not_corrected() -> None:
    requests: list[dict[str, Any]] = []

    with pytest.raises(KediExecutionError, match=r"incomplete \(reason: max_output_tokens\)"):
        await _run(TEMPLATE, ['{"code": "X'], requests, [], status="incomplete")

    assert len(requests) == 1


@pytest.mark.parametrize("output_retries", [-1, True, 1.5])
def test_output_retries_must_be_a_non_negative_integer(output_retries: Any) -> None:
    with pytest.raises(ValueError, match="output_retries must be a non-negative integer"):
        LangChainAdapter(output_retries=output_retries)
