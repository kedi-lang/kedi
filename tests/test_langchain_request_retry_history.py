from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic_ai import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from kedi.agent_adapter import HistoryProcessorContext, LangChainAdapter
from kedi.agent_adapter.adapters.langchain.request_retry import (
    RequestRetryMiddleware,
    retain_retry_continuation_reset,
)
from kedi.agent_adapter.conversation import ConversationState
from kedi.agent_adapter.invocation import observe_invocations
from kedi.run_budget import RunBudget, RunBudgetExceeded, current_run_budget, run_budget_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("server_conversation", [False, True])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("failures", [1, 2])
@pytest.mark.parametrize("ceiling", ["enough", "strict_one", "native_one"])
async def test_rewritten_history_is_retained_across_http_retries_and_admitted_each_time(
    server_conversation: bool,
    status: int,
    failures: int,
    ceiling: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies: list[dict[str, Any]] = []
    admissions: list[int] = []
    request_indices: list[int] = []

    async def sleep(delay: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr("kedi.agent_adapter.request_retry.sleep", sleep)

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        index = len(bodies)
        ledger = current_run_budget()
        if ledger is not None:
            admissions.append(ledger.spent("request"))
        if 3 <= index < 3 + failures:
            return httpx.Response(
                status,
                headers={"Retry-After": "0"},
                json={"error": {"message": "Transient request failure", "type": "server_error"}},
            )
        return httpx.Response(
            200,
            json={
                "id": f"resp_{index}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-test",
                "output": [
                    {
                        "id": f"msg_{index}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
            },
        )

    def processor(ctx: HistoryProcessorContext[BaseMessage]) -> Sequence[BaseMessage]:
        request_indices.append(ctx.request_index)
        return ctx.messages[2:] if len(bodies) == 2 else ctx.messages

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = ChatOpenAI(
            model="gpt-test",
            api_key="test-key",
            base_url="https://provider.example.test/v1",
            http_async_client=client,
            use_responses_api=True,
            use_previous_response_id=not server_conversation,
            model_kwargs={"conversation": "conv_old"} if server_conversation else {},
            max_retries=0,
        )
        adapter = LangChainAdapter(model, history_processor=processor)
        conversation = ConversationState(session_id="retry-history")

        async def invoke(prompt: str, **kwargs: Any) -> str:
            with conversation.adapter_turn("langchain", lane_identity="langchain") as context:
                with adapter.conversation_scope(context):
                    return await adapter.invoke(prompt=prompt, **kwargs)

        assert await invoke("obsolete task") == "done"
        assert await invoke("retained task") == "done"
        usages = []
        limit = 1 if ceiling == "strict_one" else failures + 1
        with run_budget_scope(RunBudget(request_limit=limit)) as ledger:
            with observe_invocations(usages.append):
                if ceiling == "enough":
                    assert await invoke("current task") == "done"
                else:
                    error = RunBudgetExceeded if ceiling == "strict_one" else UsageLimitExceeded
                    with pytest.raises(error):
                        await invoke(
                            "current task",
                            usage_limits=UsageLimits(
                                request_limit=1 if ceiling == "native_one" else 50
                            ),
                        )
            expected_requests = failures + 1 if ceiling == "enough" else 1
            assert ledger is not None and ledger.spent("request") == expected_requests

        attempts = bodies[2:]
        assert len(attempts) == expected_requests
        assert admissions == list(range(1, expected_requests + 1))
        assert usages[-1].requests == expected_requests
        assert request_indices == [1, 1, 1]
        for body in attempts:
            assert body == attempts[0]
            assert "previous_response_id" not in body
            assert "conversation" not in body
            assert "obsolete task" not in json.dumps(body["input"])
            assert "retained task" in json.dumps(body["input"])
            assert "current task" in json.dumps(body["input"])

        if ceiling == "enough":
            assert usages[-1].input_tokens == 20
            assert usages[-1].output_tokens == 3
            assert await invoke("follow-up") == "done"
            if not server_conversation:
                assert bodies[-1]["previous_response_id"] == f"resp_{2 + expected_requests}"
            assert "conversation" not in bodies[-1]
            assert conversation.cache_epoch == 1
        assert model.use_previous_response_id is not server_conversation
        assert model.model_kwargs == ({"conversation": "conv_old"} if server_conversation else {})
        assert model.max_retries == 0
        assert model.root_async_client.max_retries == 0


@pytest.mark.asyncio
async def test_continuation_reset_is_isolated_and_cleared_on_cancellation() -> None:
    middleware = RequestRetryMiddleware(0)

    async def nested(request: Any) -> None:
        assert retain_retry_continuation_reset(False) is False
        await asyncio.sleep(0)
        assert retain_retry_continuation_reset(False) is False

    async def cancelled(request: Any) -> None:
        assert retain_retry_continuation_reset(True) is True
        await asyncio.gather(
            middleware.awrap_model_call(None, nested),
            middleware.awrap_model_call(None, nested),
        )
        assert retain_retry_continuation_reset(False) is True
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await middleware.awrap_model_call(None, cancelled)
    assert retain_retry_continuation_reset(False) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outer_reset", [False, True])
async def test_same_task_nested_continuation_reset_restores_parent(outer_reset: bool) -> None:
    middleware = RequestRetryMiddleware(0)
    owner = asyncio.current_task()

    async def nested(request: Any) -> str:
        assert asyncio.current_task() is owner
        assert retain_retry_continuation_reset(False) is False
        assert retain_retry_continuation_reset(not outer_reset) is (not outer_reset)
        await asyncio.sleep(0)
        assert retain_retry_continuation_reset(False) is (not outer_reset)
        return "nested"

    async def outer(request: Any) -> str:
        assert retain_retry_continuation_reset(outer_reset) is outer_reset
        assert await middleware.awrap_model_call(None, nested) == "nested"
        assert retain_retry_continuation_reset(False) is outer_reset
        return "outer"

    assert await middleware.awrap_model_call(None, outer) == "outer"
    assert retain_retry_continuation_reset(False) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outer_reset", [False, True])
async def test_nested_terminal_error_restores_parent_continuation_reset(outer_reset: bool) -> None:
    middleware = RequestRetryMiddleware(2)
    owner = asyncio.current_task()

    async def nested(request: Any) -> None:
        assert asyncio.current_task() is owner
        assert retain_retry_continuation_reset(False) is False
        assert retain_retry_continuation_reset(True) is True
        await asyncio.sleep(0)
        raise ValueError("Terminal nested failure")

    async def outer(request: Any) -> None:
        assert retain_retry_continuation_reset(outer_reset) is outer_reset
        with pytest.raises(ValueError, match="Terminal nested failure"):
            await middleware.awrap_model_call(None, nested)
        assert retain_retry_continuation_reset(False) is outer_reset

    await middleware.awrap_model_call(None, outer)
    assert retain_retry_continuation_reset(False) is False


@pytest.mark.asyncio
async def test_concurrent_opposing_continuation_resets_remain_isolated() -> None:
    middleware = RequestRetryMiddleware(0)
    barrier = asyncio.Barrier(2)

    async def sibling(reset: bool) -> None:
        async def handler(request: Any) -> None:
            assert retain_retry_continuation_reset(False) is False
            assert retain_retry_continuation_reset(reset) is reset
            await barrier.wait()
            assert retain_retry_continuation_reset(False) is reset

        await middleware.awrap_model_call(None, handler)
        assert retain_retry_continuation_reset(False) is False

    await asyncio.wait_for(asyncio.gather(sibling(False), sibling(True)), timeout=5)
    assert retain_retry_continuation_reset(False) is False


@pytest.mark.asyncio
async def test_task_cancellation_resets_continuation_before_propagating() -> None:
    middleware = RequestRetryMiddleware(2)
    ready = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: Any) -> None:
        assert retain_retry_continuation_reset(True) is True
        ready.set()
        await release.wait()

    async def target() -> None:
        try:
            await middleware.awrap_model_call(None, handler)
        except asyncio.CancelledError:
            assert retain_retry_continuation_reset(False) is False
            raise

    task = asyncio.create_task(target())
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert retain_retry_continuation_reset(False) is False
