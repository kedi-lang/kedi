from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import httpx2
import pytest
from langchain_openai import ChatOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter import HistoryProcessorContext, LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.conversation import ConversationState


@pytest.mark.parametrize(
    ("adapter_name", "settings_on_model"),
    [("pydantic", False), ("pydantic", True), ("langchain", True)],
)
@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("server_conversation", [False, True])
def test_native_responses_wire_continuation_after_history_edit(
    adapter_name: str, settings_on_model: bool, rewrite: bool, server_conversation: bool
) -> None:
    bodies: list[dict[str, Any]] = []
    contexts: list[HistoryProcessorContext[Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        index = len(bodies)
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
                        "content": [
                            {"type": "output_text", "text": f"answer-{index}", "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 3,
                    "total_tokens": 23,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    def processor(ctx: HistoryProcessorContext[Any]):
        contexts.append(ctx)
        if rewrite and len(contexts) == 2:
            return ctx.messages[-1:]
        return ctx.messages

    async def scenario() -> None:
        def respond2(request: httpx2.Request) -> httpx2.Response:
            response = respond(httpx.Request("POST", str(request.url), content=request.content))
            return httpx2.Response(200, json=response.json())

        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client,
            httpx2.AsyncClient(transport=httpx2.MockTransport(respond2)) as client2,
        ):
            if adapter_name == "pydantic":
                settings = (
                    {"openai_conversation_id": "conv_old"}
                    if server_conversation
                    else {"openai_previous_response_id": "auto"}
                )
                model = OpenAIResponsesModel(
                    "gpt-test",
                    settings=settings if settings_on_model else None,
                    provider=OpenAIProvider(
                        api_key="test-key",
                        base_url="https://provider.example.test/v1",
                        http_client=client2,
                    ),
                )
                adapter = PydanticAdapter(
                    model,
                    history_processor=processor,
                    model_settings=None if settings_on_model else settings,
                )
            else:
                model = ChatOpenAI(
                    model="gpt-test",
                    api_key="test-key",
                    base_url="https://provider.example.test/v1",
                    http_async_client=client,
                    use_responses_api=True,
                    use_previous_response_id=not server_conversation,
                    model_kwargs={"conversation": "conv_old"} if server_conversation else {},
                )
                adapter = LangChainAdapter(model, history_processor=processor)
            conversation = ConversationState(session_id="wire-test")
            for index, prompt in enumerate(("obsolete task", "current task", "follow-up"), start=1):
                with conversation.adapter_turn(adapter_name, lane_identity=adapter_name) as context:
                    with adapter.conversation_scope(context):
                        assert await adapter.invoke(prompt=prompt) == f"answer-{index}"
            assert conversation.cache_epoch == int(rewrite)

    asyncio.run(scenario())

    assert len(bodies) == 3
    assert [ctx.request_index for ctx in contexts] == [1, 1, 1]
    if server_conversation:
        assert bodies[0]["conversation"] == "conv_old"
        if rewrite:
            assert "conversation" not in bodies[1]
            assert "conversation" not in bodies[2]
        else:
            assert bodies[1]["conversation"] == bodies[2]["conversation"] == "conv_old"
    if rewrite:
        assert "previous_response_id" not in bodies[1]
        assert "obsolete task" not in json.dumps(bodies[1]["input"])
        assert "current task" in json.dumps(bodies[1]["input"])
        assert bodies[0]["prompt_cache_key"] != bodies[1]["prompt_cache_key"]
        assert "obsolete task" not in str(contexts[2].messages)
    else:
        if not server_conversation:
            assert bodies[1]["previous_response_id"] == "resp_1"
        assert bodies[0]["prompt_cache_key"] == bodies[1]["prompt_cache_key"]
    if not server_conversation:
        assert bodies[2]["previous_response_id"] == "resp_2"
    assert bodies[2]["prompt_cache_key"] == bodies[1]["prompt_cache_key"]
