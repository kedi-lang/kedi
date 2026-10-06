import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_profile import AgentProfile


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("layer", ["model", "agent", "run", "callback"])
@pytest.mark.parametrize(
    "effort,expected", [("low", "low"), ("medium", "medium"), ("max", "xhigh")]
)
async def test_lexical_effort_overrides_native_defaults_without_mutating_them(
    api, layer, effort, expected
):
    requests = []
    defaults = {"openai_reasoning_effort": "high"}

    async def respond(request):
        requests.append(json.loads(request.content))
        if api == "chat":
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-5.2",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_fixture",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5.2",
                "output": [
                    {
                        "id": "msg_fixture",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        async with AsyncOpenAI(api_key="fixture", http_client=http) as client:
            model_type = OpenAIResponsesModel if api == "responses" else OpenAIChatModel
            model = model_type(
                "gpt-5.2",
                provider=OpenAIProvider(openai_client=client),
                settings=defaults if layer == "model" else None,
            )
            adapter = PydanticAdapter(model, model_settings=defaults if layer == "agent" else None)
            kwargs = {}
            if layer == "run":
                kwargs["model_settings"] = defaults
            elif layer == "callback":
                kwargs["model_settings"] = lambda _: defaults
            assert await adapter.invoke(prompt="Return done.", **kwargs) == "done"
            with adapter.profile_override(AgentProfile(effort=effort)):
                assert await adapter.invoke(prompt="Return done.", **kwargs) == "done"
            assert await adapter.invoke(prompt="Return done.", **kwargs) == "done"
    assert [
        r["reasoning"]["effort"] if api == "responses" else r["reasoning_effort"] for r in requests
    ] == ["high", expected, "high"]
    assert defaults == {"openai_reasoning_effort": "high"}
