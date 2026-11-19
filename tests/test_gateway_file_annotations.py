"""Gateway file annotations require a model that understands their response dialect."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.request import (
    PromptDraft,
    RunRequestAssembler,
    TextOutput,
    agent_run_request_scope,
)
from kedi.agent_adapter.request.media import MediaAttachment
from kedi.media import Document

PDF = b"%PDF-1.4\n%%EOF\n"
FILE_ANNOTATION = {
    "type": "file",
    "url_citation": None,
    "file": {
        "hash": "test-hash",
        "name": "paper.pdf",
        "content": [{"type": "text", "text": "Invoice"}],
    },
}
URL_ANNOTATION = {
    "type": "url_citation",
    "url_citation": {
        "start_index": 0,
        "end_index": 7,
        "url": "https://example.com/paper",
        "title": "Invoice",
    },
}


@pytest.mark.parametrize("surface", ["direct", "kedi"])
@pytest.mark.parametrize("dialect", ["openai", "openrouter", "openrouter_without_provider"])
@pytest.mark.parametrize("annotation", [None, FILE_ANNOTATION, URL_ANNOTATION, {"type": "unknown"}])
def test_pdf_annotation_validation_and_preservation(surface, dialect, annotation):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": "Invoice"}
        if annotation is not None:
            message["annotations"] = [annotation]
        response = {
            "id": "completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if dialect == "openrouter":
            response["provider"] = "test-upstream"
        return httpx.Response(200, json=response)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            async with AsyncOpenAI(
                api_key="test",
                base_url="https://gateway.example/v1",
                http_client=http,
                max_retries=0,
            ) as client:
                provider = OpenAIProvider(openai_client=client)
                model = (OpenRouterModel if dialect != "openai" else OpenAIChatModel)(
                    "test-model", provider=provider
                )
                if surface == "direct":
                    result = await Agent(model).run(
                        [BinaryContent(PDF, media_type="application/pdf"), "Read paper."]
                    )
                    assert result.output == "Invoice"
                    saved = result.all_messages()
                    annotations = (
                        saved[-1].provider_details.get("annotations", [])
                        if saved[-1].provider_details
                        else []
                    )
                else:
                    adapter = PydanticAdapter(model)
                    conversation = AdapterConversationContext()
                    request = RunRequestAssembler().assemble(
                        prompt=PromptDraft(content="Read paper.", kind="raw").with_media(
                            (
                                MediaAttachment(
                                    name="paper", media=Document.from_bytes(PDF, "application/pdf")
                                ),
                            )
                        ),
                        instructions=(),
                        conversation=None,
                        tool_specs=(),
                        output=TextOutput(),
                        run_id="pdf-run",
                        turn_sequence=0,
                        call_kind="invoke",
                    )
                    with agent_run_request_scope(request), adapter.conversation_scope(conversation):
                        assert await adapter.invoke(prompt="Read paper.") == "Invoice"
                    saved = conversation.result_state["pydantic_messages"]
                    annotations = (saved[-1].get("provider_details") or {}).get("annotations", [])
                if dialect == "openrouter" and annotation is not None:
                    assert annotations[0]["type"] == annotation["type"]
                    if annotation["type"] == "file":
                        assert annotations[0]["file"] == FILE_ANNOTATION["file"]

    invalid = (
        dialect == "openrouter_without_provider"
        or annotation == {"type": "unknown"}
        or (annotation == FILE_ANNOTATION and dialect == "openai")
    )
    if invalid:
        with pytest.raises(UnexpectedModelBehavior, match="Invalid response"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert len(requests) == 1
    assert requests[0]["messages"][-1]["content"][0 if surface == "direct" else 1]["type"] == "file"
