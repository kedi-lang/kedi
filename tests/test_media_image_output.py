"""Image output: ``[img: Image] << Draw a cat.`` returns an Image (#187)."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from kedi.agent_adapter import PydanticAdapter
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.media import Image
from tests.lsp_helpers import diagnostics_for
from tests.test_media_values import png_bytes

PNG = png_bytes()
DRAW = "> import: media\n[img: Image] << Draw a cat.\n= `img`\n"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


def _gemini(name: str, bodies: list[dict[str, Any]]) -> Any:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        bodies.append(body)
        modalities = (body.get("generationConfig") or {}).get("responseModalities") or []
        # Image output asks for images only; a text answer from the same model allows both.
        if modalities == ["IMAGE"]:
            part = {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode()}}
        else:
            part = {"text": "a drawn cat"}
        candidate = {"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}
        return httpx.Response(200, json={"candidates": [candidate]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle), timeout=60)
    return GoogleModel(name, provider=GoogleProvider(api_key="test", http_client=client))


# Pydantic AI


def test_a_gemini_image_model_returns_an_image_value(workspace: Path) -> None:
    bodies: list[dict[str, Any]] = []
    image = _run(DRAW, PydanticAdapter(model=_gemini("gemini-2.5-flash-image", bodies)))
    assert isinstance(image, Image)
    assert (image.data, image.media_type) == (PNG, "image/png")
    (body,) = bodies
    assert body["generationConfig"]["responseModalities"] == ["IMAGE"]
    # Image models take no tools, so Kedi's artifact tools stay out of the request.
    assert "tools" not in body


def test_a_generated_image_is_an_ordinary_media_value(workspace: Path) -> None:
    source = (
        "> import: media\n"
        "[img: Image] << Draw a cat.\n"
        '[path] = `save(img, "cat.png")`\n'
        "[caption] << Describe <<img>>.\n"
        "= <caption>\n"
    )
    bodies: list[dict[str, Any]] = []
    assert _run(source, PydanticAdapter(model=_gemini("gemini-2.5-flash-image", bodies))) == (
        "a drawn cat"
    )
    assert (workspace / "cat.png").read_bytes() == PNG
    parts = bodies[-1]["contents"][-1]["parts"]
    assert parts[1]["inlineData"]["data"] == base64.b64encode(PNG).decode()


def test_a_text_model_is_refused_before_the_request(workspace: Path) -> None:
    bodies: list[dict[str, Any]] = []
    with pytest.raises(KediExecutionError, match="gemini-3.5-flash cannot produce images"):
        _run(DRAW, PydanticAdapter(model=_gemini("gemini-3.5-flash", bodies)))
    assert bodies == []


def test_openai_responses_models_get_the_image_generation_tool(workspace: Path) -> None:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content or b"{}"))
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5",
                "status": "completed",
                "output": [
                    {
                        "type": "image_generation_call",
                        "id": "ig_1",
                        "status": "completed",
                        "result": base64.b64encode(PNG).decode(),
                        "output_format": "png",
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    model = OpenAIResponsesModel(
        "gpt-5", provider=OpenAIProvider(api_key="test", http_client=client)
    )
    image = _run(DRAW, PydanticAdapter(model=model))
    assert isinstance(image, Image) and image.data == PNG
    assert {"type": "image_generation"} in [
        {"type": tool["type"]} for tool in bodies[0]["tools"] if tool["type"] == "image_generation"
    ]


# LangChain


def _fake_langchain(content: Any, *, image_outputs: bool = True) -> Any:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = GenericFakeChatModel(messages=iter([AIMessage(content=content)]))
    model.profile = {"image_outputs": image_outputs, "tool_calling": False}
    return model


def test_langchain_returns_the_image_in_the_final_message(workspace: Path) -> None:
    from kedi.agent_adapter import LangChainAdapter

    data_url = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
    content = [{"type": "image_url", "image_url": {"url": data_url}}, "Here is a cat."]
    image = _run(DRAW, LangChainAdapter(model=_fake_langchain(content)))
    assert isinstance(image, Image) and (image.data, image.media_type) == (PNG, "image/png")

    # An image the response holds only by provider file id is skipped for one it carries.
    with_file = [*content, {"type": "image", "file_id": "file-1"}]
    image = _run(DRAW, LangChainAdapter(model=_fake_langchain(with_file)))
    assert image.data == PNG

    hosted = [{"type": "image", "url": "https://example.com/cat.png"}]
    image = _run(DRAW, LangChainAdapter(model=_fake_langchain(hosted)))
    assert image.url == "https://example.com/cat.png"


@pytest.mark.parametrize(
    ("content", "image_outputs", "message"),
    [
        ("no picture, sorry", True, "returned no image"),
        ("unused", False, "cannot produce images"),
    ],
)
def test_langchain_refuses_text_models_and_answers_without_an_image(
    workspace: Path, content: Any, image_outputs: bool, message: str
) -> None:
    from kedi.agent_adapter import LangChainAdapter

    with pytest.raises(KediExecutionError, match=message):
        _run(DRAW, LangChainAdapter(model=_fake_langchain(content, image_outputs=image_outputs)))


def test_langchain_models_without_tools_get_no_artifact_tools() -> None:
    from kedi.agent_adapter import LangChainAdapter
    from kedi.agent_adapter.adapters.langchain import adapter as langchain_adapter
    from kedi.agent_profile import AgentProfile

    adapter = LangChainAdapter(model=_fake_langchain("x"))
    assert not adapter.supports_artifact_contract(AgentProfile.empty())
    # A model id the adapter cannot build here is left to the run to report.
    assert adapter.supports_artifact_contract(AgentProfile(model="not-a-provider:model"))
    assert langchain_adapter._takes_tools("not-a-provider:model")


# Other adapters and the editor


def test_adapters_without_image_output_are_refused_before_the_call(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.agent_adapter.adapters.claude import ClaudeAdapter
    from tests.test_agent_adapters import _install_fake_claude_sdk

    events = _install_fake_claude_sdk(monkeypatch)
    with pytest.raises(KediExecutionError, match="Adapter 'claude' cannot produce images"):
        _run(DRAW, ClaudeAdapter(model="claude-sonnet-4"))
    assert events["prompts"] == []


def test_the_editor_reports_image_output_on_adapters_without_it() -> None:
    def codes(head: str) -> list[tuple[str, int]]:
        return [
            (str(diagnostic.code), diagnostic.range.start.line)
            for diagnostic in diagnostics_for(head + DRAW, "/tmp/draw.kedi")
            if str(diagnostic.code) == "K-ADAPTER-CAPABILITY"
        ]

    assert codes("> agent: claude\n") == [("K-ADAPTER-CAPABILITY", 2)]
    assert codes("> adapter: pydantic\n") == []
    # Without the media module, `Image` is an ordinary type name.
    assert [
        diagnostic
        for diagnostic in diagnostics_for(
            "> agent: claude\n[img: Image] << Draw a cat.\n", "/tmp/draw.kedi"
        )
        if "image output" in diagnostic.message
    ] == []
