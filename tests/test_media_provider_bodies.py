"""Offline provider request bodies for prompts with media, on both framework adapters."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.adapters.langchain.media import bedrock_document_name
from kedi.engine import SequentialEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.test_media_values import png_bytes

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 16
PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16
IMAGE_B64 = base64.b64encode(png_bytes()).decode()

MIXED = """> import: media
[cat_image: Image] = `image("cat.png")`
[contract_v2: Document] = `document("nda.pdf")`
[reply] << Compare <<cat_image>> with <<contract_v2>>. <<document(notes.md)>>
= <reply>
"""
CAT_OPEN = '<attachments>\n<image id="cat_image" type="image/png">\n'
PDF_OPEN = '\n</image>\n<document id="contract_v2" type="application/pdf">\n'
TAIL = (
    '\n</document>\n<document id="document_result_1" type="text/markdown">\n'
    "decision: ship\n\n</document>\n</attachments>\n"
    "Compare cat_image with contract_v2. document_result_1"
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "cat.png").write_bytes(png_bytes())
    (tmp_path / "nda.pdf").write_bytes(PDF)
    (tmp_path / "voice.wav").write_bytes(WAV)
    (tmp_path / "voice.ogg").write_bytes(b"OggS" + b"\x00" * 40)
    (tmp_path / "clip.mp4").write_bytes(MP4)
    (tmp_path / "notes.md").write_text("decision: ship\n", encoding="utf-8")
    return tmp_path


def _program(name: str, loader: str, prompt: str) -> str:
    """Bind ``loader`` (``audio(voice.wav)``) to ``name`` and reference it in ``prompt``."""
    function, argument = loader[:-1].split("(", 1)
    return (
        f'> import: media\n[{name}: Media] = `{function}("{argument}")`\n'
        f"[reply] << {prompt}\n= <reply>\n"
    )


def _run(source: str, model: Any, adapter_type: type[Any]) -> Any:
    engine = SequentialEngine()
    runtime = compile_program(parse_program(source), adapter=adapter_type(model), engine=engine)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())
        engine.shutdown()


def _chat_completion(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chat",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model"),
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class Bodies(list[dict[str, Any]]):
    def last_user(self) -> Any:
        messages = self[-1].get("messages") or self[-1].get("contents")
        return messages[-1]


def _bedrock_client(bodies: Bodies) -> Any:
    boto3 = pytest.importorskip("boto3")
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    def capture(params: dict[str, Any], **_: Any) -> None:
        bodies.append(params)
        raise RuntimeError("request captured")

    client.meta.events.register("provide-client-params.bedrock-runtime.Converse", capture)
    return client


def pydantic_model(provider: str, bodies: Bodies, name: str | None = None) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        bodies.append(body)
        if provider == "google":
            candidate = {
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP",
            }
            return httpx.Response(200, json={"candidates": [candidate]})
        return httpx.Response(200, json=_chat_completion(body))

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return OpenAIChatModel(
            name or "gpt-4o", provider=OpenAIProvider(api_key="test", http_client=client)
        )
    if provider == "google":
        pytest.importorskip("google.genai")
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle), timeout=60)
        return GoogleModel(
            "gemini-2.5-flash", provider=GoogleProvider(api_key="test", http_client=client)
        )
    if provider == "anthropic":
        httpx2 = pytest.importorskip("httpx2")
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        def capture(request: Any) -> Any:
            bodies.append(json.loads(request.content or b"{}"))
            return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error"}})

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(capture))
        return AnthropicModel(
            "claude-sonnet-4-5",
            provider=AnthropicProvider(api_key="test", http_client=client),
        )
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    return BedrockConverseModel(
        name or "amazon.nova-pro-v1:0",
        provider=BedrockProvider(bedrock_client=_bedrock_client(bodies)),
    )


def langchain_model(provider: str, bodies: Bodies, name: str | None = None) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        def handle(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content or b"{}")
            bodies.append(body)
            return httpx.Response(200, json=_chat_completion(body))

        transport = httpx.MockTransport(handle)
        return ChatOpenAI(
            model=name or "gpt-4o",
            api_key="test",
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
        )
    langchain_aws = pytest.importorskip("langchain_aws")
    return langchain_aws.ChatBedrockConverse(
        model=name or "amazon.nova-pro-v1:0",
        client=_bedrock_client(bodies),
        region_name="us-east-1",
    )


def _run_capture(adapter_type: type[Any], factory: Any, provider: str, source: str) -> Bodies:
    bodies = Bodies()
    try:
        _run(source, factory(provider, bodies), adapter_type)
    except KediExecutionError as exc:
        if "request captured" not in str(exc) and "api_error" not in str(exc).lower():
            if not bodies:
                raise
    return bodies


def test_pydantic_openai_body(workspace: Path) -> None:
    bodies = _run_capture(PydanticAdapter, pydantic_model, "openai", MIXED)
    content = bodies.last_user()["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text", "file", "text"]
    assert content[0]["text"] == CAT_OPEN
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{IMAGE_B64}"
    assert content[2]["text"] == PDF_OPEN
    assert content[3]["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert "nda" not in content[3]["file"]["filename"]  # the real file name stays local
    assert content[4]["text"] == TAIL


def test_pydantic_anthropic_body(workspace: Path) -> None:
    bodies = _run_capture(PydanticAdapter, pydantic_model, "anthropic", MIXED)
    content = bodies.last_user()["content"]
    assert [part["type"] for part in content] == ["text", "image", "text", "document", "text"]
    assert content[1]["source"] == {"type": "base64", "media_type": "image/png", "data": IMAGE_B64}
    assert content[3]["source"]["media_type"] == "application/pdf"


def test_pydantic_google_body(workspace: Path) -> None:
    bodies = _run_capture(PydanticAdapter, pydantic_model, "google", MIXED)
    parts = bodies.last_user()["parts"]
    assert parts[0] == {"text": CAT_OPEN}
    assert parts[1]["inlineData"] == {"data": IMAGE_B64, "mime_type": "image/png"}
    assert parts[3]["inlineData"]["mime_type"] == "application/pdf"
    assert parts[-1] == {"text": TAIL}


def test_pydantic_bedrock_body(workspace: Path) -> None:
    bodies = _run_capture(PydanticAdapter, pydantic_model, "bedrock", MIXED)
    content = bodies.last_user()["content"]
    assert content[0] == {"text": CAT_OPEN}
    assert content[1]["image"] == {"format": "png", "source": {"bytes": png_bytes()}}
    assert content[3]["document"]["format"] == "pdf"


def test_langchain_openai_body_names_files_after_the_kedi_name(workspace: Path) -> None:
    bodies = _run_capture(LangChainAdapter, langchain_model, "openai", MIXED)
    content = bodies.last_user()["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text", "file", "text"]
    assert content[3]["file"]["filename"] == "contract_v2.pdf"
    assert content[4]["text"] == TAIL


def test_langchain_bedrock_body_cleans_document_names(workspace: Path) -> None:
    bodies = _run_capture(LangChainAdapter, langchain_model, "bedrock", MIXED)
    content = bodies.last_user()["content"]
    assert content[1]["image"]["source"]["bytes"] == png_bytes()
    assert content[3]["document"]["name"] == "contract-v2"
    assert bedrock_document_name("a_b  c.d") == "a-b c-d"


@pytest.mark.parametrize(
    ("adapter_type", "factory", "provider", "name", "loader", "message"),
    [
        (
            PydanticAdapter,
            pydantic_model,
            "openai",
            "voice",
            "audio(voice.ogg)",
            "only WAV and MP3",
        ),
        (
            PydanticAdapter,
            pydantic_model,
            "openai",
            "clip",
            "video(clip.mp4)",
            "OpenAI models do not accept video",
        ),
        (
            PydanticAdapter,
            pydantic_model,
            "anthropic",
            "voice",
            "audio(voice.wav)",
            "Anthropic models do not accept audio",
        ),
        (
            PydanticAdapter,
            pydantic_model,
            "bedrock",
            "voice",
            "audio(voice.wav)",
            "Bedrock models do not accept audio",
        ),
        (
            LangChainAdapter,
            langchain_model,
            "openai",
            "voice",
            "audio(voice.wav)",
            "gpt-4o does not accept audio input",
        ),
        (
            LangChainAdapter,
            langchain_model,
            "bedrock",
            "voice",
            "audio(voice.wav)",
            "Bedrock models do not accept audio",
        ),
    ],
)
def test_unsupported_media_fails_before_the_request(
    workspace: Path,
    adapter_type: type[Any],
    factory: Any,
    provider: str,
    name: str,
    loader: str,
    message: str,
) -> None:
    bodies = Bodies()
    source = _program(name, loader, f"Describe <<{name}>>.")
    with pytest.raises(KediExecutionError, match=message):
        _run(source, factory(provider, bodies), adapter_type)
    assert bodies == []


@pytest.mark.parametrize(
    ("adapter_type", "factory"),
    [(PydanticAdapter, pydantic_model), (LangChainAdapter, langchain_model)],
)
def test_openai_audio_models_receive_wav(workspace: Path, adapter_type: Any, factory: Any) -> None:
    bodies = Bodies()
    _run(
        _program("voice", "audio(voice.wav)", "Summarize <<voice>>."),
        factory("openai", bodies, "gpt-audio"),
        adapter_type,
    )
    audio = bodies.last_user()["content"][1]
    assert audio["type"] == "input_audio" and audio["input_audio"]["format"] == "wav"


def test_video_goes_to_gemini_and_bedrock(workspace: Path) -> None:
    source = _program("clip", "video(clip.mp4)", "Describe <<clip>>.")
    google = _run_capture(PydanticAdapter, pydantic_model, "google", source)
    assert google.last_user()["parts"][1]["inlineData"]["mime_type"] == "video/mp4"
    for adapter_type, factory in (
        (PydanticAdapter, pydantic_model),
        (LangChainAdapter, langchain_model),
    ):
        bedrock = _run_capture(adapter_type, factory, "bedrock", source)
        assert bedrock.last_user()["content"][1]["video"]["format"] == "mp4"


def test_youtube_urls_reach_only_gemini_as_urls(workspace: Path) -> None:
    source = "> import: media\n[reply] << Describe <<video(https://youtu.be/abc123)>>.\n= <reply>\n"
    google = _run_capture(PydanticAdapter, pydantic_model, "google", source)
    assert google.last_user()["parts"][1]["fileData"]["file_uri"] == "https://youtu.be/abc123"
    with pytest.raises(KediExecutionError, match="YouTube URLs are read only by Gemini"):
        _run(source, pydantic_model("openai", Bodies()), PydanticAdapter)


def test_langchain_passes_image_urls_when_the_model_accepts_them(workspace: Path) -> None:
    bodies = _run_capture(
        LangChainAdapter,
        langchain_model,
        "openai",
        _program("photo", "image(https://example.com/cat.png)", "Describe <<photo>>."),
    )
    image = bodies.last_user()["content"][1]
    assert image == {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}


def test_langchain_downloads_urls_the_model_cannot_open(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.agent_adapter.adapters.langchain import media as langchain_media
    from kedi.media import Image

    requested: list[str] = []

    async def fake_download(media: Any) -> Any:
        requested.append(media.url)
        return Image(media_type="image/png", data=png_bytes(), name=media.name, filename=media.url)

    monkeypatch.setattr(langchain_media, "download_media", fake_download)
    bodies = _run_capture(
        LangChainAdapter,
        langchain_model,
        "bedrock",
        _program("photo", "image(https://example.com/cat.png)", "Describe <<photo>>."),
    )
    assert requested == ["https://example.com/cat.png"]
    assert bodies.last_user()["content"][1]["image"]["source"]["bytes"] == png_bytes()
