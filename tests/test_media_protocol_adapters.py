"""Prompt media on the ACP and A2A adapters (#193)."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from a2a.types import AgentCard, AgentInterface, Message, Part, Role, StreamResponse

from kedi.agent_adapter.adapters import a2a as a2a_module
from kedi.agent_adapter.adapters.a2a import A2AAdapter, a2a_prompt_parts
from kedi.agent_adapter.adapters.acp import ACPAdapter, acp_prompt_blocks
from kedi.agent_adapter.request.media import MediaAttachment
from kedi.agent_profile import A2AConnection
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.media import Audio, Document, Image, MediaError, Video
from tests.test_media_values import png_bytes

FAKE_ACP_AGENT = Path(__file__).parent / "files" / "fake_acp_agent.py"
PNG = png_bytes()
PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 16
ALL_CAPABILITIES = {"image": True, "audio": True, "embeddedContext": True}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "cat.png").write_bytes(PNG)
    (tmp_path / "nda.pdf").write_bytes(PDF)
    (tmp_path / "notes.md").write_text("decision: ship\n", encoding="utf-8")
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


PROGRAM = (
    "> import: media\n"
    '[cat: Image] = `image("cat.png")`\n'
    "[reply] << Compare <<cat>> with <<document(nda.pdf)>>. <<document(notes.md)>>\n"
    "= <reply>\n"
)


# ACP


def _acp_adapter(tmp_path: Path, capabilities: dict[str, bool] | None) -> tuple[ACPAdapter, Path]:
    log_path = tmp_path / "acp.jsonl"
    env = {"KEDI_ACP_FAKE_LOG": str(log_path)}
    if capabilities is not None:
        env["KEDI_ACP_FAKE_PROMPT_CAPABILITIES"] = json.dumps(capabilities)
    adapter = ACPAdapter(command=[sys.executable, str(FAKE_ACP_AGENT)], model_settings={"env": env})
    return adapter, log_path


def _acp_prompts(log_path: Path) -> list[list[dict[str, Any]]]:
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return [line["params"]["prompt"] for line in lines if line.get("method") == "session/prompt"]


def test_acp_sends_images_as_image_blocks_and_pdfs_as_embedded_resources(
    workspace: Path,
) -> None:
    adapter, log_path = _acp_adapter(workspace, ALL_CAPABILITIES)
    try:
        _run(PROGRAM, adapter)
    finally:
        adapter.close()
    (prompt,) = _acp_prompts(log_path)
    assert [block["type"] for block in prompt] == ["text", "image", "text", "resource", "text"]
    assert prompt[0]["text"].endswith('<image id="cat" type="image/png">\n')
    assert prompt[1] == {
        "type": "image",
        "mimeType": "image/png",
        "data": base64.b64encode(PNG).decode(),
    }
    assert prompt[3]["resource"] == {
        "uri": "kedi:attachment/document_result_1",
        "mimeType": "application/pdf",
        "blob": base64.b64encode(PDF).decode(),
    }
    assert "decision: ship" in prompt[4]["text"]


def test_acp_keeps_plain_prompts_as_one_text_block(workspace: Path) -> None:
    adapter, log_path = _acp_adapter(workspace, None)
    try:
        _run("[reply] << Hello.\n= <reply>\n", adapter)
    finally:
        adapter.close()
    assert _acp_prompts(log_path) == [[{"type": "text", "text": "Hello."}]]


@pytest.mark.parametrize(
    ("capabilities", "message"),
    [
        (None, "does not accept images in prompts \\(promptCapabilities.image is not set\\)"),
        ({"image": True}, "embedded resources such as PDFs"),
    ],
)
def test_acp_refuses_media_the_agent_does_not_advertise(
    workspace: Path, capabilities: dict[str, bool] | None, message: str
) -> None:
    adapter, log_path = _acp_adapter(workspace, capabilities)
    try:
        with pytest.raises(KediExecutionError, match=message):
            _run(PROGRAM, adapter)
    finally:
        adapter.close()
    assert _acp_prompts(log_path) == []


def test_acp_blocks_for_audio_video_urls_and_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download(media: Any) -> Any:
        return Image.from_bytes(PNG, "image/png") if media.url else media

    monkeypatch.setattr("kedi.agent_adapter.adapters.acp.download_media", download)
    attachments = [
        MediaAttachment("voice", Audio.from_bytes(WAV, "audio/wav")),
        MediaAttachment("clip", Video.from_bytes(b"\x00\x00\x00\x18ftypisom", "video/mp4")),
        MediaAttachment("cat", Image.from_url("https://example.com/cat.png")),
    ]
    blocks = asyncio.run(
        acp_prompt_blocks("Read.", attachments, capabilities=ALL_CAPABILITIES, media_limit=None)
    )
    media_blocks = [block for block in blocks if block["type"] != "text"]
    assert [block["type"] for block in media_blocks] == ["audio", "resource", "image"]
    assert media_blocks[0]["mimeType"] == "audio/wav"
    assert media_blocks[1]["resource"]["uri"] == "kedi:attachment/clip"

    uploaded = MediaAttachment(
        "cat", Image.from_file_id("file-1", provider="openai", media_type="image/png")
    )
    with pytest.raises(MediaError, match="not an uploaded file"):
        asyncio.run(
            acp_prompt_blocks("Look.", [uploaded], capabilities=ALL_CAPABILITIES, media_limit=None)
        )
    with pytest.raises(MediaError, match="media_limit"):
        asyncio.run(
            acp_prompt_blocks(
                "Look.",
                [MediaAttachment("cat", Image.from_bytes(PNG, "image/png"))],
                capabilities=ALL_CAPABILITIES,
                media_limit=10,
            )
        )


def test_acp_embedded_resources_keep_their_source_url(monkeypatch: pytest.MonkeyPatch) -> None:
    async def download(media: Any) -> Any:
        return Document.from_bytes(PDF, "application/pdf")

    monkeypatch.setattr("kedi.agent_adapter.adapters.acp.download_media", download)
    attachment = MediaAttachment("nda", Document.from_url("https://example.com/nda.pdf"))
    blocks = asyncio.run(
        acp_prompt_blocks("Read.", [attachment], capabilities=ALL_CAPABILITIES, media_limit=None)
    )
    assert blocks[1]["resource"]["uri"] == "https://example.com/nda.pdf"


# A2A


def _a2a_adapter(
    monkeypatch: pytest.MonkeyPatch, input_modes: list[str]
) -> tuple[A2AAdapter, list[Any]]:
    sent: list[Any] = []
    adapter = A2AAdapter(connection=A2AConnection(endpoint="https://agent.example"))
    card = AgentCard(
        supported_interfaces=[
            AgentInterface(url="https://agent.example/rpc", protocol_binding="JSONRPC")
        ],
        default_input_modes=input_modes,
    )
    monkeypatch.setattr(adapter, "_agent_card", AsyncMock(return_value=card))

    class Client:
        async def send_message(self, request: Any, *, context: Any = None) -> Any:
            sent.append(request)
            yield StreamResponse(message=Message(role=Role.ROLE_AGENT, parts=[Part(text="seen")]))

    async def create_client(*args: Any, **kwargs: Any) -> Client:
        return Client()

    monkeypatch.setattr("a2a.client.create_client", create_client)
    return adapter, sent


def test_a2a_sends_media_as_raw_parts_with_type_and_name(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, sent = _a2a_adapter(monkeypatch, ["text/plain", "image/*", "application/pdf"])
    assert _run(PROGRAM, adapter) == "seen"
    (request,) = sent
    parts = list(request.message.parts)
    assert [part.WhichOneof("content") for part in parts] == ["text", "raw", "text", "raw", "text"]
    assert (parts[1].raw, parts[1].media_type, parts[1].filename) == (PNG, "image/png", "cat.png")
    assert (parts[3].media_type, parts[3].filename) == ("application/pdf", "nda.pdf")
    assert "decision: ship" in parts[4].text


def test_a2a_refuses_media_types_the_card_does_not_list(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, sent = _a2a_adapter(monkeypatch, ["text/plain", "image/png"])
    with pytest.raises(
        KediExecutionError,
        match="does not accept application/pdf input \\(its card lists image/png, text/plain\\)",
    ):
        _run(PROGRAM, adapter)
    assert sent == []


def test_a2a_parts_for_urls_and_unsupported_sources() -> None:
    url = MediaAttachment("cat", Image.from_url("https://example.com/cat.png"))
    (_, part, _) = a2a_prompt_parts("Look.", [url], media_limit=None)
    assert part == {
        "url": "https://example.com/cat.png",
        "media_type": "image/png",
        "filename": "cat",
    }
    for media, held in (
        (Image.from_url("gs://bucket/cat.png"), "gs://bucket/cat.png"),
        (Image.from_file_id("file-1", provider="openai", media_type="image/png"), "an uploaded"),
    ):
        with pytest.raises(MediaError, match=f"not {held}"):
            a2a_prompt_parts("Look.", [MediaAttachment("cat", media)], media_limit=None)
    with pytest.raises(MediaError, match="media_limit"):
        a2a_prompt_parts(
            "Look.", [MediaAttachment("cat", Image.from_bytes(PNG, "image/png"))], media_limit=10
        )
    assert a2a_prompt_parts("Plain.", [], media_limit=None) == [{"text": "Plain."}]
    # A card that lists no input modes accepts no media.
    with pytest.raises(MediaError, match="its card lists none"):
        a2a_module._validate_card_input_modes(AgentCard(), [url])
