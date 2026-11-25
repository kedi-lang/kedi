"""Prompt media on the Claude Agent SDK and Codex adapters (#188)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_adapter.adapters.claude import ClaudeAdapter
from kedi.agent_adapter.adapters.claude.live import run_query
from kedi.agent_adapter.adapters.claude.media import claude_prompt
from kedi.agent_adapter.adapters.codex.media import codex_turn_input
from kedi.agent_adapter.request.media import MediaAttachment
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.media import Document, Image, MediaError
from tests.test_agent_adapters import _install_fake_claude_sdk
from tests.test_codex_adapter import _adapter as codex_adapter
from tests.test_codex_adapter import _messages, _method
from tests.test_media_values import png_bytes

PNG = png_bytes()
PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x10\x00\x00\x00\x01\x00\x01\x00" + b"\x00" * 16


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "cat.png").write_bytes(PNG)
    (tmp_path / "nda.pdf").write_bytes(PDF)
    (tmp_path / "voice.wav").write_bytes(WAV)
    (tmp_path / "notes.md").write_text("decision: ship\n", encoding="utf-8")
    return tmp_path


def _run(source: str, adapter: Any) -> Any:
    runtime = compile_program(parse_program(source), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


async def _collect(prompt: Any) -> list[dict[str, Any]]:
    return [message async for message in prompt]


# Claude Agent SDK


def test_claude_sends_images_and_pdfs_as_content_blocks(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    source = (
        "> import: media\n"
        '[cat: Image] = `image("cat.png")`\n'
        "[reply] << Compare <<cat>> with <<document(nda.pdf)>>. <<document(notes.md)>>\n"
        "= <reply>\n"
    )
    assert _run(source, ClaudeAdapter(model="claude-sonnet-4")) == "plain claude text"

    (message,) = asyncio.run(_collect(events["prompts"][-1]["prompt"]))
    assert message["type"] == "user" and message["message"]["role"] == "user"
    content = message["message"]["content"]
    assert [block["type"] for block in content] == ["text", "image", "text", "document", "text"]
    assert content[0]["text"].endswith('<image id="cat" type="image/png">\n')
    assert content[1]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(PNG).decode(),
    }
    assert content[3]["source"]["media_type"] == "application/pdf"
    # Text documents stay inline in the text, as on every adapter.
    assert "decision: ship" in content[4]["text"]
    assert content[4]["text"].endswith("Compare cat with document_result_1. document_result_2")


def test_claude_keeps_a_plain_prompt_without_media(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    _run("[reply] << Hello.\n= <reply>\n", ClaudeAdapter(model="claude-sonnet-4"))
    assert events["prompts"][-1]["prompt"] == "Hello."


@pytest.mark.parametrize(
    ("loader", "message"),
    [
        ("audio(voice.wav)", "Anthropic models do not accept audio"),
        ("image(gs://bucket/cat.png)", "data or an http\\(s\\) URL"),
    ],
)
def test_claude_refuses_media_it_cannot_send_before_the_request(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, loader: str, message: str
) -> None:
    events = _install_fake_claude_sdk(monkeypatch)
    function, argument = loader[:-1].split("(", 1)
    source = (
        f'> import: media\n[x: Media] = `{function}("{argument}")`\n[r] << Read <<x>>.\n= <r>\n'
    )
    with pytest.raises(KediExecutionError, match=message):
        _run(source, ClaudeAdapter(model="claude-sonnet-4"))
    assert events["prompts"] == []


def test_claude_prompt_blocks_for_urls_uploaded_files_and_limits() -> None:
    url = MediaAttachment("cat", Image.from_url("https://example.com/cat.png"))
    (message,) = asyncio.run(_collect(claude_prompt("Look.", [url], model=None, media_limit=None)))
    assert message["message"]["content"][1] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.png"},
    }
    uploaded = MediaAttachment(
        "cat", Image.from_file_id("file-1", provider="anthropic", media_type="image/png")
    )
    with pytest.raises(MediaError, match="not an uploaded file"):
        claude_prompt("Look.", [uploaded], model=None, media_limit=None)
    big = MediaAttachment("cat", Image.from_bytes(PNG, "image/png"))
    with pytest.raises(MediaError, match="media_limit"):
        claude_prompt("Look.", [big], model=None, media_limit=10)


def test_claude_live_query_needs_the_text_of_a_streamed_prompt() -> None:
    from claude_agent_sdk import ClaudeAgentOptions

    async def streamed() -> Any:
        yield {}

    with pytest.raises(TypeError, match="needs the prompt text"):
        asyncio.run(run_query(ClaudeAgentOptions(), streamed(), object(), lambda _: None))  # type: ignore[arg-type]


# Codex


def test_codex_sends_image_data_as_local_files_for_the_turn(workspace: Path) -> None:
    adapter, log_path = codex_adapter(workspace)
    source = '> import: media\n[cat: Image] = `image("cat.png")`\n[r] << Describe <<cat>>.\n= <r>\n'
    try:
        _run(source, adapter)
    finally:
        adapter.close()

    messages = _messages(log_path)
    (turn,) = _method(messages, "turn/start")
    items = turn["params"]["input"]
    assert [item["type"] for item in items] == ["text", "localImage", "text"]
    assert items[0]["text"].endswith('<image id="cat" type="image/png">\n')
    assert items[2]["text"].endswith("Describe cat.")
    (seen,) = [message for message in messages if "localImage" in message]
    assert seen["sha256"] == hashlib.sha256(PNG).hexdigest()
    # The file lived for the turn only.
    assert not Path(items[1]["path"]).exists()
    assert Path(items[1]["path"]).name == "cat.png"


def test_codex_sends_image_urls_and_openai_files_by_reference() -> None:
    attachments = [
        MediaAttachment("cat", Image.from_url("https://example.com/cat.png")),
        MediaAttachment(
            "dog", Image.from_file_id("file-9", provider="openai", media_type="image/png")
        ),
    ]
    with codex_turn_input("Compare.", attachments, model=None, media_limit=None) as items:
        assert [item for item in items if item["type"] == "image"] == [
            {"type": "image", "url": "https://example.com/cat.png"},
            {"type": "image", "fileId": "file-9"},
        ]
    with pytest.raises(MediaError, match="not gs://bucket/cat.png"):
        with codex_turn_input(
            "Look.",
            [MediaAttachment("cat", Image.from_url("gs://bucket/cat.png"))],
            model=None,
            media_limit=None,
        ):
            pass
    with codex_turn_input("Plain.", [], model=None, media_limit=None) as items:
        assert items == [{"type": "text", "text": "Plain."}]


@pytest.mark.parametrize("loader", ["document(nda.pdf)", "audio(voice.wav)"])
def test_codex_refuses_other_media_before_the_turn(workspace: Path, loader: str) -> None:
    adapter, log_path = codex_adapter(workspace)
    function, argument = loader[:-1].split("(", 1)
    source = (
        f'> import: media\n[x: Media] = `{function}("{argument}")`\n[r] << Read <<x>>.\n= <r>\n'
    )
    try:
        with pytest.raises(KediExecutionError, match="Codex reads only images"):
            _run(source, adapter)
    finally:
        adapter.close()
    assert not log_path.exists() or _method(_messages(log_path), "turn/start") == []


def test_codex_inlines_text_documents(workspace: Path) -> None:
    attachment = MediaAttachment("notes", Document.from_path(workspace / "notes.md"))
    with codex_turn_input("Read <<notes>>.", [attachment], model=None, media_limit=None) as items:
        assert [item["type"] for item in items] == ["text"]
        assert "decision: ship" in items[0]["text"]
