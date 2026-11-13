"""Adapter-side attachment lowering: native parts, history scans, downloads and provider checks."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kedi.agent_adapter.adapters.langchain import media as langchain_media
from kedi.agent_adapter.adapters.pydantic import media as pydantic_media
from kedi.agent_adapter.request import PromptDraft
from kedi.agent_adapter.request.media import (
    MediaAttachment,
    check_media_support,
    compose_prompt,
    download_media,
    history_keeps_media,
    media_provider,
)
from kedi.media import Audio, Document, Image, Media, MediaError, Reference, Video
from tests.test_media_values import png_bytes

CAT = Image.from_bytes(png_bytes(1), "image/png")
DOG = Image.from_bytes(png_bytes(2), "image/png")
PDF = Document.from_bytes(b"%PDF-1.4\n%%EOF\n", "application/pdf")


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _attach(name: str, media: Media) -> MediaAttachment:
    return MediaAttachment(name=name, media=media)


def test_generated_and_repeated_names_skip_taken_ones() -> None:
    bird = Image.from_bytes(png_bytes(3), "image/png")
    text, attachments = compose_prompt(
        [
            Reference((CAT,), "image_1"),
            " ",
            Reference((DOG,)),
            " ",
            Reference((CAT.with_name("photo"),)),
            Reference((DOG.with_name("photo"),)),
            Reference((bird.with_name("photo"),)),
            " ",
            Reference((PDF,), call="pick"),
            Reference((PDF, bird), call="pick", is_list=True),
        ]
    )
    # ``image_1`` is a binding here, so the unnamed image becomes ``image_2``; content already
    # attached keeps its first name, and a repeated explicit name is numbered. Calls are named
    # ``<proc>_result_<n>``.
    assert text == "image_1 image_2 image_1image_2photo pick_result_1pick_result_1, photo"
    assert [attachment.name for attachment in attachments] == [
        "image_1",
        "image_2",
        "photo",
        "pick_result_1",
    ]

    names = [
        attachment.name
        for attachment in compose_prompt(
            [Reference((media.with_name("photo"),)) for media in (CAT, DOG, bird)]
        )[1]
    ]
    assert names == ["photo", "photo_2", "photo_3"]


@pytest.mark.parametrize(
    ("system", "provider"),
    [
        (None, None),
        ("openai", "openai"),
        ("azure", "openai"),
        ("anthropic", "anthropic"),
        ("google-gla", "google"),
        ("google-vertex", "google"),
        ("bedrock", "bedrock"),
        ("groq", None),
    ],
)
def test_media_provider_families(system: str | None, provider: str | None) -> None:
    assert media_provider(system) == provider


def test_provider_checks_name_the_attachment_and_model() -> None:
    video = _attach("clip", Video.from_bytes(b"\x00\x00\x00\x18ftypisom", "video/mp4"))
    audio = _attach("voice", Audio.from_url("https://example.com/voice"))
    check_media_support([video, audio], provider=None, model=None)
    check_media_support([audio], provider="openai", model="gpt-audio")  # unknown URL type passes
    with pytest.raises(MediaError, match="clip: Anthropic models do not accept video; the model"):
        check_media_support([video], provider="anthropic", model=None)
    with pytest.raises(MediaError, match="model gpt-4o cannot open https://youtu.be/x"):
        check_media_support(
            [_attach("video_1", Video.from_url("https://youtu.be/x"))],
            provider="openai",
            model="gpt-4o",
        )


def test_history_reuse_needs_an_untouched_stateful_history() -> None:
    from kedi.agent_adapter.conversation import AdapterConversationContext

    conversation = AdapterConversationContext()
    native = SimpleNamespace(mode="native")
    assert history_keeps_media(conversation, processor=None, archive=None, compaction=None)
    assert history_keeps_media(
        conversation, processor=None, archive=None, compaction=SimpleNamespace(mode="disabled")
    )
    assert not history_keeps_media(None, processor=None, archive=None, compaction=None)
    assert not history_keeps_media(conversation, processor=print, archive=None, compaction=None)
    assert not history_keeps_media(conversation, processor=None, archive=object(), compaction=None)
    assert not history_keeps_media(conversation, processor=None, archive=None, compaction=native)


def test_download_media_uses_the_protected_downloader(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic_ai.models

    requested: list[Any] = []

    async def fake_download(item: Any, data_format: str) -> dict[str, Any]:
        requested.append((type(item).__name__, item.url, data_format))
        return {"data": png_bytes(), "data_type": "image/png"}

    monkeypatch.setattr(pydantic_ai.models, "download_item", fake_download)
    typed = asyncio.run(download_media(Image.from_url("https://example.com/cat.png", name="cat")))
    untyped = asyncio.run(download_media(Image.from_url("https://example.com/render?id=3")))
    assert (typed.data, typed.name, typed.filename) == (
        png_bytes(),
        "cat",
        "https://example.com/cat.png",
    )
    assert untyped.media_type == "image/png"
    assert requested == [
        ("ImageUrl", "https://example.com/cat.png", "bytes"),
        ("ImageUrl", "https://example.com/render?id=3", "bytes"),
    ]
    assert asyncio.run(download_media(CAT)) is CAT
    with pytest.raises(MediaError, match="YouTube URLs are never downloaded"):
        asyncio.run(download_media(Video.from_url("https://youtu.be/abc")))
    with pytest.raises(MediaError, match="downloads only http\\(s\\) media URLs"):
        asyncio.run(download_media(Image.from_url("s3://bucket/cat.png")))


def test_url_text_documents_are_downloaded_and_inlined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pydantic_ai.models

    from tests.test_media_prompts import run_pydantic

    async def fake_download(item: Any, data_format: str) -> dict[str, Any]:
        return {"data": b"decision: ship\n", "data_type": "text/markdown"}

    monkeypatch.chdir(tmp_path)
    (tmp_path / "cat.png").write_bytes(png_bytes())
    monkeypatch.setattr(pydantic_ai.models, "download_item", fake_download)
    source = """> import: media
[summary] << Summarize <<document(https://example.com/notes.md)>> for <<image(cat.png)>>.
= <summary>
"""
    _, seen = run_pydantic(source)
    assert seen == [
        [
            '<attachments>\n<document id="document_result_1" type="text/markdown">\n'
            "decision: ship\n\n</document>\n"
            '<image id="image_result_1" type="image/png">\n',
            "<binary image_result_1>",
            "\n</image>\n</attachments>\nSummarize document_result_1 for image_result_1.",
        ]
    ]


def test_pydantic_parts_for_urls_and_uploaded_files() -> None:
    from pydantic_ai.messages import DocumentUrl, ImageUrl, UploadedFile

    attachments = [
        _attach("typed", Image.from_url("https://example.com/cat.png")),
        _attach("untyped", Image.from_url("https://example.com/render?id=3")),
        _attach("paper", Document.from_url("https://example.com/paper.pdf")),
        _attach(
            "stored", Media.from_file_id("file-1", provider="openai", media_type="application/pdf")
        ),
    ]
    content = asyncio.run(
        pydantic_media.user_prompt(
            "Compare them.", attachments, history=None, model=SimpleNamespace()
        )
    )
    assert content[0::2] == [
        '<attachments>\n<image id="typed" type="image/png">\n',
        '\n</image>\n<image id="untyped">\n',
        '\n</image>\n<document id="paper" type="application/pdf">\n',
        '\n</document>\n<document id="stored" type="application/pdf">\n',
        "\n</document>\n</attachments>\nCompare them.",
    ]
    typed, untyped, paper, stored = content[1::2]
    assert isinstance(typed, ImageUrl) and typed.media_type == "image/png"
    assert (typed.identifier, untyped.identifier) == ("typed", "untyped")
    assert isinstance(untyped, ImageUrl) and untyped.url.endswith("id=3")
    assert isinstance(paper, DocumentUrl) and paper.identifier == "paper"
    assert isinstance(stored, UploadedFile)
    assert (stored.file_id, stored.provider_name, stored.identifier) == (
        "file-1",
        "openai",
        "stored",
    )


def test_pydantic_history_scan_reads_labels_urls_files_and_checkpoints() -> None:
    from pydantic_ai.messages import (
        BinaryContent,
        CompactionPart,
        ImageUrl,
        ModelRequest,
        ModelResponse,
        TextPart,
        UploadedFile,
        UserPromptPart,
    )

    first = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    '<attachments>\n<image id="old" type="image/png">\n',
                    BinaryContent(png_bytes(1), media_type="image/png"),
                    "\n</image>\n</attachments>\nDescribe old.",
                ]
            )
        ]
    )
    checkpoint = ModelRequest(
        parts=[UserPromptPart(content="summary")], metadata={"kedi_compaction_checkpoint": "c1"}
    )
    second = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    '<attachments>\n<image id="link" type="image/png">\n',
                    ImageUrl("https://example.com/cat.png"),
                    '\n</image>\n<document id="stored" type="application/pdf">\n',
                    UploadedFile("file-1", "openai", media_type="application/pdf"),
                    "\n</document>\nnot a tag",
                    BinaryContent(png_bytes(2), media_type="image/png"),
                    '<document id="note" type="text/plain">\nship\n</document>\n</attachments>\nHi',
                    '<image id="ghost" type="image/png">\n',
                    "text after an opening tag",
                    '<image id="last" type="image/png">\n',
                ]
            ),
            UserPromptPart(content="plain"),
        ]
    )
    reply = ModelResponse(parts=[TextPart("ok")])
    index = pydantic_media.sent_attachments([first, checkpoint, second, reply])
    assert index == {
        "link": ("link", "image", "url", "https://example.com/cat.png"),
        "stored": ("stored", "document", "file_id", "openai", "file-1"),
        "note": ("note", "document", "sha256", _sha256(b"ship")),
    }
    assert pydantic_media.sent_attachments([first])["old"][:3] == ("old", "image", "sha256")

    compacted = ModelResponse(parts=[CompactionPart(content="native summary")])
    assert pydantic_media.sent_attachments([first, compacted]) == {}


def test_pydantic_live_input_confirmation_reads_text_items_of_media_prompts() -> None:
    from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

    from kedi.agent_adapter.adapters.pydantic.live import InputDeliveryCapability

    confirmed: list[tuple[str, ...]] = []
    control = SimpleNamespace(confirm_contents=confirmed.append)
    capability = InputDeliveryCapability(control=control)  # type: ignore[arg-type]
    capability._record_request(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=["cat:", BinaryContent(png_bytes(), media_type="image/png"), "Hi"]
                    ),
                    UserPromptPart(content="plain"),
                ]
            )
        ]
    )
    assert confirmed == [("cat:", "Hi", "plain")]


def test_langchain_blocks_for_urls_files_and_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_download(media: Media) -> Media:
        return Document.from_bytes(b"%PDF-1.4\n%%EOF\n", "application/pdf", name=media.name)

    monkeypatch.setattr(langchain_media, "download_media", fake_download)
    model = SimpleNamespace(profile={"image_url_inputs": True}, model="demo")
    attachments = [
        _attach("link", Image.from_url("https://example.com/render?id=3")),
        _attach("bucket", Video.from_url("s3://bucket/clip.mp4")),
        _attach(
            "stored", Media.from_file_id("file-1", provider="openai", media_type="application/pdf")
        ),
        _attach("paper", Document.from_url("https://example.com/paper.pdf")),
    ]
    content = asyncio.run(
        langchain_media.user_content("Compare.", attachments, history=None, model=model)
    )
    assert isinstance(content, list)
    link, bucket, stored, paper = content[1:8:2]
    assert link == {"type": "image", "url": "https://example.com/render?id=3"}
    assert bucket == {"type": "video", "url": "s3://bucket/clip.mp4", "mime_type": "video/mp4"}
    assert stored == {
        "type": "file",
        "file_id": "file-1",
        "mime_type": "application/pdf",
        "extras": {"filename": "stored.pdf", "name": "stored"},
    }
    # A PDF URL is downloaded: only image URLs are known to be accepted as URLs.
    assert paper["base64"] == base64.b64encode(b"%PDF-1.4\n%%EOF\n").decode()
    assert langchain_media.langchain_model_name(model) == "demo"
    assert langchain_media.langchain_model_name(SimpleNamespace()) is None


def test_langchain_profile_and_client_detection() -> None:
    from langchain_openai import ChatOpenAI

    first_party = ChatOpenAI(model="gpt-4o", api_key="test")
    compatible = ChatOpenAI(model="qwen-vl", api_key="test", base_url="https://example.com/v1")
    assert langchain_media.langchain_media_provider(first_party) == "openai"
    assert langchain_media.langchain_media_provider(compatible) is None
    assert langchain_media.langchain_media_provider(SimpleNamespace()) is None

    pdf = _attach("paper", PDF)
    no_pdf = SimpleNamespace(profile={"pdf_inputs": False})
    with pytest.raises(MediaError, match="paper: the model does not accept PDF input"):
        asyncio.run(langchain_media.user_content("Read.", [pdf], history=None, model=no_pdf))
    # Without a profile, and for text documents, nothing is assumed.
    notes = _attach("notes", Document.from_bytes(b"ship", "text/plain"))
    content = asyncio.run(
        langchain_media.user_content(
            "Read.", [notes, pdf], history=None, model=SimpleNamespace(profile=None)
        )
    )
    assert isinstance(content, list)
    assert content[0]["text"].startswith('<attachments>\n<document id="notes" type="text/plain">')
    assert asyncio.run(langchain_media.user_content("Read.", [notes], history=None, model=no_pdf))


def test_langchain_history_scan_reads_labels_urls_and_inline_files() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    encoded = base64.b64encode(png_bytes(1)).decode()
    history = [
        HumanMessage(content="plain text"),
        AIMessage(content="ok"),
        HumanMessage(
            content=[
                '<attachments>\n<image id="cat" type="image/png">\n',
                {"type": "image", "base64": encoded, "mime_type": "image/png"},
                {"type": "text", "text": '\n</image>\n<image id="link" type="image/png">\n'},
                {"type": "image", "url": "https://example.com/cat.png"},
                {"type": "text", "text": '\n</image>\n<document id="stored">\n'},
                {"type": "file", "file_id": "file-1"},
                {"type": "text", "text": '\n</document>\n<image id="broken">\n'},
                {"type": "image", "base64": "@@@"},
                {"type": "text", "text": '\n</image>\n<audio id="untyped">\n'},
                {"type": "audio", "base64": encoded},
                {"type": "text", "text": '\n</audio>\n<image id="tool">\n'},
                {"type": "tool_use", "id": "t"},
                {"type": "text", "text": "words"},
                "bare string",
                {"type": "text", "text": None},
                {
                    "type": "text",
                    "text": '<document id="notes" type="text/plain">\nship\n</document>\n'
                    "</attachments>\nintro",
                },
                {"type": "text", "text": '<image id="ghost">\n'},
                "bare string after an opening tag",
                {"type": "text", "text": '<image id="last">\n'},
            ]
        ),
    ]
    index = langchain_media.sent_attachments(history)
    assert set(index) == {"cat", "link", "untyped", "notes"}
    assert index["cat"][:3] == ("cat", "image", "sha256")
    assert index["link"] == ("link", "image", "url", "https://example.com/cat.png")
    assert index["untyped"][:2] == ("untyped", "audio")
    assert index["notes"][:3] == ("notes", "document", "sha256")


def test_request_identity_includes_media_but_not_bytes() -> None:
    from kedi.agent_adapter.request import RunRequestAssembler, TextOutput, request_diagnostics
    from kedi.core.model_calls import _prompt_fingerprint
    from kedi.decisions import fingerprint_text

    def request(prompt: PromptDraft) -> Any:
        return RunRequestAssembler().assemble(
            prompt=prompt,
            instructions=(),
            conversation=None,
            tool_specs=(),
            output=TextOutput(),
            run_id="run",
            turn_sequence=0,
            call_kind="invoke",
        )

    plain = PromptDraft(content="Describe cat.", kind="raw")
    with_cat = plain.with_media((_attach("cat", CAT),))
    with_dog = plain.with_media((_attach("cat", DOG),))
    digests = {
        request_diagnostics(request(prompt)).prompt_digest for prompt in (plain, with_cat, with_dog)
    }
    assert len(digests) == 3
    assert request_diagnostics(request(with_cat)).media_count == 1
    assert _prompt_fingerprint(plain) == fingerprint_text("Describe cat.")
    assert len({_prompt_fingerprint(prompt) for prompt in (plain, with_cat, with_dog)}) == 3
    assert with_cat.with_content("Edited.").media == with_cat.media
