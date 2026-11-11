"""Media in the editor, request budgets, history codecs and notebook snapshots."""

from __future__ import annotations

import base64
import json

from kedi._interactive.persistence import SessionValueDecoder, SessionValueEncoder
from kedi.agent_adapter.compaction.langchain_codec import LangChainHistoryCodec
from kedi.agent_adapter.compaction.pydantic_codec import PydanticHistoryCodec
from kedi.agent_adapter.request.media import MediaAttachment, compose_prompt, unsent_attachments
from kedi.agent_adapter.request_budget import (
    estimate_attachment_tokens,
    estimate_media_tokens,
    project_media_payloads,
)
from kedi.media import Document, Image, MediaMention, Video
from tests.lsp_helpers import diagnostics_for, hover_text_at
from tests.test_media_values import png_bytes

LARGE = png_bytes() + b"\x00" * 600_000


def _codes(source: str) -> list[tuple[str, int]]:
    return [
        (str(diagnostic.code), diagnostic.range.start.line)
        for diagnostic in diagnostics_for(source, "/tmp/media.kedi")
        if str(diagnostic.code).startswith(("K-MEDIA", "K-ADAPTER"))
    ]


def test_editor_reports_media_rendered_into_text() -> None:
    source = """> import: media
> import: decisions

> profile: reviewer:
    > adapter: pydantic

> profile: lead:
    > adapter: pydantic
    > subagent: reviewer

> use: lead

@caption(photo: Image, style: str) -> str:
  [text] << Describe <photo> in a <style> style.
  = <text>

@bad(photo: Image) -> str:
  = See <photo>

@pick() -> Image:
  = <image(cat.png)>

@plain(text: str) -> str:
  = <text>

[cat_image] = <image(cat.png)>
[picked] = <pick()>
[title: str] = Cats
[ok] = <caption(`cat_image`, plain)>
[wrong] = <caption(<cat_image>, plain)>
>> What is <cat_image>? Compare <caption(<picked>, plain)> [what]
[reply] << Describe <caption(<cat_image>, plain)>.
[label] = See <cat_image>
[label] := Again <picked>
[copy: Image] = `cat_image`
> show: <cat_image>
> instructions:
    Look at <cat_image>.
> task [job]: reviewer:
    >> Review <cat_image>.
> send: job:
    >> Also check <picked>.
"""
    assert sorted(_codes(source)) == [
        ("K-MEDIA-TEXT", 17),  # text return in a procedure
        ("K-MEDIA-TEXT", 29),  # string argument in an assignment
        ("K-MEDIA-TEXT", 30),  # string argument inside a template
        ("K-MEDIA-TEXT", 31),  # string argument inside a raw invoke
        ("K-MEDIA-TEXT", 32),  # text assignment
        ("K-MEDIA-TEXT", 33),  # text reassignment
        ("K-MEDIA-TEXT", 37),  # instructions
        ("K-MEDIA-TEXT", 39),  # subagent task
        ("K-MEDIA-TEXT", 41),  # agent input
    ]
    messages = [d.message for d in diagnostics_for(source, "/tmp/media.kedi")]
    assert any("instructions are text only" in message for message in messages)
    assert any("subagent tasks do not take media yet" in message for message in messages)


def test_editor_ignores_media_names_without_the_media_import() -> None:
    source = """~Image(url: str)

@image(path: str) -> str:
  = <path>

@describe(photo: Image) -> str:
  = See <photo>

> agent: claude
[cat_image] = <image(cat.png)>
[reply] << Describe <cat_image> and <image(dog.png)>.
"""
    assert _codes(source) == []


def test_editor_reports_media_prompts_on_adapters_without_media_input() -> None:
    source = """> import: media
> agent: claude

@pick() -> Image:
  = <image(cat.png)>

[cat_image] = <image(cat.png)>
>> What is <cat_image>? [what]
[reply] << Describe <image(dog.png)> and <pick()>.
> if: <cat_image> shows a cat
  [mode] = cat
"""
    assert _codes(source) == [
        ("K-ADAPTER-CAPABILITY", 7),
        ("K-ADAPTER-CAPABILITY", 8),
        ("K-ADAPTER-CAPABILITY", 8),
        ("K-ADAPTER-CAPABILITY", 9),
    ]


def test_editor_hover_documents_the_loaders() -> None:
    source = "> import: media\n[cat_image] = <image(cat.png)>\n"
    hover = hover_text_at(source, "/tmp/media.kedi", "image(cat", offset=1)
    assert "@image(source: Any)" in hover and "data: URI" in hover


def test_budget_estimates_media_by_kind_not_base64() -> None:
    assert estimate_media_tokens("image", 10_000_000) == 1_600
    assert estimate_media_tokens("audio", 1_000_000) == 2_000
    assert estimate_media_tokens("video", 100) == 258
    image = MediaAttachment(name="cat", media=Image.from_bytes(LARGE, "image/png"))
    notes = MediaAttachment(name="notes", media=Document.from_bytes(b"ship", "text/plain"))
    assert estimate_attachment_tokens([image]) == len("cat:") + 1_600
    assert estimate_attachment_tokens([notes]) == len(notes.inline_text().encode())

    encoded = base64.b64encode(LARGE).decode()
    projected, tokens = project_media_payloads(
        [
            {"kind": "binary", "data": encoded, "media_type": "image/png"},
            {"type": "audio", "base64": encoded, "mime_type": "audio/wav"},
            {"type": "text", "text": f"data:video/mp4;base64,{encoded}"},
            {"type": "file", "source_type": "base64", "data": encoded},
            {"type": "file", "file_id": "file-1"},
        ]
    )
    assert encoded not in json.dumps(projected)
    assert projected[-1] == {"type": "file", "file_id": "file-1"}
    assert tokens == 1_600 + len(LARGE) // 500 + len(LARGE) // 1_000 + len(LARGE) // 50


def test_history_codecs_do_not_count_or_search_base64() -> None:
    from langchain_core.messages import HumanMessage, message_to_dict
    from pydantic_ai.messages import (
        BinaryContent,
        ModelMessagesTypeAdapter,
        ModelRequest,
        UserPromptPart,
    )

    request = ModelRequest(
        parts=[UserPromptPart(content=["cat:", BinaryContent(LARGE, media_type="image/png"), "hi"])]
    )
    pydantic_state = {
        "pydantic_messages": ModelMessagesTypeAdapter.dump_python([request], mode="json")
    }
    block = {"type": "image", "base64": base64.b64encode(LARGE).decode(), "mime_type": "image/png"}
    langchain_state = {
        "langchain_messages": [
            message_to_dict(HumanMessage(content=[{"type": "text", "text": "cat:"}, block]))
        ]
    }
    for codec, state in (
        (PydanticHistoryCodec(), pydantic_state),
        (LangChainHistoryCodec(), langchain_state),
    ):
        (entry,) = codec.inspect(state).entries
        assert entry.token_estimate < 3_000
        assert "AAAAAAAA" not in entry.searchable_text


def test_request_names_and_history_reuse_rules() -> None:
    cat = Image.from_bytes(png_bytes(1), "image/png")
    dog = Image.from_bytes(png_bytes(2), "image/png")
    named = cat.with_name("kedi")
    text, attachments = compose_prompt(
        [
            "A ",
            MediaMention((dog,)),
            " B ",
            MediaMention((named,), "cat"),
            " C ",
            MediaMention((cat,), "again"),
        ]
    )
    # Unnamed media are numbered; explicit names win; the same content keeps its first name.
    assert text == "A image_1 B kedi C kedi"
    assert [attachment.name for attachment in attachments] == ["image_1", "kedi"]

    clash_text, clash = compose_prompt(
        [MediaMention((cat.with_name("photo"),)), " ", MediaMention((dog.with_name("photo"),))]
    )
    assert clash_text == "photo photo_2" and len(clash) == 2

    sent = {"kedi": attachments[1].identity}
    assert unsent_attachments(attachments, sent) == (attachments[0],)
    assert unsent_attachments(attachments, None) == attachments
    rebound = MediaAttachment(name="kedi", media=dog)
    assert unsent_attachments([rebound], sent) == (rebound,)


def test_notebook_snapshots_round_trip_media() -> None:
    encoder = SessionValueEncoder(kedi_models={})
    decoder = SessionValueDecoder(kedi_models={})
    values = [
        Image.from_bytes(png_bytes(), "image/png", name="cat"),
        Video.from_url("https://youtu.be/abc"),
        [Document.from_bytes(b"ship", "text/markdown")],
    ]
    for value in values:
        restored = decoder.decode(json.loads(json.dumps(encoder.encode(value))))
        assert restored == value
