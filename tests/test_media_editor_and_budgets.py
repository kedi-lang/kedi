"""References in the editor, request budgets, history codecs and notebook snapshots."""

from __future__ import annotations

import base64
import json

import pytest
from lsprotocol import types as lsp

from kedi._interactive.persistence import SessionValueDecoder, SessionValueEncoder
from kedi.agent_adapter.compaction.langchain_codec import LangChainHistoryCodec
from kedi.agent_adapter.compaction.pydantic_codec import PydanticHistoryCodec
from kedi.agent_adapter.request.media import MediaAttachment, compose_prompt, unsent_attachments
from kedi.agent_adapter.request_budget import (
    estimate_attachment_tokens,
    estimate_media_tokens,
    project_media_payloads,
)
from kedi.media import Document, Image, Reference, Video
from tests.lsp_helpers import diagnostics_for, hover_text_at
from tests.test_media_values import png_bytes

LARGE = png_bytes() + b"\x00" * 600_000


def _codes(source: str) -> list[tuple[str, int]]:
    return [
        (str(diagnostic.code), diagnostic.range.start.line)
        for diagnostic in diagnostics_for(source, "/tmp/media.kedi")
        if str(diagnostic.code).startswith(("K-SUBSTITUTION", "K-ADAPTER"))
    ]


def test_editor_reports_media_substituted_as_text() -> None:
    source = """> import: media
> import: decisions

> profile: reviewer:
    > adapter: pydantic

> profile: lead:
    > adapter: pydantic
    > subagent: reviewer

> use: lead

@caption(photo: Image, style: str) -> str:
  [text] << Describe <<photo>> in a <style> style.
  = <text>

@bad(photo: Image) -> str:
  = See <photo>

@pick() -> Image:
  = `image("cat.png")`

@plain(text: str) -> str:
  = <text>

[cat_image: Image] = `image("cat.png")`
[picked: Image] = `pick()`
[title: str] = Cats
[ok] = <caption(`cat_image`, plain)>
[wrong] = <caption(<cat_image>, plain)>
>> What is <<cat_image>>? Compare <caption(<picked>, plain)> [what]
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
>> Look at <image(cat.png)> and <<image(dog.png)>> [seen]
"""
    assert sorted(_codes(source)) == [
        ("K-SUBSTITUTION-TYPE", 17),  # text return in a procedure
        ("K-SUBSTITUTION-TYPE", 29),  # argument in an assignment
        ("K-SUBSTITUTION-TYPE", 30),  # argument inside a template
        ("K-SUBSTITUTION-TYPE", 31),  # argument inside a raw invoke
        ("K-SUBSTITUTION-TYPE", 32),  # text assignment
        ("K-SUBSTITUTION-TYPE", 33),  # text reassignment
        ("K-SUBSTITUTION-TYPE", 35),  # show
        ("K-SUBSTITUTION-TYPE", 37),  # instructions
        ("K-SUBSTITUTION-TYPE", 39),  # subagent task
        ("K-SUBSTITUTION-TYPE", 41),  # agent input
        ("K-SUBSTITUTION-TYPE", 42),  # a media call as text in a prompt
    ]
    messages = [d.message for d in diagnostics_for(source, "/tmp/media.kedi")]
    assert "<cat_image> renders text, but cat_image is media" in "\n".join(messages)
    assert "pass it by reference with <<image(...)>>" in "\n".join(messages)


def test_editor_treats_media_names_without_the_media_import_as_ordinary_types() -> None:
    source = """~Image(url: str)

@image(path: str) -> str:
  = <path>

@describe(photo: Image) -> str:
  = See <photo>

> agent: claude
[cat_image] = <image(cat.png)>
[reply] << Describe <<cat_image>> and <<image(dog.png)>> with <image(dog.png)>.
"""
    # Without the import, Image is the program's own ~Image type: a model value, not media.
    assert _codes(source) == [("K-SUBSTITUTION-TYPE", 6)]
    (message,) = [
        d.message
        for d in diagnostics_for(source, "/tmp/media.kedi")
        if d.code == "K-SUBSTITUTION-TYPE"
    ]
    assert message.startswith(
        "<photo> renders text, but photo is declared Image, which has no text"
    )


def test_editor_reports_media_prompts_on_adapters_without_media_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter import registry
    from kedi.agent_adapter.meta import AdapterCapabilities

    source = """> import: media
> agent: acp:
    command: acp-server

@pick() -> Image:
  = `image("cat.png")`

[cat_image: Image] = `image("cat.png")`
>> What is <<cat_image>>? [what]
[reply] << Describe <<image(dog.png)>> and <<pick()>>.
> if: <<cat_image>> shows a cat
  [mode] = cat
"""

    def media_lines(text: str) -> list[int]:
        return [
            diagnostic.range.start.line
            for diagnostic in diagnostics_for(text, "/tmp/media.kedi")
            if str(diagnostic.code) == "K-ADAPTER-CAPABILITY"
            and "media input" in diagnostic.message
        ]

    # Every built-in adapter a program can select takes media, so the check is seen through
    # an adapter that advertises none, as a custom adapter may.
    assert media_lines(source) == []
    with monkeypatch.context() as patch:
        patch.setitem(registry._CAPABILITY_LOADERS, "acp", AdapterCapabilities)
        assert media_lines(source) == [8, 9, 9, 10]
    for agent in ("claude", "codex", "a2a:\n    endpoint: https://agent.example"):
        assert media_lines(source.replace("acp:\n    command: acp-server", agent)) == []


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
    assert (
        estimate_attachment_tokens([image])
        == len('<image id="cat" type="image/png"></image>') + 1_600
    )
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
            Reference((dog,)),
            " B ",
            Reference((named,), "cat"),
            " C ",
            Reference((cat,), "again"),
        ]
    )
    # Unnamed media are numbered; explicit names win; the same content keeps its first name.
    assert text == "A image_1 B kedi C kedi"
    assert [attachment.name for attachment in attachments] == ["image_1", "kedi"]

    clash_text, clash = compose_prompt(
        [Reference((cat.with_name("photo"),)), " ", Reference((dog.with_name("photo"),))]
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


def test_editor_reports_substitutions_whose_declared_type_has_no_text_form() -> None:
    source = """~Line(qty: int)

@lines() -> list[Line]:
  = `[]`

@tags(names: set[str]) -> str:
  = <names>

[ok_scalars: dict[str, list[int | None]]] = `{}`
[ok_union: Line | str] = `"x"`
[ok_optional: Optional[str]] = `None`
[ok_unknown: Customer] = `None`
[ok_open: list] = `[]`
[ok_literal: Literal["a"]] = `"a"`
[ok_twice: set[int]] = `set()`
[ok_twice] = text
[bad_line: Line] = `Line(qty=1)`
[bad_list: list[Line]] = `[]`
[bad_keys: dict[int, str]] = `{}`
[bad_values: Dict[str, Line]] = `{}`
[bad_path: Path] = `None`
[bad_union: Union[set[int], bytes]] = `b""`
[bad_none: None] = `None`
[bad_annotated: Annotated[frozenset[int], "ids"]] = `frozenset()`
[bad_optional: Optional[bytes]] = `None`
>> <ok_scalars> <ok_union> <ok_optional> <ok_unknown> <ok_open> <ok_literal> <ok_twice> [a]
>> <bad_line> <bad_list> <bad_keys> <bad_values> <bad_path> <bad_union> [b]
>> <bad_none> <bad_annotated> <bad_optional> <lines()> <<lines()>> <<bad_line>> [c]
"""
    found = [
        (d.range.start.line, d.range.start.character, d.message)
        for d in diagnostics_for(source, "/tmp/types.kedi")
        if d.code == "K-SUBSTITUTION-TYPE"
    ]
    assert sorted((line, character) for line, character, _ in found) == [
        (6, 4),
        (26, 3),
        (26, 14),
        (26, 25),
        (26, 36),
        (26, 49),
        (26, 60),
        (27, 3),
        (27, 14),
        (27, 30),
        (27, 45),
    ]
    messages = "\n".join(message for _, _, message in found)
    assert (
        "<names> renders text, but names is declared set[str], which has no text form." in messages
    )
    # A container of values without text renders only when empty: a warning, not an error.
    assert (
        "<lines(...)> renders text, but lines(...) returns list[Line], which has a text form "
        "only when empty" in messages
    )
    severities = {
        (d.range.start.line, d.range.start.character): d.severity
        for d in diagnostics_for(source, "/tmp/types.kedi")
        if d.code == "K-SUBSTITUTION-TYPE"
    }
    warning, error = lsp.DiagnosticSeverity.Warning, lsp.DiagnosticSeverity.Error
    assert severities[(26, 14)] == warning  # list[Line]
    assert severities[(26, 25)] == warning  # dict[int, str]
    assert severities[(26, 3)] == error  # Line
    assert severities[(6, 4)] == error  # set[str]
    assert "pass it by reference with <<bad_keys>>" in messages


def test_unparsable_annotations_are_left_to_the_run_time_check() -> None:
    from kedi.lsp.features.substitution_types import annotation_has_no_text

    assert annotation_has_no_text("list[", frozenset()) is False
    assert annotation_has_no_text("set[int]", frozenset()) is True
    # Dotted names and keys that are not plain names.
    assert annotation_has_no_text("pathlib.Path", frozenset({"Path"})) is True
    assert annotation_has_no_text("dict[tuple[int, int], str]", frozenset()) is False
    assert annotation_has_no_text("tuple[Line, ...]", frozenset({"Line"})) is True
    assert annotation_has_no_text("tuple[int, ...]", frozenset({"Line"})) is False
    assert annotation_has_no_text("Optional[Line]", frozenset({"Line"})) is True
    assert annotation_has_no_text("list[Optional[Line]]", frozenset({"Line"})) is False
    assert annotation_has_no_text("make()", frozenset()) is False
