"""Kedi media values: loading, kinds, validation and byte-free display."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from kedi.media import (
    MAX_MEDIA_BYTES,
    Audio,
    Document,
    Image,
    Media,
    MediaError,
    Reference,
    Video,
    as_media,
    media_error_detail,
    reference,
    workspace_path,
)


def png_bytes(red: int = 255) -> bytes:
    raw = b"".join(b"\x00" + bytes([red, 0, 0]) * 2 for _ in range(2))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KEDI_WORKSPACE_POLICY", raising=False)
    (tmp_path / "cat.png").write_bytes(png_bytes())
    (tmp_path / "notes.md").write_text("﻿# Notes\nDecision: yes\n", encoding="utf-8")
    return tmp_path


def test_from_path_detects_kind_from_content_and_extension(workspace: Path) -> None:
    image = Image.from_path("cat.png")
    assert (image.kind, image.media_type, image.size) == ("image", "image/png", 73)
    assert image.filename == "cat.png" and image.name is None
    assert image.sha256 == __import__("hashlib").sha256(png_bytes()).hexdigest()

    # Content wins over a misleading extension.
    (workspace / "photo.jpg").write_bytes(png_bytes())
    assert Image.from_path("photo.jpg").media_type == "image/png"

    document = Media.from_path("notes.md")
    assert isinstance(document, Document) and document.is_text
    assert document.text == "# Notes\nDecision: yes\n"  # BOM removed, stored as UTF-8


def test_media_from_path_picks_the_class_and_typed_loaders_check_it(workspace: Path) -> None:
    (workspace / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 16)
    (workspace / "voice.wav").write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24)
    (workspace / "nda.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    assert type(Media.from_path("clip.mp4")) is Video
    assert type(Media.from_path("voice.wav")) is Audio
    assert Document.from_path("nda.pdf").media_type == "application/pdf"
    with pytest.raises(MediaError, match="text/markdown, not an image"):
        Image.from_path("notes.md")
    with pytest.raises(TypeError, match="abstract"):
        Media(media_type="image/png", data=png_bytes())


def test_unsupported_formats_fail_with_a_conversion_hint(workspace: Path) -> None:
    (workspace / "report.docx").write_bytes(b"PK\x03\x04docx")
    with pytest.raises(MediaError, match="Office document; convert it to PDF or text"):
        Media.from_path("report.docx")
    (workspace / "scan.tiff").write_bytes(b"II*\x00tiff")
    with pytest.raises(MediaError, match="images must be PNG, JPEG, GIF or WebP"):
        Image.from_path("scan.tiff")
    with pytest.raises(MediaError, match="Cannot tell the media type"):
        Media.from_bytes(b"\xff\xfe\x00binary")


def test_text_documents_must_be_utf8_unless_an_encoding_is_given(workspace: Path) -> None:
    (workspace / "legacy.txt").write_bytes("şğü".encode("cp1254"))
    with pytest.raises(MediaError, match="not UTF-8 text"):
        Document.from_path("legacy.txt")
    converted = Document.from_path("legacy.txt", encoding="cp1254")
    assert converted.text == "şğü" and converted.data == "şğü".encode()
    with pytest.raises(MediaError, match="encoding= applies only to text documents"):
        Image.from_path("cat.png", encoding="latin-1")


def test_size_limit_is_checked_at_load(workspace: Path) -> None:
    big = workspace / "big.png"
    big.write_bytes(png_bytes() + b"\x00" * MAX_MEDIA_BYTES)
    with pytest.raises(MediaError, match="limited to 20.0 MB"):
        Image.from_path(big)


def test_names_must_be_identifiers(workspace: Path) -> None:
    assert Image.from_path("cat.png", name="cat").name == "cat"
    with pytest.raises(MediaError, match="must be an identifier"):
        Image.from_path("cat.png", name="cat image")


def test_urls_keep_their_kind_and_youtube_is_a_video() -> None:
    video = Video.from_url("https://youtu.be/abc")
    assert video.is_youtube and video.media_type == "video/mp4"
    assert Media.from_url("https://example.com/a.webp").media_type == "image/webp"
    unknown = Image.from_url("https://example.com/render?id=3")
    assert unknown.media_type is None and unknown.kind == "image"
    with pytest.raises(MediaError, match="Cannot tell the document type"):
        Document.from_url("https://example.com/download?id=3")
    with pytest.raises(MediaError, match="Unsupported media URL scheme 'ftp'"):
        Image.from_url("ftp://example.com/a.png")


def test_data_uris_and_file_ids() -> None:
    uri = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    image = Image.from_data_uri(uri)
    assert image.content_key == ("sha256", image.sha256)
    uploaded = Media.from_file_id("file-1", provider="openai", media_type="application/pdf")
    assert isinstance(uploaded, Document)
    assert uploaded.content_key == ("file_id", "openai", "file-1")


def test_validation_accepts_media_values_only(workspace: Path) -> None:
    from pydantic_ai.messages import BinaryContent

    adapter = TypeAdapter(Image)
    image = Image.from_path("cat.png")
    assert adapter.validate_python(image) is image
    assert adapter.validate_python(BinaryContent(png_bytes(), media_type="image/png")) == image
    assert adapter.json_schema() == {"type": "string"}
    # A string is never a path or a URL to load: loading is explicit.
    for value in ("cat.png", "https://example.com/cat.png", 3):
        with pytest.raises(ValidationError, match="load it explicitly"):
            adapter.validate_python(value)
    with pytest.raises(ValidationError) as excinfo:
        adapter.validate_python(Document.from_path("notes.md"))
    assert (
        media_error_detail(excinfo.value)
        == "Expected an image, got <document notes.md · text/markdown · 22 B>"
    )
    # JSON reads back what serialization writes, never a path.
    assert adapter.validate_json('"https://example.com/cat.png"').url is not None
    with pytest.raises(ValidationError, match="as a data URI, a URL or a file id"):
        adapter.validate_json('"cat.png"')
    with pytest.raises(ValidationError, match="is audio/mpeg, not an image"):
        adapter.validate_json('"https://example.com/a.mp3"')
    # Explicit loading keeps the workspace rules.
    with pytest.raises(PermissionError, match="escapes the working directory"):
        Image.coerce("../outside.png")
    (workspace / ".env").write_text("SECRET=1")
    with pytest.raises(PermissionError, match="secret file"):
        workspace_path(".env")
    with pytest.raises(FileNotFoundError, match="Media file not found: missing.png"):
        Image.coerce("missing.png")


def test_workspace_policy_none_allows_outside_paths(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "cat.png"
    outside.write_bytes(png_bytes())
    monkeypatch.setenv("KEDI_WORKSPACE_POLICY", "none")
    assert Image.coerce(str(outside)).filename == "cat.png"


def test_labels_and_reprs_never_contain_bytes(workspace: Path) -> None:
    image = Image.from_path("cat.png")
    assert str(image) == "<image cat.png · image/png · 73 B>"
    assert repr(image) == "Image(media_type='image/png', size=73, filename='cat.png')"
    assert "iVBOR" not in repr([image]) and "\\x89" not in repr([image])


def test_json_serialization_round_trips_through_validation(workspace: Path) -> None:
    adapter = TypeAdapter(Image)
    image = Image.from_path("cat.png")
    dumped = adapter.dump_python(image, mode="json")
    assert dumped.startswith("data:image/png;base64,")
    assert adapter.validate_json(json.dumps(dumped)) == image
    assert adapter.dump_python(image) is image


def test_pydantic_ai_and_pil_values_convert() -> None:
    from pydantic_ai.messages import BinaryContent, DocumentUrl, ImageUrl, UploadedFile

    assert isinstance(as_media(BinaryContent(png_bytes(), media_type="image/png")), Image)
    assert as_media(ImageUrl("https://example.com/a.png")).url == "https://example.com/a.png"
    assert as_media(DocumentUrl("https://example.com/a.pdf")).kind == "document"
    uploaded = as_media(UploadedFile("file-9", "anthropic", media_type="application/pdf"))
    assert uploaded is not None and uploaded.file_id == "file-9"

    class FakePilImage:  # PIL images are recognized by module and their ``save``
        __module__ = "PIL.Image"

        def save(self, buffer, format: str) -> None:
            assert format == "PNG"
            buffer.write(png_bytes())

    converted = as_media(FakePilImage())
    assert converted is not None and converted.media_type == "image/png"


def test_references_attach_media_and_documents_for_other_values(workspace: Path) -> None:
    from pydantic import BaseModel

    class Owner(BaseModel):
        name: str

    image = Image.from_path("cat.png")
    assert reference(image, binding="cat") == Reference((image,), binding="cat")
    assert reference([image, image], binding="cats") == Reference(
        (image, image), binding="cats", is_list=True
    )

    def document(value: object) -> tuple[str | None, str]:
        (item,) = reference(value, call="f").items
        return item.media_type, item.text

    assert document("plain text") == ("text/plain", "plain text")
    assert document(3) == ("text/plain", "3")
    assert document(True) == ("text/plain", "True")
    assert document(["a", 1]) == ("application/json", '["a",1]')
    assert document([]) == ("application/json", "[]")
    assert document({"k": [1]}) == ("application/json", '{"k":[1]}')
    assert document(Owner(name="Ada")) == ("application/json", '{"name":"Ada"}')
    with pytest.raises(MediaError, match="<<x>> is None"):
        reference(None, binding="x")
    with pytest.raises(MediaError, match=r"<<f\(...\)>>: media inside"):
        reference({"photo": image}, call="f")
    with pytest.raises(MediaError, match="media inside a dict, a model or a mixed list"):
        reference([image, "caption"], binding="x")
    with pytest.raises(MediaError, match="a object value has no JSON form"):
        reference(object(), binding="x")


def test_reference_ids_come_from_bindings_and_calls() -> None:
    image = Image.from_bytes(png_bytes(), "image/png")
    assert Reference((image,), "cat_image").preferred_name(1) == "cat_image"
    assert Reference((image, image), "photos", is_list=True).preferred_name(2) == "photos_2"
    assert Reference((image,), "args.invoice").preferred_name(1) == "args_invoice"
    assert Reference((image,)).preferred_name(1) is None
    assert Reference((image,), call="image").generated_base == "image_result"
    assert Reference((image,), "cat").generated_base is None


def test_attachment_elements_carry_the_id_and_type(workspace: Path) -> None:
    from kedi.agent_adapter.request.media import MediaAttachment, attachment_content

    notes = MediaAttachment("notes", Document.from_bytes(b"body", "text/plain"))
    cat = MediaAttachment("cat", Image.from_path("cat.png"))
    stored = MediaAttachment("log", notes.media, artifact_ref={"ref_id": "artifact_1"})
    assert notes.inline_text() == '<document id="notes" type="text/plain">\nbody\n</document>'
    assert stored.inline_text() == '<artifact id="log">\n{"ref_id": "artifact_1"}\n</artifact>'
    assert attachment_content("Ask.", []) == ["Ask."]
    assert attachment_content("Ask.", [notes, cat]) == [
        "<attachments>\n" + notes.inline_text() + '\n<image id="cat" type="image/png">\n',
        cat,
        "\n</image>\n</attachments>\nAsk.",
    ]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\xff\xd8\xff\xe0jpeg", "image/jpeg"),
        (b"GIF89a....", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"FORM\x00\x00\x00\x00AIFFCOMM", "audio/aiff"),
        (b"fLaC\x00\x00\x00\x22", "audio/flac"),
        (b"ID3\x04\x00\x00\x00\x00", "audio/mpeg"),
        (b"\x00\x00\x00\x14ftypqt  \x00\x00", "video/quicktime"),
        (b"\x00\x00\x00\x14ftyp3gp4\x00\x00", "video/3gpp"),
        (b"\x1a\x45\xdf\xa3\x01\x00\x00\x00webm", "video/webm"),
        (b"\x1a\x45\xdf\xa3\x01\x00\x00\x00matroska", "video/x-matroska"),
    ],
)
def test_formats_are_recognized_from_their_leading_bytes(data: bytes, expected: str) -> None:
    assert Media.from_bytes(data).media_type == expected


def test_m4a_audio_and_plain_text_fall_back_to_names_and_decoding(workspace: Path) -> None:
    # An MP4 audio container is not sniffed as video; without a name its type is unknown.
    with pytest.raises(MediaError, match="Cannot tell the media type of the bytes"):
        Media.from_bytes(b"\x00\x00\x00\x14ftypM4A \x00\x00\xff\xfe")
    (workspace / "README").write_text("plain words\n", encoding="utf-8")
    readme = Media.from_path("README")
    assert readme.media_type == "text/plain" and readme.text == "plain words\n"
    with pytest.raises(FileNotFoundError, match="Media file not found: nowhere.png"):
        Image.from_path("nowhere.png")


def test_values_are_immutable_hashable_and_copyable(workspace: Path) -> None:
    import copy
    import pickle

    image = Image.from_path("cat.png", name="cat")
    with pytest.raises(AttributeError, match="Image values are immutable"):
        image.name = "dog"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="Image values are immutable"):
        del image.name  # type: ignore[misc]
    assert pickle.loads(pickle.dumps(image)) == image == copy.deepcopy(image)
    assert len({image, Image.from_path("cat.png", name="cat")}) == 1
    assert image != "cat.png" and image != image.with_name(None)


def test_construction_checks_sources_types_and_limits() -> None:
    with pytest.raises(MediaError, match="exactly one of data, url or file_id"):
        Image(media_type="image/png")
    with pytest.raises(MediaError, match="exactly one of data, url or file_id"):
        Image(media_type="image/png", data=png_bytes(), url="https://example.com/a.png")
    with pytest.raises(MediaError, match="needs its provider"):
        Image(media_type="image/png", file_id="file-1")
    with pytest.raises(MediaError, match="image data needs a media type"):
        Image(media_type=None, data=png_bytes())
    with pytest.raises(MediaError, match="limited to 20.0 MB"):
        Image.from_bytes(png_bytes() + b"\x00" * MAX_MEDIA_BYTES, "image/png")
    with pytest.raises(MediaError, match="audio must be WAV, MP3"):
        Audio.from_bytes(b"....", "audio/midi")
    with pytest.raises(MediaError, match="video must be MP4, MOV"):
        Video.from_bytes(b"....", "video/x-msvideo")
    with pytest.raises(MediaError, match="documents must be PDF or text"):
        Document.from_bytes(b"PK\x03\x04", "application/zip")
    with pytest.raises(MediaError, match="is not latin-9000 text"):
        Document.from_bytes(b"abc", "text/plain", encoding="latin-9000")
    with pytest.raises(MediaError, match="not a loaded text document"):
        Image.from_bytes(png_bytes(), "image/png").text  # noqa: B018
    with pytest.raises(MediaError, match="use Image, Audio, Video or Document"):
        Media.from_url("https://example.com/render?id=3")


def test_data_uri_errors_and_labels_for_every_source() -> None:
    with pytest.raises(MediaError, match="must look like data:<type>;base64,<data>"):
        Image.from_data_uri("data:image/png,plain")
    with pytest.raises(MediaError, match="not valid base64"):
        Image.from_data_uri("data:image/png;base64,@@@@")
    url = Image.from_url("https://example.com/render?id=3")
    assert str(url) == "<image https://example.com/render?id=3>"
    assert repr(url) == "Image(media_type=None, url='https://example.com/render?id=3')"
    uploaded = Media.from_file_id("file-1", provider="openai", media_type="application/pdf")
    assert str(uploaded) == "<document file-1 · application/pdf>"
    named = Image.from_bytes(png_bytes() + b"\x00" * 2_000, "image/png", name="cat")
    assert str(named) == "<image cat · image/png · 2 KB>"
    assert str(named.with_name(None)) == "<image data · image/png · 2 KB>"


def test_coercion_from_bytes_and_unsupported_values(workspace: Path) -> None:
    from kedi.media import is_text_media_type

    assert Image.coerce(png_bytes()).media_type == "image/png"
    assert Image.coerce(bytearray(png_bytes())).size == len(png_bytes())
    with pytest.raises(MediaError, match="Cannot use a int value as an image"):
        Image.coerce(3)
    with pytest.raises(MediaError, match="Cannot use a float value as a media value"):
        Media.coerce(1.5)
    assert not is_text_media_type(None) and is_text_media_type("application/ld+json")
    assert not is_text_media_type("image/svg+xml")


def test_workspace_policy_must_be_known(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEDI_WORKSPACE_POLICY", "open")
    with pytest.raises(RuntimeError, match="Invalid KEDI_WORKSPACE_POLICY value 'open'"):
        workspace_path("cat.png")


def test_pydantic_ai_values_without_a_kedi_equivalent_or_type() -> None:
    from pydantic_ai.messages import ImageUrl, TextPart

    assert as_media(TextPart("words")) is None
    assert as_media(ImageUrl("https://example.com/render")).media_type is None


def test_nested_media_is_found_in_models_and_ignored_in_plain_values(workspace: Path) -> None:
    from pydantic import BaseModel, ConfigDict

    class Report(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        title: str
        photo: Image | None

    image = Image.from_path("cat.png")
    for value in (Report(title="cat", photo=image), {"photos": {image}}):
        with pytest.raises(MediaError, match="media inside"):
            reference(value, binding="r")
    (document,) = reference(Report(title="cat", photo=None), binding="r").items
    assert document.text == '{"title":"cat","photo":null}'
    assert Reference((image,), "1st").preferred_name(1) == "_1st"


def test_error_details_come_only_from_media_failures() -> None:
    adapter = TypeAdapter(int)
    with pytest.raises(ValidationError) as excinfo:
        adapter.validate_python("words")
    assert media_error_detail(excinfo.value) is None
    assert media_error_detail(KeyError("x")) is None
    assert media_error_detail(MediaError("broken")) == "broken"


def test_json_serialization_of_urls_and_file_ids() -> None:
    adapter = TypeAdapter(Media)
    url = Image.from_url("https://example.com/cat.png")
    assert adapter.dump_python(url, mode="json") == "https://example.com/cat.png"
    uploaded = Media.from_file_id("file-1", provider="openai", media_type="application/pdf")
    assert adapter.dump_python(uploaded, mode="json") == {
        "file_id": "file-1",
        "provider": "openai",
        "media_type": "application/pdf",
    }


def test_explicit_loading_and_json_round_trips_cover_every_source(workspace: Path) -> None:
    image = Image.from_path("cat.png")
    uri = TypeAdapter(Image).dump_python(image, mode="json")
    assert Image.coerce(image) is image
    assert Image.coerce(uri) == image
    with pytest.raises(MediaError, match="Expected an image, got <document"):
        Image.coerce(Document.from_path("notes.md"))
    media = TypeAdapter(Media)
    assert media.validate_json(json.dumps(uri)) == image
    uploaded = Media.from_file_id("file-1", provider="openai", media_type="application/pdf")
    assert media.validate_json(json.dumps(media.dump_python(uploaded, mode="json"))) == uploaded


def test_substitutions_open_stored_values_and_references_keep_them() -> None:
    from types import SimpleNamespace

    from kedi.artifacts import ArtifactHandle
    from kedi.lang.compiler.environment import substitution_text

    def stored(value: object) -> ArtifactHandle:
        ref = SimpleNamespace(ref_id="a1", model_dump=lambda **_: {"ref_id": "a1"})
        manager = SimpleNamespace(get_value=lambda ref_id: value)
        return ArtifactHandle(ref=ref, manager=manager)  # type: ignore[arg-type]

    assert substitution_text(stored("the whole log"), "log") == "the whole log"
    with pytest.raises(TypeError, match="<report> renders text, but the value is of type dict"):
        substitution_text(stored({"k": 1}), "report")
    handle = stored("the whole log")
    ref = reference(handle, binding="log")
    assert ref.stored is handle and ref.items[0].text == '{"ref_id": "a1"}'


def test_reference_spans_cover_the_whole_reference() -> None:
    from kedi.lang.ast import InputReference, Span, reference_span

    assert reference_span(InputReference(name="cat")) is None
    loc = Span(start_byte=5, end_byte=13, start_line=1, start_col=5, end_line=1, end_col=13)
    span = reference_span(InputReference(name="cat", loc=loc))
    assert span is not None and (span.start_byte, span.start_col) == (4, 4)


def test_raw_invokes_resolve_binding_references_when_built() -> None:
    from concurrent.futures import Future

    from kedi.core import KediRuntime
    from kedi.core.expressions import ReferenceExpr
    from kedi.promise import KediPromise

    runtime = KediRuntime()
    ref, output = runtime._evaluate_expression(ReferenceExpr(key="notes"), {"notes": "ship"})
    assert output is None and ref.binding == "notes" and ref.items[0].text == "ship"
    # A value still being produced becomes a dependency resolved when the prompt is built.
    future: Future[dict[str, str]] = Future()
    pending, _ = runtime._evaluate_expression(
        ReferenceExpr(key="notes"), {"notes": KediPromise(future, "notes")}
    )
    future.set_result({"notes": "later"})
    assert pending.resolve().items[0].text == "later"
