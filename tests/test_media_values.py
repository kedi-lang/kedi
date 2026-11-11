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
    MediaMention,
    Video,
    as_media,
    format_inline_text,
    media_error_detail,
    prompt_media,
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


def test_validation_accepts_paths_urls_and_values_with_workspace_rules(workspace: Path) -> None:
    adapter = TypeAdapter(Image)
    assert adapter.validate_python("cat.png").filename == "cat.png"
    assert adapter.validate_python("https://example.com/cat.png").url is not None
    assert adapter.json_schema() == {"type": "string"}
    with pytest.raises(PermissionError, match="escapes the working directory"):
        adapter.validate_python("../outside.png")
    (workspace / ".env").write_text("SECRET=1")
    with pytest.raises(PermissionError, match="secret file"):
        workspace_path(".env")
    with pytest.raises(FileNotFoundError, match="Media file not found: missing.png"):
        adapter.validate_python("missing.png")
    with pytest.raises(ValidationError) as excinfo:
        adapter.validate_python(Document.from_path("notes.md"))
    assert (
        media_error_detail(excinfo.value)
        == "Expected an image, got <document notes.md · text/markdown · 22 B>"
    )


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
    assert adapter.validate_python(json.loads(json.dumps(dumped))) == image
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


def test_prompt_media_accepts_values_and_lists_but_not_nested_media(workspace: Path) -> None:
    image = Image.from_path("cat.png")
    assert prompt_media(image) == (image,)
    assert prompt_media([image, image]) == (image, image)
    assert prompt_media("text") is None and prompt_media([]) is None
    with pytest.raises(MediaError, match="inside a list, dict or model"):
        prompt_media({"photo": image})
    with pytest.raises(MediaError, match="inside a list, dict or model"):
        prompt_media([image, "caption"])


def test_mention_names_come_from_bindings() -> None:
    image = Image.from_bytes(png_bytes(), "image/png")
    assert MediaMention((image,), "cat_image").preferred_name(1) == "cat_image"
    assert MediaMention((image, image), "photos", is_list=True).preferred_name(2) == "photos_2"
    assert MediaMention((image,), "args.invoice").preferred_name(1) == "args_invoice"
    assert MediaMention((image,)).preferred_name(1) is None


def test_inline_text_format_matches_pydantic_ai() -> None:
    from pydantic_ai._utils import format_inlined_text_file

    expected = format_inlined_text_file("body", media_type="text/plain", identifier="notes")
    assert format_inline_text("body", name="notes", media_type="text/plain") == expected


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

    from kedi.media import contains_media

    class Report(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        title: str
        photo: Image

    image = Image.from_path("cat.png")
    assert contains_media(Report(title="cat", photo=image))
    assert contains_media({"photos": {image}})
    assert not contains_media(Report.model_construct(title="cat", photo=None))
    assert not contains_media(object())
    assert MediaMention((image,), "1st").preferred_name(1) == "_1st"


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
