"""Media values for multimodal prompts: images, audio, video and documents.

A media value substituted into a model prompt becomes an attachment placed before the prompt
text, and the text shows only its name. Load values in Python with ``Image.from_path(...)`` and
friends, or in Kedi through the bundled ``media`` module (``> import: media``).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, cast
from urllib.parse import unquote_to_bytes, urlparse

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

MediaKind = Literal["image", "audio", "video", "document"]
MediaT = TypeVar("MediaT", bound="Media")

MAX_MEDIA_BYTES = 20 * 1024 * 1024
"""Largest media value Kedi loads: Gemini's limit for a request with inline data."""

IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
AUDIO_MEDIA_TYPES = frozenset(
    {"audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/aiff", "audio/aac"}
)
VIDEO_MEDIA_TYPES = frozenset(
    {
        "video/mp4",
        "video/quicktime",
        "video/webm",
        "video/x-matroska",
        "video/x-flv",
        "video/mpeg",
        "video/x-ms-wmv",
        "video/3gpp",
    }
)
PDF_MEDIA_TYPE = "application/pdf"

_OFFICE_MEDIA_TYPES = frozenset(
    {
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        "application/rtf",
    }
)
_MEDIA_TYPE_ALIASES = {
    "image/jpg": "image/jpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/vnd.wave": "audio/wav",
    "audio/mp3": "audio/mpeg",
    "audio/x-aiff": "audio/aiff",
    "audio/x-flac": "audio/flac",
    "audio/x-aac": "audio/aac",
    "text/xml": "application/xml",
    "application/x-yaml": "application/yaml",
}
_EXTENSION_TYPES = {
    ".webp": "image/webp",
    ".md": "text/markdown",
    ".mdx": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".toml": "application/toml",
    ".xml": "application/xml",
    ".log": "text/plain",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".mkv": "video/x-matroska",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".3gp": "video/3gpp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
}
_MIME_TYPES = mimetypes.MimeTypes()
_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_URL_SCHEMES = frozenset({"http", "https", "s3", "gs"})
_YOUTUBE_HOSTS = frozenset({"youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com"})
_WORKSPACE_POLICY_ENV = "KEDI_WORKSPACE_POLICY"


class MediaError(ValueError):
    """A media value cannot be loaded, or cannot be used where it appears."""


def media_type_for_name(name: str) -> str | None:
    """Guess a normalized media type from a file name or URL path."""

    suffix = Path(name).suffix.lower()
    guessed = _EXTENSION_TYPES.get(suffix) or _MIME_TYPES.guess_type(f"file{suffix}")[0]
    return None if guessed is None else normalize_media_type(guessed)


def normalize_media_type(media_type: str) -> str:
    """Lower-case a media type, drop its parameters and apply common aliases."""

    bare = media_type.split(";", 1)[0].strip().lower()
    return _MEDIA_TYPE_ALIASES.get(bare, bare)


def is_text_media_type(media_type: str | None) -> bool:
    """Return whether a media type holds text: ``text/*``, JSON, XML, YAML or TOML."""

    if media_type is None:
        return False
    bare = normalize_media_type(media_type)
    return (
        bare.startswith("text/")
        or bare in {"application/json", "application/xml", "application/yaml", "application/toml"}
        or bare.endswith("+json")
        or (bare.endswith("+xml") and not bare.startswith("image/"))
    )


def media_kind(media_type: str) -> MediaKind:
    """Return the media kind a media type belongs to."""

    bare = normalize_media_type(media_type)
    if bare.startswith("image/"):
        return "image"
    if bare.startswith("audio/"):
        return "audio"
    if bare.startswith("video/"):
        return "video"
    return "document"


def sniff_media_type(data: bytes) -> str | None:
    """Recognize common binary formats from their leading bytes."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"FORM" and data[8:12] in {b"AIFF", b"AIFC"}:
        return "audio/aiff"
    if data.startswith(b"%PDF-"):
        return PDF_MEDIA_TYPE
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"ID3"):
        return "audio/mpeg"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand == b"qt  ":
            return "video/quicktime"
        if brand.startswith(b"3g"):
            return "video/3gpp"
        if brand not in {b"M4A ", b"M4B "}:
            return "video/mp4"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm" if b"webm" in data[:64] else "video/x-matroska"
    return None


def format_size(size: int) -> str:
    """Render a byte count for labels and errors."""

    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{round(size / 1024)} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def format_inline_text(text: str, *, name: str, media_type: str) -> str:
    """Render a text document the way it reaches the model: a delimited block named ``name``.

    This is the inlined-file format Pydantic AI uses for providers without text-file input,
    so a document reads the same on every adapter and provider.
    """

    return "\n".join(
        (
            f'-----BEGIN FILE id="{name}" type="{media_type}"-----',
            text,
            f'-----END FILE id="{name}"-----',
        )
    )


def workspace_path(raw: str | os.PathLike[str]) -> Path:
    """Resolve a path with the bundled filesystem module's rules.

    Relative paths start at the working directory. With ``KEDI_WORKSPACE_POLICY=strict``
    (the default) a path may not leave it. ``.env`` files are never loaded as media.
    """

    root = Path.cwd().resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    policy = (os.getenv(_WORKSPACE_POLICY_ENV) or "strict").strip().lower()
    if policy not in {"none", "strict"}:
        raise RuntimeError(f"Invalid {_WORKSPACE_POLICY_ENV} value {policy!r}: use none|strict")
    if policy == "strict" and not resolved.is_relative_to(root):
        raise PermissionError(f"path escapes the working directory: {raw}")
    if resolved.name == ".env" or resolved.name.startswith(".env."):
        raise PermissionError(f"refusing to load a secret file as media: {raw}")
    return resolved


_FIELDS = ("media_type", "data", "url", "file_id", "provider", "name", "filename")


class Media:
    """An immutable media value. Exactly one of ``data``, ``url`` or ``file_id`` holds it.

    ``name`` is the name the model sees; without it a prompt uses the binding name, or
    ``image_1``, ``audio_1``, ... by order of appearance. ``filename`` is only for labels
    and errors: it never reaches the model.
    """

    kind: ClassVar[MediaKind]
    _type_label: ClassVar[str]
    __slots__ = (*_FIELDS, "_sha256")

    media_type: str | None
    data: bytes | None
    url: str | None
    file_id: str | None
    provider: str | None
    name: str | None
    filename: str | None
    _sha256: str | None

    def __init__(
        self,
        *,
        media_type: str | None,
        data: bytes | None = None,
        url: str | None = None,
        file_id: str | None = None,
        provider: str | None = None,
        name: str | None = None,
        filename: str | None = None,
    ) -> None:
        values = (media_type, data, url, file_id, provider, name, filename)
        for field_name, value in zip(_FIELDS, values):
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "_sha256", None)
        self._validate()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} values are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} values are immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild_media, (type(self), {name: getattr(self, name) for name in _FIELDS}))

    def _validate(self) -> None:
        if type(self) is Media:
            raise TypeError(
                "Media is abstract: use Image, Audio, Video or Document, or Media.from_path()"
            )
        sources = [value for value in (self.data, self.url, self.file_id) if value is not None]
        if len(sources) != 1:
            raise MediaError("A media value needs exactly one of data, url or file_id")
        if self.name is not None and not _NAME_PATTERN.fullmatch(self.name):
            raise MediaError(
                f"Media name {self.name!r} must be an identifier ([A-Za-z_][A-Za-z0-9_]*)"
            )
        if self.file_id is not None and not self.provider:
            raise MediaError("A file_id media value needs its provider")
        if self.url is not None:
            scheme = urlparse(self.url).scheme.lower()
            if scheme not in _URL_SCHEMES:
                raise MediaError(
                    f"Unsupported media URL scheme {scheme or '(none)'!r}: use http(s), s3 or gs"
                )
        media_type = None if self.media_type is None else normalize_media_type(self.media_type)
        if media_type is None and self.url is None:
            raise MediaError(f"{self._describe()} needs a media type")
        if media_type is not None:
            object.__setattr__(self, "media_type", media_type)
            self._check_media_type(media_type)
        if self.data is not None:
            if len(self.data) > MAX_MEDIA_BYTES:
                raise MediaError(
                    f"{self._describe()} is {format_size(len(self.data))}; media values are "
                    f"limited to {format_size(MAX_MEDIA_BYTES)}"
                )
            if self.is_text:
                object.__setattr__(self, "data", _utf8_text_bytes(self.data, self._describe()))

    def _check_media_type(self, media_type: str) -> None:
        if media_kind(media_type) != self.kind:
            raise MediaError(f"{self._describe()} is {media_type}, not {self._type_label}")

    # Content

    @property
    def size(self) -> int | None:
        """Byte size of ``data``; ``None`` for URL and file-id values."""

        return None if self.data is None else len(self.data)

    @property
    def sha256(self) -> str | None:
        """Hex SHA-256 of ``data``; ``None`` for URL and file-id values."""

        if self.data is None:
            return None
        if self._sha256 is None:
            object.__setattr__(self, "_sha256", hashlib.sha256(self.data).hexdigest())
        return self._sha256

    @property
    def content_key(self) -> tuple[str, ...]:
        """Identity of the content, independent of its name."""

        if self.data is not None:
            return ("sha256", self.sha256 or "")
        if self.url is not None:
            return ("url", self.url)
        return ("file_id", self.provider or "", self.file_id or "")

    @property
    def is_text(self) -> bool:
        """Whether this is a text document, which reaches the model as text."""

        return self.kind == "document" and is_text_media_type(self.media_type)

    @property
    def is_youtube(self) -> bool:
        return self.url is not None and urlparse(self.url).hostname in _YOUTUBE_HOSTS

    @property
    def text(self) -> str:
        """The UTF-8 text of a loaded text document."""

        if not self.is_text or self.data is None:
            raise MediaError(f"{self._describe()} is not a loaded text document")
        return self.data.decode("utf-8")

    def with_name(self: MediaT, name: str | None) -> MediaT:
        """Return the same content under another model-facing name."""

        return type(self)(
            media_type=self.media_type,
            data=self.data,
            url=self.url,
            file_id=self.file_id,
            provider=self.provider,
            name=name,
            filename=self.filename,
        )

    # Display

    @property
    def label(self) -> str:
        """A byte-free description for ``> show:``, logs and errors."""

        parts = [self._source_label()]
        if self.media_type is not None:
            parts.append(self.media_type)
        if self.size is not None:
            parts.append(format_size(self.size))
        return f"<{self.kind} {' · '.join(parts)}>"

    def _source_label(self) -> str:
        if self.filename:
            return self.filename
        if self.url is not None:
            return self.url
        if self.file_id is not None:
            return self.file_id
        return self.name or "data"

    def _describe(self) -> str:
        return self.filename or self.url or self.file_id or self.name or f"{self.kind} data"

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        fields = [f"media_type={self.media_type!r}"]
        if self.size is not None:
            fields.append(f"size={self.size}")
        for attribute in ("url", "file_id", "provider", "name", "filename"):
            value = getattr(self, attribute)
            if value is not None:
                fields.append(f"{attribute}={value!r}")
        return f"{type(self).__name__}({', '.join(fields)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Media):
            return NotImplemented
        return (type(self), self.media_type, self.content_key, self.name) == (
            type(other),
            other.media_type,
            other.content_key,
            other.name,
        )

    def __hash__(self) -> int:
        return hash((type(self), self.media_type, self.content_key, self.name))

    # Construction

    @classmethod
    def from_path(
        cls: type[MediaT],
        path: str | os.PathLike[str],
        *,
        name: str | None = None,
        media_type: str | None = None,
        encoding: str | None = None,
    ) -> MediaT:
        """Read a file. The media type comes from ``media_type``, the content or the extension."""

        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise FileNotFoundError(f"Media file not found: {path}")
        size = file_path.stat().st_size
        if size > MAX_MEDIA_BYTES:
            raise MediaError(
                f"{file_path.name} is {format_size(size)}; media values are limited to "
                f"{format_size(MAX_MEDIA_BYTES)}"
            )
        data = file_path.read_bytes()
        return cls._from_data(
            data,
            media_type=media_type,
            name=name,
            filename=file_path.name,
            encoding=encoding,
        )

    @classmethod
    def from_bytes(
        cls: type[MediaT],
        data: bytes,
        media_type: str | None = None,
        *,
        name: str | None = None,
        encoding: str | None = None,
    ) -> MediaT:
        """Wrap bytes; without ``media_type`` the format must be recognizable from them."""

        return cls._from_data(
            bytes(data), media_type=media_type, name=name, filename=None, encoding=encoding
        )

    @classmethod
    def from_url(
        cls: type[MediaT],
        url: str,
        *,
        name: str | None = None,
        media_type: str | None = None,
    ) -> MediaT:
        """Refer to an http(s), ``s3://`` or ``gs://`` URL. Kedi downloads it only when needed."""

        parsed = urlparse(url)
        resolved_type = media_type
        if resolved_type is None and parsed.hostname in _YOUTUBE_HOSTS:
            resolved_type = "video/mp4"
        if resolved_type is None:
            resolved_type = media_type_for_name(parsed.path)
        target = cls._target_class(resolved_type, url)
        if resolved_type is None and target is Document:
            raise MediaError(f"Cannot tell the document type of {url}; pass media_type=")
        return target(media_type=resolved_type, url=url, name=name)

    @classmethod
    def from_data_uri(cls: type[MediaT], uri: str, *, name: str | None = None) -> MediaT:
        """Decode a base64 ``data:`` URI."""

        if not uri.startswith("data:") or ";base64," not in uri:
            raise MediaError("A media data URI must look like data:<type>;base64,<data>")
        header, payload = uri[5:].split(";base64,", 1)
        try:
            data = base64.b64decode(unquote_to_bytes(payload), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise MediaError("The media data URI is not valid base64") from exc
        return cls._from_data(data, media_type=header or None, name=name, filename=None)

    @classmethod
    def from_file_id(
        cls: type[MediaT],
        file_id: str,
        *,
        provider: str,
        media_type: str,
        name: str | None = None,
    ) -> MediaT:
        """Refer to a file already uploaded to ``provider`` (for example ``openai``)."""

        target = cls._target_class(media_type, file_id)
        return target(media_type=media_type, file_id=file_id, provider=provider, name=name)

    @classmethod
    def _from_data(
        cls: type[MediaT],
        data: bytes,
        *,
        media_type: str | None,
        name: str | None,
        filename: str | None,
        encoding: str | None = None,
    ) -> MediaT:
        resolved = (
            normalize_media_type(media_type)
            if media_type is not None
            else sniff_media_type(data)
            or (None if filename is None else media_type_for_name(filename))
        )
        if resolved is None and _decodes(data, encoding):
            resolved = "text/plain"
        if resolved is None:
            source = filename or "the bytes"
            raise MediaError(f"Cannot tell the media type of {source}; pass media_type=")
        target = cls._target_class(resolved, filename or "the bytes")
        if encoding is not None:
            if not (target is Document and is_text_media_type(resolved)):
                raise MediaError("encoding= applies only to text documents")
            try:
                data = data.decode(encoding).encode("utf-8")
            except (LookupError, UnicodeDecodeError) as exc:
                raise MediaError(f"{filename or 'the bytes'} is not {encoding} text") from exc
        return target(media_type=resolved, data=data, name=name, filename=filename)

    @classmethod
    def _target_class(cls: type[MediaT], media_type: str | None, source: str) -> type[MediaT]:
        if cls is not Media:
            return cls
        if media_type is None:
            raise MediaError(
                f"Cannot tell the media type of {source}; use Image, Audio, Video or Document"
            )
        return cast("type[MediaT]", _KIND_CLASSES[media_kind(media_type)])

    @classmethod
    def coerce(cls: type[MediaT], value: Any) -> MediaT:
        """Convert a supported value, as typed parameters and validation do.

        Accepts media values, Pydantic AI media objects, PIL images, data URIs, URLs and
        workspace paths (see :func:`workspace_path`).
        """

        media = as_media(value)
        if media is None:
            if isinstance(value, (bytes, bytearray)):
                media = cls.from_bytes(bytes(value))
            elif isinstance(value, str) and value.startswith("data:"):
                media = cls.from_data_uri(value)
            elif isinstance(value, str) and urlparse(value).scheme.lower() in _URL_SCHEMES:
                media = cls.from_url(value)
            elif isinstance(value, (str, os.PathLike)):
                resolved = workspace_path(value)
                if not resolved.is_file():
                    raise FileNotFoundError(f"Media file not found: {value}")
                media = cls.from_path(resolved)
            else:
                raise MediaError(f"Cannot use a {type(value).__name__} value as {cls._label()}")
        if not isinstance(media, cls):
            raise MediaError(f"Expected {cls._label()}, got {media.label}")
        return media

    @classmethod
    def _label(cls) -> str:
        return getattr(cls, "_type_label", "a media value")

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls.coerce,
            json_schema_input_schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, info_arg=True
            ),
        )


class Image(Media):
    """An image: PNG, JPEG, GIF or WebP."""

    __slots__ = ()
    kind: ClassVar[MediaKind] = "image"
    _type_label: ClassVar[str] = "an image"

    def _check_media_type(self, media_type: str) -> None:
        super()._check_media_type(media_type)
        if media_type not in IMAGE_MEDIA_TYPES:
            raise MediaError(
                f"{self._describe()} is {media_type}; images must be PNG, JPEG, GIF or WebP"
            )


class Audio(Media):
    """Audio: WAV, MP3, OGG, FLAC, AIFF or AAC."""

    __slots__ = ()
    kind: ClassVar[MediaKind] = "audio"
    _type_label: ClassVar[str] = "audio"

    def _check_media_type(self, media_type: str) -> None:
        super()._check_media_type(media_type)
        if media_type not in AUDIO_MEDIA_TYPES:
            raise MediaError(
                f"{self._describe()} is {media_type}; audio must be WAV, MP3, OGG, FLAC, "
                "AIFF or AAC"
            )


class Video(Media):
    """Video: MP4, MOV, WebM, MKV, FLV, MPEG, WMV or 3GP."""

    __slots__ = ()
    kind: ClassVar[MediaKind] = "video"
    _type_label: ClassVar[str] = "a video"

    def _check_media_type(self, media_type: str) -> None:
        super()._check_media_type(media_type)
        if media_type not in VIDEO_MEDIA_TYPES:
            raise MediaError(
                f"{self._describe()} is {media_type}; video must be MP4, MOV, WebM, MKV, FLV, "
                "MPEG, WMV or 3GP"
            )


class Document(Media):
    """A PDF, or a text document (text, Markdown, CSV, HTML, JSON, XML, YAML, TOML)."""

    __slots__ = ()
    kind: ClassVar[MediaKind] = "document"
    _type_label: ClassVar[str] = "a document"

    def _check_media_type(self, media_type: str) -> None:
        super()._check_media_type(media_type)
        if media_type in _OFFICE_MEDIA_TYPES:
            raise MediaError(
                f"{self._describe()} is an Office document; convert it to PDF or text first"
            )
        if media_type != PDF_MEDIA_TYPE and not is_text_media_type(media_type):
            raise MediaError(f"{self._describe()} is {media_type}; documents must be PDF or text")


_KIND_CLASSES: dict[MediaKind, type[Media]] = {
    "image": Image,
    "audio": Audio,
    "video": Video,
    "document": Document,
}


@dataclass(frozen=True, slots=True)
class MediaMention:
    """Media substituted into a prompt, before the request assigns names."""

    items: tuple[Media, ...]
    binding: str | None = None
    is_list: bool = False

    def preferred_name(self, index: int) -> str | None:
        if self.binding is None:
            return None
        # A dotted lookup such as ``args.invoice`` still yields an identifier.
        base = re.sub(r"\W", "_", self.binding)
        if base[:1].isdigit():
            base = f"_{base}"
        return f"{base}_{index}" if self.is_list else base


def as_media(value: Any) -> Media | None:
    """Return ``value`` as a media value when it is one, or a Pydantic AI or PIL equivalent."""

    if isinstance(value, Media):
        return value
    module = type(value).__module__
    if module.startswith("pydantic_ai"):
        return _from_pydantic_ai(value)
    if module.startswith("PIL.") and callable(getattr(value, "save", None)):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return Image.from_bytes(buffer.getvalue(), "image/png")
    return None


def prompt_media(value: Any) -> tuple[Media, ...] | None:
    """Return the media a prompt substitution attaches, or ``None`` for an ordinary value.

    A media value or a non-empty list or tuple of media values is attached. Media anywhere
    else in a value (in a dict, a model or a mixed list) is an error: it would otherwise reach
    the model as its label only.
    """

    media = as_media(value)
    if media is not None:
        return (media,)
    if isinstance(value, (list, tuple)) and value:
        items = [as_media(item) for item in value]
        if all(item is not None for item in items):
            return tuple(item for item in items if item is not None)
    if contains_media(value):
        raise MediaError(
            "Media inside a list, dict or model cannot be used in a prompt yet; "
            "substitute each media value directly"
        )
    return None


def contains_media(value: Any, *, _depth: int = 0) -> bool:
    """Return whether a container holds a media value (bounded depth)."""

    if _depth > 8 or isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return False
    if as_media(value) is not None:
        return True
    if isinstance(value, dict):
        return any(contains_media(item, _depth=_depth + 1) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(contains_media(item, _depth=_depth + 1) for item in value)
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return any(
            contains_media(item, _depth=_depth + 1)
            for item in object.__getattribute__(value, "__dict__").values()
        )
    return False


def media_error_detail(exc: BaseException) -> str | None:
    """Return the reason a media value failed validation, for wrapped type errors."""

    if isinstance(exc, (MediaError, OSError)):
        return str(exc)
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        for error in exc.errors():
            cause = (error.get("ctx") or {}).get("error")
            if isinstance(cause, MediaError):
                return str(cause)
    return None


def text_embedding_error(value: Any, *, hint: str | None = None) -> MediaError:
    """The error for media rendered into text outside a model prompt."""

    media = as_media(value)
    subject = media.label if media is not None else "A media value"
    message = (
        f"{subject} cannot be embedded in text: use it directly in a model prompt "
        "(>>, <<, a typed capture or a > if: claim)"
    )
    return MediaError(message if hint is None else f"{message}; {hint}")


def _from_pydantic_ai(value: Any) -> Media | None:
    from pydantic_ai import messages

    if isinstance(value, messages.BinaryContent):
        return Media.from_bytes(value.data, value.media_type)
    url_types: dict[type[Any], type[Media]] = {
        messages.ImageUrl: Image,
        messages.AudioUrl: Audio,
        messages.VideoUrl: Video,
        messages.DocumentUrl: Document,
    }
    for url_type, media_class in url_types.items():
        if isinstance(value, url_type):
            try:
                media_type: str | None = value.media_type
            except ValueError:
                media_type = None
            return media_class.from_url(value.url, media_type=media_type)
    if isinstance(value, messages.UploadedFile):
        return Media.from_file_id(
            value.file_id, provider=value.provider_name, media_type=value.media_type
        )
    return None


def _decodes(data: bytes, encoding: str | None) -> bool:
    try:
        data.decode(encoding or "utf-8")
    except (LookupError, UnicodeDecodeError):
        return False
    return True


def _utf8_text_bytes(data: bytes, source: str) -> bytes:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MediaError(
            f"{source} is not UTF-8 text; convert it, or pass encoding= in Python"
        ) from exc
    return text.encode("utf-8")


def _rebuild_media(cls: type[Media], values: dict[str, Any]) -> Media:
    return cls(**values)


def _serialize(value: Media, info: core_schema.SerializationInfo) -> Any:
    """Keep the value in Python mode; JSON gets a data URI, the URL or the file id."""
    if info.mode != "json":
        return value
    if value.data is not None:
        return f"data:{value.media_type};base64,{base64.b64encode(value.data).decode('ascii')}"
    if value.url is not None:
        return value.url
    return {"file_id": value.file_id, "provider": value.provider, "media_type": value.media_type}


__all__ = [
    "MAX_MEDIA_BYTES",
    "Audio",
    "Document",
    "Image",
    "Media",
    "MediaError",
    "MediaKind",
    "MediaMention",
    "Video",
    "as_media",
    "contains_media",
    "format_inline_text",
    "is_text_media_type",
    "prompt_media",
    "workspace_path",
]
