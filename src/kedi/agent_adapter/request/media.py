"""Prompt attachments: the values a prompt references, named for one model request.

Every value a prompt references with ``<<x>>`` becomes an attachment placed before the prompt
text, inside one ``<attachments>`` element and tagged with the name the text uses::

    <attachments>
    <image id="cat1" type="image/png">
    [image part]
    </image>
    <document id="spec" type="text/plain">
    ...
    </document>
    </attachments>
    Does cat1 match spec?

Naming happens once per request: an explicit ``name=``, else the binding name, else
``<proc>_result_1``, ... for procedure calls. The same content is attached once even when two
names reference it. Contents are not escaped.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlparse

from kedi.media import Media, MediaError, MediaKind, Reference

MediaSource = Literal["data", "url", "file_id"]
MediaProvider = Literal["openai", "anthropic", "google", "bedrock", "typesafe", "laya"]


@dataclass(frozen=True, slots=True)
class MediaSummary:
    """Byte-free description of one attachment for hooks, diagnostics and telemetry."""

    name: str
    kind: MediaKind
    media_type: str | None
    source: MediaSource
    size: int | None = None
    sha256: str | None = None
    url: str | None = None


@dataclass(frozen=True, slots=True)
class MediaAttachment:
    """One media value attached to a prompt under the name the prompt text uses.

    ``stored`` is the artifact handle of a value referenced with ``<<x>>`` that is already stored
    as an artifact. When the model can read artifacts, ``artifact_ref`` replaces the content: the
    model sees the reference and reads the value with the artifact tools.
    """

    name: str
    media: Media
    artifact_ref: Mapping[str, Any] | None = None
    stored: Any = field(default=None, compare=False, repr=False)

    @property
    def summary(self) -> MediaSummary:
        media = self.media
        return MediaSummary(
            name=self.name,
            kind=media.kind,
            media_type=media.media_type,
            source="data" if media.data is not None else "url" if media.url else "file_id",
            size=media.size,
            sha256=media.sha256,
            url=media.url,
        )

    @property
    def identity(self) -> tuple[str, ...]:
        """What makes two attachments interchangeable for history reuse."""

        return (self.name, self.media.kind, *self.media.content_key)

    @property
    def is_text(self) -> bool:
        return self.artifact_ref is not None or self.media.is_text

    @property
    def tag(self) -> str:
        return "artifact" if self.artifact_ref is not None else self.media.kind

    def open_tag(self) -> str:
        if self.artifact_ref is not None:
            return f'<artifact id="{self.name}">'
        media_type = self.media.media_type
        kind = self.media.kind
        if media_type is None:
            return f'<{kind} id="{self.name}">'
        return f'<{kind} id="{self.name}" type="{media_type}">'

    def close_tag(self) -> str:
        return f"</{self.tag}>"

    def inline_text(self) -> str:
        """The element a text document or an artifact reference reaches the model as."""

        if self.artifact_ref is not None:
            content = json.dumps(self.artifact_ref, ensure_ascii=False, sort_keys=True)
        else:
            content = self.media.text
        return f"{self.open_tag()}\n{content}\n{self.close_tag()}"


ATTACHMENTS_OPEN = "<attachments>"
ATTACHMENTS_CLOSE = "</attachments>"


def attachment_content(
    prompt: str, attachments: Sequence[MediaAttachment]
) -> list[str | MediaAttachment]:
    """Lay out one user message: text pieces, and the attachments whose native part goes there.

    Adjacent text is merged, so a binary attachment's opening tag ends the text before it.
    """

    if not attachments:
        return [prompt]
    pieces: list[str | MediaAttachment] = []
    text = ATTACHMENTS_OPEN + "\n"
    for attachment in attachments:
        if attachment.is_text:
            text += attachment.inline_text() + "\n"
            continue
        pieces.append(text + attachment.open_tag() + "\n")
        pieces.append(attachment)
        text = "\n" + attachment.close_tag() + "\n"
    pieces.append(text + ATTACHMENTS_CLOSE + "\n" + prompt)
    return pieces


_TEXT_ELEMENT = re.compile(
    r'<document id="([A-Za-z_][A-Za-z0-9_]*)" type="[^"]*">\n(.*?)\n</document>', re.DOTALL
)
_OPEN_TAG_AT_END = re.compile(
    r'<(?:image|audio|video|document) id="([A-Za-z_][A-Za-z0-9_]*)"(?: type="[^"]*")?>\n\Z'
)


def scan_sent_content(
    items: Sequence[Any],
    index: dict[str, tuple[str, ...]],
    *,
    text_of: Callable[[Any], str | None],
    identity_of: Callable[[Any], tuple[str, ...] | None],
) -> None:
    """Record the attachments one sent user message holds, by name, into ``index``."""

    for position, item in enumerate(items):
        text = text_of(item)
        if text is None:
            continue
        for name, content in _TEXT_ELEMENT.findall(text):
            index[name] = (name, "document", "sha256", _sha256(content.encode("utf-8")))
        opened = _OPEN_TAG_AT_END.search(text)
        following = items[position + 1] if position + 1 < len(items) else None
        identity = None if following is None or opened is None else identity_of(following)
        if identity is not None and opened is not None:
            index[opened.group(1)] = (opened.group(1), *identity)


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class MediaNamer:
    """Assign request-unique names and drop repeated content."""

    def __init__(self) -> None:
        self._by_content: dict[tuple[str, ...], str] = {}
        self._names: set[str] = set()
        self._generated: Counter[str] = Counter()
        self._attachments: list[MediaAttachment] = []

    def render(self, ref: Reference) -> str:
        names = [
            self.name_for(
                item, ref.preferred_name(index), generated=ref.generated_base, stored=ref.stored
            )
            for index, item in enumerate(ref.items, start=1)
        ]
        return ", ".join(names)

    def name_for(
        self,
        media: Media,
        preferred: str | None,
        *,
        generated: str | None = None,
        stored: Any = None,
    ) -> str:
        key = (media.kind, *media.content_key)
        existing = self._by_content.get(key)
        if existing is not None:
            return existing
        base = media.name or preferred
        if base is None:
            name = self._next_generated(generated or media.kind)
        else:
            name = self._unique(base)
        self._by_content[key] = name
        self._names.add(name)
        self._attachments.append(MediaAttachment(name=name, media=media, stored=stored))
        return name

    @property
    def attachments(self) -> tuple[MediaAttachment, ...]:
        return tuple(self._attachments)

    def _unique(self, base: str) -> str:
        if base not in self._names:
            return base
        suffix = 2
        while f"{base}_{suffix}" in self._names:
            suffix += 1
        return f"{base}_{suffix}"

    def _next_generated(self, kind: str) -> str:
        while True:
            self._generated[kind] += 1
            candidate = f"{kind}_{self._generated[kind]}"
            if candidate not in self._names:
                return candidate


def compose_prompt(parts: Iterable[Any]) -> tuple[str, tuple[MediaAttachment, ...]]:
    """Join rendered prompt parts, naming each reference once for this request."""

    namer = MediaNamer()
    rendered: list[str] = []
    for part in parts:
        rendered.append(namer.render(part) if isinstance(part, Reference) else part)
    return "".join(rendered), namer.attachments


def media_summaries(attachments: Sequence[MediaAttachment]) -> tuple[MediaSummary, ...]:
    return tuple(attachment.summary for attachment in attachments)


def media_digest_material(attachments: Sequence[MediaAttachment]) -> list[list[str]]:
    """Stable, byte-free identity of a prompt's attachments for digests and fingerprints."""

    return [list(attachment.identity) for attachment in attachments]


def unsent_attachments(
    attachments: Sequence[MediaAttachment],
    history: Mapping[str, tuple[str, ...]] | None,
) -> tuple[MediaAttachment, ...]:
    """Drop attachments the history being sent already holds under the same name.

    ``history`` maps each attachment name to the identity of its latest attachment in the
    replayed history, or is ``None`` when the replayed history cannot be trusted to keep
    them (for example with a custom history processor or native compaction).
    """

    if not history:
        return tuple(attachments)
    return tuple(
        attachment
        for attachment in attachments
        if history.get(attachment.name) != attachment.identity
    )


def history_keeps_media(
    conversation: Any, *, processor: Any, archive: Any, compaction: Any
) -> bool:
    """Whether replayed history reaches the model intact, so its attachments can be reused.

    A history processor, history archiving or native compaction may drop earlier messages
    without Kedi knowing, so their attachments are sent again.
    """

    from kedi.agent_adapter.conversation import AdapterConversationContext

    return (
        isinstance(conversation, AdapterConversationContext)
        and processor is None
        and archive is None
        and not (compaction is not None and compaction.mode == "native")
    )


def media_provider(system: str | None) -> MediaProvider | None:
    """Map a model system or client family to the providers Kedi checks media for."""

    if system is None:
        return None
    normalized = system.lower()
    if normalized in {"openai", "azure"}:
        return "openai"
    if normalized == "anthropic":
        return "anthropic"
    if normalized.startswith("google") or normalized in {"gemini", "vertexai"}:
        return "google"
    if normalized.startswith("bedrock") or normalized == "aws":
        return "bedrock"
    if normalized == "typesafe":
        return "typesafe"
    if normalized == "laya":
        return "laya"
    return None


def check_media_support(
    attachments: Sequence[MediaAttachment],
    *,
    provider: MediaProvider | None,
    model: str | None,
) -> None:
    """Fail before model I/O when a known provider cannot take an attachment."""

    for attachment in attachments:
        media = attachment.media
        if attachment.is_text:
            continue
        if media.is_youtube and provider != "google":
            raise MediaError(
                f"{attachment.name}: YouTube URLs are read only by Gemini models; "
                f"{_model_text(model)} cannot open {media.url}"
            )
        if provider is None:
            continue
        if provider == "laya" or (provider == "typesafe" and media.kind != "image"):
            accepted = "text" if provider == "laya" else "images and text documents"
            raise MediaError(
                f"{attachment.name}: {_provider_text(provider)} decision models read only "
                f"{accepted}; {_model_text(model)} cannot read {media.label}"
            )
        if provider == "typesafe":
            _check_jev_image(attachment)
        if media.kind == "video" and provider in {"openai", "anthropic"}:
            raise MediaError(
                f"{attachment.name}: {_provider_text(provider)} models do not accept video; "
                f"{_model_text(model)} cannot read {media.label}"
            )
        if media.kind == "audio" and provider in {"anthropic", "bedrock"}:
            raise MediaError(
                f"{attachment.name}: {_provider_text(provider)} models do not accept audio; "
                f"{_model_text(model)} cannot read {media.label}"
            )
        if (
            media.kind == "audio"
            and provider == "openai"
            and media.media_type not in {"audio/wav", "audio/mpeg", None}
        ):
            raise MediaError(
                f"{attachment.name}: OpenAI accepts only WAV and MP3 audio, not {media.media_type}"
            )


# TypeSafe's limits for image input: https://docs.typesafe.ai/images
_JEV_IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})
_JEV_IMAGE_BYTES = 16 * 1024 * 1024
_JEV_IMAGE_SIDES = (280, 1568)


def _check_jev_image(attachment: MediaAttachment) -> None:
    media = attachment.media
    if media.media_type is not None and media.media_type not in _JEV_IMAGE_TYPES:
        raise MediaError(
            f"{attachment.name}: Jev reads PNG and JPEG images, not {media.media_type}"
        )
    if media.data is None:
        return
    if len(media.data) > _JEV_IMAGE_BYTES:
        raise MediaError(f"{attachment.name}: Jev images are limited to 16 MiB")
    size = image_size(media.data)
    low, high = _JEV_IMAGE_SIDES
    if size is not None and not all(low <= side <= high for side in size):
        width, height = size
        raise MediaError(
            f"{attachment.name}: Jev needs each image side between {low} and {high} pixels, "
            f"not {width}x{height}"
        )


def image_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG or JPEG header; None for other or unreadable data."""

    if data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            return None
        marker = data[index + 1]
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return width, height
        index += 2 + length
    return None


async def download_image_urls(
    attachments: Sequence[MediaAttachment],
) -> list[MediaAttachment]:
    """Download URL images for a model that reads image data only (Jev)."""
    return [
        replace(item, media=await download_media(item.media))
        if item.media.kind == "image" and item.media.url is not None and not item.media.is_youtube
        else item
        for item in attachments
    ]


async def download_media(media: Media) -> Media:
    """Fetch an http(s) media URL as data through Pydantic AI's SSRF-protected download."""

    from pydantic_ai.messages import AudioUrl, DocumentUrl, ImageUrl, VideoUrl
    from pydantic_ai.models import download_item

    if media.url is None:
        return media
    if media.is_youtube:
        raise MediaError(f"{media.url}: YouTube URLs are never downloaded; only Gemini reads them")
    if urlparse(media.url).scheme.lower() not in {"http", "https"}:
        raise MediaError(f"{media.url}: Kedi downloads only http(s) media URLs")
    url_type: Any = {
        "image": ImageUrl,
        "audio": AudioUrl,
        "video": VideoUrl,
        "document": DocumentUrl,
    }[media.kind]
    item = (
        url_type(media.url)
        if media.media_type is None
        else url_type(media.url, media_type=media.media_type)
    )
    downloaded = await download_item(item, data_format="bytes")
    return type(media)(
        media_type=downloaded["data_type"],
        data=downloaded["data"],
        name=media.name,
        filename=media.url,
    )


def load_text_documents(attachments: Sequence[MediaAttachment]) -> tuple[MediaAttachment, ...]:
    """Download URL text documents: they reach every model as text, never as a URL."""

    if not any(item.media.is_text and item.media.data is None for item in attachments):
        return tuple(attachments)
    from kedi.agent_adapter.execution import run_sync

    async def load() -> tuple[MediaAttachment, ...]:
        loaded = []
        for attachment in attachments:
            media = attachment.media
            if media.is_text and media.data is None:
                attachment = replace(attachment, media=await download_media(media))
            loaded.append(attachment)
        return tuple(loaded)

    return run_sync(load())


def _provider_text(provider: MediaProvider) -> str:
    return {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Gemini",
        "bedrock": "Bedrock",
        "typesafe": "Jev",
        "laya": "Laya",
    }[provider]


def _model_text(model: str | None) -> str:
    return f"model {model}" if model else "the model"


__all__ = [
    "MediaAttachment",
    "ATTACHMENTS_CLOSE",
    "ATTACHMENTS_OPEN",
    "MediaNamer",
    "MediaProvider",
    "MediaSummary",
    "attachment_content",
    "check_media_support",
    "download_image_urls",
    "image_size",
    "compose_prompt",
    "history_keeps_media",
    "media_digest_material",
    "media_provider",
    "media_summaries",
    "scan_sent_content",
    "unsent_attachments",
]
