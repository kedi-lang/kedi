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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from kedi.media import (
    Audio,
    Media,
    MediaError,
    MediaKind,
    Reference,
    Video,
    format_seconds,
    format_size,
    video_settings_key,
)

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
        if ref.layout is not None:
            # Media inside a value: each is attached under its path name, then the value goes
            # as JSON naming them (two mentions of one image share its first name).
            names = [
                self.name_for(item, preferred)
                for item, preferred in zip(ref.items, ref.layout.preferred, strict=True)
            ]
            document = ref.layout.document(names)
            return self.name_for(
                document, ref.preferred_name(1), generated=ref.generated_base, stored=ref.stored
            )
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

    def reserve(self, names: Iterable[str]) -> None:
        """Keep ``names`` (the prompt's attachments) from being given to other media."""

        self._names.update(names)

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


class RequestMedia(Mapping[str, Media]):
    """The media a model can name in its request: the prompt's attachments and the media its
    tools returned so far.

    Tool media are read on each lookup, so a name a tool gave after validation began still
    resolves.
    """

    def __init__(self, attachments: Sequence[MediaAttachment], namer: MediaNamer) -> None:
        self._attachments = tuple(attachments)
        self._namer = namer

    def _named(self) -> dict[str, Media]:
        named = {attachment.name: attachment.media for attachment in self._attachments}
        for attachment in self._namer.attachments:
            named.setdefault(attachment.name, attachment.media)
        return named

    def __getitem__(self, name: str) -> Media:
        return self._named()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._named())

    def __len__(self) -> int:
        return len(self._named())


def request_media() -> RequestMedia:
    """The current request's media by name; outside a request, only tool media of this scope."""

    from .context import current_agent_run_request, tool_media_namer

    request = current_agent_run_request()
    return RequestMedia(request.prompt.media if request is not None else (), tool_media_namer())


def media_validation_context(context: Any = None) -> Any:
    """``context`` with the current request's media names, to validate what a model wrote.

    A media-typed value a model gives (an output field or a tool argument) names one of its
    request's media (see :data:`kedi.media.MEDIA_NAMES`). A caller's own mapping context keeps
    its keys; any other context is returned unchanged.
    """

    from kedi.media import MEDIA_NAMES

    if context is None:
        return {MEDIA_NAMES: request_media()}
    if isinstance(context, Mapping):
        return {**context, MEDIA_NAMES: request_media()}
    return context


def require_media_input(adapter: Any) -> None:
    """Fail before any model I/O when ``adapter`` cannot take prompt media."""

    if not adapter.capabilities.supports("media_input"):
        raise MediaError(
            f"Adapter {adapter.shortname!r} does not accept media input; "
            "Pydantic AI, LangChain, the Claude Agent SDK, Codex, ACP and A2A take "
            "prompts with media"
        )


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


def check_media_limit(attachments: Sequence[MediaAttachment], limit: int | None) -> None:
    """Fail before model I/O when an attachment's bytes exceed the ``media_limit`` setting.

    URLs and provider file IDs have no local size and are not checked.
    """

    if limit is None:
        return
    for attachment in attachments:
        size = attachment.media.size
        if size is not None and size > limit:
            raise MediaError(
                f"{attachment.name}: {attachment.media.label} is larger than the "
                f"media_limit setting ({format_size(limit)})"
            )


OutputMediaKind = Literal["image", "audio"]

NamedAudio = Annotated[Audio, "an audio of the request, by name"]
"""``[x: Audio] <<`` when the model names an audio of its request instead of speaking."""


def output_media_kind(output_type: Any) -> OutputMediaKind | None:
    """The media a typed call asks the model itself to produce, if any.

    ``[img: Image] << Draw a cat.`` asks for an image, ``[speech: Audio] << Read this aloud.``
    for speech.
    """

    from kedi.media import Audio, Image

    if output_type is Image:
        return "image"
    if output_type is Audio:
        return "audio"
    return None


def require_media_output(
    adapter: Any, kind: OutputMediaKind, attachments: Sequence[MediaAttachment]
) -> None:
    """Fail before any model I/O when ``adapter`` cannot produce ``kind``.

    An adapter with image output decides per model whether the model draws the image or names
    one of its request. Any other adapter that validates structured output can only have the
    model name an attachment: its tool bridge sends text only.
    """

    if kind == "image":
        capabilities = adapter.capabilities
        if capabilities.supports("image_output"):
            return
        if not (
            capabilities.supports("structured_output") and capabilities.supports("media_input")
        ):
            raise MediaError(
                f"Adapter {adapter.shortname!r} cannot produce images; "
                "Pydantic AI and LangChain return images from image models"
            )
        if not any(attachment.media.kind == "image" for attachment in attachments):
            raise MediaError(
                f"Adapter {adapter.shortname!r} cannot produce images, and this request holds "
                "no image to name; attach the images to choose from, or use Pydantic AI or "
                "LangChain with an image model"
            )
        return
    if not adapter.capabilities.supports("audio_output"):
        raise MediaError(
            f"Adapter {adapter.shortname!r} cannot produce speech; Pydantic AI returns it from "
            "realtime models and LangChain from audio models"
        )


def require_audio_to_name(attachments: Sequence[MediaAttachment], *, tools: Iterable[Any]) -> None:
    """Fail before the request when ``[x: Audio] <<`` names an audio and none could be named.

    The request has attachments, so it does not speak: a speech request sends text only.
    """

    if tools_return(tools, "audio") or any(item.media.kind == "audio" for item in attachments):
        return
    names = ", ".join(item.name for item in attachments)
    raise MediaError(
        f"{names}: a speech request sends text only, and no attachment is audio the model could "
        "name; describe the media in an earlier request and speak its answer"
    )


RETURN_MEDIA_METADATA = "kedi_return_media"
"""Tool metadata key: the media kinds a tool's declared return type can hold."""


def tools_return(tools: Iterable[Any], kind: MediaKind) -> bool:
    """Whether one of ``tools`` (tool specs) declares a return type that can hold ``kind``."""

    return any(
        kind in (getattr(tool, "metadata", None) or {}).get(RETURN_MEDIA_METADATA, ())
        for tool in tools
    )


def tools_may_return_media(tools: Iterable[Any]) -> bool:
    """Whether one of ``tools`` (tool specs) may return media for a model to name.

    Kedi's artifact tools return media only from archived history, which callers check
    separately.
    """

    return any(
        (getattr(tool, "metadata", None) or {}).get("kedi_tool_kind") != "artifact_management"
        for tool in tools
    )


def require_image_to_name(model: str, *, example: str) -> None:
    """Fail before the request when ``model`` cannot draw and its request holds no image to name.

    Called only when no tool could return one: the prompt's attachments are all it can name.
    """

    from .context import current_agent_run_request

    request = current_agent_run_request()
    if request is not None and any(item.media.kind == "image" for item in request.prompt.media):
        return
    raise MediaError(
        f"Model {model} cannot produce images; use an image model such as {example}, or give "
        "it images to choose from: attachments, or a tool that returns one"
    )


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
        if isinstance(media, Video) and media.has_settings:
            # Above the unknown-provider exit: a model that ignores the settings would
            # silently read the whole video instead of the clip.
            _check_video_settings(attachment, media, provider=provider, model=model)
        if media.is_youtube and provider != "google":
            raise MediaError(
                f"{attachment.name}: YouTube URLs are read only by Gemini models; "
                f"{_model_text(model)} cannot open {media.url}"
            )
        if provider is None:
            continue
        if provider == "laya":
            _check_laya_media(attachment, model)
            continue
        if provider == "typesafe" and media.kind != "image":
            raise MediaError(
                f"{attachment.name}: Jev decision models read only images and text documents; "
                f"{_model_text(model)} cannot read {media.label}"
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


LAYA_MEDIA_MODEL = "vega"
"""Laya's engine checkpoint, the one Laya model that reads images, audio and video."""


def _check_laya_media(attachment: MediaAttachment, model: str | None) -> None:
    media = attachment.media
    if model != LAYA_MEDIA_MODEL or media.kind not in {"image", "audio", "video"}:
        raise MediaError(
            f"{attachment.name}: Laya decision models read only text, except laya/vega, "
            f"which also reads images, audio and video; {_model_text(model)} cannot read "
            f"{media.label}"
        )
    if media.data is None:
        raise MediaError(
            f"{attachment.name}: Laya reads media data, not a URL or file ID; load "
            f"{media.label} as a file first"
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
        # A downloaded clip keeps its settings: the bytes replace only the source.
        **{name: getattr(media, name) for name in media._SETTINGS},
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


_GEMINI_BEFORE_3 = re.compile(r"(?:^|[/:])gemini-[12](?:[.-]|$)")
_GEMINI_RESOLUTIONS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}


def _check_video_settings(
    attachment: MediaAttachment,
    media: Video,
    *,
    provider: MediaProvider | None,
    model: str | None,
) -> None:
    if provider != "google":
        raise MediaError(
            f"{attachment.name}: video settings (start, end, fps, resolution) are read only by "
            f"Gemini models; {_model_text(model)} cannot use them"
        )
    if media.resolution is not None and model and _GEMINI_BEFORE_3.search(model.lower()):
        raise MediaError(
            f"{attachment.name}: video resolution needs Gemini 3 or later; "
            f"{_model_text(model)} ignores it"
        )


def gemini_video_metadata(media: Media) -> dict[str, Any]:
    """Gemini's ``video_metadata`` for a video's clip settings; empty without them."""

    if not isinstance(media, Video):
        return {}
    metadata: dict[str, Any] = {}
    if media.start is not None:
        metadata["start_offset"] = f"{format_seconds(media.start)}s"
    if media.end is not None:
        metadata["end_offset"] = f"{format_seconds(media.end)}s"
    if media.fps is not None:
        metadata["fps"] = media.fps
    return metadata


def gemini_media_resolution(media: Media) -> str | None:
    """Gemini's per-part media resolution level for a video, when it sets one."""

    if not isinstance(media, Video) or media.resolution is None:
        return None
    return _GEMINI_RESOLUTIONS[media.resolution]


def gemini_settings_key(metadata: Any, resolution: Any) -> tuple[str, ...]:
    """Rebuild a video's settings identity from the Gemini fields that carried it."""

    def seconds(value: Any) -> float | None:
        if isinstance(value, str) and value.endswith("s"):
            try:
                return float(value[:-1])
            except ValueError:
                return None
        return None

    fields = metadata if isinstance(metadata, Mapping) else {}
    if isinstance(resolution, Mapping):
        resolution = resolution.get("level")
    names = {level: name for name, level in _GEMINI_RESOLUTIONS.items()}
    fps = fields.get("fps")
    return video_settings_key(
        start=seconds(fields.get("start_offset")),
        end=seconds(fields.get("end_offset")),
        fps=float(fps) if isinstance(fps, (int, float)) and not isinstance(fps, bool) else None,
        resolution=names.get(resolution) if isinstance(resolution, str) else None,
    )


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
    "NamedAudio",
    "RETURN_MEDIA_METADATA",
    "ATTACHMENTS_CLOSE",
    "ATTACHMENTS_OPEN",
    "MediaNamer",
    "MediaProvider",
    "MediaSummary",
    "RequestMedia",
    "attachment_content",
    "check_media_support",
    "download_image_urls",
    "gemini_media_resolution",
    "gemini_settings_key",
    "gemini_video_metadata",
    "image_size",
    "output_media_kind",
    "require_audio_to_name",
    "require_image_to_name",
    "require_media_output",
    "tools_may_return_media",
    "tools_return",
    "compose_prompt",
    "history_keeps_media",
    "media_digest_material",
    "media_provider",
    "media_summaries",
    "media_validation_context",
    "request_media",
    "require_media_input",
    "scan_sent_content",
    "unsent_attachments",
]
