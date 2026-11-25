"""LangChain standard content blocks for prompts with attachments.

One human message carries the ``<attachments>`` element (see ``kedi.agent_adapter.request.media``)
and then the prompt text as standard blocks; each binary attachment is a media block inside its
tag.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from kedi.agent_adapter.request.media import (
    MediaAttachment,
    MediaProvider,
    attachment_content,
    check_media_support,
    download_image_urls,
    download_media,
    gemini_media_resolution,
    gemini_settings_key,
    gemini_video_metadata,
    scan_sent_content,
    unsent_attachments,
)
from kedi.media import Media, MediaError, Video, media_kind

if TYPE_CHECKING:
    from langchain_core.messages import HumanMessage

_SOURCE_IDENTITIES = "kedi_media_sources"

_BLOCK_TYPES = {"image": "image", "audio": "audio", "video": "video", "document": "file"}
_PROFILE_FLAGS = {
    "image": "image_inputs",
    "audio": "audio_inputs",
    "video": "video_inputs",
    "document": "pdf_inputs",
}
_CLIENT_PROVIDERS: Mapping[str, MediaProvider] = {
    "ChatOpenAI": "openai",
    "AzureChatOpenAI": "openai",
    "ChatAnthropic": "anthropic",
    "ChatAnthropicVertex": "anthropic",
    "ChatGoogleGenerativeAI": "google",
    "ChatVertexAI": "google",
    "ChatBedrockConverse": "bedrock",
    "ChatBedrock": "bedrock",
    "TypeSafeChatModel": "typesafe",
    "LayaChatModel": "laya",
}


async def user_message(
    prompt: str,
    attachments: Sequence[MediaAttachment],
    *,
    history: Sequence[Any] | None,
    model: Any,
) -> HumanMessage:
    """Build a human message with replayable source identities for lowered media.

    ``history`` is the replayed message history when it is known to reach the model intact;
    attachments it already holds under the same name and content are not sent again.
    """

    from langchain_core.messages import HumanMessage

    provider = langchain_media_provider(model)
    model_name = langchain_model_name(model)
    if provider == "typesafe":
        attachments = await download_image_urls(attachments)
    check_media_support(attachments, provider=provider, model=model_name)
    _check_profile(attachments, model, model_name)
    pending = unsent_attachments(
        attachments, None if history is None else sent_attachments(history)
    )
    if not pending:
        return HumanMessage(content=prompt)
    content: list[str | dict[str, Any]] = [
        {"type": "text", "text": piece}
        if isinstance(piece, str)
        else await _media_block(piece, model)
        for piece in attachment_content(prompt, pending)
    ]
    message = HumanMessage(content=content)
    lowered = sent_attachments([message])
    # Downloading changes a URL identity to a byte digest. Bind its source identity
    # to the media still present in this message, outside model-visible content.
    sources = {
        item.name: {"source": list(item.identity), "lowered": list(lowered[item.name])}
        for item in pending
        if item.name in lowered and item.identity != lowered[item.name]
    }
    if sources:
        message.additional_kwargs[_SOURCE_IDENTITIES] = sources
    return message


def langchain_media_provider(model: Any) -> MediaProvider | None:
    """Recognize first-party chat model clients; OpenAI-compatible endpoints are unknown."""

    for cls in type(model).__mro__:
        provider = _CLIENT_PROVIDERS.get(cls.__name__)
        if provider is None:
            continue
        base_url = getattr(model, "openai_api_base", None)
        if provider == "openai" and base_url and urlparse(base_url).hostname != "api.openai.com":
            return None
        return provider
    return None


def langchain_model_name(model: Any) -> str | None:
    for attribute in ("model_name", "model", "model_id"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def sent_attachments(messages: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """Map each attachment name to its latest identity in replayed human messages."""

    from langchain_core.messages import HumanMessage

    index: dict[str, tuple[str, ...]] = {}
    for message in messages:
        # A compaction checkpoint hides everything before it, so the scan starts
        # over there — same as the pydantic history scan.
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, Mapping) and isinstance(
            additional_kwargs.get("kedi_compaction_checkpoint"), str
        ):
            index.clear()
            continue
        if not isinstance(message, HumanMessage) or not isinstance(message.content, list):
            continue
        present: dict[str, tuple[str, ...]] = {}
        scan_sent_content(
            list(message.content), present, text_of=_block_text, identity_of=_block_identity
        )
        sources = message.additional_kwargs.get(_SOURCE_IDENTITIES, {})
        if isinstance(sources, Mapping):
            for name, identity in present.items():
                source = sources.get(name)
                if (
                    isinstance(source, Mapping)
                    and source.get("lowered") == list(identity)
                    and isinstance(source.get("source"), list)
                    and all(isinstance(part, str) for part in source["source"])
                ):
                    present[name] = tuple(source["source"])
        index.update(present)
    return index


def bedrock_document_name(name: str) -> str:
    """Bedrock document names allow letters, digits, single spaces, hyphens and brackets."""

    cleaned = re.sub(r"[^A-Za-z0-9\s\-()\[\]]", "-", name)
    return re.sub(r"\s+", " ", cleaned).strip() or "document"


def _check_profile(
    attachments: Sequence[MediaAttachment], model: Any, model_name: str | None
) -> None:
    profile = getattr(model, "profile", None)
    if not isinstance(profile, Mapping):
        return
    for attachment in attachments:
        if attachment.is_text:
            continue
        kind = attachment.media.kind
        if profile.get(_PROFILE_FLAGS[kind]) is False:
            subject = f"model {model_name}" if model_name else "the model"
            label = "PDF" if kind == "document" else kind
            raise MediaError(f"{attachment.name}: {subject} does not accept {label} input")


async def _media_block(attachment: MediaAttachment, model: Any) -> dict[str, Any]:
    media = attachment.media
    if media.url is not None and not _sends_url(media, model):
        media = await download_media(media)
    if isinstance(media, Video) and media.has_settings:
        return _gemini_video_block(media)
    block: dict[str, Any] = {"type": _BLOCK_TYPES[media.kind]}
    if media.data is not None:
        block["base64"] = base64.b64encode(media.data).decode("ascii")
    elif media.url is not None:
        block["url"] = media.url
    else:
        block["file_id"] = media.file_id
    if media.media_type is not None:
        block["mime_type"] = media.media_type
    if media.kind == "document":
        # The real file name never reaches the model; providers that require one get the
        # Kedi name. OpenAI reads ``filename``, Bedrock reads ``name``.
        block["extras"] = {
            "filename": f"{attachment.name}.pdf",
            "name": bedrock_document_name(attachment.name),
        }
    return block


def _gemini_video_block(media: Video) -> dict[str, Any]:
    """langchain-google-genai reads clip settings only from its own ``media`` block.

    check_media_support admits a video with settings only for Gemini models.
    """

    block: dict[str, Any] = {"type": "media", "mime_type": media.media_type or "video/mp4"}
    if media.data is not None:
        block["data"] = base64.b64encode(media.data).decode("ascii")
    else:
        block["file_uri"] = media.url if media.url is not None else media.file_id
    metadata = gemini_video_metadata(media)
    if metadata:
        block["video_metadata"] = metadata
    resolution = gemini_media_resolution(media)
    if resolution is not None:
        block["media_resolution"] = resolution
    return block


def _sends_url(media: Media, model: Any) -> bool:
    """Send a URL as is when the model can open it; Kedi downloads the rest."""

    if media.is_youtube or urlparse(media.url or "").scheme.lower() in {"s3", "gs"}:
        return True
    profile = getattr(model, "profile", None)
    return (
        media.kind == "image"
        and isinstance(profile, Mapping)
        and profile.get("image_url_inputs") is True
    )


def _block_text(block: Any) -> str | None:
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping) and block.get("type") == "text":
        text = block.get("text")
        return text if isinstance(text, str) else None
    return None


def _block_identity(block: Any) -> tuple[str, ...] | None:
    if not isinstance(block, Mapping):
        return None
    block_type = block.get("type")
    if block_type == "media":
        return _gemini_block_identity(block)
    kinds = {"image": "image", "audio": "audio", "video": "video", "file": "document"}
    kind = kinds.get(str(block_type))
    if kind is None:
        return None
    if isinstance(block.get("base64"), str):
        try:
            data = base64.b64decode(block["base64"], validate=True)
        except ValueError:
            return None
        mime_type = block.get("mime_type")
        if isinstance(mime_type, str):
            kind = media_kind(mime_type)
        return (kind, "sha256", _sha256(data))
    if isinstance(block.get("url"), str):
        return (kind, "url", block["url"])
    if isinstance(block.get("file_id"), str):
        return (kind, "file_id", block["file_id"])
    return None


def _gemini_block_identity(block: Mapping[str, Any]) -> tuple[str, ...] | None:
    mime_type = block.get("mime_type")
    kind = media_kind(mime_type) if isinstance(mime_type, str) else "video"
    if isinstance(block.get("data"), str):
        try:
            source: tuple[str, ...] = (
                "sha256",
                _sha256(base64.b64decode(block["data"], validate=True)),
            )
        except ValueError:
            return None
    elif isinstance(block.get("file_uri"), str):
        # A file id sent as a file_uri is mapped back through the message's source identities.
        source = ("url", block["file_uri"])
    else:
        return None
    settings = gemini_settings_key(block.get("video_metadata"), block.get("media_resolution"))
    return (kind, *source, *settings)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "bedrock_document_name",
    "langchain_media_provider",
    "langchain_model_name",
    "sent_attachments",
    "user_message",
]
