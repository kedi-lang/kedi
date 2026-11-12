"""LangChain standard content blocks for prompts with media attachments.

One human message carries ``[label, media]* + [prompt text]`` as standard blocks: each binary
attachment is introduced by a ``name:`` text block, and text documents arrive as
inlined-file text blocks.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from kedi.agent_adapter.request.media import (
    MediaAttachment,
    MediaProvider,
    check_media_support,
    download_media,
    unsent_attachments,
)
from kedi.media import Media, MediaError, media_kind

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
_INLINE_FILE = re.compile(
    r'-----BEGIN FILE id="([^"]+)" type="([^"]+)"-----\n(.*?)\n-----END FILE id="\1"-----',
    re.DOTALL,
)


async def user_content(
    prompt: str,
    attachments: Sequence[MediaAttachment],
    *,
    history: Sequence[Any] | None,
    model: Any,
) -> str | list[Any]:
    """Build the human message content for one run.

    ``history`` is the replayed message history when it is known to reach the model intact;
    attachments it already holds under the same name and content are not sent again.
    """

    provider = langchain_media_provider(model)
    model_name = langchain_model_name(model)
    check_media_support(attachments, provider=provider, model=model_name)
    _check_profile(attachments, model, model_name)
    pending = unsent_attachments(
        attachments, None if history is None else sent_attachments(history)
    )
    if not pending:
        return prompt
    blocks: list[dict[str, Any]] = []
    for attachment in pending:
        if attachment.is_text:
            blocks.append({"type": "text", "text": attachment.inline_text()})
            continue
        blocks.append({"type": "text", "text": attachment.label()})
        blocks.append(await _media_block(attachment, model))
    blocks.append({"type": "text", "text": prompt})
    return blocks


def langchain_media_provider(model: Any) -> MediaProvider | None:
    """Recognize first-party chat model clients; OpenAI-compatible endpoints are unknown."""

    for cls in type(model).__mro__:
        provider = _CLIENT_PROVIDERS.get(cls.__name__)
        if provider is None:
            continue
        base_url = getattr(model, "openai_api_base", None)
        if provider == "openai" and base_url and "api.openai.com" not in base_url:
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
        if not isinstance(message, HumanMessage) or not isinstance(message.content, list):
            continue
        blocks = list(message.content)
        for position, block in enumerate(blocks):
            text = _block_text(block)
            if text is None:
                continue
            for name, _media_type, inline in _INLINE_FILE.findall(text):
                index[name] = (name, "document", "sha256", _sha256(inline.encode("utf-8")))
            following = blocks[position + 1] if position + 1 < len(blocks) else None
            identity = None if following is None else _block_identity(following)
            if identity is not None and text.endswith(":") and text[:-1].isidentifier():
                index[text[:-1]] = (text[:-1], *identity)
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
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "bedrock_document_name",
    "langchain_media_provider",
    "langchain_model_name",
    "sent_attachments",
    "user_content",
]
