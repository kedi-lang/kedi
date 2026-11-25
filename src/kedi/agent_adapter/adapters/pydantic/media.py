"""Pydantic AI user content for prompts with attachments.

One user message carries the ``<attachments>`` element (see ``kedi.agent_adapter.request.media``)
and then the prompt text; each binary attachment is a native item inside its tag.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, cast

from kedi.agent_adapter.request.media import (
    MediaAttachment,
    attachment_content,
    check_media_support,
    download_image_urls,
    gemini_media_resolution,
    gemini_settings_key,
    gemini_video_metadata,
    media_provider,
    scan_sent_content,
    unsent_attachments,
)
from kedi.media import media_kind


async def user_prompt(
    prompt: str,
    attachments: Sequence[MediaAttachment],
    *,
    history: Sequence[Any] | None,
    model: Any,
) -> str | list[Any]:
    """Build the user content for one run.

    ``history`` is the replayed message history when it is known to reach the model intact;
    attachments it already holds under the same name and content are not sent again.
    """

    provider = media_provider(getattr(model, "system", None))
    if provider == "typesafe":
        attachments = await download_image_urls(attachments)
    check_media_support(attachments, provider=provider, model=getattr(model, "model_name", None))
    pending = unsent_attachments(
        attachments, None if history is None else sent_attachments(history)
    )
    if not pending:
        return prompt
    return [
        piece if isinstance(piece, str) else _native_content(piece)
        for piece in attachment_content(prompt, pending)
    ]


def sent_attachments(messages: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    """Map each attachment name to its latest identity in replayed history.

    A compaction checkpoint hides everything before it, so the scan starts over there.
    """

    from pydantic_ai.messages import CompactionPart, ModelRequest, UserPromptPart

    index: dict[str, tuple[str, ...]] = {}
    for message in messages:
        metadata = getattr(message, "metadata", None) or {}
        if metadata.get("kedi_compaction_checkpoint") or any(
            isinstance(part, CompactionPart) for part in message.parts
        ):
            index.clear()
            continue
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                scan_sent_content(
                    list(part.content),
                    index,
                    text_of=lambda item: item if isinstance(item, str) else None,
                    identity_of=_content_identity,
                )
    return index


def _content_identity(item: Any) -> tuple[str, ...] | None:
    from pydantic_ai.messages import (
        AudioUrl,
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        UploadedFile,
        VideoUrl,
    )

    url_kinds = {ImageUrl: "image", AudioUrl: "audio", VideoUrl: "video", DocumentUrl: "document"}
    if isinstance(item, BinaryContent):
        identity = (media_kind(item.media_type), "sha256", _sha256(item.data))
    elif isinstance(item, UploadedFile):
        identity = (media_kind(item.media_type), "file_id", item.provider_name, item.file_id)
    elif isinstance(item, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl)):
        kind = next(kind for url_type, kind in url_kinds.items() if isinstance(item, url_type))
        identity = (kind, "url", item.url)
    else:
        return None
    metadata = dict(getattr(item, "vendor_metadata", None) or {})
    resolution = metadata.pop("media_resolution", None)
    return (*identity, *gemini_settings_key(metadata, resolution))


def _native_content(attachment: MediaAttachment) -> Any:
    from pydantic_ai.messages import (
        AudioUrl,
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        UploadedFile,
        VideoUrl,
    )

    media = attachment.media
    # Gemini clip settings travel as vendor metadata; check_media_support admits them only
    # for Gemini models.
    vendor_metadata: dict[str, Any] | None = gemini_video_metadata(media) or None
    resolution = gemini_media_resolution(media)
    if resolution is not None:
        vendor_metadata = {**(vendor_metadata or {}), "media_resolution": {"level": resolution}}
    if media.data is not None:
        return BinaryContent.narrow_type(
            BinaryContent(
                media.data,
                media_type=media.media_type or "",
                identifier=attachment.name,
                vendor_metadata=vendor_metadata,
            )
        )
    if media.url is not None:
        url_type: Any = {
            "image": ImageUrl,
            "audio": AudioUrl,
            "video": VideoUrl,
            "document": DocumentUrl,
        }[media.kind]
        if media.media_type is None:
            return url_type(media.url, identifier=attachment.name, vendor_metadata=vendor_metadata)
        return url_type(
            media.url,
            media_type=media.media_type,
            identifier=attachment.name,
            vendor_metadata=vendor_metadata,
        )
    assert media.file_id is not None
    return UploadedFile(
        media.file_id,
        cast(Any, media.provider),
        media_type=media.media_type,
        identifier=attachment.name,
        vendor_metadata=vendor_metadata,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["sent_attachments", "user_prompt"]
