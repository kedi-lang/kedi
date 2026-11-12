"""Pydantic AI user content for prompts with media attachments.

One user message carries ``[label, media]* + [prompt text]``: each binary attachment is
introduced by a ``name:`` text item, and text documents arrive as inlined-file text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, cast

from kedi.agent_adapter.request.media import (
    MediaAttachment,
    check_media_support,
    download_image_urls,
    media_provider,
    unsent_attachments,
)
from kedi.media import media_kind

_INLINE_FILE = re.compile(
    r'-----BEGIN FILE id="([^"]+)" type="([^"]+)"-----\n(.*?)\n-----END FILE id="\1"-----',
    re.DOTALL,
)


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
    content: list[Any] = []
    for attachment in pending:
        if attachment.is_text:
            content.append(attachment.inline_text())
            continue
        content.append(attachment.label())
        content.append(_native_content(attachment))
    content.append(prompt)
    return content


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
                _scan_content(list(part.content), index)
    return index


def _scan_content(items: list[Any], index: dict[str, tuple[str, ...]]) -> None:
    for position, item in enumerate(items):
        if not isinstance(item, str):
            continue
        for name, _media_type, text in _INLINE_FILE.findall(item):
            index[name] = (name, "document", "sha256", _sha256(text.encode("utf-8")))
        following = items[position + 1] if position + 1 < len(items) else None
        identity = None if following is None else _content_identity(following)
        if identity is not None and item.endswith(":") and item[:-1].isidentifier():
            index[item[:-1]] = (item[:-1], *identity)


def _content_identity(item: Any) -> tuple[str, ...] | None:
    from pydantic_ai.messages import (
        AudioUrl,
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        UploadedFile,
        VideoUrl,
    )

    if isinstance(item, BinaryContent):
        return (media_kind(item.media_type), "sha256", _sha256(item.data))
    url_kinds = {ImageUrl: "image", AudioUrl: "audio", VideoUrl: "video", DocumentUrl: "document"}
    for url_type, kind in url_kinds.items():
        if isinstance(item, url_type):
            return (kind, "url", item.url)
    if isinstance(item, UploadedFile):
        return (media_kind(item.media_type), "file_id", item.provider_name, item.file_id)
    return None


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
    if media.data is not None:
        return BinaryContent.narrow_type(
            BinaryContent(media.data, media_type=media.media_type or "", identifier=attachment.name)
        )
    if media.url is not None:
        url_type: Any = {
            "image": ImageUrl,
            "audio": AudioUrl,
            "video": VideoUrl,
            "document": DocumentUrl,
        }[media.kind]
        if media.media_type is None:
            return url_type(media.url, identifier=attachment.name)
        return url_type(media.url, media_type=media.media_type, identifier=attachment.name)
    assert media.file_id is not None
    return UploadedFile(
        media.file_id,
        cast(Any, media.provider),
        media_type=media.media_type,
        identifier=attachment.name,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = ["sent_attachments", "user_prompt"]
