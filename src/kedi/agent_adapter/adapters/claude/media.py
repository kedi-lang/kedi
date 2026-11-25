"""Prompt media for the Claude Agent SDK: Anthropic image and document content blocks."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Sequence
from typing import Any
from urllib.parse import urlparse

from kedi.agent_adapter.request.media import (
    MediaAttachment,
    attachment_content,
    check_media_limit,
    check_media_support,
)
from kedi.media import MediaError


def claude_prompt(
    prompt: str,
    attachments: Sequence[MediaAttachment],
    *,
    model: str | None,
    media_limit: int | None,
) -> str | AsyncIterator[dict[str, Any]]:
    """The prompt as a string, or as one user message whose content carries the attachments.

    The Claude Agent SDK takes content blocks only through a streamed user message.
    """

    if not attachments:
        return prompt
    check_media_support(attachments, provider="anthropic", model=model)
    check_media_limit(attachments, media_limit)
    content = [
        {"type": "text", "text": piece} if isinstance(piece, str) else _content_block(piece)
        for piece in attachment_content(prompt, attachments)
    ]

    async def messages() -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
        }

    return messages()


def _content_block(attachment: MediaAttachment) -> dict[str, Any]:
    # check_media_support refuses audio and video for Anthropic, and text documents are
    # inline text, so only images and PDFs reach here.
    media = attachment.media
    block_type = "image" if media.kind == "image" else "document"
    if media.data is not None:
        source: dict[str, Any] = {
            "type": "base64",
            "media_type": media.media_type,
            "data": base64.b64encode(media.data).decode("ascii"),
        }
    elif media.url is not None and urlparse(media.url).scheme.lower() in {"http", "https"}:
        source = {"type": "url", "url": media.url}
    else:
        held = "an uploaded file" if media.file_id is not None else media.url
        raise MediaError(
            f"{attachment.name}: the Claude Agent SDK takes media as data or an http(s) URL, "
            f"not {held}"
        )
    return {"type": block_type, "source": source}


__all__ = ["claude_prompt"]
