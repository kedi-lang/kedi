"""Prompt media for Codex: images as turn input, local ones through files Kedi manages."""

from __future__ import annotations

import mimetypes
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kedi.agent_adapter.request.media import (
    MediaAttachment,
    attachment_content,
    check_media_limit,
    check_media_support,
)
from kedi.media import MediaError


@contextmanager
def codex_turn_input(
    prompt: str,
    attachments: Sequence[MediaAttachment],
    *,
    model: str | None,
    media_limit: int | None,
) -> Generator[list[dict[str, Any]], None, None]:
    """Codex ``turn/start`` input for a prompt and its attachments.

    Codex reads images only. Image data goes as ``localImage`` files in a directory that
    exists for the turn and is removed after it.
    """

    if not attachments:
        yield [{"type": "text", "text": prompt}]
        return
    for attachment in attachments:
        if not attachment.is_text and attachment.media.kind != "image":
            raise MediaError(
                f"{attachment.name}: Codex reads only images; "
                f"{f'model {model}' if model else 'the model'} cannot read "
                f"{attachment.media.label}"
            )
    check_media_support(attachments, provider="openai", model=model)
    check_media_limit(attachments, media_limit)
    with tempfile.TemporaryDirectory(prefix="kedi-codex-media-") as directory:
        yield [
            {"type": "text", "text": piece}
            if isinstance(piece, str)
            else _image_input(piece, Path(directory))
            for piece in attachment_content(prompt, attachments)
        ]


def _image_input(attachment: MediaAttachment, directory: Path) -> dict[str, Any]:
    media = attachment.media
    if media.data is not None:
        suffix = mimetypes.guess_extension(media.media_type or "") or ""
        path = directory / f"{attachment.name}{suffix}"
        path.write_bytes(media.data)
        return {"type": "localImage", "path": str(path)}
    if media.url is not None and urlparse(media.url).scheme.lower() in {"http", "https"}:
        return {"type": "image", "url": media.url}
    if media.file_id is not None and media.provider == "openai":
        return {"type": "image", "fileId": media.file_id}
    held = media.url or f"a {media.provider} file"
    raise MediaError(
        f"{attachment.name}: Codex takes images as data, an http(s) URL or an OpenAI file, not {held}"
    )


__all__ = ["codex_turn_input"]
