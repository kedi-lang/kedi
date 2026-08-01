from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .errors import ArtifactStreamError
from .serialization import JsonPointerRange, encode_json_with_pointers
from .stream import ArtifactStream, ArtifactStreamKind

_MAX_JSON_POINTER_INDEX_ENTRIES = 100_000


@dataclass(frozen=True)
class StreamedArtifact:
    codec: str
    media_type: str
    logical_type: str
    size_bytes: int
    char_count: int | None
    preview: str
    summary: str
    digest: str
    json_pointer_ranges: dict[str, JsonPointerRange] | None = None
    json_pointer_index_complete: bool = True


class ArtifactStreamEncoder:
    """Incrementally encode one ArtifactStream into its canonical payload."""

    def __init__(self, stream: ArtifactStream[Any], *, preview_chars: int) -> None:
        self.stream = stream
        self.preview_chars = preview_chars
        self._digest = hashlib.sha256()
        self._size_bytes = 0
        self._char_count = 0 if stream.kind is not ArtifactStreamKind.BYTES else None
        self._preview_text = ""
        self._binary_preview = bytearray()
        self._json_item_count = 0
        self._json_pointer_ranges: dict[str, JsonPointerRange] = {}
        self._json_pointer_index_complete = True
        self._finished = False

    @property
    def codec(self) -> str:
        if self.stream.kind is ArtifactStreamKind.TEXT:
            return "text"
        if self.stream.kind is ArtifactStreamKind.BYTES:
            return "bytes"
        return "json"

    def start(self) -> bytes:
        if self.stream.kind is ArtifactStreamKind.JSON_ITEMS:
            self._json_pointer_ranges[""] = (0, 0, 0, 0)
            return self._observe(b"[", rendered="[")
        return b""

    def encode(self, chunk: Any) -> bytes:
        if self._finished:
            raise RuntimeError("artifact stream encoder is already finished")
        if self.stream.kind is ArtifactStreamKind.TEXT:
            return self._observe(chunk.encode("utf-8"), rendered=chunk)
        if self.stream.kind is ArtifactStreamKind.BYTES:
            payload = bytes(chunk)
            remaining = max(
                0, _binary_preview_bytes(self.preview_chars) - len(self._binary_preview)
            )
            self._binary_preview.extend(payload[:remaining])
            return self._observe(payload)

        item_pointer = f"/{self._json_item_count}"
        prefix = "," if self._json_item_count else ""
        remaining_entries = _MAX_JSON_POINTER_INDEX_ENTRIES - len(self._json_pointer_ranges)
        if remaining_entries > 0:
            encoded = encode_json_with_pointers(
                chunk,
                root_pointer=item_pointer,
                max_entries=remaining_entries,
            )
            byte_base = self._size_bytes + len(prefix.encode("utf-8"))
            char_base = (self._char_count or 0) + len(prefix)
            for pointer, (
                byte_start,
                byte_end,
                char_start,
                char_end,
            ) in encoded.pointer_ranges.items():
                self._json_pointer_ranges[pointer] = (
                    byte_base + byte_start,
                    byte_base + byte_end,
                    char_base + char_start,
                    char_base + char_end,
                )
            if not encoded.index_complete:
                self._json_pointer_index_complete = False
            rendered = encoded.payload.decode("utf-8")
        else:
            self._json_pointer_index_complete = False
            rendered = json.dumps(
                chunk,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        self._json_item_count += 1
        text = prefix + rendered
        return self._observe(text.encode("utf-8"), rendered=text)

    def finish(self) -> tuple[bytes, StreamedArtifact]:
        if self._finished:
            raise RuntimeError("artifact stream encoder is already finished")
        self._finished = True
        tail = (
            self._observe(b"]", rendered="]")
            if self.stream.kind is ArtifactStreamKind.JSON_ITEMS
            else b""
        )
        if self.stream.kind is ArtifactStreamKind.JSON_ITEMS:
            self._json_pointer_ranges[""] = (
                0,
                self._size_bytes,
                0,
                self._char_count or 0,
            )
        if (
            self.stream.content_length is not None
            and self.stream.content_length != self._size_bytes
        ):
            raise ArtifactStreamError(
                "artifact stream content_length does not match encoded payload size "
                f"({self.stream.content_length} declared, {self._size_bytes} received)"
            )
        if self.stream.kind is ArtifactStreamKind.BYTES:
            preview = base64.b64encode(self._binary_preview).decode("ascii")
            summary = f"{self._size_bytes} bytes of binary data"
        else:
            preview = _finish_preview(
                self._preview_text,
                total_chars=self._char_count or 0,
                limit=self.preview_chars,
            )
            if self.stream.kind is ArtifactStreamKind.JSON_ITEMS:
                summary = (
                    f"{self.stream.logical_type} value with {self._json_item_count} JSON items"
                )
            else:
                summary = (
                    f"{self.stream.logical_type} value with {self._char_count or 0} characters"
                )
        return tail, StreamedArtifact(
            codec=self.codec,
            media_type=self.stream.media_type,
            logical_type=self.stream.logical_type,
            size_bytes=self._size_bytes,
            char_count=self._char_count,
            preview=preview,
            summary=summary,
            digest=self._digest.hexdigest(),
            json_pointer_ranges=(
                self._json_pointer_ranges
                if self.stream.kind is ArtifactStreamKind.JSON_ITEMS
                else None
            ),
            json_pointer_index_complete=self._json_pointer_index_complete,
        )

    def _observe(self, payload: bytes, *, rendered: str | None = None) -> bytes:
        self._digest.update(payload)
        self._size_bytes += len(payload)
        if rendered is not None:
            if self._char_count is None:
                raise RuntimeError("binary artifact stream cannot observe rendered text")
            self._char_count += len(rendered)
            remaining = max(0, self.preview_chars - len(self._preview_text))
            self._preview_text += rendered[:remaining]
        return payload


def materialize_stream_payload(streamed: StreamedArtifact, payload: bytes) -> Any:
    if streamed.codec == "text":
        return payload.decode("utf-8")
    if streamed.codec == "bytes":
        return payload
    if streamed.codec == "json":
        return json.loads(payload)
    raise RuntimeError(f"unsupported stream codec {streamed.codec!r}")


def _finish_preview(value: str, *, total_chars: int, limit: int) -> str:
    if total_chars <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1] + "\u2026"


def _binary_preview_bytes(preview_chars: int) -> int:
    return max(1, preview_chars * 3 // 4)
