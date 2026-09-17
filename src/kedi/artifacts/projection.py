from __future__ import annotations

import base64
import codecs
import json
import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, NoReturn, cast

from kedi.artifacts.errors import ArtifactSerializationError
from kedi.artifacts.serialization import JsonPointerRange

_PROJECTION_INDEX_VERSION = 1


_UTF8_CHECKPOINT_INTERVAL = 4_096


@dataclass(frozen=True)
class _ProjectionIndex:
    utf8_checkpoints: tuple[tuple[int, int], ...]
    json_pointer_ranges: dict[str, JsonPointerRange]
    json_pointer_index_complete: bool


def _atomic_write(path: Path, payload: bytes, *, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _slice_projection(
    rendered: str,
    offset: int,
    offset_from: Literal["start", "end"],
    limit: int,
) -> tuple[str, int | None, bool]:
    start = _projection_start(len(rendered), offset, offset_from, limit)
    length = _projection_length(len(rendered), start, offset, offset_from, limit)
    content = rendered[start : start + length]
    return _projection_page(content, start, len(rendered), offset, offset_from)


def _projection_start(
    total_chars: int,
    offset: int,
    offset_from: Literal["start", "end"],
    limit: int,
) -> int:
    if offset_from == "start":
        return min(offset, total_chars)
    end = max(0, total_chars - offset)
    return max(0, end - limit)


def _projection_length(
    total_chars: int,
    start: int,
    offset: int,
    offset_from: Literal["start", "end"],
    limit: int,
) -> int:
    end = total_chars if offset_from == "start" else max(0, total_chars - offset)
    return min(limit, max(0, end - start))


def _projection_page(
    content: str,
    start: int,
    total_chars: int,
    offset: int,
    offset_from: Literal["start", "end"],
) -> tuple[str, int | None, bool]:
    if offset_from == "start":
        end = start + len(content)
        complete = end >= total_chars
        return content, None if complete else end, complete
    complete = start == 0
    return content, None if complete else offset + len(content), complete


def _count_utf8_chars(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        while chunk := stream.read(65_536):
            total += len(chunk)
    return total


def _read_utf8_chars(
    path: Path,
    *,
    start: int,
    limit: int,
    checkpoints: tuple[tuple[int, int], ...] | None = None,
) -> str:
    if limit <= 0:
        return ""
    checkpoints = checkpoints or ((0, 0),)
    checkpoint_index = max(0, bisect_right(checkpoints, (start, 2**63)) - 1)
    checkpoint_char, checkpoint_byte = checkpoints[checkpoint_index]
    skip = start - checkpoint_char
    required = skip + limit
    decoder = codecs.getincrementaldecoder("utf-8")()
    pieces: list[str] = []
    decoded_chars = 0
    try:
        with path.open("rb") as stream:
            stream.seek(checkpoint_byte)
            while decoded_chars < required:
                chunk = stream.read(65_536)
                if not chunk:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        pieces.append(tail)
                    break
                text = decoder.decode(chunk)
                if text:
                    pieces.append(text)
                    decoded_chars += len(text)
    except UnicodeDecodeError as exc:
        raise ArtifactSerializationError("artifact payload is not valid UTF-8") from exc
    return "".join(pieces)[skip:required]


def _read_base64_chars(path: Path, *, start: int, limit: int) -> str:
    end = start + limit
    position = 0
    pieces: list[str] = []
    with path.open("rb") as stream:
        while position < end:
            chunk = stream.read(49_152)
            if not chunk:
                break
            encoded = base64.b64encode(chunk).decode("ascii")
            chunk_end = position + len(encoded)
            if chunk_end > start:
                pieces.append(encoded[max(0, start - position) : max(0, end - position)])
            position = chunk_end
    return "".join(pieces)


def _write_projection_index(
    index_path: Path,
    payload_path: Path,
    *,
    json_pointer_ranges: dict[str, JsonPointerRange] | None,
    json_pointer_index_complete: bool,
) -> None:
    index = {
        "version": _PROJECTION_INDEX_VERSION,
        "utf8_checkpoints": _build_utf8_checkpoints(payload_path),
        "json_pointer_ranges": json_pointer_ranges or {},
        "json_pointer_index_complete": json_pointer_index_complete,
    }
    _atomic_write(
        index_path,
        json.dumps(index, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        ),
        mode=0o600,
    )


def _build_utf8_checkpoints(path: Path) -> tuple[tuple[int, int], ...]:
    checkpoints = [(0, 0)]
    byte_position = 0
    char_position = 0
    with path.open("rb") as stream:
        while chunk := stream.read(65_536):
            if chunk.isascii():
                next_checkpoint = (
                    _UTF8_CHECKPOINT_INTERVAL
                    if char_position == 0
                    else (
                        (char_position + _UTF8_CHECKPOINT_INTERVAL - 1) // _UTF8_CHECKPOINT_INTERVAL
                    )
                    * _UTF8_CHECKPOINT_INTERVAL
                )
                chunk_end = char_position + len(chunk)
                while next_checkpoint < chunk_end:
                    checkpoints.append(
                        (
                            next_checkpoint,
                            byte_position + next_checkpoint - char_position,
                        )
                    )
                    next_checkpoint += _UTF8_CHECKPOINT_INTERVAL
                char_position = chunk_end
                byte_position += len(chunk)
                continue
            for offset, byte in enumerate(chunk):
                if byte & 0xC0 == 0x80:
                    continue
                if char_position and char_position % _UTF8_CHECKPOINT_INTERVAL == 0:
                    checkpoints.append((char_position, byte_position + offset))
                char_position += 1
            byte_position += len(chunk)
    return tuple(checkpoints)


def _load_projection_index(path: Path) -> _ProjectionIndex:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != _PROJECTION_INDEX_VERSION:
            raise ValueError("unsupported projection index version")
        checkpoints = _validated_checkpoints(raw["utf8_checkpoints"])
        pointer_ranges = _validated_pointer_ranges(raw["json_pointer_ranges"])
        complete = raw["json_pointer_index_complete"]
        if not isinstance(complete, bool):
            raise TypeError("json_pointer_index_complete must be boolean")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactSerializationError(
            f"artifact projection index {path.name!r} is corrupt"
        ) from exc
    return _ProjectionIndex(
        utf8_checkpoints=checkpoints,
        json_pointer_ranges=pointer_ranges,
        json_pointer_index_complete=complete,
    )


def _validated_checkpoints(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise TypeError("utf8_checkpoints must be a non-empty list")
    checkpoints: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(position, int) and position >= 0 for position in item)
        ):
            raise TypeError("invalid UTF-8 checkpoint")
        positions = cast(list[int], item)
        checkpoint = (positions[0], positions[1])
        if checkpoints and (
            checkpoint[0] <= checkpoints[-1][0] or checkpoint[1] <= checkpoints[-1][1]
        ):
            raise ValueError("UTF-8 checkpoints must be strictly increasing")
        checkpoints.append(checkpoint)
    if checkpoints[0] != (0, 0):
        raise ValueError("UTF-8 checkpoints must start at zero")
    return tuple(checkpoints)


def _validated_pointer_ranges(value: object) -> dict[str, JsonPointerRange]:
    if not isinstance(value, dict):
        raise TypeError("json_pointer_ranges must be an object")
    ranges: dict[str, JsonPointerRange] = {}
    for pointer, raw_range in value.items():
        if (
            not isinstance(pointer, str)
            or not isinstance(raw_range, list)
            or len(raw_range) != 4
            or not all(isinstance(position, int) and position >= 0 for position in raw_range)
        ):
            raise TypeError("invalid JSON pointer range")
        byte_start, byte_end, char_start, char_end = cast(list[int], raw_range)
        if byte_end < byte_start or char_end < char_start:
            raise ValueError("JSON pointer ranges must be ordered")
        ranges[pointer] = (byte_start, byte_end, char_start, char_end)
    return ranges


def _locate_json_pointer_range(path: Path, pointer: str) -> JsonPointerRange | None:
    parts = (
        ()
        if pointer == ""
        else tuple(part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:])
    )
    with path.open("rb") as stream:
        locator = _JsonPointerLocator(stream, target=parts)
        result = locator.locate()
    return result


class _JsonPointerLocator:
    """Locate one value range in UTF-8 JSON without materializing the document."""

    def __init__(self, stream: BinaryIO, *, target: tuple[str, ...]) -> None:
        self._stream = stream
        self._target = target
        self._peeked: int | None = None
        self._byte_position = 0
        self._char_position = 0

    def locate(self) -> JsonPointerRange | None:
        result = self._parse_value(())
        if result is not None:
            return result
        self._skip_whitespace()
        if self._peek() != -1:
            self._invalid("unexpected trailing JSON content")
        return None

    def _parse_value(self, current_path: tuple[str, ...]) -> JsonPointerRange | None:
        self._skip_whitespace()
        byte_start = self._byte_position
        char_start = self._char_position
        marker = self._peek()
        if marker == ord("{"):
            nested = self._parse_object(current_path)
        elif marker == ord("["):
            nested = self._parse_array(current_path)
        elif marker == ord('"'):
            self._parse_string(decode=False)
            nested = None
        elif marker in {ord("t"), ord("f"), ord("n"), ord("-")} or ord("0") <= marker <= ord("9"):
            self._parse_literal()
            nested = None
        else:
            self._invalid("expected a JSON value")
        if nested is not None:
            return nested
        if current_path == self._target:
            return (
                byte_start,
                self._byte_position,
                char_start,
                self._char_position,
            )
        return None

    def _parse_object(self, current_path: tuple[str, ...]) -> JsonPointerRange | None:
        self._take(ord("{"))
        self._skip_whitespace()
        if self._peek() == ord("}"):
            self._take()
            return None
        while True:
            key = self._parse_string(decode=True)
            if key is None:
                self._invalid("expected a JSON object key")
            self._skip_whitespace()
            self._take(ord(":"))
            nested = self._parse_value((*current_path, key))
            if nested is not None:
                return nested
            self._skip_whitespace()
            marker = self._take()
            if marker == ord("}"):
                return None
            if marker != ord(","):
                self._invalid("expected ',' or '}' in JSON object")
            self._skip_whitespace()

    def _parse_array(self, current_path: tuple[str, ...]) -> JsonPointerRange | None:
        self._take(ord("["))
        self._skip_whitespace()
        if self._peek() == ord("]"):
            self._take()
            return None
        index = 0
        while True:
            nested = self._parse_value((*current_path, str(index)))
            if nested is not None:
                return nested
            index += 1
            self._skip_whitespace()
            marker = self._take()
            if marker == ord("]"):
                return None
            if marker != ord(","):
                self._invalid("expected ',' or ']' in JSON array")

    def _parse_string(self, *, decode: bool) -> str | None:
        raw = bytearray() if decode else None
        quote = self._take(ord('"'))
        if raw is not None:
            raw.append(quote)
        escaped = False
        while True:
            byte = self._take()
            if raw is not None:
                raw.append(byte)
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                break
        if raw is None:
            return None
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactSerializationError("artifact payload contains invalid JSON") from exc
        if not isinstance(decoded, str):
            self._invalid("JSON object key is not a string")
        return decoded

    def _parse_literal(self) -> None:
        consumed = False
        while self._peek() not in {
            -1,
            ord(" "),
            ord("\t"),
            ord("\r"),
            ord("\n"),
            ord(","),
            ord("]"),
            ord("}"),
        }:
            self._take()
            consumed = True
        if not consumed:
            self._invalid("expected a JSON literal")

    def _skip_whitespace(self) -> None:
        while self._peek() in {ord(" "), ord("\t"), ord("\r"), ord("\n")}:
            self._take()

    def _peek(self) -> int:
        if self._peeked is None:
            byte = self._stream.read(1)
            self._peeked = byte[0] if byte else -1
        return self._peeked

    def _take(self, expected: int | None = None) -> int:
        byte = self._peek()
        if byte == -1:
            self._invalid("unexpected end of JSON payload")
        if expected is not None and byte != expected:
            self._invalid(f"expected {chr(expected)!r} in JSON payload")
        self._peeked = None
        self._byte_position += 1
        if byte & 0xC0 != 0x80:
            self._char_position += 1
        return byte

    @staticmethod
    def _invalid(message: str) -> NoReturn:
        raise ArtifactSerializationError(message)
