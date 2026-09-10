from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

from pydantic import BaseModel

from .errors import ArtifactInputError, ArtifactSerializationError

JsonPointerRange = tuple[int, int, int, int]
_MAX_JSON_POINTER_INDEX_ENTRIES = 100_000


@dataclass(frozen=True)
class EncodedJson:
    payload: bytes
    pointer_ranges: dict[str, JsonPointerRange]
    index_complete: bool


@dataclass(frozen=True)
class SerializedArtifact:
    codec: str
    media_type: str
    logical_type: str
    size_bytes: int
    char_count: int | None
    preview: str
    summary: str
    payload_bytes: bytes | None
    memory_value: Any
    persistable: bool
    digest: str | None
    json_pointer_ranges: dict[str, JsonPointerRange] | None = None
    json_pointer_index_complete: bool = True


def serialize_artifact(
    value: Any,
    *,
    preview_chars: int,
    require_persistable: bool = False,
) -> SerializedArtifact:
    logical_type = _logical_type_name(value)
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return _serialized_text(
            value,
            payload,
            codec="text",
            media_type="text/plain; charset=utf-8",
            logical_type=logical_type,
            preview_chars=preview_chars,
            memory_value=value,
        )
    if isinstance(value, bytes):
        preview = base64.b64encode(value[: _binary_preview_bytes(preview_chars)]).decode("ascii")
        return SerializedArtifact(
            codec="bytes",
            media_type="application/octet-stream",
            logical_type=logical_type,
            size_bytes=len(value),
            char_count=None,
            preview=preview,
            summary=f"{len(value)} bytes of binary data",
            payload_bytes=value,
            memory_value=value,
            persistable=True,
            digest=_digest(value),
        )

    json_value = _json_value(value)
    encoded_json: EncodedJson | None = None
    if json_value is not _UNSUPPORTED:
        try:
            encoded_json = encode_json_with_pointers(json_value)
        except (TypeError, ValueError):
            pass
    if encoded_json is not None:
        payload = encoded_json.payload
        rendered = payload.decode("utf-8")
        return _serialized_text(
            rendered,
            payload,
            codec="json",
            media_type="application/json",
            logical_type=logical_type,
            preview_chars=preview_chars,
            memory_value=_snapshot(value),
            json_pointer_ranges=encoded_json.pointer_ranges,
            json_pointer_index_complete=encoded_json.index_complete,
        )

    if require_persistable:
        raise ArtifactSerializationError(
            f"{logical_type} cannot be safely stored in a file artifact store"
        )

    preview = _safe_repr(value, preview_chars)
    size = estimate_object_size(value)
    return SerializedArtifact(
        codec="opaque",
        media_type="application/x-python-object",
        logical_type=logical_type,
        size_bytes=size,
        char_count=None,
        preview=preview,
        summary=f"{logical_type} Python object ({size} estimated bytes)",
        payload_bytes=None,
        memory_value=value,
        persistable=False,
        digest=None,
    )


def deserialize_artifact(codec: str, payload: bytes) -> Any:
    if codec == "text":
        return payload.decode("utf-8")
    if codec == "bytes":
        return payload
    if codec == "json":
        return json.loads(payload)
    raise ArtifactSerializationError(f"unsupported persisted artifact codec {codec!r}")


def render_artifact_content(codec: str, value: Any) -> str:
    if codec == "text":
        return value
    if codec == "bytes":
        return base64.b64encode(value).decode("ascii")
    if codec == "json":
        json_value = _json_value(value)
        if json_value is _UNSUPPORTED:
            raise ArtifactSerializationError("JSON artifact lost its serializable form")
        return json.dumps(json_value, ensure_ascii=False, indent=2, sort_keys=True)
    return _safe_repr(value, max_chars=None)


def encode_json_with_pointers(
    value: Any,
    *,
    root_pointer: str = "",
    max_entries: int = _MAX_JSON_POINTER_INDEX_ENTRIES,
) -> EncodedJson:
    """Encode canonical JSON while recording RFC 6901 value ranges."""

    if max_entries <= 0:
        raise ValueError("JSON pointer index max_entries must be positive")
    payload = bytearray()
    pointer_ranges: dict[str, JsonPointerRange] = {}
    char_position = 0
    index_complete = True

    def append(text: str) -> None:
        nonlocal char_position
        payload.extend(text.encode("utf-8"))
        char_position += len(text)

    def encode(item: Any, pointer: str) -> None:
        nonlocal index_complete
        byte_start = len(payload)
        char_start = char_position
        indexed = pointer in pointer_ranges or len(pointer_ranges) < max_entries
        if indexed:
            pointer_ranges[pointer] = (byte_start, byte_start, char_start, char_start)
        else:
            index_complete = False

        if isinstance(item, dict):
            append("{")
            for position, key in enumerate(sorted(item)):
                if position:
                    append(",")
                append(_encode_json_scalar(key))
                append(":")
                encode(item[key], f"{pointer}/{_escape_json_pointer_part(key)}")
            append("}")
        elif isinstance(item, list):
            append("[")
            for position, child in enumerate(item):
                if position:
                    append(",")
                encode(child, f"{pointer}/{position}")
            append("]")
        else:
            append(_encode_json_scalar(item))

        if indexed:
            pointer_ranges[pointer] = (
                byte_start,
                len(payload),
                char_start,
                char_position,
            )

    encode(value, root_pointer)
    return EncodedJson(
        payload=bytes(payload),
        pointer_ranges=pointer_ranges,
        index_complete=index_complete,
    )


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ArtifactInputError("artifact JSON path must be an RFC 6901 pointer")
    current = value
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            try:
                current = current[part]
            except KeyError as exc:
                raise ArtifactInputError(f"artifact JSON path {pointer!r} was not found") from exc
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ArtifactInputError(f"artifact JSON path {pointer!r} was not found") from exc
        else:
            raise ArtifactInputError(
                f"artifact JSON path cannot descend through {type(current).__name__}"
            )
    return current


def estimate_object_size(value: Any, *, max_nodes: int = 10_000) -> int:
    seen: set[int] = set()
    remaining = max_nodes
    exhausted = False

    def visit(item: Any) -> int:
        nonlocal exhausted, remaining
        if remaining <= 0:
            exhausted = True
            return 0
        remaining -= 1
        item_id = id(item)
        if item_id in seen:
            return 0
        seen.add(item_id)
        size = sys.getsizeof(item)
        if isinstance(item, Mapping):
            return size + sum(visit(key) + visit(val) for key, val in item.items())
        if isinstance(item, (list, tuple, set, frozenset)):
            return size + sum(visit(child) for child in item)
        if isinstance(item, BaseModel):
            return size + visit(item.__dict__)
        if is_dataclass(item) and not isinstance(item, type):
            return size + sum(visit(getattr(item, item_field.name)) for item_field in fields(item))
        return size

    size = visit(value)
    return sys.maxsize if exhausted else size


class _Unsupported:
    pass


_UNSUPPORTED = _Unsupported()


def _json_value(value: Any, *, _active: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    active = set() if _active is None else _active
    value_id = id(value)
    if value_id in active:
        return _UNSUPPORTED
    if isinstance(value, BaseModel):
        try:
            dumped = value.model_dump(mode="json")
        except (TypeError, ValueError, RecursionError):
            return _UNSUPPORTED
        return _json_value(dumped, _active=active)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            dumped = asdict(value)
        except (TypeError, ValueError, RecursionError):
            return _UNSUPPORTED
        return _json_value(dumped, _active=active)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            return _UNSUPPORTED
        active.add(value_id)
        try:
            converted: dict[str, Any] = {}
            for key, item in value.items():
                child = _json_value(item, _active=active)
                if child is _UNSUPPORTED:
                    return _UNSUPPORTED
                converted[key] = child
            return converted
        finally:
            active.remove(value_id)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        active.add(value_id)
        try:
            converted_items: list[Any] = []
            for item in value:
                child = _json_value(item, _active=active)
                if child is _UNSUPPORTED:
                    return _UNSUPPORTED
                converted_items.append(child)
            return converted_items
        finally:
            active.remove(value_id)
    return _UNSUPPORTED


def _snapshot(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    return copy.deepcopy(value)


def _serialized_text(
    rendered: str,
    payload: bytes,
    *,
    codec: str,
    media_type: str,
    logical_type: str,
    preview_chars: int,
    memory_value: Any,
    json_pointer_ranges: dict[str, JsonPointerRange] | None = None,
    json_pointer_index_complete: bool = True,
) -> SerializedArtifact:
    preview = _truncate(rendered, preview_chars)
    return SerializedArtifact(
        codec=codec,
        media_type=media_type,
        logical_type=logical_type,
        size_bytes=len(payload),
        char_count=len(rendered),
        preview=preview,
        summary=f"{logical_type} value with {len(rendered)} characters",
        payload_bytes=payload,
        memory_value=memory_value,
        persistable=True,
        digest=_digest(payload),
        json_pointer_ranges=json_pointer_ranges,
        json_pointer_index_complete=json_pointer_index_complete,
    )


def _safe_repr(value: Any, max_chars: int | None) -> str:
    rendered = f"<opaque {_logical_type_name(value)}>"
    return _truncate(rendered, max_chars) if max_chars is not None else rendered


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return value[:max_chars]
    return value[: max_chars - 1] + "\u2026"


def _binary_preview_bytes(preview_chars: int) -> int:
    return max(1, preview_chars * 3 // 4)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _logical_type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _encode_json_scalar(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _escape_json_pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
