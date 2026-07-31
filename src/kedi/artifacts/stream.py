from __future__ import annotations

import inspect
import json
import builtins
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator, Mapping
from enum import Enum
from functools import wraps
from types import UnionType
from threading import Lock
from typing import Annotated, Any, Callable, Generic, TypeVar, Union, cast, get_args, get_origin

from .errors import ArtifactStreamConsumedError, ArtifactStreamError

T = TypeVar("T")
ChunkT = TypeVar("ChunkT")


class ArtifactStreamKind(str, Enum):
    TEXT = "text"
    BYTES = "bytes"
    JSON_ITEMS = "json_items"


_RESERVED_METADATA_KEYS = frozenset(
    {
        "ref_id",
        "session_id",
        "owner",
        "tenant",
        "status",
        "size_bytes",
        "char_count",
        "created_at",
        "expires_at",
        "digest",
    }
)


class ArtifactStream(Generic[T]):
    """Explicit single-consumption transport for incrementally stored tool output."""

    def __init__(
        self,
        chunks: Iterable[Any] | AsyncIterable[Any],
        *,
        kind: ArtifactStreamKind,
        media_type: str,
        logical_type: str,
        content_length: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not media_type.strip():
            raise ArtifactStreamError("artifact stream media_type cannot be empty")
        if not logical_type.strip():
            raise ArtifactStreamError("artifact stream logical_type cannot be empty")
        if content_length is not None and content_length < 0:
            raise ArtifactStreamError("artifact stream content_length cannot be negative")
        if isinstance(chunks, (str, bytes, bytearray, memoryview)):
            raise ArtifactStreamError("artifact stream chunks must be an iterable of chunks")
        if not isinstance(chunks, (Iterable, AsyncIterable)):
            raise ArtifactStreamError("artifact stream chunks must be iterable or async iterable")

        self._chunks = chunks
        self.kind = kind
        self.media_type = media_type
        self.logical_type = logical_type
        self.content_length = content_length
        self.metadata = _validated_metadata(metadata)
        self._claim_lock = Lock()
        self._claimed = False
        self._closed = False

    @classmethod
    def text(
        cls,
        chunks: Iterable[str] | AsyncIterable[str],
        *,
        media_type: str = "text/plain; charset=utf-8",
        metadata: Mapping[str, Any] | None = None,
        content_length: int | None = None,
        logical_type: str = "str",
    ) -> ArtifactStream[str]:
        return cast(
            ArtifactStream[str],
            cls(
                chunks,
                kind=ArtifactStreamKind.TEXT,
                media_type=media_type,
                logical_type=logical_type,
                content_length=content_length,
                metadata=metadata,
            ),
        )

    @classmethod
    def bytes(
        cls,
        chunks: Iterable[builtins.bytes | bytearray | memoryview]
        | AsyncIterable[builtins.bytes | bytearray | memoryview],
        *,
        media_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
        content_length: int | None = None,
        logical_type: str = "bytes",
    ) -> ArtifactStream[builtins.bytes]:
        return cast(
            ArtifactStream[builtins.bytes],
            cls(
                chunks,
                kind=ArtifactStreamKind.BYTES,
                media_type=media_type,
                logical_type=logical_type,
                content_length=content_length,
                metadata=metadata,
            ),
        )

    @classmethod
    def json_items(
        cls,
        items: Iterable[Any] | AsyncIterable[Any],
        *,
        metadata: Mapping[str, Any] | None = None,
        content_length: int | None = None,
        logical_type: str = "list",
    ) -> ArtifactStream[list[Any]]:
        return cast(
            ArtifactStream[list[Any]],
            cls(
                items,
                kind=ArtifactStreamKind.JSON_ITEMS,
                media_type="application/json",
                logical_type=logical_type,
                content_length=content_length,
                metadata=metadata,
            ),
        )

    @property
    def consumed(self) -> bool:
        with self._claim_lock:
            return self._claimed

    @property
    def is_async(self) -> bool:
        return isinstance(self._chunks, AsyncIterable)

    def iter_chunks(self) -> Iterator[Any]:
        if isinstance(self._chunks, AsyncIterable):
            raise ArtifactStreamError("an async artifact stream requires async iteration")
        self._claim()
        iterator = iter(self._chunks)
        try:
            for chunk in iterator:
                yield self._normalize_chunk(chunk)
        finally:
            self._closed = True
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    async def aiter_chunks(self) -> AsyncIterator[Any]:
        self._claim()
        if isinstance(self._chunks, AsyncIterable):
            iterator = aiter(self._chunks)
            try:
                async for chunk in iterator:
                    yield self._normalize_chunk(chunk)
            finally:
                self._closed = True
                close = getattr(iterator, "aclose", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result
            return

        iterator = iter(self._chunks)
        try:
            for chunk in iterator:
                yield self._normalize_chunk(chunk)
        finally:
            self._closed = True
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._chunks, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        if self._closed:
            return
        close = getattr(self._chunks, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            finally:
                self._closed = True
            return
        self.close()

    def _claim(self) -> None:
        with self._claim_lock:
            if self._claimed:
                raise ArtifactStreamConsumedError("artifact stream has already been consumed")
            if self._closed:
                raise ArtifactStreamConsumedError("artifact stream is closed")
            self._claimed = True

    def _normalize_chunk(self, chunk: Any) -> Any:
        if self.kind is ArtifactStreamKind.TEXT:
            if not isinstance(chunk, str):
                raise ArtifactStreamError(
                    f"text artifact stream chunks must be str, got {type(chunk).__name__}"
                )
            return chunk
        if self.kind is ArtifactStreamKind.BYTES:
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise ArtifactStreamError(
                    f"binary artifact stream chunks must be bytes-like, got {type(chunk).__name__}"
                )
            return bytes(chunk)
        return _json_compatible(chunk, label="JSON artifact stream item")


def unwrap_artifact_stream_annotation(annotation: Any) -> tuple[Any, bool]:
    """Return the logical result annotation and whether transport wrapping was present."""

    if annotation is ArtifactStream:
        return Any, True
    if get_origin(annotation) is ArtifactStream:
        arguments = get_args(annotation)
        return (arguments[0] if arguments else Any), True
    return annotation, False


def artifact_stream_matches_annotation(stream: ArtifactStream[Any], annotation: Any) -> bool:
    """Return whether a transport stream can represent one logical return annotation."""

    origin = get_origin(annotation)
    if origin is Annotated:
        return artifact_stream_matches_annotation(stream, get_args(annotation)[0])
    if origin in (Union, UnionType):
        return any(
            artifact_stream_matches_annotation(stream, argument)
            for argument in get_args(annotation)
        )
    if annotation is Any:
        return True
    if stream.kind is ArtifactStreamKind.TEXT:
        return annotation is str
    if stream.kind is ArtifactStreamKind.BYTES:
        return annotation is builtins.bytes
    return annotation is list or origin is list


def callable_with_logical_stream_return(fn: Callable[..., Any]) -> tuple[Callable[..., Any], Any]:
    """Hide ArtifactStream transport metadata from adapter callable introspection."""

    signature = inspect.signature(fn)
    try:
        annotations = inspect.get_annotations(fn, eval_str=True)
    except (NameError, TypeError):
        annotations = dict(getattr(fn, "__annotations__", {}))
    return_annotation = annotations.get("return", signature.return_annotation)
    logical_annotation, wrapped = unwrap_artifact_stream_annotation(return_annotation)
    if not wrapped:
        return fn, None

    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

        wrapper = cast(Callable[..., Any], async_wrapper)
    else:

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper = sync_wrapper

    setattr(wrapper, "__signature__", signature.replace(return_annotation=logical_annotation))
    wrapper.__annotations__ = {**annotations, "return": logical_annotation}
    return wrapper, logical_annotation


def _validated_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    invalid = _RESERVED_METADATA_KEYS.intersection(metadata)
    if invalid:
        names = ", ".join(sorted(invalid))
        raise ArtifactStreamError(f"artifact stream metadata cannot override: {names}")
    if not all(isinstance(key, str) for key in metadata):
        raise ArtifactStreamError("artifact stream metadata keys must be strings")
    return cast(dict[str, Any], _json_compatible(dict(metadata), label="artifact stream metadata"))


def _json_compatible(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ArtifactStreamError(f"{label} must be JSON-compatible") from exc
