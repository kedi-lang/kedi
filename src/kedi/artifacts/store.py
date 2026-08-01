from __future__ import annotations

import json
import os
import secrets
import time
import base64
import codecs
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Literal, NoReturn, Protocol, cast

from .errors import (
    ArtifactAccessError,
    ArtifactNotFoundError,
    ArtifactSerializationError,
)
from .models import ArtifactRecord, ArtifactRef, ArtifactStatus
from .serialization import (
    JsonPointerRange,
    SerializedArtifact,
    deserialize_artifact,
    render_artifact_content,
    resolve_json_pointer,
)
from .streaming import StreamedArtifact, materialize_stream_payload


_PROJECTION_INDEX_VERSION = 1
_UTF8_CHECKPOINT_INTERVAL = 4_096


@dataclass(frozen=True)
class _ProjectionIndex:
    utf8_checkpoints: tuple[tuple[int, int], ...]
    json_pointer_ranges: dict[str, JsonPointerRange]
    json_pointer_index_complete: bool


class ArtifactStoreWriter(Protocol):
    def write(self, chunk: bytes) -> None: ...

    def commit(self, record: ArtifactRecord, streamed: StreamedArtifact) -> None: ...

    def abort(self) -> None: ...


class ArtifactStore(Protocol):
    def put(
        self,
        record: ArtifactRecord,
        serialized: SerializedArtifact,
    ) -> None: ...

    def begin_write(self, session_id: str, ref_id: str) -> ArtifactStoreWriter: ...

    def get_record(self, session_id: str, ref_id: str) -> ArtifactRecord: ...

    def get_value(self, session_id: str, ref_id: str) -> Any: ...

    def read_projection(
        self,
        session_id: str,
        ref_id: str,
        *,
        offset: int,
        offset_from: Literal["start", "end"],
        max_chars: int,
    ) -> tuple[str, int | None, bool]: ...

    def read_json_pointer_projection(
        self,
        session_id: str,
        ref_id: str,
        *,
        pointer: str,
        offset: int,
        offset_from: Literal["start", "end"],
        max_chars: int,
    ) -> tuple[str, int | None, bool]: ...

    def records(self, session_id: str) -> tuple[ArtifactRecord, ...]: ...

    def delete_payload(self, session_id: str, ref_id: str) -> None: ...

    def discard(self, session_id: str, ref_id: str) -> None: ...

    def close(self) -> None: ...


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self._records: dict[str, ArtifactRecord] = {}
        self._lock = RLock()

    def put(
        self,
        record: ArtifactRecord,
        serialized: SerializedArtifact,
    ) -> None:
        with self._lock:
            if record.ref_id in self._records:
                raise ArtifactSerializationError(
                    f"artifact reference {record.ref_id!r} already exists"
                )
            record.payload = serialized.memory_value
            self._records[record.ref_id] = record

    def begin_write(self, session_id: str, ref_id: str) -> ArtifactStoreWriter:
        return _MemoryArtifactWriter(self, session_id=session_id, ref_id=ref_id)

    def get_record(self, session_id: str, ref_id: str) -> ArtifactRecord:
        with self._lock:
            return _owned_record(self._records, session_id, ref_id)

    def get_value(self, session_id: str, ref_id: str) -> Any:
        with self._lock:
            return _owned_record(self._records, session_id, ref_id).payload

    def read_projection(
        self,
        session_id: str,
        ref_id: str,
        *,
        offset: int,
        offset_from: Literal["start", "end"],
        max_chars: int,
    ) -> tuple[str, int | None, bool]:
        record = self.get_record(session_id, ref_id)
        rendered = render_artifact_content(record.codec, self.get_value(session_id, ref_id))
        return _slice_projection(rendered, offset, offset_from, max_chars)

    def read_json_pointer_projection(
        self,
        session_id: str,
        ref_id: str,
        *,
        pointer: str,
        offset: int,
        offset_from: Literal["start", "end"],
        max_chars: int,
    ) -> tuple[str, int | None, bool]:
        record = self.get_record(session_id, ref_id)
        if record.codec != "json":
            raise ArtifactSerializationError("JSON paths can only be read from JSON artifacts")
        selected = resolve_json_pointer(self.get_value(session_id, ref_id), pointer)
        return _slice_projection(
            render_artifact_content(record.codec, selected),
            offset,
            offset_from,
            max_chars,
        )

    def records(self, session_id: str) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            return tuple(
                record for record in self._records.values() if record.session_id == session_id
            )

    def delete_payload(self, session_id: str, ref_id: str) -> None:
        with self._lock:
            record = _owned_record(self._records, session_id, ref_id)
            record.payload = None

    def discard(self, session_id: str, ref_id: str) -> None:
        with self._lock:
            _owned_record(self._records, session_id, ref_id)
            self._records.pop(ref_id)

    def close(self) -> None:
        with self._lock:
            self._records.clear()


class FileArtifactStore:
    """Safe file-backed payload store rooted under one application directory."""

    def __init__(self, root: Path) -> None:
        requested_root = root.expanduser()
        if requested_root.is_symlink():
            raise ArtifactAccessError("artifact store root cannot be a symlink")
        self.root = requested_root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._records: dict[str, ArtifactRecord] = {}
        self._projection_indexes: dict[tuple[str, str], _ProjectionIndex] = {}
        self._restored_sessions: set[str] = set()
        self._active_temporary_paths: set[Path] = set()
        self._lock = RLock()

    def put(
        self,
        record: ArtifactRecord,
        serialized: SerializedArtifact,
    ) -> None:
        if not serialized.persistable or serialized.payload_bytes is None:
            raise ArtifactSerializationError(
                f"{record.ref.logical_type} cannot be stored in the file artifact store"
            )
        with self._lock:
            if record.ref_id in self._records:
                raise ArtifactSerializationError(
                    f"artifact reference {record.ref_id!r} already exists"
                )
            directory = self._session_directory(record.session_id)
            payload_path = self._payload_path(record.session_id, record.ref_id)
            metadata_path = self._metadata_path(record.session_id, record.ref_id)
            index_path = self._projection_index_path(record.session_id, record.ref_id)
            _atomic_write(payload_path, serialized.payload_bytes, mode=0o600)
            metadata = {
                "ref": record.ref.model_dump(mode="json"),
                "session_id": record.session_id,
                "codec": record.codec,
                "status": record.status.value,
                "source": record.source,
                "digest": record.digest,
                "provenance": record.provenance,
                "idle_ttl_seconds": record.idle_ttl_seconds,
            }
            try:
                if record.codec in {"text", "json"}:
                    _write_projection_index(
                        index_path,
                        payload_path,
                        json_pointer_ranges=serialized.json_pointer_ranges,
                        json_pointer_index_complete=(serialized.json_pointer_index_complete),
                    )
                _atomic_write(
                    metadata_path,
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8"),
                    mode=0o600,
                )
            except BaseException:
                payload_path.unlink(missing_ok=True)
                index_path.unlink(missing_ok=True)
                raise
            record.payload = payload_path
            self._records[record.ref_id] = record
            os.chmod(directory, 0o700)

    def begin_write(self, session_id: str, ref_id: str) -> ArtifactStoreWriter:
        with self._lock:
            writer = _FileArtifactWriter(self, session_id=session_id, ref_id=ref_id)
            self._active_temporary_paths.add(writer._temporary_path)
            return writer

    def get_record(self, session_id: str, ref_id: str) -> ArtifactRecord:
        with self._lock:
            self._restore_session(session_id)
            return _owned_record(self._records, session_id, ref_id)

    def get_value(self, session_id: str, ref_id: str) -> Any:
        with self._lock:
            self._restore_session(session_id)
            record = _owned_record(self._records, session_id, ref_id)
            payload_path = self._payload_path(session_id, ref_id)
            payload = payload_path.read_bytes()
            return deserialize_artifact(record.codec, payload)

    def read_projection(
        self,
        session_id: str,
        ref_id: str,
        *,
        offset: int,
        offset_from: Literal["start", "end"],
        max_chars: int,
    ) -> tuple[str, int | None, bool]:
        with self._lock:
            self._restore_session(session_id)
            record = _owned_record(self._records, session_id, ref_id)
            payload_path = self._payload_path(session_id, ref_id)
            if record.codec in {"text", "json"}:
                index = self._read_projection_index(session_id, ref_id)
                total_chars = record.ref.char_count
                if total_chars is None:
                    total_chars = _count_utf8_chars(payload_path)
                start = _projection_start(total_chars, offset, offset_from, max_chars)
                length = _projection_length(total_chars, start, offset, offset_from, max_chars)
                content = _read_utf8_chars(
                    payload_path,
                    start=start,
                    limit=length,
                    checkpoints=(index.utf8_checkpoints if index is not None else None),
                )
                return _projection_page(content, start, total_chars, offset, offset_from)
            if record.codec == "bytes":
                total_chars = 4 * ((payload_path.stat().st_size + 2) // 3)
                start = _projection_start(total_chars, offset, offset_from, max_chars)
                length = _projection_length(total_chars, start, offset, offset_from, max_chars)
                content = _read_base64_chars(payload_path, start=start, limit=length)
                return _projection_page(content, start, total_chars, offset, offset_from)
            raise ArtifactSerializationError(
                f"artifact codec {record.codec!r} has no file-backed text projection"
            )

    def read_json_pointer_projection(
        self,
        session_id: str,
        ref_id: str,
        *,
        pointer: str,
        offset: int,
        offset_from: Literal["start", "end"],
        max_chars: int,
    ) -> tuple[str, int | None, bool]:
        if pointer and not pointer.startswith("/"):
            raise ValueError("artifact JSON path must be an RFC 6901 pointer")
        with self._lock:
            self._restore_session(session_id)
            record = _owned_record(self._records, session_id, ref_id)
            if record.codec != "json":
                raise ArtifactSerializationError("JSON paths can only be read from JSON artifacts")
            index = self._read_projection_index(session_id, ref_id)
            pointer_range = index.json_pointer_ranges.get(pointer) if index is not None else None
            if pointer_range is None:
                if index is not None and index.json_pointer_index_complete:
                    raise ValueError(f"artifact JSON path {pointer!r} was not found")
                pointer_range = _locate_json_pointer_range(
                    self._payload_path(session_id, ref_id),
                    pointer,
                )
                if pointer_range is None:
                    raise ValueError(f"artifact JSON path {pointer!r} was not found")

            _, _, char_start, char_end = pointer_range
            total_chars = char_end - char_start
            start = _projection_start(total_chars, offset, offset_from, max_chars)
            length = _projection_length(total_chars, start, offset, offset_from, max_chars)
            content = _read_utf8_chars(
                self._payload_path(session_id, ref_id),
                start=char_start + start,
                limit=length,
                checkpoints=(index.utf8_checkpoints if index is not None else None),
            )
            return _projection_page(content, start, total_chars, offset, offset_from)

    def records(self, session_id: str) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            self._restore_session(session_id)
            return tuple(
                record for record in self._records.values() if record.session_id == session_id
            )

    def delete_payload(self, session_id: str, ref_id: str) -> None:
        with self._lock:
            self._restore_session(session_id)
            _owned_record(self._records, session_id, ref_id)
            self._payload_path(session_id, ref_id).unlink(missing_ok=True)
            self._metadata_path(session_id, ref_id).unlink(missing_ok=True)
            self._projection_index_path(session_id, ref_id).unlink(missing_ok=True)
            self._projection_indexes.pop((session_id, ref_id), None)

    def discard(self, session_id: str, ref_id: str) -> None:
        with self._lock:
            self._restore_session(session_id)
            _owned_record(self._records, session_id, ref_id)
            self._payload_path(session_id, ref_id).unlink(missing_ok=True)
            self._metadata_path(session_id, ref_id).unlink(missing_ok=True)
            self._projection_index_path(session_id, ref_id).unlink(missing_ok=True)
            self._projection_indexes.pop((session_id, ref_id), None)
            self._records.pop(ref_id)

    def close(self) -> None:
        with self._lock:
            self._records.clear()
            self._projection_indexes.clear()
            self._restored_sessions.clear()

    def restore_session(
        self,
        session_id: str,
        *,
        monotonic_now: float,
        wall_now: datetime,
    ) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            self._restore_session(
                session_id,
                monotonic_now=monotonic_now,
                wall_now=wall_now,
            )
            return tuple(
                record for record in self._records.values() if record.session_id == session_id
            )

    def _session_directory(self, session_id: str) -> Path:
        return _safe_child(self.root, session_id, create=True)

    def _payload_path(self, session_id: str, ref_id: str) -> Path:
        return _safe_child(self._session_directory(session_id), f"{ref_id}.payload")

    def _metadata_path(self, session_id: str, ref_id: str) -> Path:
        return _safe_child(self._session_directory(session_id), f"{ref_id}.json")

    def _projection_index_path(self, session_id: str, ref_id: str) -> Path:
        return _safe_child(self._session_directory(session_id), f"{ref_id}.index")

    def _read_projection_index(
        self,
        session_id: str,
        ref_id: str,
    ) -> _ProjectionIndex | None:
        key = (session_id, ref_id)
        cached = self._projection_indexes.get(key)
        if cached is not None:
            return cached
        path = self._projection_index_path(session_id, ref_id)
        if not path.is_file():
            return None
        index = _load_projection_index(path)
        self._projection_indexes[key] = index
        return index

    def _restore_session(
        self,
        session_id: str,
        *,
        monotonic_now: float | None = None,
        wall_now: datetime | None = None,
    ) -> None:
        if session_id in self._restored_sessions:
            return
        directory = self._session_directory(session_id)
        monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
        wall_now = datetime.now(timezone.utc) if wall_now is None else wall_now
        restored_ref_ids = {
            record.ref_id for record in self._records.values() if record.session_id == session_id
        }
        for metadata_path in directory.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("session_id") != session_id:
                    continue
                ref = ArtifactRef[Any].model_validate(metadata["ref"])
                payload_path = self._payload_path(session_id, ref.ref_id)
                if not payload_path.is_file():
                    continue
                remaining = (ref.expires_at - wall_now).total_seconds()
                status = ArtifactStatus(metadata.get("status", ArtifactStatus.ACTIVE.value))
                if remaining <= 0 and status is ArtifactStatus.ACTIVE:
                    status = ArtifactStatus.EXPIRED
                self._records[ref.ref_id] = ArtifactRecord(
                    ref=ref,
                    session_id=session_id,
                    codec=str(metadata["codec"]),
                    status=status,
                    created_monotonic=monotonic_now,
                    last_accessed_monotonic=monotonic_now,
                    expires_monotonic=monotonic_now + max(0.0, remaining),
                    idle_ttl_seconds=metadata.get("idle_ttl_seconds"),
                    source=metadata.get("source"),
                    digest=metadata.get("digest"),
                    provenance=metadata.get("provenance"),
                    payload=payload_path,
                )
                restored_ref_ids.add(ref.ref_id)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        self._remove_interrupted_session_files(directory, restored_ref_ids)
        self._restored_sessions.add(session_id)

    def _remove_interrupted_session_files(
        self,
        directory: Path,
        restored_ref_ids: set[str],
    ) -> None:
        # Restoring a session assumes one owning process. Cross-process session
        # ownership requires a lease protocol rather than filesystem heuristics.
        for path in directory.iterdir():
            if path in self._active_temporary_paths:
                continue
            name = path.name
            interrupted = name.startswith(".") and name.endswith(".tmp")
            orphaned = False
            for suffix in (".payload", ".index"):
                if name.endswith(suffix):
                    orphaned = name[: -len(suffix)] not in restored_ref_ids
                    break
            if (interrupted or orphaned) and (path.is_file() or path.is_symlink()):
                path.unlink(missing_ok=True)


class _MemoryArtifactWriter:
    def __init__(self, store: InMemoryArtifactStore, *, session_id: str, ref_id: str) -> None:
        self._store = store
        self._session_id = session_id
        self._ref_id = ref_id
        self._buffer = BytesIO()
        self._finished = False

    def write(self, chunk: bytes) -> None:
        self._ensure_open()
        self._buffer.write(chunk)

    def commit(self, record: ArtifactRecord, streamed: StreamedArtifact) -> None:
        self._ensure_open()
        if record.session_id != self._session_id or record.ref_id != self._ref_id:
            raise ArtifactSerializationError("artifact writer identity does not match record")
        payload = self._buffer.getvalue()
        with self._store._lock:
            if record.ref_id in self._store._records:
                raise ArtifactSerializationError(
                    f"artifact reference {record.ref_id!r} already exists"
                )
            record.payload = materialize_stream_payload(streamed, payload)
            self._store._records[record.ref_id] = record
        self._finish()

    def abort(self) -> None:
        self._finish()

    def _ensure_open(self) -> None:
        if self._finished:
            raise ArtifactSerializationError("artifact writer is already closed")

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._buffer.close()


class _FileArtifactWriter:
    def __init__(self, store: FileArtifactStore, *, session_id: str, ref_id: str) -> None:
        self._store = store
        self._session_id = session_id
        self._ref_id = ref_id
        directory = store._session_directory(session_id)
        self._temporary_path = _safe_child(
            directory,
            f".{ref_id}.{secrets.token_urlsafe(8)}.tmp",
        )
        descriptor = os.open(
            self._temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        self._stream: BinaryIO | None = os.fdopen(descriptor, "wb")
        self._finished = False

    def write(self, chunk: bytes) -> None:
        stream = self._open_stream()
        stream.write(chunk)

    def commit(self, record: ArtifactRecord, streamed: StreamedArtifact) -> None:
        stream = self._open_stream()
        if record.session_id != self._session_id or record.ref_id != self._ref_id:
            raise ArtifactSerializationError("artifact writer identity does not match record")
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        self._stream = None
        payload_path = self._store._payload_path(self._session_id, self._ref_id)
        metadata_path = self._store._metadata_path(self._session_id, self._ref_id)
        index_path = self._store._projection_index_path(self._session_id, self._ref_id)
        metadata = {
            "ref": record.ref.model_dump(mode="json"),
            "session_id": record.session_id,
            "codec": record.codec,
            "status": record.status.value,
            "source": record.source,
            "digest": record.digest,
            "provenance": record.provenance,
            "idle_ttl_seconds": record.idle_ttl_seconds,
        }
        published = False
        try:
            with self._store._lock:
                if record.ref_id in self._store._records:
                    raise ArtifactSerializationError(
                        f"artifact reference {record.ref_id!r} already exists"
                    )
                self._temporary_path.replace(payload_path)
                published = True
                if record.codec in {"text", "json"}:
                    _write_projection_index(
                        index_path,
                        payload_path,
                        json_pointer_ranges=streamed.json_pointer_ranges,
                        json_pointer_index_complete=(streamed.json_pointer_index_complete),
                    )
                _atomic_write(
                    metadata_path,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                    mode=0o600,
                )
                record.payload = payload_path
                self._store._records[record.ref_id] = record
                self._finished = True
                self._store._active_temporary_paths.discard(self._temporary_path)
        except BaseException:
            if published:
                payload_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                index_path.unlink(missing_ok=True)
            self.abort()
            raise

    def abort(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._temporary_path.unlink(missing_ok=True)
        with self._store._lock:
            self._store._active_temporary_paths.discard(self._temporary_path)

    def _open_stream(self) -> BinaryIO:
        if self._finished or self._stream is None:
            raise ArtifactSerializationError("artifact writer is already closed")
        return self._stream


def store_size(records: Iterable[ArtifactRecord]) -> tuple[int, int]:
    active = tuple(
        record
        for record in records
        if record.status in {ArtifactStatus.ACTIVE, ArtifactStatus.PENDING_RELEASE}
    )
    return len(active), sum(record.ref.size_bytes for record in active)


def _owned_record(
    records: dict[str, ArtifactRecord],
    session_id: str,
    ref_id: str,
) -> ArtifactRecord:
    try:
        record = records[ref_id]
    except KeyError as exc:
        raise ArtifactNotFoundError(f"artifact {ref_id!r} was not found") from exc
    if record.session_id != session_id:
        raise ArtifactAccessError(f"artifact {ref_id!r} does not belong to the active session")
    return record


def _safe_child(root: Path, name: str, *, create: bool = False) -> Path:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ArtifactAccessError(f"invalid artifact path component {name!r}")
    candidate = root / name
    if create:
        candidate.mkdir(mode=0o700, parents=False, exist_ok=True)
        if candidate.is_symlink():
            raise ArtifactAccessError("artifact directory cannot be a symlink")
    return candidate


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
    content = rendered[start : start + limit]
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
