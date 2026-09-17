from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

from kedi.artifacts.errors import (
    ArtifactAccessError,
    ArtifactInputError,
    ArtifactSerializationError,
)
from kedi.artifacts.models import ArtifactRecord, ArtifactRef, ArtifactStatus
from kedi.artifacts.projection import (
    _atomic_write,
    _count_utf8_chars,
    _load_projection_index,
    _locate_json_pointer_range,
    _projection_length,
    _projection_page,
    _projection_start,
    _ProjectionIndex,
    _read_base64_chars,
    _read_utf8_chars,
    _write_projection_index,
)
from kedi.artifacts.serialization import SerializedArtifact, deserialize_artifact
from kedi.artifacts.streaming import StreamedArtifact

if TYPE_CHECKING:
    from kedi.artifacts.store import ArtifactStoreWriter

from kedi.artifacts.store import _owned_record


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
            raise ArtifactInputError("artifact JSON path must be an RFC 6901 pointer")
        with self._lock:
            self._restore_session(session_id)
            record = _owned_record(self._records, session_id, ref_id)
            if record.codec != "json":
                raise ArtifactInputError("JSON paths can only be read from JSON artifacts")
            index = self._read_projection_index(session_id, ref_id)
            pointer_range = index.json_pointer_ranges.get(pointer) if index is not None else None
            if pointer_range is None:
                if index is not None and index.json_pointer_index_complete:
                    raise ArtifactInputError(f"artifact JSON path {pointer!r} was not found")
                pointer_range = _locate_json_pointer_range(
                    self._payload_path(session_id, ref_id),
                    pointer,
                )
                if pointer_range is None:
                    raise ArtifactInputError(f"artifact JSON path {pointer!r} was not found")

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


def _safe_child(root: Path, name: str, *, create: bool = False) -> Path:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ArtifactAccessError(f"invalid artifact path component {name!r}")
    candidate = root / name
    if create:
        candidate.mkdir(mode=0o700, parents=False, exist_ok=True)
        if candidate.is_symlink():
            raise ArtifactAccessError("artifact directory cannot be a symlink")
    return candidate
