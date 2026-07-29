from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from .errors import (
    ArtifactAccessError,
    ArtifactNotFoundError,
    ArtifactSerializationError,
)
from .models import ArtifactRecord, ArtifactRef, ArtifactStatus
from .serialization import SerializedArtifact, deserialize_artifact


class ArtifactStore(Protocol):
    def put(
        self,
        record: ArtifactRecord,
        serialized: SerializedArtifact,
    ) -> None: ...

    def get_record(self, session_id: str, ref_id: str) -> ArtifactRecord: ...

    def get_value(self, session_id: str, ref_id: str) -> Any: ...

    def records(self, session_id: str) -> tuple[ArtifactRecord, ...]: ...

    def delete_payload(self, session_id: str, ref_id: str) -> None: ...

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

    def get_record(self, session_id: str, ref_id: str) -> ArtifactRecord:
        with self._lock:
            return _owned_record(self._records, session_id, ref_id)

    def get_value(self, session_id: str, ref_id: str) -> Any:
        with self._lock:
            return _owned_record(self._records, session_id, ref_id).payload

    def records(self, session_id: str) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            return tuple(
                record for record in self._records.values() if record.session_id == session_id
            )

    def delete_payload(self, session_id: str, ref_id: str) -> None:
        with self._lock:
            record = _owned_record(self._records, session_id, ref_id)
            record.payload = None

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
        self._native_cache: dict[str, Any] = {}
        self._restored_sessions: set[str] = set()
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
            _atomic_write(payload_path, serialized.payload_bytes, mode=0o600)
            metadata = {
                "ref": record.ref.model_dump(mode="json"),
                "session_id": record.session_id,
                "codec": record.codec,
                "status": record.status.value,
                "source": record.source,
                "digest": record.digest,
                "idle_ttl_seconds": record.idle_ttl_seconds,
            }
            try:
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
                raise
            record.payload = payload_path
            self._records[record.ref_id] = record
            self._native_cache[record.ref_id] = serialized.memory_value
            os.chmod(directory, 0o700)

    def get_record(self, session_id: str, ref_id: str) -> ArtifactRecord:
        with self._lock:
            self._restore_session(session_id)
            return _owned_record(self._records, session_id, ref_id)

    def get_value(self, session_id: str, ref_id: str) -> Any:
        with self._lock:
            self._restore_session(session_id)
            record = _owned_record(self._records, session_id, ref_id)
            if ref_id in self._native_cache:
                return self._native_cache[ref_id]
            payload_path = self._payload_path(session_id, ref_id)
            payload = payload_path.read_bytes()
            value = deserialize_artifact(record.codec, payload)
            self._native_cache[ref_id] = value
            return value

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
            self._native_cache.pop(ref_id, None)
            self._payload_path(session_id, ref_id).unlink(missing_ok=True)
            self._metadata_path(session_id, ref_id).unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            self._native_cache.clear()
            self._records.clear()
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
        for metadata_path in directory.glob("art_*.json"):
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
                    payload=payload_path,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        self._restored_sessions.add(session_id)


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
