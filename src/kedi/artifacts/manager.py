from __future__ import annotations

import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

from .cleanup import cleanup_service
from .errors import (
    ArtifactExpiredError,
    ArtifactQuotaExceededError,
    ArtifactReleasedError,
)
from .history import ArtifactHistory
from .models import (
    ArtifactChunk,
    ArtifactHandle,
    ArtifactRecord,
    ArtifactRef,
    ArtifactReleaseResult,
    ArtifactSearchResult,
    ArtifactStatus,
)
from .policy import ArtifactPolicy
from .serialization import (
    render_artifact_content,
    serialize_artifact,
)
from . import telemetry
from .store import (
    ArtifactStore,
    FileArtifactStore,
    InMemoryArtifactStore,
    store_size,
)


class ArtifactManager:
    """Session-scoped artifact creation, lookup, and lifecycle coordinator."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        history: ArtifactHistory | None = None,
        monotonic: Any = time.monotonic,
        wall_clock: Any = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.session_id = session_id or f"session_{secrets.token_urlsafe(18)}"
        self.history = history or ArtifactHistory()
        self._monotonic = monotonic
        self._wall_clock = wall_clock
        self._memory_store = InMemoryArtifactStore()
        self._file_stores: dict[Path, FileArtifactStore] = {}
        self._ref_stores: dict[str, ArtifactStore] = {}
        self._ref_policies: dict[str, ArtifactPolicy] = {}
        self._digest_refs: dict[tuple[str, str], str] = {}
        self._metered_active_refs: dict[str, int] = {}
        self._lock = RLock()
        self._last_cleanup = 0.0
        self._cleanup_interval_seconds = 60.0
        self._closed = False

    @property
    def cleanup_interval_seconds(self) -> float:
        with self._lock:
            return self._cleanup_interval_seconds

    def artifactize(
        self,
        value: Any,
        *,
        policy: ArtifactPolicy,
        source: str | None = None,
        sensitive: bool = False,
    ) -> Any:
        effective = policy.resolved()
        if not effective.enabled:
            return value
        serialized = serialize_artifact(
            value,
            preview_chars=effective.preview_chars,
            require_persistable=effective.store == "file",
        )
        if serialized.size_bytes < effective.threshold_bytes:
            return value

        with telemetry.artifact_span(
            "create",
            session_id=self.session_id,
            source=source or "",
            size_bytes=serialized.size_bytes,
            store=effective.store,
        ):
            with self._lock:
                self._ensure_open()
                self._cleanup_interval_seconds = min(
                    self._cleanup_interval_seconds,
                    effective.cleanup_interval_seconds,
                )
                self._maybe_cleanup_locked()
                store = self._store_for_policy(effective)
                if serialized.digest is not None:
                    digest_key = (self._store_identity(effective), serialized.digest)
                    existing_ref = self._digest_refs.get(digest_key)
                    if existing_ref is not None:
                        existing = self._record(existing_ref, allow_inactive=False)
                        return ArtifactHandle(ref=existing.ref, manager=self)
                try:
                    self._enforce_quota(effective, serialized.size_bytes)
                except ArtifactQuotaExceededError:
                    telemetry.record_quota_rejection()
                    raise

                now_monotonic = self._monotonic()
                now_wall = self._wall_clock()
                ref_id = f"art_{secrets.token_urlsafe(18)}"
                ref = ArtifactRef[Any](
                    ref_id=ref_id,
                    logical_type=serialized.logical_type,
                    media_type=serialized.media_type,
                    summary=serialized.summary,
                    preview=serialized.preview,
                    size_bytes=serialized.size_bytes,
                    char_count=serialized.char_count,
                    created_at=now_wall,
                    expires_at=now_wall + timedelta(seconds=effective.ttl_seconds),
                    sensitive=sensitive,
                )
                record = ArtifactRecord(
                    ref=ref,
                    session_id=self.session_id,
                    codec=serialized.codec,
                    status=ArtifactStatus.ACTIVE,
                    created_monotonic=now_monotonic,
                    last_accessed_monotonic=now_monotonic,
                    expires_monotonic=now_monotonic + effective.ttl_seconds,
                    idle_ttl_seconds=effective.idle_ttl_seconds,
                    source=source,
                    digest=serialized.digest,
                )
                store.put(record, serialized)
                self._ref_stores[ref_id] = store
                self._ref_policies[ref_id] = effective
                self._metered_active_refs[ref_id] = ref.size_bytes
                if serialized.digest is not None:
                    self._digest_refs[(self._store_identity(effective), serialized.digest)] = ref_id
                self.history.append(
                    "artifact_created",
                    ref_id=ref_id,
                    data={
                        "logical_type": ref.logical_type,
                        "media_type": ref.media_type,
                        "size_bytes": ref.size_bytes,
                        "source": source,
                    },
                )
                telemetry.record_created(
                    size_bytes=ref.size_bytes,
                    context_bytes_avoided=ref.size_bytes - len(ref.preview.encode("utf-8")),
                )
                cleanup_service.register(self)
                return ArtifactHandle(ref=ref, manager=self)

    def get_value(self, ref_id: str) -> Any:
        with self._lease(ref_id) as (store, record):
            return store.get_value(self.session_id, record.ref_id)

    def read(
        self,
        ref_id: str,
        *,
        offset: int = 0,
        max_chars: int = -1,
        path: str | None = None,
    ) -> ArtifactChunk:
        with telemetry.artifact_span(
            "read",
            session_id=self.session_id,
            ref_id=ref_id,
            offset=offset,
        ):
            if offset < 0:
                raise ValueError("artifact read offset must not be negative")
            with self._lease(ref_id) as (store, record):
                policy = self._ref_policies[ref_id]
                limit = (
                    policy.read_max_chars
                    if max_chars == -1
                    else min(max_chars, policy.read_max_chars)
                )
                if limit <= 0:
                    raise ValueError("artifact max_chars must be -1 or positive")
                value = store.get_value(self.session_id, ref_id)
                selected = _resolve_json_pointer(value, path) if path is not None else value
                rendered = render_artifact_content(record.codec, selected)
                content = rendered[offset : offset + limit]
                next_offset = offset + len(content)
                complete = next_offset >= len(rendered)
                chunk = ArtifactChunk(
                    ref_id=ref_id,
                    content=content,
                    offset=offset,
                    next_offset=None if complete else next_offset,
                    complete=complete,
                    media_type=record.ref.media_type,
                    path=path,
                )
                self.history.append(
                    "artifact_read",
                    ref_id=ref_id,
                    data={
                        "offset": offset,
                        "chars": len(content),
                        "complete": complete,
                        "path": path,
                    },
                )
                telemetry.record_read(bytes_returned=len(content.encode("utf-8")))
                return chunk

    def search(
        self,
        query: str | None = None,
        *,
        limit: int = 20,
    ) -> list[ArtifactSearchResult]:
        with telemetry.artifact_span(
            "search",
            session_id=self.session_id,
            limit=limit,
            has_query=query is not None,
        ) as span:
            if limit <= 0:
                raise ValueError("artifact search limit must be positive")
            self.cleanup_expired()
            needle = query.casefold() if query else None
            records = sorted(
                self._all_records(),
                key=lambda record: record.ref.created_at,
                reverse=True,
            )
            results: list[ArtifactSearchResult] = []
            for record in records:
                if record.status is not ArtifactStatus.ACTIVE:
                    continue
                haystack = " ".join(
                    (
                        record.ref.ref_id,
                        record.ref.logical_type,
                        record.ref.media_type,
                        record.ref.summary,
                        record.source or "",
                    )
                ).casefold()
                if needle is not None and needle not in haystack:
                    continue
                results.append(
                    ArtifactSearchResult(
                        ref_id=record.ref_id,
                        logical_type=record.ref.logical_type,
                        media_type=record.ref.media_type,
                        summary=record.ref.summary,
                        size_bytes=record.ref.size_bytes,
                        source=record.source,
                        status=record.status,
                        created_at=record.ref.created_at,
                        expires_at=record.ref.expires_at,
                    )
                )
                if len(results) >= min(limit, 100):
                    break
            span.set_attribute("kedi.artifact.search_results", len(results))
            return results

    def release(
        self,
        ref_id: str,
        *,
        compact_history: bool = True,
    ) -> ArtifactReleaseResult:
        with telemetry.artifact_span(
            "release",
            session_id=self.session_id,
            ref_id=ref_id,
        ):
            with self._lock:
                record = self._record(ref_id, allow_inactive=True)
                if record.status is ArtifactStatus.RELEASED:
                    return ArtifactReleaseResult(
                        ref_id=ref_id,
                        status=record.status,
                        already_released=True,
                    )
                if record.status is ArtifactStatus.EXPIRED:
                    raise ArtifactExpiredError(f"artifact {ref_id!r} has expired")
                record.status = (
                    ArtifactStatus.PENDING_RELEASE
                    if record.active_readers
                    else ArtifactStatus.RELEASED
                )
                if record.status is ArtifactStatus.RELEASED:
                    self._delete_payload(record)
                self.history.append("artifact_released", ref_id=ref_id)
                compacted = (
                    self.history.compact(ref_id, reason="released") if compact_history else 0
                )
                telemetry.record_released()
                return ArtifactReleaseResult(
                    ref_id=ref_id,
                    status=record.status,
                    compacted_events=compacted,
                )

    def cleanup_expired(self) -> int:
        started = time.perf_counter()
        with telemetry.artifact_span("cleanup", session_id=self.session_id) as span:
            try:
                with self._lock:
                    if self._closed:
                        return 0
                    now = self._monotonic()
                    expired = 0
                    for record in self._all_records():
                        if record.status not in {
                            ArtifactStatus.ACTIVE,
                            ArtifactStatus.PENDING_RELEASE,
                        }:
                            continue
                        idle_deadline = (
                            record.last_accessed_monotonic + record.idle_ttl_seconds
                            if record.idle_ttl_seconds is not None
                            else float("inf")
                        )
                        if now < min(record.expires_monotonic, idle_deadline):
                            continue
                        with telemetry.artifact_span(
                            "expire",
                            session_id=self.session_id,
                            ref_id=record.ref_id,
                        ):
                            record.status = ArtifactStatus.EXPIRED
                            if not record.active_readers:
                                self._delete_payload(record)
                            self.history.append("artifact_expired", ref_id=record.ref_id)
                            self.history.compact(record.ref_id, reason="expired")
                            telemetry.record_expired()
                            expired += 1
                    self._last_cleanup = now
                    span.set_attribute("kedi.artifact.expired_count", expired)
                    return expired
            finally:
                telemetry.record_cleanup(duration_seconds=time.perf_counter() - started)

    def stats(self) -> dict[str, int]:
        records = self._all_records()
        active_count, active_bytes = store_size(records)
        return {
            "active_artifacts": active_count,
            "active_bytes": active_bytes,
            "released_artifacts": sum(
                record.status is ArtifactStatus.RELEASED for record in records
            ),
            "expired_artifacts": sum(record.status is ArtifactStatus.EXPIRED for record in records),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            cleanup_service.unregister(self)
            for size_bytes in self._metered_active_refs.values():
                telemetry.record_payload_removed(size_bytes=size_bytes)
            self._metered_active_refs.clear()
            self._memory_store.close()
            for store in self._file_stores.values():
                store.close()
            self._file_stores.clear()
            self._ref_stores.clear()
            self._ref_policies.clear()
            self._digest_refs.clear()

    @contextmanager
    def _lease(
        self,
        ref_id: str,
    ) -> Iterator[tuple[ArtifactStore, ArtifactRecord]]:
        with self._lock:
            self._maybe_cleanup_locked()
            record = self._record(ref_id, allow_inactive=False)
            record.active_readers += 1
            record.last_accessed_monotonic = self._monotonic()
            store = self._ref_stores[ref_id]
        try:
            yield store, record
        finally:
            with self._lock:
                record.active_readers -= 1
                if record.active_readers == 0 and record.status in {
                    ArtifactStatus.PENDING_RELEASE,
                    ArtifactStatus.EXPIRED,
                }:
                    if record.status is ArtifactStatus.PENDING_RELEASE:
                        record.status = ArtifactStatus.RELEASED
                    self._delete_payload(record)

    def _record(self, ref_id: str, *, allow_inactive: bool) -> ArtifactRecord:
        try:
            store = self._ref_stores[ref_id]
        except KeyError:
            return self._memory_store.get_record(self.session_id, ref_id)
        record = store.get_record(self.session_id, ref_id)
        if allow_inactive:
            return record
        if record.status is ArtifactStatus.EXPIRED:
            raise ArtifactExpiredError(f"artifact {ref_id!r} has expired")
        if record.status in {
            ArtifactStatus.RELEASED,
            ArtifactStatus.PENDING_RELEASE,
        }:
            raise ArtifactReleasedError(f"artifact {ref_id!r} was released")
        return record

    def _store_for_policy(self, policy: ArtifactPolicy) -> ArtifactStore:
        if policy.store == "memory":
            return self._memory_store
        path = policy.path.expanduser().resolve()
        store = self._file_stores.get(path)
        if store is None:
            store = FileArtifactStore(path)
            self._file_stores[path] = store
            for record in store.restore_session(
                self.session_id,
                monotonic_now=self._monotonic(),
                wall_now=self._wall_clock(),
            ):
                self._ref_stores[record.ref_id] = store
                self._ref_policies[record.ref_id] = policy
                if record.status in {
                    ArtifactStatus.ACTIVE,
                    ArtifactStatus.PENDING_RELEASE,
                }:
                    self._metered_active_refs[record.ref_id] = record.ref.size_bytes
                    telemetry.record_active(size_bytes=record.ref.size_bytes)
                else:
                    self._delete_payload(record)
                if record.digest is not None and record.status is ArtifactStatus.ACTIVE:
                    self._digest_refs[(self._store_identity(policy), record.digest)] = record.ref_id
        return store

    def _enforce_quota(
        self,
        policy: ArtifactPolicy,
        new_size: int,
    ) -> None:
        count, size = store_size(self._all_records())
        if count + 1 > policy.max_artifacts:
            raise ArtifactQuotaExceededError(
                f"artifact count would exceed limit {policy.max_artifacts}"
            )
        if size + new_size > policy.session_quota_bytes:
            raise ArtifactQuotaExceededError(
                f"artifact payload would exceed session quota {policy.session_quota_bytes} bytes"
            )

    def _all_records(self) -> tuple[ArtifactRecord, ...]:
        stores: tuple[ArtifactStore, ...] = (
            self._memory_store,
            *self._file_stores.values(),
        )
        return tuple(record for store in stores for record in store.records(self.session_id))

    def _delete_payload(self, record: ArtifactRecord) -> None:
        store = self._ref_stores[record.ref_id]
        store.delete_payload(self.session_id, record.ref_id)
        size_bytes = self._metered_active_refs.pop(record.ref_id, None)
        if size_bytes is not None:
            telemetry.record_payload_removed(size_bytes=size_bytes)
        if record.digest is not None:
            policy = self._ref_policies[record.ref_id]
            self._digest_refs.pop((self._store_identity(policy), record.digest), None)

    @staticmethod
    def _store_identity(policy: ArtifactPolicy) -> str:
        if policy.store == "memory":
            return "memory"
        return f"file:{policy.path.expanduser().resolve()}"

    def _maybe_cleanup_locked(self) -> None:
        if self._monotonic() - self._last_cleanup >= self._cleanup_interval_seconds:
            self.cleanup_expired()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("artifact manager is closed")


def _resolve_json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("artifact JSON path must be an RFC 6901 pointer")
    current = value
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise ValueError(f"artifact JSON path cannot descend through {type(current).__name__}")
    return current
