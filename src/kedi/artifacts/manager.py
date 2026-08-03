from __future__ import annotations

import secrets
import time
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Literal, cast

from . import telemetry
from .cleanup import cleanup_service
from .errors import (
    ArtifactExpiredError,
    ArtifactQuotaExceededError,
    ArtifactReleasedError,
    ToolOutputTooLargeError,
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
    SerializedArtifact,
    serialize_artifact,
)
from .store import (
    ArtifactStore,
    FileArtifactStore,
    InMemoryArtifactStore,
    store_size,
)
from .stream import ArtifactStream
from .streaming import ArtifactStreamEncoder, StreamedArtifact, materialize_stream_payload

MAX_INLINE_TOOL_RESULT_BYTES = 100_000


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
        self._ref_counters: dict[str, int] = defaultdict(int)
        self._reserved_artifacts = 0
        self._reserved_bytes = 0
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
        provenance: dict[str, Any] | None = None,
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

        return self._store_serialized(
            value,
            serialized=serialized,
            policy=effective,
            source=source,
            sensitive=sensitive,
            provenance=provenance,
        )

    def admit_tool_result(
        self,
        value: Any,
        *,
        policy: ArtifactPolicy,
        source: str | None = None,
        sensitive: bool = False,
    ) -> Any:
        """Admit a materialized tool result without allowing oversized inline history."""

        effective = policy.resolved()
        serialized = serialize_artifact(
            value,
            preview_chars=effective.preview_chars,
            require_persistable=effective.enabled and effective.store == "file",
        )
        should_artifact = effective.enabled and (
            serialized.size_bytes >= effective.threshold_bytes
            or serialized.size_bytes > MAX_INLINE_TOOL_RESULT_BYTES
        )
        if not should_artifact and serialized.size_bytes <= MAX_INLINE_TOOL_RESULT_BYTES:
            telemetry.record_output_admission(
                decision="inline",
                size_bytes=serialized.size_bytes,
            )
            return value
        if not effective.enabled:
            telemetry.record_output_admission(
                decision="rejected",
                size_bytes=serialized.size_bytes,
            )
            raise ToolOutputTooLargeError(
                size_bytes=serialized.size_bytes,
                limit_bytes=MAX_INLINE_TOOL_RESULT_BYTES,
            )
        telemetry.record_output_admission(
            decision="artifact",
            size_bytes=serialized.size_bytes,
        )
        return self._store_serialized(
            value,
            serialized=serialized,
            policy=effective,
            source=source,
            sensitive=sensitive,
        )

    def _store_serialized(
        self,
        value: Any,
        *,
        serialized: SerializedArtifact,
        policy: ArtifactPolicy,
        source: str | None,
        sensitive: bool,
        provenance: dict[str, Any] | None = None,
    ) -> Any:
        effective = policy

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
                ref_id = self._new_ref_id(source)
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
                    provenance=dict(provenance) if provenance is not None else None,
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

    def artifactize_stream(
        self,
        stream: ArtifactStream[Any],
        *,
        policy: ArtifactPolicy,
        source: str | None = None,
        sensitive: bool = False,
    ) -> Any:
        """Consume a synchronous stream once and store it without full materialization."""

        return self._consume_stream(
            stream,
            stream.iter_chunks(),
            policy=policy,
            source=source,
            sensitive=sensitive,
        )

    async def artifactize_stream_async(
        self,
        stream: ArtifactStream[Any],
        *,
        policy: ArtifactPolicy,
        source: str | None = None,
        sensitive: bool = False,
    ) -> Any:
        """Consume a sync or async stream once and store it without full materialization."""

        effective = policy.resolved()
        encoder = ArtifactStreamEncoder(stream, preview_chars=effective.preview_chars)
        state = _StreamingWrite(self, effective, source, sensitive, stream, encoder)
        try:
            state.accept(encoder.start())
            async for chunk in stream.aiter_chunks():
                state.accept_chunk(encoder.encode(chunk))
            tail, streamed = encoder.finish()
            state.accept(tail)
            return state.finish(streamed)
        except BaseException as exc:
            state.abort(exc)
            await stream.aclose()
            raise

    def get_value(self, ref_id: str) -> Any:
        with self._lease(ref_id) as (store, record):
            return store.get_value(self.session_id, record.ref_id)

    def metadata(self, ref_id: str) -> dict[str, Any]:
        with self._lease(ref_id) as (_, record):
            return {
                **record.ref.model_dump(mode="json"),
                "status": record.status.value,
                "source": record.source,
                "provenance": dict(record.provenance) if record.provenance is not None else None,
            }

    def find(
        self,
        ref_id: str,
        pattern: str,
        *,
        max_matches: int = 20,
        context_chars: int = 120,
    ) -> list[dict[str, Any]]:
        """Find literal text with bounded store reads and cross-chunk matching."""

        if not pattern:
            raise ValueError("artifact search pattern cannot be empty")
        if max_matches <= 0:
            raise ValueError("artifact max_matches must be positive")
        max_matches = min(max_matches, 100)
        with self._lease(ref_id) as (store, record):
            policy = self._ref_policies[ref_id]
            chunk_chars = max(65_536, policy.read_max_chars, len(pattern) + context_chars * 2)
            chunk_chars = min(chunk_chars, 65_536)
            offset = 0
            overlap = ""
            results: list[dict[str, Any]] = []
            seen_offsets: set[int] = set()
            while len(results) < max_matches:  # pragma: no branch - validated positive limit
                content, next_offset, complete = store.read_projection(
                    self.session_id,
                    record.ref_id,
                    offset=offset,
                    offset_from="start",
                    max_chars=chunk_chars,
                )
                window = overlap + content
                base = offset - len(overlap)
                cursor = 0
                while len(results) < max_matches:
                    index = window.find(pattern, cursor)
                    if index < 0:
                        break
                    absolute = base + index
                    if absolute not in seen_offsets:
                        seen_offsets.add(absolute)
                        left = max(0, index - context_chars)
                        right = min(len(window), index + len(pattern) + context_chars)
                        results.append(
                            {
                                "offset": absolute,
                                "match": pattern,
                                "context": window[left:right],
                            }
                        )
                    cursor = index + max(1, len(pattern))
                if complete or next_offset is None:
                    break
                overlap = window[-max(0, len(pattern) - 1) :]
                offset = next_offset
            return results

    def read(
        self,
        ref_id: str,
        *,
        offset: int = 0,
        offset_from: Literal["start", "end"] = "start",
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
            if offset_from not in {"start", "end"}:
                raise ValueError("artifact offset_from must be 'start' or 'end'")
            with self._lease(ref_id) as (store, record):
                policy = self._ref_policies[ref_id]
                limit = (
                    policy.read_max_chars
                    if max_chars == -1
                    else min(max_chars, policy.read_max_chars)
                )
                if limit <= 0:
                    raise ValueError("artifact max_chars must be -1 or positive")
                if path is None:
                    content, next_offset, complete = store.read_projection(
                        self.session_id,
                        ref_id,
                        offset=offset,
                        offset_from=offset_from,
                        max_chars=limit,
                    )
                else:
                    content, next_offset, complete = store.read_json_pointer_projection(
                        self.session_id,
                        ref_id,
                        pointer=path,
                        offset=offset,
                        offset_from=offset_from,
                        max_chars=limit,
                    )
                chunk = ArtifactChunk(
                    ref_id=ref_id,
                    content=content,
                    offset=offset,
                    next_offset=None if complete else next_offset,
                    complete=complete,
                    media_type=record.ref.media_type,
                    path=path,
                    offset_from=offset_from,
                )
                self.history.append(
                    "artifact_read",
                    ref_id=ref_id,
                    data={
                        "offset": offset,
                        "chars": len(content),
                        "complete": complete,
                        "path": path,
                        "offset_from": offset_from,
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

    def has_active_artifacts(self) -> bool:
        """Return whether this session owns content that an agent may need to read."""

        self.cleanup_expired()
        with self._lock:
            return any(record.status is ArtifactStatus.ACTIVE for record in self._all_records())

    def release(self, ref_id: str) -> ArtifactReleaseResult:
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
                telemetry.record_released()
                return ArtifactReleaseResult(
                    ref_id=ref_id,
                    status=record.status,
                )

    def cleanup_expired(self) -> int:
        started = time.perf_counter()
        try:
            with self._lock:
                if self._closed:
                    return 0
                now = self._monotonic()
                candidates = tuple(
                    record for record in self._all_records() if _is_expired(record, now=now)
                )
                self._last_cleanup = now
                if not candidates:
                    return 0

                for record in candidates:
                    with telemetry.artifact_span(
                        "expire",
                        session_id=self.session_id,
                        ref_id=record.ref_id,
                    ):
                        record.status = ArtifactStatus.EXPIRED
                        if not record.active_readers:
                            self._delete_payload(record)
                        self.history.append("artifact_expired", ref_id=record.ref_id)
                        telemetry.record_expired()
                return len(candidates)
        except BaseException as exc:
            with telemetry.artifact_span(
                "cleanup",
                session_id=self.session_id,
                outcome="error",
            ) as span:
                span.record_exception(exc)
            raise
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
            self._ref_counters.clear()
            self._reserved_artifacts = 0
            self._reserved_bytes = 0

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
        if count + self._reserved_artifacts + 1 > policy.max_artifacts:
            raise ArtifactQuotaExceededError(
                f"artifact count would exceed limit {policy.max_artifacts}"
            )
        if size + self._reserved_bytes + new_size > policy.session_quota_bytes:
            raise ArtifactQuotaExceededError(
                "artifact payload would exceed session quota "
                f"{policy.session_quota_bytes} bytes "
                f"(active={size}, requested={new_size})"
            )

    def _reserve_stream(self, policy: ArtifactPolicy, initial_size: int) -> None:
        with self._lock:
            self._enforce_quota(policy, initial_size)
            self._reserved_artifacts += 1
            self._reserved_bytes += initial_size

    def _grow_stream_reservation(self, policy: ArtifactPolicy, size: int) -> None:
        if size == 0:
            return
        with self._lock:
            _, active_size = store_size(self._all_records())
            if active_size + self._reserved_bytes + size > policy.session_quota_bytes:
                telemetry.record_quota_rejection()
                raise ArtifactQuotaExceededError(
                    "artifact payload would exceed session quota "
                    f"{policy.session_quota_bytes} bytes "
                    f"(active={active_size}, reserved={self._reserved_bytes}, requested={size})"
                )
            self._reserved_bytes += size

    def _release_stream_reservation(self, size: int) -> None:
        with self._lock:
            self._reserved_artifacts -= 1
            self._reserved_bytes -= size

    def _consume_stream(
        self,
        stream: ArtifactStream[Any],
        chunks: Iterator[Any],
        *,
        policy: ArtifactPolicy,
        source: str | None,
        sensitive: bool,
    ) -> Any:
        effective = policy.resolved()
        encoder = ArtifactStreamEncoder(stream, preview_chars=effective.preview_chars)
        state = _StreamingWrite(self, effective, source, sensitive, stream, encoder)
        try:
            state.accept(encoder.start())
            for chunk in chunks:
                state.accept_chunk(encoder.encode(chunk))
            tail, streamed = encoder.finish()
            state.accept(tail)
            return state.finish(streamed)
        except BaseException as exc:
            state.abort(exc)
            stream.close()
            raise

    def _new_ref_id(self, source: str | None) -> str:
        kind = _artifact_kind(source)
        while True:
            self._ref_counters[kind] += 1
            ref_id = f"{kind}_{self._ref_counters[kind]}"
            if ref_id not in self._ref_stores:
                return ref_id

    def _register_streamed(
        self,
        *,
        record: ArtifactRecord,
        store: ArtifactStore,
        policy: ArtifactPolicy,
    ) -> ArtifactHandle[Any]:
        ref_id = record.ref_id
        self._ref_stores[ref_id] = store
        self._ref_policies[ref_id] = policy
        self._metered_active_refs[ref_id] = record.ref.size_bytes
        if record.digest is not None:
            self._digest_refs[(self._store_identity(policy), record.digest)] = ref_id
        cleanup_service.register(self)
        telemetry.record_created(
            size_bytes=record.ref.size_bytes,
            context_bytes_avoided=(record.ref.size_bytes - len(record.ref.preview.encode("utf-8"))),
        )
        self.history.append(
            "artifact_created",
            ref_id=ref_id,
            data={
                "logical_type": record.ref.logical_type,
                "media_type": record.ref.media_type,
                "size_bytes": record.ref.size_bytes,
                "source": record.source,
                "streamed": True,
            },
        )
        return ArtifactHandle(ref=record.ref, manager=self)

    def _discard_streamed(
        self,
        *,
        record: ArtifactRecord,
        store: ArtifactStore,
        policy: ArtifactPolicy,
    ) -> None:
        ref_id = record.ref_id
        store.discard(self.session_id, ref_id)
        self._ref_stores.pop(ref_id, None)
        self._ref_policies.pop(ref_id, None)
        size_bytes = self._metered_active_refs.pop(ref_id, None)
        if size_bytes is not None:
            telemetry.record_payload_removed(size_bytes=size_bytes)
        if record.digest is not None:
            self._digest_refs.pop((self._store_identity(policy), record.digest), None)

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


class _StreamingWrite:
    """Threshold buffer and transactional store write for one consumed stream."""

    def __init__(
        self,
        manager: ArtifactManager,
        policy: ArtifactPolicy,
        source: str | None,
        sensitive: bool,
        stream: ArtifactStream[Any],
        encoder: ArtifactStreamEncoder,
    ) -> None:
        self.manager = manager
        self.policy = policy
        self.source = source
        self.sensitive = sensitive
        self.stream = stream
        self.encoder = encoder
        self.buffer = bytearray()
        self.writer: Any = None
        self.store: ArtifactStore | None = None
        self.ref_id: str | None = None
        self.reserved_bytes = 0
        self.finished = False
        self.chunk_count = 0
        self._store_span_context: AbstractContextManager[telemetry.TelemetrySpan] | None = None
        self._store_span: telemetry.TelemetrySpan | None = None

    def accept(self, payload: bytes) -> None:
        if self.finished or not payload:
            return
        if self.writer is None:
            if not self.policy.enabled:
                self.buffer.extend(payload)
                if len(self.buffer) > MAX_INLINE_TOOL_RESULT_BYTES:
                    telemetry.record_output_admission(
                        decision="rejected",
                        size_bytes=len(self.buffer),
                    )
                    raise ToolOutputTooLargeError(
                        size_bytes=len(self.buffer),
                        limit_bytes=MAX_INLINE_TOOL_RESULT_BYTES,
                    )
                return
            prospective_size = len(self.buffer) + len(payload)
            if (
                prospective_size < self.policy.threshold_bytes
                and prospective_size <= MAX_INLINE_TOOL_RESULT_BYTES
            ):
                self.buffer.extend(payload)
                return
            telemetry.record_output_admission(
                decision="artifact",
                size_bytes=prospective_size,
            )
            self._start_store_span(prospective_size)
            with self.manager._lock:
                self.manager._ensure_open()
                self.manager._cleanup_interval_seconds = min(
                    self.manager._cleanup_interval_seconds,
                    self.policy.cleanup_interval_seconds,
                )
                self.manager._maybe_cleanup_locked()
                self.store = self.manager._store_for_policy(self.policy)
                self.manager._reserve_stream(self.policy, prospective_size)
                self.reserved_bytes = prospective_size
                self.ref_id = self.manager._new_ref_id(self.source)
                self.writer = self.store.begin_write(self.manager.session_id, self.ref_id)
            if self.buffer:
                self.writer.write(bytes(self.buffer))
            self.buffer.clear()
            self.writer.write(payload)
            return
        self.manager._grow_stream_reservation(self.policy, len(payload))
        self.reserved_bytes += len(payload)
        self.writer.write(payload)

    def accept_chunk(self, payload: bytes) -> None:
        self.chunk_count += 1
        self.accept(payload)

    def finish(self, streamed: StreamedArtifact) -> Any:
        if self.finished:
            raise RuntimeError("artifact stream write is already finished")
        if self.writer is None:
            self.finished = True
            return materialize_stream_payload(streamed, bytes(self.buffer))
        if self.store is None or self.ref_id is None:
            raise RuntimeError("artifact stream writer has incomplete state")

        now_monotonic = self.manager._monotonic()
        now_wall = self.manager._wall_clock()
        ref = ArtifactRef[Any](
            ref_id=self.ref_id,
            logical_type=streamed.logical_type,
            media_type=streamed.media_type,
            summary=streamed.summary,
            preview=streamed.preview,
            size_bytes=streamed.size_bytes,
            char_count=streamed.char_count,
            created_at=now_wall,
            expires_at=now_wall + timedelta(seconds=self.policy.ttl_seconds),
            sensitive=self.sensitive,
        )
        record = ArtifactRecord(
            ref=ref,
            session_id=self.manager.session_id,
            codec=streamed.codec,
            status=ArtifactStatus.ACTIVE,
            created_monotonic=now_monotonic,
            last_accessed_monotonic=now_monotonic,
            expires_monotonic=now_monotonic + self.policy.ttl_seconds,
            idle_ttl_seconds=self.policy.idle_ttl_seconds,
            source=self.source,
            digest=streamed.digest,
        )
        try:
            self.writer.commit(record, streamed)
            try:
                with self.manager._lock:
                    result = self.manager._register_streamed(
                        record=record,
                        store=self.store,
                        policy=self.policy,
                    )
            except BaseException:
                with self.manager._lock:
                    self.manager._discard_streamed(
                        record=record,
                        store=self.store,
                        policy=self.policy,
                    )
                raise
            self.finished = True
            self._finish_store_span(streamed)
            return result
        finally:
            self.manager._release_stream_reservation(self.reserved_bytes)
            self.reserved_bytes = 0

    def abort(self, exc: BaseException | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            if self.writer is not None:
                self.writer.abort()
            if self.reserved_bytes:
                self.manager._release_stream_reservation(self.reserved_bytes)
                self.reserved_bytes = 0
        finally:
            self._close_store_span(exc)

    def _start_store_span(self, size_bytes: int) -> None:
        self._store_span_context = telemetry.artifact_span(
            "create",
            source=self.source or "",
            size_bytes=size_bytes,
            store=self.policy.store,
            streamed=True,
        )
        self._store_span = self._store_span_context.__enter__()

    def _finish_store_span(self, streamed: StreamedArtifact) -> None:
        active = cast(telemetry.TelemetrySpan, self._store_span)
        active.set_attribute("kedi.artifact.size_bytes", streamed.size_bytes)
        active.set_attribute("kedi.artifact.chunk_count", self.chunk_count)
        self._close_store_span()

    def _close_store_span(self, exc: BaseException | None = None) -> None:
        context = self._store_span_context
        active = self._store_span
        self._store_span_context = None
        self._store_span = None
        if context is None:
            return
        if exc is None:
            context.__exit__(None, None, None)
        else:
            cast(telemetry.TelemetrySpan, active).set_attribute("error.type", type(exc).__name__)
            context.__exit__(type(exc), exc, exc.__traceback__)


def _is_expired(record: ArtifactRecord, *, now: float) -> bool:
    if record.status not in {
        ArtifactStatus.ACTIVE,
        ArtifactStatus.PENDING_RELEASE,
    }:
        return False
    idle_deadline = (
        record.last_accessed_monotonic + record.idle_ttl_seconds
        if record.idle_ttl_seconds is not None
        else float("inf")
    )
    return now >= min(record.expires_monotonic, idle_deadline)


def _artifact_kind(source: str | None) -> str:
    if source is not None and source.startswith("tool:"):
        return "tool_call_result"
    if source is not None and source.startswith(("template", "output:")):
        return "template_output"
    if source is not None and source.startswith("artifact_code"):
        return "artifact_code_result"
    return "artifact"
