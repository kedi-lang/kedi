from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast

from kedi.artifacts import telemetry
from kedi.artifacts.errors import ToolOutputTooLargeError
from kedi.artifacts.models import ArtifactRecord, ArtifactRef, ArtifactStatus
from kedi.artifacts.policy import MAX_INLINE_TOOL_RESULT_BYTES, ArtifactPolicy
from kedi.artifacts.store import ArtifactStore
from kedi.artifacts.stream import ArtifactStream
from kedi.artifacts.streaming import (
    ArtifactStreamEncoder,
    StreamedArtifact,
    materialize_stream_payload,
)

if TYPE_CHECKING:
    from kedi.artifacts.manager import ArtifactManager


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


def _artifact_kind(source: str | None) -> str:
    if source is not None and source.startswith("tool:"):
        return "tool_call_result"
    if source is not None and source.startswith("template"):
        return "template_output"
    if source is not None and source.startswith("artifact_code"):
        return "artifact_code_result"
    return "artifact"
