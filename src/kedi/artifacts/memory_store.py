from __future__ import annotations

from io import BytesIO
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal

from kedi.artifacts.errors import ArtifactInputError, ArtifactSerializationError
from kedi.artifacts.models import ArtifactRecord
from kedi.artifacts.projection import _slice_projection
from kedi.artifacts.serialization import (
    SerializedArtifact,
    render_artifact_content,
    resolve_json_pointer,
)
from kedi.artifacts.streaming import StreamedArtifact, materialize_stream_payload

if TYPE_CHECKING:
    from kedi.artifacts.store import ArtifactStoreWriter

from kedi.artifacts.store import _owned_record


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
            raise ArtifactInputError("JSON paths can only be read from JSON artifacts")
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
