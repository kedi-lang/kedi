from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, Protocol

from kedi.artifacts.errors import ArtifactAccessError, ArtifactNotFoundError
from kedi.artifacts.models import ArtifactRecord, ArtifactStatus
from kedi.artifacts.serialization import SerializedArtifact
from kedi.artifacts.streaming import StreamedArtifact

__all__ = [
    "ArtifactStore",
    "ArtifactStoreWriter",
    "FileArtifactStore",
    "InMemoryArtifactStore",
    "store_size",
]

if TYPE_CHECKING:
    from .file_store import FileArtifactStore as FileArtifactStore
    from .memory_store import InMemoryArtifactStore as InMemoryArtifactStore


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


def __getattr__(name: str) -> Any:
    if name == "InMemoryArtifactStore":
        from .memory_store import InMemoryArtifactStore

        return InMemoryArtifactStore
    if name == "FileArtifactStore":
        from .file_store import FileArtifactStore

        return FileArtifactStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
