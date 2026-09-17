from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ArtifactStatus(str, Enum):
    ACTIVE = "active"
    PENDING_RELEASE = "pending_release"
    RELEASED = "released"
    EXPIRED = "expired"


class ArtifactRef(BaseModel, Generic[T]):
    """Compact model-visible metadata for an artifact-backed value."""

    model_config = ConfigDict(frozen=True)

    ref_id: str
    logical_type: str
    media_type: str
    summary: str
    preview: str
    size_bytes: int = Field(ge=0)
    char_count: int | None = Field(default=None, ge=0)
    created_at: datetime
    expires_at: datetime
    sensitive: bool = False
    usage_hint: str = (
        "Copy this exact ref_id into artifact tools; never use a source or tool name as an ID. "
        "Use read_artifact for a known literal or bounded inspection; use run_artifact_code for "
        "filtering, aggregation, joins, or multi-artifact transformations. A read with "
        "complete=false is partial; follow its continuation only when more content is relevant. "
        "Release when done."
    )


class ArtifactChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref_id: str
    content: str
    offset: int = Field(ge=0)
    next_offset: int | None = Field(default=None, ge=0)
    complete: bool
    media_type: str
    path: str | None = None
    offset_from: Literal["start", "end"] = "start"
    requested_max_chars: int = Field(ge=-1)
    applied_max_chars: int = Field(gt=0)
    returned_chars: int = Field(ge=0)
    continuation: dict[str, Any] | None = None


class ArtifactSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref_id: str
    logical_type: str
    media_type: str
    summary: str
    size_bytes: int = Field(ge=0)
    source: str | None = None
    status: ArtifactStatus
    created_at: datetime
    expires_at: datetime


class ArtifactReleaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref_id: str
    status: ArtifactStatus
    already_released: bool = False


class ArtifactCodeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: Any
    stdout: str = ""
    source_refs: tuple[str, ...] = ()


class ArtifactManagerProtocol(Protocol):
    session_id: str

    def get_value(self, ref_id: str) -> Any: ...


@dataclass(frozen=True)
class ArtifactHandle(Generic[T]):
    """Internal lazy value stored in Kedi environments."""

    ref: ArtifactRef[T]
    manager: ArtifactManagerProtocol

    def get_value(self) -> T:
        return self.manager.get_value(self.ref.ref_id)

    def as_ref(self) -> ArtifactRef[T]:
        return self.ref


@dataclass
class ArtifactRecord:
    ref: ArtifactRef[Any]
    session_id: str
    codec: str
    status: ArtifactStatus
    created_monotonic: float
    last_accessed_monotonic: float
    expires_monotonic: float
    idle_ttl_seconds: float | None
    source: str | None = None
    digest: str | None = None
    provenance: dict[str, Any] | None = None
    payload: Any = None
    active_readers: int = 0

    @property
    def ref_id(self) -> str:
        return self.ref.ref_id
