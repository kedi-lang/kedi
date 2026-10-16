"""Deterministic, adapter-neutral conversation compaction primitives."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .archive import HistoryArchiveSettings
from .backends import CompactionBackend, resolve_compaction_backend
from .codecs import AdapterHistoryCodec, HistoryCodecRegistry
from .conversation import (
    ConversationCompactionTransaction,
    PreparedConversationCompaction,
)
from .coordinator import CompactionCoordinator
from .kedi_backend import KediCompactionTransaction, PreparedCompaction
from .langchain_archive import LangChainArtifactHistoryProcessor
from .langchain_codec import LangChainHistoryCodec
from .models import (
    CheckpointAcceptancePolicy,
    CheckpointAnchor,
    CheckpointAnchorKind,
    CompactionCapabilities,
    CompactionCheckpoint,
    CompactionError,
    CompactionOutcome,
    CompactionPreflightError,
    CompactionResult,
    CompactionUsage,
    CompactionValidationError,
    CompactionWindow,
    HistoryRange,
    HistoryReductionPlan,
    HistoryReductionPolicy,
    HistorySnapshot,
    HistoryUnit,
    HistoryUnitGroup,
    NativeHistoryEntryRef,
    ProcessedHistory,
)
from .native_backend import NativeCompactionBackend
from .processor import HistoryProcessor
from .pydantic_codec import PydanticHistoryCodec

if TYPE_CHECKING:
    from .pydantic_archive import (
        PydanticArtifactHistoryCapability,
        PydanticArtifactHistoryProcessor,
        PydanticHistoryArchiveSettings,
    )


def __getattr__(name: str) -> Any:
    if name not in {
        "PydanticArtifactHistoryCapability",
        "PydanticArtifactHistoryProcessor",
        "PydanticHistoryArchiveSettings",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".pydantic_archive", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "AdapterHistoryCodec",
    "CompactionBackend",
    "CompactionCapabilities",
    "CompactionCheckpoint",
    "CompactionCoordinator",
    "CompactionError",
    "CompactionOutcome",
    "CompactionPreflightError",
    "CompactionResult",
    "CompactionUsage",
    "CompactionValidationError",
    "CompactionWindow",
    "CheckpointAcceptancePolicy",
    "CheckpointAnchor",
    "CheckpointAnchorKind",
    "ConversationCompactionTransaction",
    "HistoryCodecRegistry",
    "HistoryProcessor",
    "HistoryArchiveSettings",
    "LangChainArtifactHistoryProcessor",
    "HistoryRange",
    "HistoryReductionPlan",
    "HistoryReductionPolicy",
    "HistorySnapshot",
    "HistoryUnit",
    "HistoryUnitGroup",
    "KediCompactionTransaction",
    "LangChainHistoryCodec",
    "NativeHistoryEntryRef",
    "NativeCompactionBackend",
    "PreparedCompaction",
    "PreparedConversationCompaction",
    "ProcessedHistory",
    "PydanticArtifactHistoryCapability",
    "PydanticArtifactHistoryProcessor",
    "PydanticHistoryArchiveSettings",
    "PydanticHistoryCodec",
    "resolve_compaction_backend",
]
