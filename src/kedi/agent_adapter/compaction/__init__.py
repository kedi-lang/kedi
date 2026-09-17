"""Deterministic, adapter-neutral conversation compaction primitives."""

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
from .pydantic_archive import (
    PydanticArtifactHistoryCapability,
    PydanticArtifactHistoryProcessor,
    PydanticHistoryArchiveSettings,
)
from .pydantic_codec import PydanticHistoryCodec

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
