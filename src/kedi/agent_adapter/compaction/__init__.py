"""Deterministic, adapter-neutral conversation compaction primitives."""

from .backends import CompactionBackend, resolve_compaction_backend
from .codecs import AdapterHistoryCodec, HistoryCodecRegistry
from .coordinator import CompactionCoordinator
from .conversation import (
    ConversationCompactionTransaction,
    PreparedConversationCompaction,
)
from .kedi_backend import KediCompactionTransaction, PreparedCompaction
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
    "PydanticHistoryCodec",
    "resolve_compaction_backend",
]
