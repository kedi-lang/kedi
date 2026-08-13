from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

CompactionBackendKind: TypeAlias = Literal["native", "kedi", "disabled"]
CompactionMode: TypeAlias = Literal["auto", "native", "kedi", "disabled"]
CheckpointAnchorKind: TypeAlias = Literal[
    "approval",
    "constraint",
    "decision",
    "error",
    "fact",
    "lifecycle",
    "pending-work",
    "resource",
]
HistoryEntryKind: TypeAlias = Literal[
    "system",
    "user",
    "assistant",
    "reasoning",
    "tool-call",
    "tool-result",
    "retry",
    "checkpoint",
    "unknown",
]
HistoryNoOpReason: TypeAlias = Literal[
    "below-threshold",
    "empty-history",
    "no-safe-window",
]
HistoryStorageFormat: TypeAlias = Literal["serialized", "native", "mixed"]


class CompactionError(RuntimeError):
    """Base error for adapter-neutral compaction failures."""


class CompactionPreflightError(CompactionError):
    """Raised before model I/O when a requested compaction mode is unavailable."""


class CompactionValidationError(CompactionError):
    """Raised when a planned history replacement is no longer safe to commit."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointAnchor:
    """Canonical continuation state that a semantic summary cannot paraphrase away."""

    anchor_id: str
    kind: CheckpointAnchorKind
    content: str

    def __post_init__(self) -> None:
        if not self.anchor_id.strip():
            raise ValueError("checkpoint anchor ID must not be empty")
        if not self.content.strip():
            raise ValueError("checkpoint anchor content must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointAcceptancePolicy:
    """Machine-checkable requirements applied before a checkpoint can commit."""

    required_anchors: tuple[CheckpointAnchor, ...] = ()

    def __post_init__(self) -> None:
        anchor_ids = [anchor.anchor_id for anchor in self.required_anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("checkpoint anchor IDs must be unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionCapabilities:
    """Compaction operations an adapter can honor without transcript loss."""

    native_managed: bool = False
    manual_trigger: bool = False
    configurable_threshold: bool = False
    can_disable: bool = False
    history_export: bool = False
    history_replace: bool = False

    @property
    def supports_kedi_backend(self) -> bool:
        return self.history_export and self.history_replace


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionUsage:
    """Token accounting reported by a compaction backend when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionOutcome:
    """Normalized result from either native or Kedi-owned compaction."""

    backend: CompactionBackendKind
    applied: bool
    trigger: str | None = None
    checkpoint_id: str | None = None
    source_tokens: int | None = None
    result_tokens: int | None = None
    usage: CompactionUsage | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeHistoryEntryRef:
    """Stable metadata for one untouched adapter-native history entry."""

    index: int
    entry_id: str
    fingerprint: str
    kind: HistoryEntryKind
    token_estimate: int
    part_kinds: tuple[str, ...] = ()
    lifecycle_ids: tuple[str, ...] = ()
    closes_lifecycle_ids: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    complete: bool = True
    checkpoint: bool = False
    unknown: bool = False
    provider_required: bool = False
    searchable_text: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryUnit:
    """Smallest entry range classified before lifecycle grouping."""

    unit_id: str
    entry_start: int
    entry_end: int
    entry_ids: tuple[str, ...]
    kinds: tuple[HistoryEntryKind, ...]
    token_estimate: int
    lifecycle_ids: tuple[str, ...] = ()
    closes_lifecycle_ids: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    complete: bool = True
    checkpoint: bool = False
    unknown: bool = False
    provider_required: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryUnitGroup:
    """Atomic replay interval that selection must retain or replace as a whole."""

    group_id: str
    unit_ids: tuple[str, ...]
    entry_start: int
    entry_end: int
    token_estimate: int
    closed: bool
    lifecycle_ids: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    protected_reasons: tuple[str, ...] = ()

    @property
    def protected(self) -> bool:
        return bool(self.protected_reasons)


@dataclass(frozen=True, slots=True, kw_only=True)
class HistorySnapshot:
    """Immutable inspection result over an adapter-native replay list."""

    adapter_shortname: str
    state_key: str
    source_version: int
    source_digest: str
    native_entries: tuple[Any, ...] = field(repr=False, compare=False)
    entries: tuple[NativeHistoryEntryRef, ...]
    units: tuple[HistoryUnit, ...] = ()
    groups: tuple[HistoryUnitGroup, ...] = ()
    total_tokens: int = 0
    estimator_version: str = "utf8-div3-v1"
    storage_format: HistoryStorageFormat = "serialized"


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryRange:
    """Half-open adapter-native history interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or not isinstance(self.start, int)
            or isinstance(self.end, bool)
            or not isinstance(self.end, int)
        ):
            raise TypeError("history range boundaries must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("history range must be a non-empty half-open interval")


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryReductionPolicy:
    """Deterministic Track A selection policy."""

    threshold_tokens: int
    target_tokens: int
    preserve_recent_tokens: int
    checkpoint_budget_tokens: int
    minimum_reduction_tokens: int = 1
    protected_identifiers: frozenset[str] = frozenset()
    live_artifact_refs: frozenset[str] = frozenset()
    missing_artifact_refs: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        positive = {
            "threshold_tokens": self.threshold_tokens,
            "target_tokens": self.target_tokens,
            "checkpoint_budget_tokens": self.checkpoint_budget_tokens,
            "minimum_reduction_tokens": self.minimum_reduction_tokens,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.preserve_recent_tokens, bool)
            or not isinstance(self.preserve_recent_tokens, int)
            or self.preserve_recent_tokens < 0
        ):
            raise ValueError("preserve_recent_tokens must be a non-negative integer")
        if self.target_tokens >= self.threshold_tokens:
            raise ValueError("target_tokens must be lower than threshold_tokens")


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionWindow:
    """One contiguous group-aligned interval selected for replacement."""

    native_range: HistoryRange
    group_ids: tuple[str, ...]
    source_tokens: int
    predicted_result_tokens: int
    predicted_reduction_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryReductionPlan:
    """Reproducible output of deterministic history selection."""

    source_version: int
    source_digest: str
    estimator_version: str
    total_tokens: int
    policy: HistoryReductionPolicy
    groups: tuple[HistoryUnitGroup, ...]
    window: CompactionWindow | None
    unchanged_prefix_fingerprints: tuple[str, ...]
    unchanged_tail_fingerprints: tuple[str, ...]
    no_op_reason: HistoryNoOpReason | None = None

    @property
    def applicable(self) -> bool:
        return self.window is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionCheckpoint:
    """Structured continuation checkpoint supplied to the Track A coordinator."""

    checkpoint_id: str
    version: int
    source_start_id: str
    source_end_id: str
    source_digest: str
    summary: str
    source_tokens: int
    checkpoint_tokens: int
    durable_facts: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    pending_work: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    completed_outcomes: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    errors_and_lessons: tuple[str, ...] = ()
    anchors: tuple[CheckpointAnchor, ...] = ()

    def __post_init__(self) -> None:
        required_text = {
            "checkpoint_id": self.checkpoint_id,
            "source_start_id": self.source_start_id,
            "source_end_id": self.source_end_id,
            "source_digest": self.source_digest,
        }
        for name, value in required_text.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("checkpoint version must be an integer")
        if self.version <= 0:
            raise ValueError("checkpoint version must be positive")
        if not self.summary.strip():
            raise ValueError("checkpoint summary must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.source_tokens, self.checkpoint_tokens)
        ):
            raise TypeError("checkpoint token counts must be integers")
        if self.source_tokens <= 0 or self.checkpoint_tokens <= 0:
            raise ValueError("checkpoint token counts must be positive")
        anchor_ids = [anchor.anchor_id for anchor in self.anchors]
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("checkpoint anchor IDs must be unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessedHistory:
    """Committed adapter state plus evidence that retained slices were unchanged."""

    state: Mapping[str, object]
    replaced_range: HistoryRange
    checkpoint_id: str
    source_digest: str
    result_digest: str
    prefix_fingerprints: tuple[str, ...]
    tail_fingerprints: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CompactionResult:
    """Track A transaction result."""

    outcome: CompactionOutcome
    processed: ProcessedHistory | None = None
