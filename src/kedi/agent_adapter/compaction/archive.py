"""Shared exact-history archival; native bridges retain message ownership."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Generic, TypeVar, cast

from kedi.artifacts import ArtifactError, ArtifactHandle, ArtifactPolicy

from ..tools.scope import current_artifact_admission
from .checkpoint import render_checkpoint
from .codecs import AdapterHistoryCodec, conservative_token_estimate, fingerprint_sequence
from .models import CompactionCheckpoint, CompactionOutcome, HistoryRange, HistorySnapshot
from .telemetry import compact_history_span

MessageT = TypeVar("MessageT")
_CACHE_EPOCH_SUFFIX = re.compile(r"_h[0-9a-f]{8}$")
PROVIDER_INPUT_BASELINE_KEY = "kedi_history_provider_input_baseline_tokens"


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryArchiveSettings:
    """Bounded, deterministic archival policy for long native tool loops."""

    threshold_tokens: int = 32_000
    preserve_recent_tokens: int = 12_000
    minimum_reduction_tokens: int = 8_000

    def __post_init__(self) -> None:
        for name, value in (
            ("threshold_tokens", self.threshold_tokens),
            ("preserve_recent_tokens", self.preserve_recent_tokens),
            ("minimum_reduction_tokens", self.minimum_reduction_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.preserve_recent_tokens >= self.threshold_tokens:
            raise ValueError("preserve_recent_tokens must be lower than threshold_tokens")
        if self.minimum_reduction_tokens >= self.threshold_tokens:
            raise ValueError("minimum_reduction_tokens must be lower than threshold_tokens")


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArchiveWindow:
    range: HistoryRange
    source_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderInputPressure:
    baseline_tokens: int
    current_tokens: int
    growth_tokens: int


class ArtifactHistoryProcessor(Generic[MessageT], ABC):
    """Archive old, completed tool exchanges without inventing a semantic summary."""

    def __init__(
        self, codec: AdapterHistoryCodec, settings: HistoryArchiveSettings | None = None
    ) -> None:
        self.settings = settings or HistoryArchiveSettings()
        self.codec = codec

    def __call__(self, messages: list[MessageT]) -> list[MessageT]:
        snapshot = self.codec.inspect({self.codec.state_key: messages})
        provider_pressure = self.provider_input_pressure(messages)
        trigger = "threshold"
        if provider_pressure is not None:
            if provider_pressure.growth_tokens < self.settings.threshold_tokens:
                return messages
            trigger = "provider"
        elif snapshot.total_tokens < self.settings.threshold_tokens:
            return messages
        if not snapshot.entries:
            return messages
        window = self._select_window(snapshot)
        if window is None:
            return messages
        admission = current_artifact_admission()
        if admission is None:
            return messages
        manager, configured_policy = admission
        if not isinstance(configured_policy, ArtifactPolicy):
            return messages
        effective_policy = configured_policy.resolved()
        if not effective_policy.enabled:
            return messages

        selected = messages[window.range.start : window.range.end]
        serialized = self.serialize(selected)
        source_digest = fingerprint_sequence(list(selected))
        checkpoint_id = f"history_{source_digest[:16]}"
        archive_payload = {
            "schema": f"kedi.{self.codec.adapter_shortname}-history-archive.v1",
            "checkpoint_id": checkpoint_id,
            "source_digest": source_digest,
            "source_tokens": window.source_tokens,
            "messages": serialized,
        }
        if provider_pressure is not None:
            archive_payload["provider_input_baseline_tokens"] = provider_pressure.baseline_tokens
        forced_policy = effective_policy.overlay(ArtifactPolicy.patch({"threshold_bytes": 0}))

        with compact_history_span(
            mode="kedi",
            trigger=trigger,
            adapter_name=self.codec.adapter_shortname,
            input_messages=len(messages),
            input_tokens=snapshot.total_tokens,
        ) as telemetry:
            if provider_pressure is not None:
                telemetry.span.set_attribute(
                    "kedi.history.compaction.provider_input_tokens",
                    provider_pressure.current_tokens,
                )
                telemetry.span.set_attribute(
                    "kedi.history.compaction.provider_input_baseline_tokens",
                    provider_pressure.baseline_tokens,
                )
                telemetry.span.set_attribute(
                    "kedi.history.compaction.provider_history_growth_tokens",
                    provider_pressure.growth_tokens,
                )
            try:
                stored = manager.artifactize(
                    archive_payload,
                    policy=forced_policy,
                    source="history:compaction",
                    sensitive=True,
                    provenance={
                        "checkpoint_id": checkpoint_id,
                        "source_digest": source_digest,
                        "source_messages": len(selected),
                        "source_tokens": window.source_tokens,
                    },
                )
            except (ArtifactError, OSError) as exc:
                telemetry.span.add_event(
                    "compaction rejected",
                    {
                        "kedi.history.compaction.rejection_reason": "artifact_error",
                        "error.type": type(exc).__name__,
                    },
                )
                telemetry.complete(
                    CompactionOutcome(
                        backend="kedi",
                        applied=False,
                        trigger=trigger,
                        source_tokens=window.source_tokens,
                        result_tokens=snapshot.total_tokens,
                    ),
                    input_messages=len(messages),
                    output_messages=len(messages),
                    input_tokens=snapshot.total_tokens,
                    output_tokens=snapshot.total_tokens,
                    selected_tokens=window.source_tokens,
                    prefix_preserved=True,
                    cache_epoch_changed=False,
                    checkpoint_valid=False,
                )
                return messages
            if not isinstance(stored, ArtifactHandle):
                raise RuntimeError("forced history archival did not create an artifact")

            checkpoint = self._checkpoint(
                snapshot,
                window=window,
                selected=selected,
                checkpoint_id=checkpoint_id,
                source_digest=source_digest,
                archive_ref=stored.ref.ref_id,
            )
            updated_state = self.codec.replace_range(
                {self.codec.state_key: messages},
                snapshot=snapshot,
                window=window.range,
                checkpoint=checkpoint,
            )
            updated = cast("list[MessageT]", updated_state[self.codec.state_key])
            if provider_pressure is not None:
                updated[window.range.start] = self.with_provider_input_baseline(
                    updated[window.range.start],
                    provider_pressure.baseline_tokens,
                )
            telemetry.checkpoint_validated()
            telemetry.checkpoint_committed()
            telemetry.cache_epoch_advanced()
            telemetry.complete(
                CompactionOutcome(
                    backend="kedi",
                    applied=True,
                    trigger=trigger,
                    checkpoint_id=checkpoint_id,
                    source_tokens=window.source_tokens,
                    result_tokens=(
                        snapshot.total_tokens - window.source_tokens + checkpoint.checkpoint_tokens
                    ),
                ),
                input_messages=len(messages),
                output_messages=len(updated),
                input_tokens=snapshot.total_tokens,
                output_tokens=(
                    snapshot.total_tokens - window.source_tokens + checkpoint.checkpoint_tokens
                ),
                selected_tokens=window.source_tokens,
                prefix_preserved=True,
                cache_epoch_changed=True,
                checkpoint_valid=True,
            )
            return updated

    def _select_window(self, snapshot: HistorySnapshot) -> _ArchiveWindow | None:
        entries = snapshot.entries
        latest_user = max(
            (entry.index for entry in entries if entry.kind == "user"),
            default=None,
        )
        if latest_user is None:
            return None
        latest_checkpoint = max(
            (entry.index for entry in entries if entry.checkpoint),
            default=-1,
        )
        start = max(latest_user, latest_checkpoint) + 1
        if start >= len(entries) - 1:
            return None

        recent_tokens = 0
        recent_start = len(entries)
        for entry in reversed(entries):
            recent_tokens += entry.token_estimate
            recent_start = entry.index
            if recent_tokens >= self.settings.preserve_recent_tokens:
                break
        if recent_start <= start:
            return None

        open_calls: Counter[str] = Counter()
        blocked = False
        candidate_start = start
        candidate_tokens = 0
        candidate_saw_call = False
        best_range: HistoryRange | None = None
        best_tokens = 0
        for entry in entries[start:recent_start]:
            barrier = (
                not entry.complete
                or entry.unknown
                or entry.kind in {"system", "user", "checkpoint", "assistant", "retry"}
            )
            open_calls.update(entry.lifecycle_ids)
            for identifier in entry.closes_lifecycle_ids:
                if open_calls[identifier] <= 0:
                    barrier = True
                    continue
                open_calls[identifier] -= 1
                if open_calls[identifier] == 0:
                    del open_calls[identifier]

            blocked = blocked or barrier
            if blocked:
                if not open_calls:
                    candidate_start = entry.index + 1
                    candidate_tokens = 0
                    candidate_saw_call = False
                    blocked = False
                continue

            candidate_saw_call = candidate_saw_call or bool(entry.lifecycle_ids)
            candidate_tokens += entry.token_estimate
            if candidate_saw_call and not open_calls and entry.kind == "tool-result":
                if candidate_tokens > best_tokens:
                    best_range = HistoryRange(start=candidate_start, end=entry.index + 1)
                    best_tokens = candidate_tokens

        if best_range is None or best_tokens < self.settings.minimum_reduction_tokens:
            return None
        return _ArchiveWindow(
            range=best_range,
            source_tokens=best_tokens,
        )

    def _checkpoint(
        self,
        snapshot: HistorySnapshot,
        *,
        window: _ArchiveWindow,
        selected: list[MessageT],
        checkpoint_id: str,
        source_digest: str,
        archive_ref: str,
    ) -> CompactionCheckpoint:
        outcomes = self.tool_outcomes(selected)
        inherited_refs = {
            ref
            for entry in snapshot.entries[window.range.start : window.range.end]
            for ref in entry.artifact_refs
        }
        checkpoint = CompactionCheckpoint(
            checkpoint_id=checkpoint_id,
            version=1,
            source_start_id=snapshot.entries[window.range.start].entry_id,
            source_end_id=snapshot.entries[window.range.end - 1].entry_id,
            source_digest=source_digest,
            summary=(
                "Older completed tool interactions were archived exactly. The original task, "
                "retained recent messages, and current workspace remain authoritative. Do not "
                "repeat completed work solely because details were archived; inspect the archive "
                "only when exact prior evidence is needed."
            ),
            source_tokens=window.source_tokens,
            checkpoint_tokens=1,
            resources=(f"Exact archived transcript: {archive_ref}",),
            completed_outcomes=outcomes,
            artifact_refs=tuple(sorted({archive_ref, *inherited_refs})),
        )
        for _ in range(3):
            estimate = conservative_token_estimate(render_checkpoint(checkpoint).encode("utf-8"))
            if estimate == checkpoint.checkpoint_tokens:
                break
            checkpoint = replace(checkpoint, checkpoint_tokens=estimate)
        return checkpoint

    @abstractmethod
    def serialize(self, messages: list[MessageT]) -> object: ...

    @abstractmethod
    def provider_input_pressure(self, messages: list[MessageT]) -> ProviderInputPressure | None: ...

    @abstractmethod
    def with_provider_input_baseline(self, message: MessageT, baseline_tokens: int) -> MessageT: ...

    @abstractmethod
    def tool_outcomes(self, messages: list[MessageT]) -> tuple[str, ...]: ...


def cache_epoch_key(cache_key: str, checkpoint_id: str) -> str:
    suffix = f"_h{sha256(checkpoint_id.encode()).hexdigest()[:8]}"
    while _CACHE_EPOCH_SUFFIX.search(cache_key):
        cache_key = _CACHE_EPOCH_SUFFIX.sub("", cache_key)
    return f"{cache_key}{suffix}"
