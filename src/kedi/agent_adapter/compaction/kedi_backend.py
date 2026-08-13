from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...telemetry import span
from .codecs import AdapterHistoryCodec
from .coordinator import CompactionCoordinator
from .models import (
    CheckpointAcceptancePolicy,
    CompactionCheckpoint,
    CompactionResult,
    HistoryReductionPlan,
    HistoryReductionPolicy,
    HistorySnapshot,
)
from .processor import HistoryProcessor


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedCompaction:
    snapshot: HistorySnapshot
    plan: HistoryReductionPlan
    acceptance_policy: CheckpointAcceptancePolicy


class KediCompactionTransaction:
    """Track A prepare/commit facade; it deliberately does not summarize or auto-run."""

    def __init__(
        self,
        codec: AdapterHistoryCodec,
        *,
        processor: HistoryProcessor | None = None,
        coordinator: CompactionCoordinator | None = None,
    ) -> None:
        self.codec = codec
        self.processor = processor or HistoryProcessor()
        self.coordinator = coordinator or CompactionCoordinator()

    def prepare(
        self,
        state: Mapping[str, object],
        policy: HistoryReductionPolicy,
        *,
        acceptance_policy: CheckpointAcceptancePolicy | None = None,
    ) -> PreparedCompaction:
        snapshot = self.processor.inspect(state, codec=self.codec)
        return self.prepare_snapshot(
            snapshot,
            policy,
            acceptance_policy=acceptance_policy,
        )

    def prepare_snapshot(
        self,
        snapshot: HistorySnapshot,
        policy: HistoryReductionPolicy,
        *,
        acceptance_policy: CheckpointAcceptancePolicy | None = None,
    ) -> PreparedCompaction:
        """Plan from one already-inspected snapshot without reading native state again."""

        if snapshot.adapter_shortname != self.codec.adapter_shortname:
            raise ValueError("history snapshot belongs to a different adapter codec")
        with span(
            "agent",
            "process history",
            "process_history",
            level="detailed",
            attributes={
                "gen_ai.operation.name": "process_history",
                "kedi.compaction.adapter": self.codec.adapter_shortname,
            },
        ) as current:
            plan = self.processor.plan(snapshot, policy)
            current.set_attribute("kedi.compaction.source_entries", len(snapshot.entries))
            current.set_attribute("kedi.compaction.source_tokens", snapshot.total_tokens)
            current.set_attribute("kedi.compaction.group_count", len(snapshot.groups))
            current.set_attribute(
                "kedi.compaction.protected_group_count",
                sum(group.protected for group in plan.groups),
            )
            current.set_attribute("kedi.compaction.applicable", plan.applicable)
            if plan.no_op_reason is not None:
                current.set_attribute("kedi.compaction.no_op_reason", plan.no_op_reason)
            return PreparedCompaction(
                snapshot=snapshot,
                plan=plan,
                acceptance_policy=acceptance_policy or CheckpointAcceptancePolicy(),
            )

    def commit(
        self,
        state: Mapping[str, object],
        *,
        prepared: PreparedCompaction,
        checkpoint: CompactionCheckpoint,
    ) -> CompactionResult:
        with span(
            "agent",
            "compact history",
            "compact_history",
            level="detailed",
            attributes={
                "gen_ai.operation.name": "compact_history",
                "kedi.compaction.adapter": self.codec.adapter_shortname,
                "kedi.compaction.source_digest": prepared.snapshot.source_digest,
            },
        ) as current:
            try:
                result = self.coordinator.apply_checkpoint(
                    state,
                    snapshot=prepared.snapshot,
                    plan=prepared.plan,
                    checkpoint=checkpoint,
                    codec=self.codec,
                    acceptance_policy=prepared.acceptance_policy,
                )
            except BaseException as exc:
                current.record_exception(exc)
                raise
            current.set_attribute("kedi.compaction.applied", result.outcome.applied)
            if result.outcome.source_tokens is not None:
                current.set_attribute(
                    "kedi.compaction.source_tokens",
                    result.outcome.source_tokens,
                )
            if result.outcome.result_tokens is not None:
                current.set_attribute(
                    "kedi.compaction.result_tokens",
                    result.outcome.result_tokens,
                )
            return result
