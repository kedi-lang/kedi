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
from .telemetry import compact_history_span


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
        if not snapshot.entries or snapshot.total_tokens < policy.threshold_tokens:
            plan = self.processor.plan(snapshot, policy)
        else:
            with span(
                "agent",
                "process history",
                "process_history",
                level="detailed",
                attributes={
                    "kedi.adapter.name": self.codec.adapter_shortname,
                    "kedi.history.compaction.mode": "kedi",
                    "kedi.history.compaction.input_messages": len(snapshot.entries),
                    "kedi.history.compaction.input_tokens": snapshot.total_tokens,
                },
            ) as current:
                plan = self.processor.plan(snapshot, policy)
                current.set_attribute(
                    "kedi.history.compaction.group_count",
                    len(snapshot.groups),
                )
                current.set_attribute(
                    "kedi.history.compaction.protected_group_count",
                    sum(group.protected for group in plan.groups),
                )
                current.set_attribute("kedi.history.compaction.applicable", plan.applicable)
                if plan.no_op_reason is not None:
                    current.set_attribute(
                        "kedi.history.compaction.no_op_reason",
                        plan.no_op_reason,
                    )
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
        cache_epoch_changed: bool | None = None,
    ) -> CompactionResult:
        with compact_history_span(
            mode="kedi",
            trigger="threshold",
            adapter_name=self.codec.adapter_shortname,
            input_messages=len(prepared.snapshot.entries),
            input_tokens=prepared.snapshot.total_tokens,
        ) as telemetry:
            result = self.coordinator.apply_checkpoint(
                state,
                snapshot=prepared.snapshot,
                plan=prepared.plan,
                checkpoint=checkpoint,
                codec=self.codec,
                acceptance_policy=prepared.acceptance_policy,
            )
            output_messages = None
            if result.processed is not None:
                replaced = result.processed.replaced_range
                output_messages = (
                    len(prepared.snapshot.entries) - (replaced.end - replaced.start) + 1
                )
            checkpoint_valid = result.outcome.applied and result.processed is not None
            telemetry.complete(
                result.outcome,
                input_messages=len(prepared.snapshot.entries),
                output_messages=output_messages,
                input_tokens=prepared.snapshot.total_tokens,
                output_tokens=result.outcome.result_tokens,
                selected_tokens=result.outcome.source_tokens,
                prefix_preserved=checkpoint_valid,
                cache_epoch_changed=(cache_epoch_changed if checkpoint_valid else False),
                checkpoint_valid=checkpoint_valid,
            )
            if checkpoint_valid:
                telemetry.checkpoint_validated()
                telemetry.checkpoint_committed()
                if cache_epoch_changed:
                    telemetry.cache_epoch_advanced()
            return result
