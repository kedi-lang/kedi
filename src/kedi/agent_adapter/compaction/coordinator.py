from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from .codecs import AdapterHistoryCodec
from .models import (
    CheckpointAcceptancePolicy,
    CompactionCheckpoint,
    CompactionOutcome,
    CompactionResult,
    CompactionValidationError,
    CompactionWindow,
    HistoryReductionPlan,
    HistorySnapshot,
    ProcessedHistory,
)
from .validation import validate_compaction_transaction


class CompactionCoordinator:
    """Validate and atomically compose one adapter-native checkpoint splice."""

    def apply_checkpoint(
        self,
        state: Mapping[str, object],
        *,
        snapshot: HistorySnapshot,
        plan: HistoryReductionPlan,
        checkpoint: CompactionCheckpoint,
        codec: AdapterHistoryCodec,
        acceptance_policy: CheckpointAcceptancePolicy | None = None,
    ) -> CompactionResult:
        validate_compaction_transaction(
            state,
            snapshot=snapshot,
            plan=plan,
            checkpoint=checkpoint,
            codec=codec,
            acceptance_policy=acceptance_policy,
        )
        window = cast(CompactionWindow, plan.window)
        current_snapshot = codec.inspect(state)
        updated = codec.replace_range(
            state,
            snapshot=snapshot,
            window=window.native_range,
            checkpoint=checkpoint,
        )
        result_snapshot = codec.inspect(updated)
        start = window.native_range.start
        prefix = tuple(entry.fingerprint for entry in result_snapshot.entries[:start])
        tail = tuple(entry.fingerprint for entry in result_snapshot.entries[start + 1 :])
        if prefix != plan.unchanged_prefix_fingerprints:
            raise CompactionValidationError("codec mutated the retained history prefix")
        if tail != plan.unchanged_tail_fingerprints:
            raise CompactionValidationError("codec mutated the retained history tail")
        if not all(
            retained is source
            for retained, source in zip(
                result_snapshot.native_entries[:start],
                current_snapshot.native_entries[:start],
            )
        ):
            raise CompactionValidationError("codec replaced retained native prefix values")
        if not all(
            retained is source
            for retained, source in zip(
                result_snapshot.native_entries[start + 1 :],
                current_snapshot.native_entries[window.native_range.end :],
            )
        ):
            raise CompactionValidationError("codec replaced retained native tail values")
        checkpoint_entry = result_snapshot.entries[start]
        if not checkpoint_entry.checkpoint:
            raise CompactionValidationError("codec did not produce a checkpoint marker")
        actual_checkpoint_tokens = max(
            checkpoint.checkpoint_tokens,
            checkpoint_entry.token_estimate,
        )
        if actual_checkpoint_tokens > plan.policy.checkpoint_budget_tokens:
            raise CompactionValidationError("serialized checkpoint exceeds its token budget")
        processed = ProcessedHistory(
            state=updated,
            replaced_range=window.native_range,
            checkpoint_id=checkpoint.checkpoint_id,
            source_digest=snapshot.source_digest,
            result_digest=result_snapshot.source_digest,
            prefix_fingerprints=prefix,
            tail_fingerprints=tail,
        )
        return CompactionResult(
            outcome=CompactionOutcome(
                backend="kedi",
                applied=True,
                trigger="threshold",
                checkpoint_id=checkpoint.checkpoint_id,
                source_tokens=window.source_tokens,
                result_tokens=(
                    snapshot.total_tokens - window.source_tokens + actual_checkpoint_tokens
                ),
            ),
            processed=processed,
        )
