from __future__ import annotations

from dataclasses import dataclass, replace

from kedi.agent_adapter.conversation import ConversationState

from .codecs import AdapterHistoryCodec
from .kedi_backend import KediCompactionTransaction, PreparedCompaction
from .models import (
    CheckpointAcceptancePolicy,
    CompactionCheckpoint,
    CompactionResult,
    CompactionValidationError,
    HistoryReductionPolicy,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedConversationCompaction:
    """A Track A plan bound to one conversation lane and cache epoch."""

    adapter_shortname: str
    lane_identity: str
    cache_epoch: int
    prepared: PreparedCompaction


class ConversationCompactionTransaction:
    """Atomically apply deterministic Track A splices to ``ConversationState``."""

    def __init__(
        self,
        codec: AdapterHistoryCodec,
        *,
        transaction: KediCompactionTransaction | None = None,
    ) -> None:
        self.codec = codec
        self.transaction = transaction or KediCompactionTransaction(codec)
        if self.transaction.codec.adapter_shortname != codec.adapter_shortname:
            raise ValueError("compaction transaction and codec must own the same adapter")

    def prepare(
        self,
        conversation: ConversationState,
        policy: HistoryReductionPolicy,
        *,
        acceptance_policy: CheckpointAcceptancePolicy | None = None,
    ) -> PreparedConversationCompaction:
        lane = conversation.adapter_state_snapshot(self.codec.adapter_shortname)
        snapshot = self.transaction.processor.inspect(lane.state, codec=self.codec)
        live_refs, known_refs = conversation.artifacts.lifecycle_snapshot()
        referenced = {ref_id for entry in snapshot.entries for ref_id in entry.artifact_refs}
        effective_policy = replace(
            policy,
            live_artifact_refs=policy.live_artifact_refs.union(live_refs),
            missing_artifact_refs=policy.missing_artifact_refs.union(
                referenced.difference(known_refs)
            ),
        )
        return PreparedConversationCompaction(
            adapter_shortname=self.codec.adapter_shortname,
            lane_identity=lane.lane_identity,
            cache_epoch=lane.cache_epoch,
            prepared=self.transaction.prepare_snapshot(
                snapshot,
                effective_policy,
                acceptance_policy=acceptance_policy,
            ),
        )

    def commit(
        self,
        conversation: ConversationState,
        *,
        prepared: PreparedConversationCompaction,
        checkpoint: CompactionCheckpoint,
    ) -> CompactionResult:
        if prepared.adapter_shortname != self.codec.adapter_shortname:
            raise CompactionValidationError(
                "prepared conversation compaction belongs to a different adapter"
            )
        with conversation.adapter_turn(
            self.codec.adapter_shortname,
            lane_identity=prepared.lane_identity,
        ) as lane:
            if lane.cache_epoch != prepared.cache_epoch:
                raise CompactionValidationError("conversation cache epoch changed")
            result = self.transaction.commit(
                lane.resume_state,
                prepared=prepared.prepared,
                checkpoint=checkpoint,
                cache_epoch_changed=True,
            )
            if result.processed is None:
                raise CompactionValidationError("compaction produced no replacement state")
            lane.result_state.update(result.processed.state)
            lane.cache_prefix_changed = True
        return result
