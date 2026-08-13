from __future__ import annotations

from dataclasses import dataclass, replace

from kedi.agent_adapter.conversation import ConversationState
from kedi.artifacts import ArtifactHistoryEvent

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
        live_refs, known_refs = _artifact_lifecycle(conversation.history.snapshot())
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
        with conversation.adapter_turn(self.codec.adapter_shortname) as lane:
            if lane.cache_epoch != prepared.cache_epoch:
                raise CompactionValidationError("conversation cache epoch changed")
            result = self.transaction.commit(
                lane.resume_state,
                prepared=prepared.prepared,
                checkpoint=checkpoint,
            )
            if result.processed is None:
                raise CompactionValidationError("compaction produced no replacement state")
            lane.result_state.update(result.processed.state)
            lane.cache_prefix_changed = True
        return result


def _artifact_lifecycle(
    events: tuple[ArtifactHistoryEvent, ...],
) -> tuple[frozenset[str], frozenset[str]]:
    statuses: dict[str, str] = {}
    for event in events:
        if event.ref_id is None:
            continue
        if event.kind == "artifact_created":
            statuses[event.ref_id] = "active"
        elif event.kind in {"artifact_released", "artifact_expired"}:
            statuses[event.ref_id] = event.kind
    return (
        frozenset(ref_id for ref_id, status in statuses.items() if status == "active"),
        frozenset(statuses),
    )
