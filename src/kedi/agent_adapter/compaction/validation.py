from __future__ import annotations

import re
from collections.abc import Mapping

from .checkpoint import checkpoint_payload
from .codecs import (
    AdapterHistoryCodec,
    canonical_json_bytes,
    extract_artifact_refs,
)
from .models import (
    CheckpointAcceptancePolicy,
    CompactionCheckpoint,
    CompactionValidationError,
    HistoryReductionPlan,
    HistorySnapshot,
    NativeHistoryEntryRef,
)
from .selection import plan_history_reduction

_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?:api[_-]?(?:key|token)|access[_-]?token|auth[_-]?token|password|secret)"
    r"\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9][A-Za-z0-9._~+/=-]{11,})"
)
_SECRET_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<value>(?:github_pat|sk|pk|ghp|gho|ghu|ghs|ghr|glpat|xoxb|"
    r"xoxp|xoxa|xoxr)[_-][A-Za-z0-9_-]{12,})(?![A-Za-z0-9_])"
)
_AWS_ACCESS_KEY_PATTERN = re.compile(r"(?<![A-Z0-9])(?P<value>AKIA[A-Z0-9]{16})(?![A-Z0-9])")


def validate_compaction_transaction(
    state: Mapping[str, object],
    *,
    snapshot: HistorySnapshot,
    plan: HistoryReductionPlan,
    checkpoint: CompactionCheckpoint,
    codec: AdapterHistoryCodec,
    acceptance_policy: CheckpointAcceptancePolicy | None = None,
) -> None:
    """Reject stale, sparse, protected, or uneconomical history replacement."""

    current = codec.inspect(state)
    if (
        current.adapter_shortname != snapshot.adapter_shortname
        or current.state_key != snapshot.state_key
    ):
        raise CompactionValidationError("compaction codec no longer owns the source history")
    if current.source_version != snapshot.source_version:
        raise CompactionValidationError("compaction codec version changed")
    if current.estimator_version != snapshot.estimator_version:
        raise CompactionValidationError("compaction estimator version changed")
    if snapshot.storage_format == "mixed":
        raise CompactionValidationError("mixed native and serialized history cannot be compacted")
    if current.source_digest != snapshot.source_digest:
        raise CompactionValidationError("compaction snapshot is stale")
    if (
        plan.source_digest != snapshot.source_digest
        or plan.source_version != snapshot.source_version
    ):
        raise CompactionValidationError("compaction plan does not match its source snapshot")
    if plan.estimator_version != snapshot.estimator_version:
        raise CompactionValidationError("compaction estimator version changed")
    if plan != plan_history_reduction(snapshot, plan.policy):
        raise CompactionValidationError("compaction plan is not the deterministic source plan")
    window = plan.window
    if window is None:
        raise CompactionValidationError("compaction plan has no selected window")
    native_range = window.native_range
    source_entries = snapshot.entries[native_range.start : native_range.end]
    if checkpoint.source_start_id != source_entries[0].entry_id:
        raise CompactionValidationError("checkpoint source start does not match the plan")
    if checkpoint.source_end_id != source_entries[-1].entry_id:
        raise CompactionValidationError("checkpoint source end does not match the plan")
    if checkpoint.source_digest != snapshot.source_digest:
        raise CompactionValidationError("checkpoint source digest does not match the snapshot")
    if checkpoint.source_tokens != window.source_tokens:
        raise CompactionValidationError("checkpoint source token count does not match the window")
    if checkpoint.checkpoint_tokens > plan.policy.checkpoint_budget_tokens:
        raise CompactionValidationError("checkpoint exceeds its token budget")
    _validate_checkpoint_acceptance(
        source_entries=source_entries,
        checkpoint=checkpoint,
        policy=acceptance_policy or CheckpointAcceptancePolicy(),
    )


def _validate_checkpoint_acceptance(
    *,
    source_entries: tuple[NativeHistoryEntryRef, ...],
    checkpoint: CompactionCheckpoint,
    policy: CheckpointAcceptancePolicy,
) -> None:
    searchable_texts = tuple(entry.searchable_text for entry in source_entries)
    source_artifact_refs = {ref_id for entry in source_entries for ref_id in entry.artifact_refs}
    explicit_refs = checkpoint.artifact_refs
    if len(explicit_refs) != len(set(explicit_refs)):
        raise CompactionValidationError("checkpoint contains duplicate artifact references")
    if any(ref_id not in source_artifact_refs for ref_id in explicit_refs):
        raise CompactionValidationError(
            "checkpoint contains an artifact reference outside its source window"
        )

    payload = checkpoint_payload(checkpoint)
    rendered_refs = set(extract_artifact_refs(payload))
    if not rendered_refs.issubset(source_artifact_refs):
        raise CompactionValidationError(
            "checkpoint text contains an artifact reference outside its source window"
        )

    if checkpoint.anchors != policy.required_anchors:
        expected_ids = tuple(anchor.anchor_id for anchor in policy.required_anchors)
        received_ids = tuple(anchor.anchor_id for anchor in checkpoint.anchors)
        if not set(expected_ids).issubset(received_ids):
            raise CompactionValidationError("checkpoint omits required continuation anchors")
        raise CompactionValidationError(
            "checkpoint continuation anchors do not match the acceptance policy"
        )

    checkpoint_text = canonical_json_bytes(payload).decode("utf-8")
    source_secrets = {value for text in searchable_texts for value in _secret_like_values(text)}
    if any(secret in checkpoint_text for secret in source_secrets):
        raise CompactionValidationError(
            "checkpoint reproduces secret-like material from its source window"
        )


def _secret_like_values(text: str) -> frozenset[str]:
    values: set[str] = set()
    for pattern in (
        _SECRET_ASSIGNMENT_PATTERN,
        _SECRET_TOKEN_PATTERN,
        _AWS_ACCESS_KEY_PATTERN,
    ):
        values.update(match.group("value") for match in pattern.finditer(text))
    return frozenset(values)
