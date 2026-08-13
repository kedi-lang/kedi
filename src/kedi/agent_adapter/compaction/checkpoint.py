from __future__ import annotations

import json

from .models import CompactionCheckpoint


def checkpoint_payload(checkpoint: CompactionCheckpoint) -> dict[str, object]:
    """Return the canonical provider-visible checkpoint payload."""

    payload: dict[str, object] = {
        "id": checkpoint.checkpoint_id,
        "version": checkpoint.version,
        "summary": checkpoint.summary,
        "durable_facts": list(checkpoint.durable_facts),
        "decisions": list(checkpoint.decisions),
        "constraints": list(checkpoint.constraints),
        "pending_work": list(checkpoint.pending_work),
        "resources": list(checkpoint.resources),
        "completed_outcomes": list(checkpoint.completed_outcomes),
        "artifact_refs": list(checkpoint.artifact_refs),
        "errors_and_lessons": list(checkpoint.errors_and_lessons),
    }
    if checkpoint.anchors:
        payload["anchors"] = [
            {
                "id": anchor.anchor_id,
                "kind": anchor.kind,
                "content": anchor.content,
            }
            for anchor in checkpoint.anchors
        ]
    return payload


def render_checkpoint(checkpoint: CompactionCheckpoint) -> str:
    """Render one deterministic replay message shared by every history codec."""

    return (
        "<kedi_compaction_checkpoint>\n"
        + json.dumps(
            checkpoint_payload(checkpoint),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n</kedi_compaction_checkpoint>"
    )
