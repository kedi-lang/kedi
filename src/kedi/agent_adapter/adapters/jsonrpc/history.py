"""History processing over Kedi's transcript schema (JSON-RPC bridges in transcript mode)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ...compaction.codecs import (
    ESTIMATOR_VERSION,
    conservative_token_estimate,
    extract_artifact_refs,
    fingerprint_sequence,
    fingerprint_value,
    projected_json_bytes,
)
from ...compaction.models import (
    HistoryEntryKind,
    HistoryOrigin,
    HistorySnapshot,
    NativeHistoryEntryRef,
)

JsonObject = dict[str, Any]


class TranscriptHistoryCodec:
    """Inspects transcript messages for Kedi's history processor runner.

    A ``toolCall`` part opens a lifecycle that the ``toolResult`` with its ``id`` closes, so
    the runner keeps a call and its result together.
    """

    adapter_shortname = "rpc"
    state_key = "transcript"

    def clone(self, messages: Sequence[JsonObject]) -> list[JsonObject]:
        if not all(isinstance(message, Mapping) for message in messages):
            raise TypeError("all values must be transcript messages")
        return copy.deepcopy([dict(message) for message in messages])

    def inspect(self, messages: Sequence[JsonObject]) -> HistorySnapshot:
        native = tuple(messages)
        entries = tuple(_entry(index, message) for index, message in enumerate(native))
        return HistorySnapshot(
            adapter_shortname=self.adapter_shortname,
            state_key=self.state_key,
            source_version=1,
            source_digest=fingerprint_sequence(list(native)),
            native_entries=native,
            entries=entries,
            total_tokens=sum(entry.token_estimate for entry in entries),
            estimator_version=ESTIMATOR_VERSION,
            # Transcript messages are this adapter's native form.
            storage_format="native",
        )


def _entry(index: int, message: Any) -> NativeHistoryEntryRef:
    encoded, media_tokens = projected_json_bytes(message)
    fingerprint = fingerprint_value(message)
    parts = message.get("parts") if isinstance(message, Mapping) else None
    role = message.get("role") if isinstance(message, Mapping) else None
    if not isinstance(parts, list) or role not in ("user", "assistant", "tool"):
        return NativeHistoryEntryRef(
            index=index,
            entry_id=f"rpc_{index}_{fingerprint[:16]}",
            fingerprint=fingerprint,
            kind="unknown",
            token_estimate=conservative_token_estimate(encoded) + media_tokens,
            unknown=True,
            searchable_text=encoded.decode("utf-8", errors="replace"),
        )
    kinds = [part.get("type") for part in parts if isinstance(part, Mapping)]
    calls = [part for part in parts if isinstance(part, Mapping) and part.get("type") == "toolCall"]
    results = [
        part for part in parts if isinstance(part, Mapping) and part.get("type") == "toolResult"
    ]
    reasoning = bool(kinds) and all(kind == "reasoning" for kind in kinds)
    kind: HistoryEntryKind = (
        "user"
        if role == "user"
        else "tool-result"
        if role == "tool"
        else "tool-call"
        if calls
        else "reasoning"
        if reasoning
        else "assistant"
    )
    return NativeHistoryEntryRef(
        index=index,
        entry_id=f"rpc_{index}_{fingerprint[:16]}",
        fingerprint=fingerprint,
        kind=kind,
        token_estimate=conservative_token_estimate(encoded) + media_tokens,
        part_kinds=(str(role), *(str(kind) for kind in kinds)),
        lifecycle_ids=tuple(str(call.get("id")) for call in calls),
        closes_lifecycle_ids=tuple(str(result.get("id")) for result in results),
        artifact_refs=extract_artifact_refs(message),
        # A backend's own data (an encrypted reasoning block) must go back as it was.
        provider_required="providerData" in message and "reasoning" in kinds,
        searchable_text=encoded.decode("utf-8", errors="replace"),
        origins=tuple(
            HistoryOrigin(kind="tool-call", tool_name=call.get("name"), tool_call_id=call.get("id"))
            for call in calls
        )
        + tuple(HistoryOrigin(kind="tool-result", tool_call_id=part.get("id")) for part in results),
    )
