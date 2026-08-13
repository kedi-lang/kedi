from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

from .checkpoint import render_checkpoint
from .codecs import (
    ESTIMATOR_VERSION,
    canonical_json_bytes,
    conservative_token_estimate,
    extract_artifact_refs,
    fingerprint_sequence,
    fingerprint_value,
)
from .models import (
    CompactionCheckpoint,
    HistoryEntryKind,
    HistoryRange,
    HistorySnapshot,
    HistoryStorageFormat,
    NativeHistoryEntryRef,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelRequest, ModelResponse


class PydanticHistoryCodec:
    """Lossless inspector and range splicer for Pydantic AI message history."""

    adapter_shortname = "pydantic"
    state_key = "pydantic_messages"

    def inspect(self, state: Mapping[str, object]) -> HistorySnapshot:
        from pydantic_ai.messages import ModelRequest, ModelResponse

        raw = state.get(self.state_key, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raw = ()
        native_entries = tuple(raw)
        entries = tuple(self._inspect_entry(index, value) for index, value in enumerate(raw))
        return HistorySnapshot(
            adapter_shortname=self.adapter_shortname,
            state_key=self.state_key,
            source_version=1,
            source_digest=fingerprint_sequence(list(native_entries)),
            native_entries=native_entries,
            entries=entries,
            total_tokens=sum(entry.token_estimate for entry in entries),
            estimator_version=ESTIMATOR_VERSION,
            storage_format=_storage_format(
                native_entries,
                native_types=(ModelRequest, ModelResponse),
            ),
        )

    def replace_range(
        self,
        state: Mapping[str, object],
        *,
        snapshot: HistorySnapshot,
        window: HistoryRange,
        checkpoint: CompactionCheckpoint,
    ) -> dict[str, object]:
        if snapshot.storage_format == "mixed":
            raise TypeError("Pydantic history mixes native and serialized message values")
        replacement = self._checkpoint_message(
            checkpoint,
            serialized=snapshot.storage_format == "serialized",
        )
        current = state.get(self.state_key)
        if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
            raise TypeError("Pydantic history state is not a message sequence")
        result = dict(state)
        result[self.state_key] = [
            *current[: window.start],
            replacement,
            *current[window.end :],
        ]
        return result

    @staticmethod
    def _inspect_entry(index: int, raw: object) -> NativeHistoryEntryRef:
        from pydantic_ai.messages import (
            BaseToolCallPart,
            BaseToolReturnPart,
            CompactionPart,
            FilePart,
            ModelMessagesTypeAdapter,
            ModelRequest,
            RetryPromptPart,
            SystemPromptPart,
            TextPart,
            ThinkingPart,
            ToolAvailabilityDeltaPart,
            UserPromptPart,
        )

        try:
            message = ModelMessagesTypeAdapter.validate_python([raw])[0]
            canonical = ModelMessagesTypeAdapter.dump_python([message], mode="json")[0]
        except (TypeError, ValueError):
            encoded = canonical_json_bytes(raw)
            fingerprint = fingerprint_value(raw)
            return NativeHistoryEntryRef(
                index=index,
                entry_id=f"pydantic_{index}_{fingerprint[:16]}",
                fingerprint=fingerprint,
                kind="unknown",
                token_estimate=conservative_token_estimate(encoded),
                complete=False,
                unknown=True,
                searchable_text=encoded.decode("utf-8", errors="replace"),
            )

        part_kinds = tuple(
            getattr(part, "part_kind", type(part).__name__) for part in message.parts
        )
        open_ids = tuple(
            part.tool_call_id
            for part in message.parts
            if isinstance(part, BaseToolCallPart) and part.tool_call_id
        )
        close_ids = tuple(
            part.tool_call_id
            for part in message.parts
            if isinstance(part, (BaseToolReturnPart, RetryPromptPart)) and part.tool_call_id
        )
        checkpoint = any(isinstance(part, CompactionPart) for part in message.parts) or bool(
            message.metadata and message.metadata.get("kedi_compaction_checkpoint")
        )
        unknown = any(
            not isinstance(
                part,
                (
                    CompactionPart,
                    FilePart,
                    RetryPromptPart,
                    SystemPromptPart,
                    TextPart,
                    ThinkingPart,
                    BaseToolCallPart,
                    BaseToolReturnPart,
                    ToolAvailabilityDeltaPart,
                    UserPromptPart,
                ),
            )
            for part in message.parts
        )
        kind = _pydantic_entry_kind(message, checkpoint=checkpoint, unknown=unknown)
        unknown = unknown or kind == "unknown"
        has_mixed_system_prompt = any(
            isinstance(part, SystemPromptPart) for part in message.parts
        ) and any(not isinstance(part, SystemPromptPart) for part in message.parts)
        encoded = canonical_json_bytes(canonical)
        fingerprint = fingerprint_value(canonical)
        return NativeHistoryEntryRef(
            index=index,
            entry_id=f"pydantic_{index}_{fingerprint[:16]}",
            fingerprint=fingerprint,
            kind=kind,
            token_estimate=conservative_token_estimate(encoded),
            part_kinds=part_kinds,
            lifecycle_ids=open_ids,
            closes_lifecycle_ids=close_ids,
            artifact_refs=extract_artifact_refs(canonical),
            complete=getattr(message, "state", "complete") == "complete",
            checkpoint=checkpoint,
            unknown=unknown,
            provider_required=(
                (isinstance(message, ModelRequest) and bool(message.instructions))
                or has_mixed_system_prompt
                or any(isinstance(part, ToolAvailabilityDeltaPart) for part in message.parts)
                or any(
                    isinstance(part, ThinkingPart)
                    and bool(part.signature or part.id or part.provider_details)
                    for part in message.parts
                )
                or _contains_provider_replay_data(canonical)
            ),
            searchable_text=encoded.decode("utf-8", errors="replace"),
        )

    @staticmethod
    def _checkpoint_message(
        checkpoint: CompactionCheckpoint,
        *,
        serialized: bool,
    ) -> object:
        from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

        message = ModelRequest(
            parts=[UserPromptPart(content=render_checkpoint(checkpoint))],
            timestamp=datetime.fromtimestamp(0, tz=timezone.utc),
            metadata={"kedi_compaction_checkpoint": checkpoint.checkpoint_id},
        )
        if serialized:
            return ModelMessagesTypeAdapter.dump_python([message], mode="json")[0]
        return message


def _pydantic_entry_kind(
    message: ModelRequest | ModelResponse,
    *,
    checkpoint: bool,
    unknown: bool,
) -> HistoryEntryKind:
    from pydantic_ai.messages import (
        BaseToolCallPart,
        BaseToolReturnPart,
        FilePart,
        ModelRequest,
        RetryPromptPart,
        SystemPromptPart,
        TextPart,
        ThinkingPart,
        ToolAvailabilityDeltaPart,
        UserPromptPart,
    )

    if unknown:
        return "unknown"
    if checkpoint:
        return "checkpoint"
    parts = message.parts
    if isinstance(message, ModelRequest):
        if any(isinstance(part, UserPromptPart) for part in parts):
            return "user"
        if any(isinstance(part, BaseToolReturnPart) for part in parts):
            return "tool-result"
        if any(isinstance(part, RetryPromptPart) for part in parts):
            return "retry"
        if any(isinstance(part, SystemPromptPart) for part in parts):
            return "system"
        if any(isinstance(part, ToolAvailabilityDeltaPart) for part in parts):
            return "system"
    if any(isinstance(part, BaseToolCallPart) for part in parts):
        return "tool-call"
    if any(isinstance(part, BaseToolReturnPart) for part in parts):
        return "tool-result"
    if any(isinstance(part, TextPart) for part in parts):
        return "assistant"
    if any(isinstance(part, FilePart) for part in parts):
        return "assistant"
    if any(isinstance(part, ThinkingPart) for part in parts):
        return "reasoning"
    return "unknown"


def _contains_provider_replay_data(value: object) -> bool:
    """Detect opaque provider payloads that must remain byte-for-byte replayable."""

    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            mapping = cast("Mapping[object, object]", current)
            if any(
                mapping.get(key) not in (None, "", [], {})
                for key in (
                    "signature",
                    "reasoning_signature",
                    "thinking_signature",
                    "encrypted_content",
                )
            ):
                return True
            pending.extend(mapping.values())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            pending.extend(current)
    return False


def _storage_format(
    values: tuple[object, ...],
    *,
    native_types: tuple[type[object], ...],
) -> HistoryStorageFormat:
    if not values or all(isinstance(value, Mapping) for value in values):
        return "serialized"
    if all(isinstance(value, native_types) for value in values):
        return "native"
    return "mixed"
