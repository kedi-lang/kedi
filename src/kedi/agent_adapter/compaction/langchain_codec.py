from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

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


class LangChainHistoryCodec:
    """Lossless inspector and range splicer for LangChain message history."""

    adapter_shortname = "langchain"
    state_key = "langchain_messages"

    def inspect(self, state: Mapping[str, object]) -> HistorySnapshot:
        from langchain_core.messages import BaseMessage

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
            storage_format=_storage_format(native_entries, native_type=BaseMessage),
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
            raise TypeError("LangChain history mixes native and serialized message values")
        replacement = self._checkpoint_message(
            checkpoint,
            serialized=snapshot.storage_format == "serialized",
        )
        current = state.get(self.state_key)
        if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
            raise TypeError("LangChain history state is not a message sequence")
        result = dict(state)
        result[self.state_key] = [
            *current[: window.start],
            replacement,
            *current[window.end :],
        ]
        return result

    @staticmethod
    def _inspect_entry(index: int, raw: object) -> NativeHistoryEntryRef:
        from langchain_core.messages import (
            AIMessage,
            AIMessageChunk,
            BaseMessage,
            ToolMessage,
            message_to_dict,
            messages_from_dict,
        )

        try:
            if isinstance(raw, BaseMessage):
                message = raw
                canonical = message_to_dict(raw)
            elif isinstance(raw, Mapping):
                message = messages_from_dict([cast("dict[str, Any]", dict(raw))])[0]
                canonical = raw
            else:
                raise TypeError
        except (KeyError, TypeError, ValueError):
            encoded = canonical_json_bytes(raw)
            fingerprint = fingerprint_value(raw)
            return NativeHistoryEntryRef(
                index=index,
                entry_id=f"langchain_{index}_{fingerprint[:16]}",
                fingerprint=fingerprint,
                kind="unknown",
                token_estimate=conservative_token_estimate(encoded),
                complete=False,
                unknown=True,
                searchable_text=encoded.decode("utf-8", errors="replace"),
            )

        open_ids: tuple[str, ...] = ()
        close_ids: tuple[str, ...] = ()
        part_kinds: list[str] = [message.type]
        content_open_ids, content_close_ids, malformed_lifecycle = _langchain_content_lifecycle(
            message.content
        )
        unknown = malformed_lifecycle or not _langchain_content_is_known(message.content)
        checkpoint = bool(message.additional_kwargs.get("kedi_compaction_checkpoint"))
        if isinstance(message, AIMessage):
            open_ids = _unique_ids(
                *(str(call["id"]) for call in message.tool_calls if call.get("id")),
                *content_open_ids,
            )
            part_kinds.extend("tool-call" for _ in message.tool_calls)
            if message.invalid_tool_calls:
                unknown = True
                part_kinds.extend("invalid-tool-call" for _ in message.invalid_tool_calls)
            part_kinds.extend(_langchain_content_kinds(message.content))
        elif isinstance(message, ToolMessage):
            close_ids = _unique_ids(message.tool_call_id, *content_close_ids)
            part_kinds.append("tool-result")
        else:
            open_ids = content_open_ids
            close_ids = content_close_ids
            part_kinds.extend(_langchain_content_kinds(message.content))

        if isinstance(message, AIMessage):
            close_ids = content_close_ids
        if _langchain_has_interrupt_metadata(message):
            part_kinds.append("interrupt")
            unknown = True

        kind = _langchain_entry_kind(message, checkpoint=checkpoint, unknown=unknown)
        unknown = unknown or kind == "unknown"
        encoded = canonical_json_bytes(canonical)
        fingerprint = fingerprint_value(canonical)
        return NativeHistoryEntryRef(
            index=index,
            entry_id=f"langchain_{index}_{fingerprint[:16]}",
            fingerprint=fingerprint,
            kind=kind,
            token_estimate=conservative_token_estimate(encoded),
            part_kinds=tuple(part_kinds),
            lifecycle_ids=open_ids,
            closes_lifecycle_ids=close_ids,
            artifact_refs=extract_artifact_refs(canonical),
            complete=not isinstance(message, AIMessageChunk),
            checkpoint=checkpoint,
            unknown=unknown,
            provider_required=_langchain_has_signed_reasoning(message),
            searchable_text=encoded.decode("utf-8", errors="replace"),
        )

    @staticmethod
    def _checkpoint_message(
        checkpoint: CompactionCheckpoint,
        *,
        serialized: bool,
    ) -> object:
        from langchain_core.messages import HumanMessage, message_to_dict

        message = HumanMessage(
            content=render_checkpoint(checkpoint),
            additional_kwargs={"kedi_compaction_checkpoint": checkpoint.checkpoint_id},
        )
        return message_to_dict(message) if serialized else message


def _langchain_entry_kind(
    message: object,
    *,
    checkpoint: bool,
    unknown: bool,
) -> HistoryEntryKind:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if unknown:
        return "unknown"
    if checkpoint:
        return "checkpoint"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, ToolMessage):
        return "tool-result"
    if isinstance(message, AIMessage):
        if message.tool_calls:
            return "tool-call"
        if _langchain_has_reasoning(message) and not _langchain_has_text(message):
            return "reasoning"
        return "assistant"
    return "unknown"


def _langchain_content_kinds(content: object) -> list[str]:
    if isinstance(content, str):
        return ["text"] if content else []
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return ["unknown-content"]
    kinds: list[str] = []
    for block in content:
        if isinstance(block, Mapping):
            mapping = cast("Mapping[str, object]", block)
            kinds.append(str(mapping.get("type", "mapping")))
        else:
            kinds.append(type(block).__name__)
    return kinds


_KNOWN_LANGCHAIN_CONTENT_TYPES = frozenset(
    {
        "audio",
        "citation",
        "file",
        "image",
        "input_audio",
        "input_file",
        "input_image",
        "input_text",
        "input_video",
        "output_text",
        "reasoning",
        "reasoning_content",
        "redacted_thinking",
        "server_tool_call",
        "server_tool_call_chunk",
        "server_tool_result",
        "text",
        "text-plain",
        "thinking",
        "tool_call",
        "tool_call_chunk",
        "tool_result",
        "tool_use",
        "video",
    }
)


def _langchain_content_is_known(content: object) -> bool:
    if isinstance(content, str):
        return True
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return False
    for block in content:
        if isinstance(block, str):
            continue
        if not isinstance(block, Mapping):
            return False
        mapping = cast("Mapping[object, object]", block)
        if mapping.get("type") not in _KNOWN_LANGCHAIN_CONTENT_TYPES:
            return False
    return True


def _langchain_content_lifecycle(
    content: object,
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        return (), (), False
    opened: list[str] = []
    closed: list[str] = []
    malformed = False
    for block in content:
        if not isinstance(block, Mapping):
            continue
        mapping = cast("Mapping[object, object]", block)
        kind = mapping.get("type")
        if kind in {"tool_call", "server_tool_call", "tool_use"}:
            identifier = mapping.get("id") or mapping.get("tool_call_id")
            if identifier:
                opened.append(str(identifier))
            else:
                malformed = True
        elif kind in {"server_tool_result", "tool_result"}:
            identifier = mapping.get("tool_call_id") or mapping.get("tool_use_id")
            if identifier:
                closed.append(str(identifier))
            else:
                malformed = True
    return _unique_ids(*opened), _unique_ids(*closed), malformed


def _unique_ids(*identifiers: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(identifier for identifier in identifiers if identifier))


def _langchain_has_reasoning(message: object) -> bool:
    content = getattr(message, "content", None)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") in {
                "reasoning",
                "thinking",
                "reasoning_content",
            }:
                return True
    additional = getattr(message, "additional_kwargs", {})
    return isinstance(additional, Mapping) and any(
        key in additional for key in ("reasoning", "reasoning_content", "thinking")
    )


def _langchain_has_signed_reasoning(message: object) -> bool:
    additional = getattr(message, "additional_kwargs", {})
    response = getattr(message, "response_metadata", {})
    if _contains_provider_replay_data(additional) or _contains_provider_replay_data(response):
        return True
    content = getattr(message, "content", None)
    return _contains_provider_replay_data(content)


def _contains_provider_replay_data(value: object) -> bool:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            mapping = cast("Mapping[object, object]", current)
            if any(
                mapping.get(key) not in (None, "", [], {})
                for key in (
                    "encrypted_content",
                    "reasoning_signature",
                    "signature",
                    "thinking_signature",
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


def _langchain_has_interrupt_metadata(message: object) -> bool:
    for metadata in (
        getattr(message, "additional_kwargs", {}),
        getattr(message, "response_metadata", {}),
    ):
        if isinstance(metadata, Mapping) and any(
            key in metadata
            for key in ("__interrupt__", "approval_request", "interrupt", "interrupts")
        ):
            return True
    return False


def _langchain_has_text(message: object) -> bool:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return bool(content)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        return any(
            isinstance(block, str)
            or (isinstance(block, Mapping) and block.get("type") in {"text", "input_text"})
            for block in content
        )
    return False


def _storage_format(
    values: tuple[object, ...],
    *,
    native_type: type[object],
) -> HistoryStorageFormat:
    if not values or all(isinstance(value, Mapping) for value in values):
        return "serialized"
    if all(isinstance(value, native_type) for value in values):
        return "native"
    return "mixed"
