from __future__ import annotations

import asyncio
import inspect
import re
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from kedi.agent_adapter.compaction.classification import classify_snapshot
from kedi.agent_adapter.compaction.models import (
    HistorySnapshot,
    HistoryUnitGroup,
    NativeHistoryEntryRef,
)
from kedi.agent_adapter.conversation import AdapterConversationContext

MessageT = TypeVar("MessageT")


class HistoryProcessingError(RuntimeError):
    """Base error raised by Kedi's user-defined history processing boundary."""


class HistoryProcessingValidationError(HistoryProcessingError):
    """Raised before model I/O when processed native history is unsafe to replay."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryProcessorContext(Generic[MessageT]):
    """Detached native history plus framework-neutral editing metadata."""

    messages: tuple[MessageT, ...]
    adapter_shortname: str
    model_id: str | None
    request_index: int
    cache_epoch: int | None
    estimated_tokens: int
    entries: tuple[NativeHistoryEntryRef, ...]
    groups: tuple[HistoryUnitGroup, ...]


HistoryProcessorResult: TypeAlias = Sequence[MessageT] | Awaitable[Sequence[MessageT]]
HistoryProcessorCallback: TypeAlias = Callable[
    [HistoryProcessorContext[MessageT]], HistoryProcessorResult[MessageT]
]


class HistoryMessageCodec(Protocol[MessageT]):
    """Native cloning and inspection required by the shared processor runner."""

    adapter_shortname: str
    state_key: str

    def clone(self, messages: Sequence[MessageT]) -> list[MessageT]: ...

    def inspect(self, messages: Sequence[MessageT]) -> HistorySnapshot: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryProcessingOutcome(Generic[MessageT]):
    messages: list[MessageT]
    changed: bool
    source_digest: str
    result_digest: str


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryTransportDirective:
    reset_continuation: bool
    rewritten: bool
    cache_key: str | None


@dataclass(slots=True)
class _HistoryProcessingRunState:
    conversation: AdapterConversationContext | None
    native_scope_active: bool = False
    request_index: int = 0
    rewritten: bool = False
    reset_continuation: bool = False


_CACHE_SUFFIX = re.compile(r"_h[0-9a-f]{8}$")
_REWRITTEN_STATE_KEY = "history_processor_rewritten"


class HistoryProcessorRunner(Generic[MessageT]):
    """Task-local executor for one adapter's native history callback."""

    def __init__(
        self,
        processor: HistoryProcessorCallback[MessageT],
        codec: HistoryMessageCodec[MessageT],
    ) -> None:
        if not callable(processor):
            raise TypeError("history_processor must be callable")
        self.processor = processor
        self.codec = codec
        self._run_state: ContextVar[_HistoryProcessingRunState | None] = ContextVar(
            f"kedi_history_processor_run_{id(self)}",
            default=None,
        )
        self._inside_processor: ContextVar[bool] = ContextVar(
            f"kedi_history_processor_active_{id(self)}",
            default=False,
        )

    @contextmanager
    def run_scope(
        self,
        conversation: AdapterConversationContext | None,
        *,
        reuse_existing: bool = False,
    ) -> Iterator[None]:
        current = self._run_state.get()
        if reuse_existing and current is not None and not current.native_scope_active:
            # invoke() enters the native iter() once; tool-driven child runs must
            # get their own counters and pending continuation reset instead.
            current.native_scope_active = True
            try:
                yield
            finally:
                current.native_scope_active = False
            return
        rewritten = bool(
            conversation is not None and conversation.resume_state.get(_REWRITTEN_STATE_KEY)
        )
        if rewritten and conversation is not None:
            conversation.result_state[_REWRITTEN_STATE_KEY] = True
        token = self._run_state.set(
            _HistoryProcessingRunState(
                conversation=conversation,
                native_scope_active=reuse_existing,
                rewritten=rewritten,
            )
        )
        try:
            yield
        finally:
            self._run_state.reset(token)

    async def process(
        self,
        messages: Sequence[MessageT],
        *,
        model_id: str | None,
    ) -> HistoryProcessingOutcome[MessageT]:
        state = self._run_state.get()
        if state is None:
            raise HistoryProcessingError("history processor executed outside an adapter run scope")
        if self._inside_processor.get():
            raise HistoryProcessingError(
                "history_processor cannot recursively invoke the same configured adapter"
            )

        state.request_index += 1
        request_index = state.request_index
        source = self._clone(messages, label="source")
        if not source:
            raise HistoryProcessingValidationError(
                "history_processor source history must contain the current request"
            )
        source_snapshot = self._inspect(source, label="source")
        context = HistoryProcessorContext(
            messages=tuple(source),
            adapter_shortname=self.codec.adapter_shortname,
            model_id=model_id,
            request_index=request_index,
            cache_epoch=(
                state.conversation.cache_epoch if state.conversation is not None else None
            ),
            estimated_tokens=source_snapshot.total_tokens,
            entries=source_snapshot.entries,
            groups=source_snapshot.groups,
        )

        active_token = self._inside_processor.set(True)
        try:
            if _is_async_callable(self.processor):
                result = self.processor(context)
            else:
                # History policies may serialize, redact, or summarize large native
                # histories. Keep synchronous policy work off the adapter event loop;
                # asyncio.to_thread also propagates this run's ContextVars.
                result = await asyncio.to_thread(self.processor, context)
            if inspect.isawaitable(result):
                result = await cast("Awaitable[Sequence[MessageT]]", result)
        except asyncio.CancelledError:
            raise
        except HistoryProcessingError:
            raise
        except Exception as exc:
            raise HistoryProcessingError(
                f"history_processor failed for {self.codec.adapter_shortname} "
                f"request {request_index}: {exc}"
            ) from exc
        finally:
            self._inside_processor.reset(active_token)

        if isinstance(result, (str, bytes, bytearray)) or not isinstance(result, Sequence):
            raise HistoryProcessingValidationError(
                "history_processor must return a sequence of native messages"
            )
        if not result:
            raise HistoryProcessingValidationError("history_processor cannot return empty history")

        candidate = self._clone(result, label="result")
        candidate_snapshot = self._inspect(candidate, label="result")
        _validate_history_rewrite(source_snapshot, candidate_snapshot)

        changed = candidate_snapshot.source_digest != source_snapshot.source_digest
        if not changed:
            # Preserve native object identity and framework bookkeeping on semantic no-ops.
            return HistoryProcessingOutcome(
                messages=(messages if isinstance(messages, list) else list(messages)),
                changed=False,
                source_digest=source_snapshot.source_digest,
                result_digest=candidate_snapshot.source_digest,
            )

        state.rewritten = True
        state.reset_continuation = True
        if state.conversation is not None:
            state.conversation.advance_cache_generation()
            state.conversation.result_state[_REWRITTEN_STATE_KEY] = True
        return HistoryProcessingOutcome(
            messages=candidate,
            changed=True,
            source_digest=source_snapshot.source_digest,
            result_digest=candidate_snapshot.source_digest,
        )

    def prepare_transport(self) -> HistoryTransportDirective:
        state = self._run_state.get()
        if state is None:
            raise HistoryProcessingError(
                "history processor transport prepared outside an adapter run scope"
            )
        reset = state.reset_continuation
        state.reset_continuation = False
        return HistoryTransportDirective(
            reset_continuation=reset,
            rewritten=state.rewritten,
            cache_key=(
                state.conversation.cache_key
                if state.rewritten and state.conversation is not None
                else None
            ),
        )

    def _clone(self, messages: Sequence[MessageT], *, label: str) -> list[MessageT]:
        try:
            return self.codec.clone(messages)
        except Exception as exc:
            raise HistoryProcessingValidationError(
                f"history_processor {label} is not valid {self.codec.adapter_shortname} "
                f"native history: {exc}"
            ) from exc

    def _inspect(self, messages: Sequence[MessageT], *, label: str) -> HistorySnapshot:
        try:
            snapshot = classify_snapshot(self.codec.inspect(messages))
        except Exception as exc:
            raise HistoryProcessingValidationError(
                f"history_processor {label} could not be inspected: {exc}"
            ) from exc
        if snapshot.storage_format != "native":
            raise HistoryProcessingValidationError(
                f"history_processor {label} must contain only native "
                f"{self.codec.adapter_shortname} messages"
            )
        return snapshot


def rebase_history_cache_key(staged_key: str, current_key: str) -> str:
    """Apply a staged conversation key without losing an archive checkpoint suffix."""

    suffix = _CACHE_SUFFIX.search(current_key)
    return f"{staged_key}{suffix.group(0) if suffix is not None else ''}"


def _validate_history_rewrite(
    source: HistorySnapshot,
    candidate: HistorySnapshot,
) -> None:
    if candidate.entries[-1].fingerprint != source.entries[-1].fingerprint:
        raise HistoryProcessingValidationError(
            "history_processor must preserve the current request tail exactly"
        )

    source_positions = _match_source_entries(source.entries, candidate.entries)
    _validate_protected_entries(source, candidate, source_positions)
    _validate_protected_barriers(source, source_positions)
    _validate_atomic_groups(source, source_positions)
    _validate_new_entries(candidate, source_positions)


def _match_source_entries(
    source: tuple[NativeHistoryEntryRef, ...],
    candidate: tuple[NativeHistoryEntryRef, ...],
) -> list[int | None]:
    by_fingerprint: dict[str, deque[int]] = defaultdict(deque)
    for index, entry in enumerate(source[:-1]):
        by_fingerprint[entry.fingerprint].append(index)

    positions: list[int | None] = []
    for entry in candidate[:-1]:
        matches = by_fingerprint.get(entry.fingerprint)
        positions.append(matches.popleft() if matches else None)
    positions.append(len(source) - 1)
    return positions


def _validate_protected_entries(
    source: HistorySnapshot,
    candidate: HistorySnapshot,
    source_positions: list[int | None],
) -> None:
    candidate_position_for_source = {
        source_index: candidate_index
        for candidate_index, source_index in enumerate(source_positions)
        if source_index is not None
    }
    protected = sorted(_protected_source_indices(source))
    missing = [index for index in protected if index not in candidate_position_for_source]
    if missing:
        raise HistoryProcessingValidationError(
            "history_processor removed provider-required, checkpoint, system, or pending history"
        )
    protected_positions = [candidate_position_for_source[index] for index in protected]
    if protected_positions != sorted(protected_positions):
        raise HistoryProcessingValidationError(
            "history_processor reordered protected native history"
        )
    if source.entries[0].kind == "system" and candidate_position_for_source.get(0) != 0:
        raise HistoryProcessingValidationError(
            "history_processor must preserve the system-history prefix"
        )


def _validate_protected_barriers(
    source: HistorySnapshot,
    source_positions: list[int | None],
) -> None:
    protected = _protected_source_indices(source)
    source_segments: dict[int, int] = {}
    segment = 0
    for source_index in range(len(source.entries)):
        if source_index in protected:
            segment += 1
        else:
            source_segments[source_index] = segment

    candidate_segment = 0
    for source_index in source_positions:
        if source_index in protected:
            candidate_segment += 1
        elif source_index is not None and source_segments[source_index] != candidate_segment:
            raise HistoryProcessingValidationError(
                "history_processor moved native history across a protected barrier"
            )


def _protected_source_indices(source: HistorySnapshot) -> set[int]:
    protected = {
        entry.index
        for entry in source.entries
        if entry.kind in {"system", "checkpoint"}
        or entry.unknown
        or entry.provider_required
        or not entry.complete
    }
    for group in source.groups:
        if group.protected:
            protected.update(range(group.entry_start, group.entry_end))
    return protected


def _validate_atomic_groups(
    source: HistorySnapshot,
    source_positions: list[int | None],
) -> None:
    candidate_positions: dict[int, int] = {
        source_index: candidate_index
        for candidate_index, source_index in enumerate(source_positions)
        if source_index is not None
    }
    for group in source.groups:
        source_indices = list(range(group.entry_start, group.entry_end))
        retained = [index for index in source_indices if index in candidate_positions]
        if retained and len(retained) != len(source_indices):
            raise HistoryProcessingValidationError(
                f"history_processor split atomic history group {group.group_id!r}"
            )
        if not retained:
            continue
        positions = [candidate_positions[index] for index in source_indices]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise HistoryProcessingValidationError(
                f"history_processor changed ordering inside atomic group {group.group_id!r}"
            )


def _validate_new_entries(
    candidate: HistorySnapshot,
    source_positions: list[int | None],
) -> None:
    for position, source_index in enumerate(source_positions):
        if source_index is not None:
            continue
        entry = candidate.entries[position]
        if (
            entry.kind in {"system", "checkpoint", "unknown", "tool-call", "tool-result"}
            or entry.lifecycle_ids
            or entry.closes_lifecycle_ids
            or entry.provider_required
            or not entry.complete
        ):
            raise HistoryProcessingValidationError(
                "history_processor may add only complete text/reasoning summary messages; "
                "it cannot fabricate system, checkpoint, provider, or tool lifecycle entries"
            )


def _is_async_callable(processor: HistoryProcessorCallback[MessageT]) -> bool:
    return inspect.iscoroutinefunction(processor) or inspect.iscoroutinefunction(
        getattr(processor, "__call__", None)
    )
