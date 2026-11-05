from __future__ import annotations

import asyncio
import inspect
import re
import warnings
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from kedi.agent_adapter.compaction.classification import classify_snapshot
from kedi.agent_adapter.compaction.models import (
    HistorySnapshot,
    HistoryUnitGroup,
    NativeHistoryEntryRef,
)
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.history_provenance import LINEAGE_STATE_KEY, HistoryProvenance

MessageT = TypeVar("MessageT")


class HistoryProcessingError(RuntimeError):
    """Base error raised by Kedi's user-defined history processing boundary."""


class HistoryProcessingValidationError(HistoryProcessingError):
    """Raised before model I/O when processed native history is unsafe to replay."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryProcessorState:
    """Read-only metadata for deciding whether native history needs processing."""

    adapter_shortname: str
    model_id: str | None
    request_index: int
    cache_epoch: int | None
    estimated_tokens: int
    history_version: str


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
    history_version: str | None = None

    def keep_recent_groups(self, n: int) -> list[MessageT]:
        """Retain recent closed groups plus protected, unfinished, and current history."""

        if type(n) is not int:
            raise TypeError("history group count must be an integer")
        if n < 0:
            raise ValueError("history group count must be non-negative")
        eligible = [group for group in self.groups if group.closed and not group.protected]
        selected = {group.group_id for group in eligible[-n:]} if n else set()
        retained = {len(self.messages) - 1}
        for group in self.groups:
            if (
                group.group_id in selected
                or group.protected
                or not group.closed
                or group.entry_end == len(self.messages)
            ):
                retained.update(range(group.entry_start, group.entry_end))
        return [message for index, message in enumerate(self.messages) if index in retained]


HistoryProcessorResult: TypeAlias = Sequence[MessageT] | Awaitable[Sequence[MessageT]]
HistoryProcessorCallback: TypeAlias = Callable[
    [HistoryProcessorContext[MessageT]], HistoryProcessorResult[MessageT]
]
HistoryProcessorCondition: TypeAlias = Callable[[HistoryProcessorState], bool | Awaitable[bool]]

HISTORY_PROCESSOR_CACHE_WARNING = (
    "History processing can change earlier messages and reduce prefix-cache reuse. "
    "processor_condition can reduce how often the processor runs, but neither option "
    "guarantees a provider cache hit; a semantic no-op preserves native history."
)


def warn_history_processor_cache() -> None:
    """Warn once when a Python adapter configuration installs a processor."""

    warnings.warn(HISTORY_PROCESSOR_CACHE_WARNING, UserWarning, stacklevel=3)


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
class _HistoryProcessingRunState(Generic[MessageT]):
    conversation: AdapterConversationContext | None
    processor: HistoryProcessorCallback[MessageT] | None
    condition: HistoryProcessorCondition | None
    provenance: HistoryProvenance
    native_scope_active: bool = False
    request_index: int = 0
    rewritten: bool = False
    reset_continuation: bool = False


_CACHE_SUFFIX = re.compile(r"_h[0-9a-f]{8}$")
_REWRITTEN_STATE_KEY = "history_processor_rewritten"


class _UseDefault:
    pass


_USE_DEFAULT = _UseDefault()


class HistoryProcessorRunner(Generic[MessageT]):
    """Task-local executor for one adapter's native history callback."""

    def __init__(
        self,
        processor: HistoryProcessorCallback[MessageT] | None,
        codec: HistoryMessageCodec[MessageT],
        condition: HistoryProcessorCondition | None = None,
    ) -> None:
        if processor is not None and not callable(processor):
            raise TypeError("history_processor must be callable")
        if condition is not None and not callable(condition):
            raise TypeError("history_processor_condition must be callable")
        if processor is None and condition is not None:
            raise ValueError("history_processor_condition requires a history_processor")
        self.processor = processor
        self.condition = condition
        self.codec = codec
        self._run_state: ContextVar[_HistoryProcessingRunState[MessageT] | None] = ContextVar(
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
        processor: HistoryProcessorCallback[MessageT] | None | _UseDefault = _USE_DEFAULT,
        condition: HistoryProcessorCondition | None | _UseDefault = _USE_DEFAULT,
    ) -> Generator[None, None, None]:
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
        selected_processor = self.processor if isinstance(processor, _UseDefault) else processor
        selected_condition = self.condition if isinstance(condition, _UseDefault) else condition
        if selected_processor is None and selected_condition is not None:
            raise ValueError("history_processor_condition requires a history_processor")
        state = _HistoryProcessingRunState(
            conversation=conversation,
            processor=selected_processor,
            condition=selected_condition,
            provenance=HistoryProvenance(
                conversation.resume_state.get(LINEAGE_STATE_KEY) if conversation else None
            ),
            native_scope_active=reuse_existing,
            rewritten=rewritten,
        )
        token = self._run_state.set(state)
        try:
            with state.provenance.bind(enabled=state.processor is not None):
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
        processor = state.processor
        if processor is None:
            return HistoryProcessingOutcome(
                messages=messages if isinstance(messages, list) else list(messages),
                changed=False,
                source_digest="",
                result_digest="",
            )
        current_digest = ""
        if state.condition is not None:
            attempt = 0
            while True:
                inspected = self._inspect(messages, label="source", classify=False)
                condition_state = HistoryProcessorState(
                    adapter_shortname=self.codec.adapter_shortname,
                    model_id=model_id,
                    request_index=request_index,
                    cache_epoch=(
                        state.conversation.cache_epoch if state.conversation is not None else None
                    ),
                    estimated_tokens=inspected.total_tokens,
                    history_version=inspected.source_digest,
                )
                should_process = await self._evaluate_condition(state.condition, condition_state)
                current_digest = self._inspect(
                    messages, label="source", classify=False
                ).source_digest
                if current_digest == condition_state.history_version:
                    if not should_process:
                        state.provenance.inspect(inspected)
                        return HistoryProcessingOutcome(
                            messages=messages if isinstance(messages, list) else list(messages),
                            changed=False,
                            source_digest=current_digest,
                            result_digest=current_digest,
                        )
                    break
                attempt += 1
                if attempt == 2:
                    raise HistoryProcessingError(
                        "history changed repeatedly while evaluating history_processor_condition"
                    )
        source = self._clone(messages, label="source")
        if not source:
            raise HistoryProcessingValidationError(
                "history_processor source history must contain the current request"
            )
        source_snapshot = classify_snapshot(
            state.provenance.inspect(self._inspect(source, label="source"))
        )
        if state.condition is not None and source_snapshot.source_digest != current_digest:
            raise HistoryProcessingError(
                "history changed between history_processor_condition and processor snapshot"
            )
        context = HistoryProcessorContext(
            messages=tuple(source),
            adapter_shortname=self.codec.adapter_shortname,
            model_id=model_id,
            request_index=request_index,
            cache_epoch=(
                state.conversation.cache_epoch if state.conversation is not None else None
            ),
            estimated_tokens=source_snapshot.total_tokens,
            entries=deepcopy(source_snapshot.entries),
            groups=deepcopy(source_snapshot.groups),
            history_version=source_snapshot.source_digest,
        )

        active_token = self._inside_processor.set(True)
        try:
            if _is_async_callable(processor):
                result = processor(context)
            else:
                # History policies may serialize, redact, or summarize large native
                # histories. Keep synchronous policy work off the adapter event loop;
                # asyncio.to_thread also propagates this run's ContextVars.
                result = await asyncio.to_thread(processor, context)
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
        state.provenance.accept(source_snapshot, result, candidate_snapshot)
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

    async def _evaluate_condition(
        self,
        condition: HistoryProcessorCondition,
        state: HistoryProcessorState,
    ) -> bool:
        active_token = self._inside_processor.set(True)
        try:
            result = (
                condition(state)
                if _is_async_callable(condition)
                else await asyncio.to_thread(condition, state)
            )
            if inspect.isawaitable(result):
                result = await result
        except asyncio.CancelledError:
            raise
        except HistoryProcessingError:
            raise
        except Exception as exc:
            raise HistoryProcessingError(
                f"history_processor_condition failed for {self.codec.adapter_shortname} "
                f"request {state.request_index}: {exc}"
            ) from exc
        finally:
            self._inside_processor.reset(active_token)
        if type(result) is not bool:
            raise HistoryProcessingError("history_processor_condition must return a bool")
        return result

    def capture(self, messages: Sequence[MessageT], stored_messages: Sequence[object]) -> None:
        """Stage lineage alongside the exact successful native continuation snapshot."""

        state = self._run_state.get()
        if state is None or state.processor is None or state.conversation is None:
            return

        from kedi.agent_adapter.history_provenance import HistoryLineage, HistoryLineageEntry

        snapshot = state.provenance.inspect(self._inspect(messages, label="completed"))
        # Serialization and cache-control normalization can change fingerprints,
        # but the adapter's saved sequence has the same native occurrence order.
        stored = self.codec.inspect(cast("Sequence[MessageT]", stored_messages))
        lineage = HistoryLineage(
            tuple(
                HistoryLineageEntry(saved.fingerprint, entry.origins)
                for saved, entry in zip(stored.entries, snapshot.entries, strict=True)
            )
        )
        state.conversation.result_state[LINEAGE_STATE_KEY] = lineage

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

    def _inspect(
        self, messages: Sequence[MessageT], *, label: str, classify: bool = True
    ) -> HistorySnapshot:
        try:
            snapshot = self.codec.inspect(messages)
            if classify:
                snapshot = classify_snapshot(snapshot)
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


def _is_async_callable(processor: Callable[..., object]) -> bool:
    return inspect.iscoroutinefunction(processor) or inspect.iscoroutinefunction(
        getattr(processor, "__call__", None)
    )
