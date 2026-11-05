from __future__ import annotations

from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .compaction.models import HistoryEntryKind, HistoryOrigin, HistorySnapshot
from .events import current_agent_run_identity, current_bound_agent_run_identity

if TYPE_CHECKING:
    from kedi.lang.ast import Span

LINEAGE_STATE_KEY = "history_processor_origins"
_SOURCE: ContextVar[tuple[str, Span | None] | None] = ContextVar(
    "kedi_history_source", default=None
)
_ACTIVE: ContextVar[HistoryProvenance | None] = ContextVar("kedi_history_provenance", default=None)


@contextmanager
def history_source_scope(run_id: str, location: Span | None) -> Generator[None, None, None]:
    token = _SOURCE.set((run_id, location))
    try:
        yield
    finally:
        _SOURCE.reset(token)


@dataclass(frozen=True, slots=True)
class HistoryLineageEntry:
    fingerprint: str
    origins: tuple[HistoryOrigin, ...]


@dataclass(frozen=True, slots=True)
class HistoryLineage:
    entries: tuple[HistoryLineageEntry, ...]


def _origin(kind: HistoryEntryKind = "user") -> HistoryOrigin:
    source = _SOURCE.get()
    identity = current_bound_agent_run_identity() or current_agent_run_identity()
    return HistoryOrigin(
        kind=kind,
        run_id=identity.run_id if identity is not None else source[0] if source else None,
        source_location=source[1] if source else None,
    )


def record_history_tool_origin(tool_name: str, tool_call_id: str) -> None:
    """Record exact tool-boundary identity without storing arguments or results."""

    active = _ACTIVE.get()
    if active is not None:
        active.tools[tool_call_id] = replace(
            _origin("tool-call"), tool_name=tool_name, tool_call_id=tool_call_id
        )


class HistoryProvenance:
    """Bounded invocation lineage; only immutable entries cross a successful turn."""

    def __init__(self, saved: object = None) -> None:
        self.saved = saved if isinstance(saved, HistoryLineage) else None
        self.snapshot: HistorySnapshot | None = None
        self.tools: dict[str, HistoryOrigin] = {}

    @contextmanager
    def bind(self, *, enabled: bool = True) -> Generator[None, None, None]:
        token = _ACTIVE.set(self if enabled else None)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def inspect(self, snapshot: HistorySnapshot) -> HistorySnapshot:
        previous = self.snapshot
        saved = self.saved.entries if previous is None and self.saved is not None else ()
        prefix = 0
        if previous is not None:
            saved = tuple(HistoryLineageEntry(e.fingerprint, e.origins) for e in previous.entries)
        for old, new in zip(saved, snapshot.entries):
            if old.fingerprint != new.fingerprint:
                break
            prefix += 1
        source = _origin()
        entries = []
        for index, entry in enumerate(snapshot.entries):
            origins = saved[index].origins if index < prefix else entry.origins
            # Only the new current request (first boundary) or appended native
            # responses (subsequent boundary) belong to this invocation. Imported
            # history and archive replacements must not acquire invented lineage.
            known_new = (
                previous is None
                and index >= prefix
                and index == len(snapshot.entries) - 1
                and entry.kind == "user"
            ) or (previous is not None and prefix == len(previous.entries) and index >= prefix)
            if known_new and entry.kind not in {"system", "checkpoint", "unknown"}:
                origins = origins or (HistoryOrigin(kind=entry.kind),)
                origins = tuple(
                    replace(origin, run_id=source.run_id, source_location=source.source_location)
                    for origin in origins
                )
            origins = tuple(
                replace(self.tools[origin.tool_call_id], kind=origin.kind)
                if origin.tool_call_id in self.tools
                and (
                    index >= prefix
                    or (source.run_id is not None and origin.run_id == source.run_id)
                )
                else origin
                for origin in origins
            )
            entries.append(replace(entry, origins=origins))
        self.snapshot = replace(snapshot, entries=tuple(entries))
        self.saved = None
        self._prune_tools()
        return self.snapshot

    def accept(
        self, source: HistorySnapshot, returned: Sequence[object], candidate: HistorySnapshot
    ) -> None:
        identity_counts = Counter(id(message) for message in source.native_entries)
        identities = {
            id(message): index
            for index, message in enumerate(source.native_entries)
            if identity_counts[id(message)] == 1
        }
        counts = Counter(entry.fingerprint for entry in source.entries)
        unique = {
            entry.fingerprint: entry.origins
            for entry in source.entries
            if counts[entry.fingerprint] == 1
        }
        entries = []
        for message, entry in zip(returned, candidate.entries):
            index = identities.get(id(message))
            if index is not None and source.entries[index].fingerprint == entry.fingerprint:
                origins = source.entries[index].origins
            else:
                # Copies and aliased occurrences cannot be distinguished safely.
                # Only unambiguous content can recover lineage without identity.
                origins = unique.get(entry.fingerprint, ())
            entries.append(replace(entry, origins=origins))
        self.snapshot = replace(candidate, entries=tuple(entries))
        self._prune_tools()

    def _prune_tools(self) -> None:
        assert self.snapshot is not None
        retained = {
            call_id
            for entry in self.snapshot.entries
            for call_id in (*entry.lifecycle_ids, *entry.closes_lifecycle_ids)
        }
        self.tools = {
            call_id: origin for call_id, origin in self.tools.items() if call_id in retained
        }
