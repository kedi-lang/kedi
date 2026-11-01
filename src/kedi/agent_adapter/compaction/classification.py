from __future__ import annotations

from collections import Counter
from dataclasses import replace

from .models import HistorySnapshot, HistoryUnit, HistoryUnitGroup, NativeHistoryEntryRef


def classify_snapshot(snapshot: HistorySnapshot) -> HistorySnapshot:
    """Build deterministic entry units and logical replay-cycle groups."""

    units = tuple(_unit_for_entry(entry) for entry in snapshot.entries)
    groups = _group_units(units)
    return replace(snapshot, units=units, groups=groups)


def _unit_for_entry(entry: NativeHistoryEntryRef) -> HistoryUnit:
    return HistoryUnit(
        unit_id=f"unit_{entry.entry_id}",
        entry_start=entry.index,
        entry_end=entry.index + 1,
        entry_ids=(entry.entry_id,),
        kinds=(entry.kind,),
        token_estimate=entry.token_estimate,
        lifecycle_ids=entry.lifecycle_ids,
        closes_lifecycle_ids=entry.closes_lifecycle_ids,
        artifact_refs=entry.artifact_refs,
        complete=entry.complete,
        checkpoint=entry.checkpoint,
        unknown=entry.unknown,
        provider_required=entry.provider_required,
        origins=entry.origins,
    )


def _group_units(units: tuple[HistoryUnit, ...]) -> tuple[HistoryUnitGroup, ...]:
    groups: list[HistoryUnitGroup] = []
    cycle: list[HistoryUnit] = []

    def flush() -> None:
        if not cycle:
            return
        kinds = {kind for unit in cycle for kind in unit.kinds}
        if "user" in kinds and not kinds & {"assistant", "checkpoint"}:
            # An unfinished cycle keeps its prompt protected, but each tool exchange it has
            # completed is its own group, so a long tool loop can be trimmed while it runs.
            for part, exchange in _split_unfinished_cycle(tuple(cycle)):
                groups.append(_build_group(len(groups), part, exchange=exchange))
        else:
            groups.append(_build_group(len(groups), tuple(cycle)))
        cycle.clear()

    for unit in units:
        kind = unit.kinds[0]
        if kind in {"system", "checkpoint"}:
            flush()
            groups.append(_build_group(len(groups), (unit,)))
            continue
        if kind == "user" and cycle:
            flush()
        cycle.append(unit)
    flush()
    return tuple(groups)


def _split_unfinished_cycle(
    units: tuple[HistoryUnit, ...],
) -> list[tuple[tuple[HistoryUnit, ...], bool]]:
    """The prompt, then one part per run of units that opens and closes every tool call."""

    prompt_end = next(i for i, unit in enumerate(units) if "user" in unit.kinds) + 1
    parts: list[tuple[tuple[HistoryUnit, ...], bool]] = [(units[:prompt_end], False)]
    exchange: list[HistoryUnit] = []
    open_calls: Counter[str] = Counter()
    for unit in units[prompt_end:]:
        exchange.append(unit)
        open_calls.update(unit.lifecycle_ids)
        open_calls.subtract(unit.closes_lifecycle_ids)
        if not +open_calls and any(part.lifecycle_ids for part in exchange):
            parts.append((tuple(exchange), True))
            exchange.clear()
            open_calls.clear()
    if exchange:
        parts.append((tuple(exchange), True))
    return parts


def _build_group(
    index: int,
    units: tuple[HistoryUnit, ...],
    *,
    exchange: bool = False,
) -> HistoryUnitGroup:
    opened = Counter(identifier for unit in units for identifier in unit.lifecycle_ids)
    closed = Counter(identifier for unit in units for identifier in unit.closes_lifecycle_ids)
    unresolved = tuple(sorted((opened - closed).elements()))
    orphaned_results = tuple(sorted((closed - opened).elements()))
    kinds = tuple(kind for unit in units for kind in unit.kinds)
    complete = all(unit.complete for unit in units)
    has_user = "user" in kinds
    has_terminal_response = "assistant" in kinds or "checkpoint" in kinds
    reasons: list[str] = []
    if "system" in kinds:
        reasons.append("immutable-prefix")
    if any(unit.checkpoint for unit in units):
        reasons.append("existing-checkpoint")
    if any(unit.unknown for unit in units):
        reasons.append("unknown-entry")
    if any(unit.provider_required for unit in units):
        reasons.append("provider-required")
    if not complete:
        reasons.append("incomplete-entry")
    if unresolved:
        reasons.append("open-lifecycle")
    if orphaned_results:
        reasons.append("orphan-lifecycle-result")
    if exchange:
        # A tool exchange of an unfinished cycle: closed once every call it makes is answered.
        is_closed = complete and not unresolved and not orphaned_results and bool(opened)
        if not opened:
            reasons.append("incomplete-exchange")
    else:
        if not has_user and "system" not in kinds and not any(unit.checkpoint for unit in units):
            reasons.append("missing-user-cycle")
        if has_user and not has_terminal_response:
            reasons.append("incomplete-response-cycle")
        is_closed = (
            complete
            and not unresolved
            and not orphaned_results
            and has_user
            and has_terminal_response
        )
    return HistoryUnitGroup(
        group_id=f"group_{index}_{units[0].entry_start}_{units[-1].entry_end}",
        unit_ids=tuple(unit.unit_id for unit in units),
        entry_start=units[0].entry_start,
        entry_end=units[-1].entry_end,
        token_estimate=sum(unit.token_estimate for unit in units),
        closed=is_closed,
        lifecycle_ids=tuple(sorted(set((*opened.keys(), *closed.keys())))),
        artifact_refs=tuple(sorted({ref_id for unit in units for ref_id in unit.artifact_refs})),
        protected_reasons=tuple(reasons),
        origins=tuple(dict.fromkeys(origin for unit in units for origin in unit.origins)),
    )
