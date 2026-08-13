from __future__ import annotations

from dataclasses import replace

from .models import (
    CompactionWindow,
    HistoryNoOpReason,
    HistoryRange,
    HistoryReductionPlan,
    HistoryReductionPolicy,
    HistorySnapshot,
    HistoryUnitGroup,
)


def plan_history_reduction(
    snapshot: HistorySnapshot,
    policy: HistoryReductionPolicy,
) -> HistoryReductionPlan:
    """Select the oldest worthwhile contiguous closed history interval."""

    groups = _protect_groups(snapshot, policy)
    if not snapshot.entries:
        return _no_op(snapshot, policy, groups, "empty-history")
    if snapshot.total_tokens < policy.threshold_tokens:
        return _no_op(snapshot, policy, groups, "below-threshold")

    segments = _eligible_segments(groups)
    for segment in segments:
        selected = _select_segment_prefix(snapshot.total_tokens, segment, policy)
        if selected is None:
            continue
        source_tokens = sum(group.token_estimate for group in selected)
        reduction = source_tokens - policy.checkpoint_budget_tokens
        native_range = HistoryRange(
            start=selected[0].entry_start,
            end=selected[-1].entry_end,
        )
        window = CompactionWindow(
            native_range=native_range,
            group_ids=tuple(group.group_id for group in selected),
            source_tokens=source_tokens,
            predicted_result_tokens=(
                snapshot.total_tokens - source_tokens + policy.checkpoint_budget_tokens
            ),
            predicted_reduction_tokens=reduction,
        )
        return HistoryReductionPlan(
            source_version=snapshot.source_version,
            source_digest=snapshot.source_digest,
            estimator_version=snapshot.estimator_version,
            total_tokens=snapshot.total_tokens,
            policy=policy,
            groups=groups,
            window=window,
            unchanged_prefix_fingerprints=tuple(
                entry.fingerprint for entry in snapshot.entries[: native_range.start]
            ),
            unchanged_tail_fingerprints=tuple(
                entry.fingerprint for entry in snapshot.entries[native_range.end :]
            ),
        )
    return _no_op(snapshot, policy, groups, "no-safe-window")


def _protect_groups(
    snapshot: HistorySnapshot,
    policy: HistoryReductionPolicy,
) -> tuple[HistoryUnitGroup, ...]:
    groups = list(snapshot.groups)
    unit_by_id = {unit.unit_id: unit for unit in snapshot.units}

    latest_user_group: int | None = None
    for index, group in enumerate(groups):
        if any("user" in unit_by_id[unit_id].kinds for unit_id in group.unit_ids):
            latest_user_group = index
    if latest_user_group is not None:
        groups[latest_user_group] = _with_reason(groups[latest_user_group], "latest-cycle")

    retained_tokens = 0
    if policy.preserve_recent_tokens:
        for index in range(len(groups) - 1, -1, -1):
            groups[index] = _with_reason(groups[index], "recent-tail")
            retained_tokens += groups[index].token_estimate
            if retained_tokens >= policy.preserve_recent_tokens:
                break

    if policy.protected_identifiers:
        for index, group in enumerate(groups):
            text = "\n".join(
                snapshot.entries[entry_index].searchable_text
                for entry_index in range(group.entry_start, group.entry_end)
            )
            if any(identifier in text for identifier in policy.protected_identifiers):
                groups[index] = _with_reason(groups[index], "protected-reference")

    for index, group in enumerate(groups):
        refs = set(group.artifact_refs)
        if refs.intersection(policy.live_artifact_refs):
            groups[index] = _with_reason(groups[index], "live-artifact")
        if refs.intersection(policy.missing_artifact_refs):
            groups[index] = _with_reason(groups[index], "missing-artifact")

    lifecycle_groups: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        for identifier in group.lifecycle_ids:
            lifecycle_groups.setdefault(identifier, []).append(index)
    for indexes in lifecycle_groups.values():
        if len(indexes) <= 1:
            continue
        for index in indexes:
            groups[index] = _with_reason(groups[index], "cross-group-lifecycle")

    for index, group in enumerate(groups):
        if not group.closed:
            groups[index] = _with_reason(group, "open-group")
    return tuple(groups)


def _with_reason(group: HistoryUnitGroup, reason: str) -> HistoryUnitGroup:
    if reason in group.protected_reasons:
        return group
    return replace(group, protected_reasons=(*group.protected_reasons, reason))


def _eligible_segments(
    groups: tuple[HistoryUnitGroup, ...],
) -> tuple[tuple[HistoryUnitGroup, ...], ...]:
    segments: list[tuple[HistoryUnitGroup, ...]] = []
    current: list[HistoryUnitGroup] = []
    for group in groups:
        if group.closed and not group.protected:
            current.append(group)
        else:
            if current:
                segments.append(tuple(current))
                current.clear()
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def _select_segment_prefix(
    total_tokens: int,
    segment: tuple[HistoryUnitGroup, ...],
    policy: HistoryReductionPolicy,
) -> tuple[HistoryUnitGroup, ...] | None:
    selected: list[HistoryUnitGroup] = []
    source_tokens = 0
    first_worthwhile: tuple[HistoryUnitGroup, ...] | None = None
    for group in segment:
        selected.append(group)
        source_tokens += group.token_estimate
        reduction = source_tokens - policy.checkpoint_budget_tokens
        if reduction >= policy.minimum_reduction_tokens and first_worthwhile is None:
            first_worthwhile = tuple(selected)
        predicted = total_tokens - source_tokens + policy.checkpoint_budget_tokens
        if reduction >= policy.minimum_reduction_tokens and predicted <= policy.target_tokens:
            return tuple(selected)
    return first_worthwhile if first_worthwhile is not None else None


def _no_op(
    snapshot: HistorySnapshot,
    policy: HistoryReductionPolicy,
    groups: tuple[HistoryUnitGroup, ...],
    reason: HistoryNoOpReason,
) -> HistoryReductionPlan:
    return HistoryReductionPlan(
        source_version=snapshot.source_version,
        source_digest=snapshot.source_digest,
        estimator_version=snapshot.estimator_version,
        total_tokens=snapshot.total_tokens,
        policy=policy,
        groups=groups,
        window=None,
        unchanged_prefix_fingerprints=tuple(entry.fingerprint for entry in snapshot.entries),
        unchanged_tail_fingerprints=(),
        no_op_reason=reason,
    )
