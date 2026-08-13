from __future__ import annotations

from collections.abc import Mapping

from .classification import classify_snapshot
from .codecs import AdapterHistoryCodec
from .models import HistoryReductionPlan, HistoryReductionPolicy, HistorySnapshot
from .selection import plan_history_reduction


class HistoryProcessor:
    """Pure deterministic inspection and selection over native replay history."""

    def inspect(
        self,
        state: Mapping[str, object],
        *,
        codec: AdapterHistoryCodec,
    ) -> HistorySnapshot:
        return classify_snapshot(codec.inspect(state))

    def plan(
        self,
        snapshot: HistorySnapshot,
        policy: HistoryReductionPolicy,
    ) -> HistoryReductionPlan:
        if not snapshot.groups and snapshot.entries:
            snapshot = classify_snapshot(snapshot)
        return plan_history_reduction(snapshot, policy)
