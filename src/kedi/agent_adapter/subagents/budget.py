"""Delegation invocation and concurrency budgets."""

from __future__ import annotations

from dataclasses import dataclass

MAX_SUBAGENT_INVOCATIONS = 100
MAX_SUBAGENT_CONCURRENCY = 32


@dataclass(slots=True)
class DelegationBudget:
    owner_profile: str
    limit: int
    started: int = 0

    @classmethod
    def for_profile(
        cls,
        *,
        profile_name: str | None,
        max_agents: int | None,
    ) -> DelegationBudget:
        return cls(
            owner_profile=profile_name or "<anonymous>",
            limit=min(max_agents or MAX_SUBAGENT_INVOCATIONS, MAX_SUBAGENT_INVOCATIONS),
        )


@dataclass(slots=True)
class DelegationConcurrency:
    limit: int
    active: int = 0

    @classmethod
    def for_limit(cls, limit: int) -> DelegationConcurrency:
        if limit < 1:
            raise ValueError("Subagent max concurrency must be at least 1")
        return cls(limit=min(limit, MAX_SUBAGENT_CONCURRENCY))
