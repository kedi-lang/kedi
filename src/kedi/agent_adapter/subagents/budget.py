"""Delegation invocation and concurrency budgets."""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import KediExecutionError

MAX_SUBAGENT_INVOCATIONS = 100
MAX_SUBAGENT_CONCURRENCY = 32


class SubagentBudgetExceeded(KediExecutionError):
    """Raised when a parent has spent its complete child invocation budget."""


class SubagentConcurrencyExceeded(KediExecutionError):
    """Raised when a parent cannot admit another concurrent child run."""


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
        if max_agents is None:
            limit = MAX_SUBAGENT_INVOCATIONS
        elif max_agents < 0:
            raise ValueError("Subagent max agents cannot be negative")
        elif max_agents > MAX_SUBAGENT_INVOCATIONS:
            raise ValueError(f"Subagent max agents cannot exceed {MAX_SUBAGENT_INVOCATIONS}")
        else:
            limit = max_agents
        return cls(
            owner_profile=profile_name or "<anonymous>",
            limit=limit,
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
