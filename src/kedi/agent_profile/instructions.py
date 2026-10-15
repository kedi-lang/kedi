from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityInstruction:
    """One user instruction contribution, identified by its defining capability."""

    identity: str
    content: str


def merge_capability_instructions(
    current: tuple[CapabilityInstruction, ...],
    incoming: tuple[CapabilityInstruction, ...],
) -> tuple[CapabilityInstruction, ...]:
    """Reusing a capability updates its contribution without changing its position."""
    by_identity = {part.identity: part for part in current}
    by_identity.update((part.identity, part) for part in incoming)
    return tuple(by_identity.values())
