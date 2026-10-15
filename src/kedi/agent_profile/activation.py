from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from kedi.agent_adapter.tools.errors import CapabilityActivationError

from .capabilities import deferred_boundaries
from .profile import AgentProfile


_active_capabilities: ContextVar[frozenset[str]] = ContextVar(
    "kedi_active_capabilities", default=frozenset()
)


@contextmanager
def capability_activation_scope(identities: frozenset[str]) -> Iterator[None]:
    token = _active_capabilities.set(identities)
    try:
        yield
    finally:
        _active_capabilities.reset(token)


def available_subagents(profile: AgentProfile, active: frozenset[str]) -> tuple[str, ...]:
    names = dict.fromkeys(profile.subagents)
    for boundary in deferred_boundaries(profile):
        if boundary.capability.identity in active and boundary.discoverable(active):
            names.update(dict.fromkeys(boundary.capability.profile.subagents))
    return tuple(names)


def require_subagent_activation(profile: AgentProfile, name: str) -> None:
    if name in available_subagents(profile, _active_capabilities.get()):
        return
    if not any(
        name in boundary.capability.profile.subagents for boundary in deferred_boundaries(profile)
    ):
        return
    raise CapabilityActivationError(
        f"Subagent {name!r} is not active; load its capability before calling it."
    )
