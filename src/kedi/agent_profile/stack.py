from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .profile import AgentProfile

_block_agent_profile: ContextVar[AgentProfile | None] = ContextVar(
    "kedi_block_agent_profile",
    default=None,
)


def current_block_agent_profile() -> AgentProfile:
    profile = _block_agent_profile.get()
    if profile is None:
        return AgentProfile.empty()
    return profile


@contextmanager
def block_agent_profile_scope(profile: AgentProfile) -> Iterator[None]:
    current = _block_agent_profile.get()
    effective = profile if current is None else current.merge(profile)
    token = _block_agent_profile.set(effective)
    try:
        yield
    finally:
        _block_agent_profile.reset(token)
