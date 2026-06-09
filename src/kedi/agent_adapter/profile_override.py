from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol

from typing_extensions import runtime_checkable

if TYPE_CHECKING:
    from kedi.agent_profile import AgentProfile


@runtime_checkable
class SupportsProfileOverride(Protocol):
    def default_model(self) -> str | None:
        """Return the adapter's configured default model, if any."""
        ...

    def profile_override(self, profile: AgentProfile) -> AbstractContextManager[None]:
        """Return a context manager that applies ``profile`` for nested agent calls."""
        ...
