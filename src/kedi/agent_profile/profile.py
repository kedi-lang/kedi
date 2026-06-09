from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, kw_only=True)
class AgentProfile:
    """Runtime agent configuration merged incrementally by profile directives."""

    model: str | None = None

    @classmethod
    def empty(cls) -> AgentProfile:
        return cls()

    def merge(self, other: AgentProfile) -> AgentProfile:
        if other.model is not None:
            return replace(self, model=other.model)
        return self

    def frozen_copy(self) -> AgentProfile:
        return replace(self)

    def needs_model_override(self, adapter_default: str | None) -> bool:
        if self.model is None:
            return False
        if adapter_default is None:
            return True
        return self.model != adapter_default
