from __future__ import annotations

from dataclasses import dataclass, replace

from ..effort import EffortValue


@dataclass(frozen=True, kw_only=True)
class AgentProfile:
    """Runtime agent configuration merged incrementally by profile directives."""

    model: str | None = None
    effort: EffortValue | None = None
    system: str | None = None
    tools: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> AgentProfile:
        return cls()

    def merge(self, other: AgentProfile) -> AgentProfile:
        updates: dict[str, object] = {}
        if other.model is not None:
            updates["model"] = other.model
        if other.effort is not None:
            updates["effort"] = other.effort
        if other.system is not None:
            updates["system"] = other.system
        if other.tools:
            updates["tools"] = _merge_tool_names(self.tools, other.tools)
        if not updates:
            return self
        return replace(self, **updates)

    def frozen_copy(self) -> AgentProfile:
        return replace(self)

    def needs_model_override(self, adapter_default: str | None) -> bool:
        if self.model is None:
            return False
        if adapter_default is None:
            return True
        return self.model != adapter_default

    def needs_adapter_override(self, adapter_default: str | None) -> bool:
        return self.effort is not None or self.needs_model_override(adapter_default)


def _merge_tool_names(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    merged = list(left)
    for name in right:
        if name in merged:
            merged.remove(name)
        merged.append(name)
    return tuple(merged)
