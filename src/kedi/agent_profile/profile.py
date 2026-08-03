from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from ..agent_adapter.approval import ApprovalPolicy
from ..agent_adapter.meta import AgentAdapterKind
from ..artifacts import ArtifactPolicy
from ..effort import EffortValue
from .mcp import McpServerSpec

if TYPE_CHECKING:
    from ..agent_adapter.tool_override import ToolSpec
    from .registry import ProfileDefinition

AgentCommand = str | tuple[str, ...]
WorkflowMode = Literal["delegate", "dynamic"]
SKILLS_USE_TARGET = "skills"


@dataclass(frozen=True, kw_only=True)
class AgentProfile:
    """Runtime agent configuration merged incrementally by profile directives."""

    name: str | None = None
    model: str | None = None
    adapter_kind: AgentAdapterKind | None = None
    adapter_shortname: str | None = None
    agent_command: AgentCommand | None = None
    effort: EffortValue | None = None
    system: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    # Imported profiles keep their private dependencies here instead of publishing
    # them into the importer's flat procedure/environment namespaces.
    tool_bindings: Mapping[str, ToolSpec] = field(default_factory=dict)
    mcp_servers: tuple[McpServerSpec, ...] = ()
    approval: ApprovalPolicy | None = None
    skills: bool | None = None
    subagents: tuple[str, ...] = ()
    subagent_bindings: Mapping[str, ProfileDefinition] = field(default_factory=dict)
    max_agents: int | None = None
    workflow_mode: WorkflowMode | None = None
    output_type: Any | None = None
    artifacts: ArtifactPolicy | None = None

    @classmethod
    def empty(cls) -> AgentProfile:
        return cls(artifacts=ArtifactPolicy.patch({}))

    def merge(self, other: AgentProfile) -> AgentProfile:
        updates: dict[str, object] = {}
        if other.name is not None:
            updates["name"] = other.name
        if other.adapter_shortname is not None:
            updates["adapter_shortname"] = other.adapter_shortname
            updates["adapter_kind"] = other.adapter_kind
        if other.agent_command is not None:
            updates["agent_command"] = other.agent_command
        if other.model is not None:
            updates["model"] = other.model
        if other.effort is not None:
            updates["effort"] = other.effort
        if other.system is not None:
            updates["system"] = other.system
        if other.settings:
            updates["settings"] = {**(self.settings or {}), **other.settings}
        if other.tools or other.tool_bindings:
            bindings = dict(self.tool_bindings)
            # A later unbound ``> use: tool`` deliberately restores normal
            # caller-scope lookup for that name.
            for tool_name in other.tools:
                bindings.pop(tool_name, None)
            bindings.update(other.tool_bindings)
            if other.tools:
                updates["tools"] = _merge_tool_names(self.tools, other.tools)
            updates["tool_bindings"] = bindings
        if other.mcp_servers:
            updates["mcp_servers"] = self.mcp_servers + other.mcp_servers
        if other.approval is not None:
            updates["approval"] = other.approval
        if other.skills is not None:
            updates["skills"] = other.skills
        if other.subagents or other.subagent_bindings:
            bindings = dict(self.subagent_bindings)
            for subagent_name in other.subagents:
                bindings.pop(subagent_name, None)
            bindings.update(other.subagent_bindings)
            if other.subagents:
                updates["subagents"] = _merge_names(self.subagents, other.subagents)
            updates["subagent_bindings"] = bindings
        if other.max_agents is not None:
            updates["max_agents"] = other.max_agents
        if other.workflow_mode is not None:
            updates["workflow_mode"] = other.workflow_mode
        if other.output_type is not None:
            updates["output_type"] = other.output_type
        if other.artifacts is not None:
            updates["artifacts"] = (
                other.artifacts
                if self.artifacts is None
                else self.artifacts.overlay(other.artifacts)
            )
        if not updates:
            return self
        return replace(self, **updates)

    def frozen_copy(self) -> AgentProfile:
        return replace(
            self,
            settings=dict(self.settings or {}),
            tool_bindings=dict(self.tool_bindings),
            subagent_bindings=dict(self.subagent_bindings),
        )

    def needs_adapter_selection(self, adapter_shortname: str | None) -> bool:
        if self.adapter_shortname is None:
            return False
        return self.adapter_shortname != adapter_shortname

    def needs_model_override(self, adapter_default: str | None) -> bool:
        if self.model is None:
            return False
        if adapter_default is None:
            return True
        return self.model != adapter_default

    def needs_adapter_override(self, adapter_default: str | None) -> bool:
        return (
            self.effort is not None
            or self.agent_command is not None
            or bool(self.mcp_servers)
            or bool(self.settings)
            or self.needs_model_override(adapter_default)
        )


def _merge_tool_names(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return _merge_names(left, right)


def _merge_names(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    merged = list(left)
    for name in right:
        if name in merged:
            merged.remove(name)
        merged.append(name)
    return tuple(merged)
