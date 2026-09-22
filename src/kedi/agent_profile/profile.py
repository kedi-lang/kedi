from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from ..agent_adapter.approvals.policy import ApprovalPolicy
from ..agent_adapter.meta import AdapterCapability, AgentAdapterKind
from ..agent_adapter.requirements import normalize_adapter_requirements
from ..artifacts import ArtifactPolicy
from ..effort import EffortValue
from .codemode import CodeModeSettings, normalize_codemode_settings
from .compaction import CompactionSettings
from .connection import A2AConnection
from .hooks import HookSettings, normalize_hook_settings
from .mcp import McpServerSpec
from .settings import tool_reason_enabled
from .skills import SkillsSettings, skills_settings

if TYPE_CHECKING:
    from ..agent_adapter.tools.specs import ToolSpec
    from .definitions import ProfileDefinition

AgentCommand = str | tuple[str, ...]
WorkflowMode = Literal["delegate", "dynamic"]


@dataclass(frozen=True, kw_only=True)
class AgentProfile:
    """Runtime agent configuration merged incrementally by profile directives."""

    name: str | None = None
    model: str | None = None
    adapter_kind: AgentAdapterKind | None = None
    adapter_shortname: str | None = None
    agent_command: AgentCommand | None = None
    a2a_connection: A2AConnection | None = None
    effort: EffortValue | None = None
    system: str | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    # Imported profiles keep their private dependencies here instead of publishing
    # them into the importer's flat procedure/environment namespaces.
    tool_bindings: Mapping[str, ToolSpec] = field(default_factory=dict)
    mcp_servers: tuple[McpServerSpec, ...] = ()
    approval: ApprovalPolicy | None = None
    hooks: HookSettings | None = None
    _hook_enforcement: HookSettings | None = field(default=None, repr=False)
    skills: SkillsSettings | bool | None = None
    subagents: tuple[str, ...] = ()
    subagent_bindings: Mapping[str, ProfileDefinition] = field(default_factory=dict)
    max_agents: int | None = None
    workflow_mode: WorkflowMode | None = None
    history: bool | None = None
    compaction: CompactionSettings | None = None
    codemode: CodeModeSettings | bool | None = None
    requirements: tuple[AdapterCapability, ...] = ()
    requirement_origins: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    output_type: Any | None = None
    artifacts: ArtifactPolicy | None = None

    def __post_init__(self) -> None:
        tool_reason_enabled(self.settings)
        object.__setattr__(self, "skills", skills_settings(self.skills))
        object.__setattr__(self, "codemode", normalize_codemode_settings(self.codemode))
        normalized_requirements = normalize_adapter_requirements(list(self.requirements))
        object.__setattr__(self, "requirements", normalized_requirements)
        object.__setattr__(
            self,
            "requirement_origins",
            {
                requirement: tuple(self.requirement_origins.get(requirement, ()))
                for requirement in normalized_requirements
            },
        )
        object.__setattr__(self, "hooks", normalize_hook_settings(self.hooks))
        object.__setattr__(
            self,
            "_hook_enforcement",
            normalize_hook_settings(self._hook_enforcement),
        )

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
            # A transport selection owns its whole connection atomically. A
            # later selection must never inherit another lexical scope's
            # command, endpoint, or credential reference.
            updates["agent_command"] = other.agent_command
            updates["a2a_connection"] = other.a2a_connection
        elif other.agent_command is not None:
            updates["agent_command"] = other.agent_command
            updates["a2a_connection"] = None
        elif other.a2a_connection is not None:
            updates["agent_command"] = None
            updates["a2a_connection"] = other.a2a_connection
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
        if other.hooks is not None:
            updates["hooks"] = other.hooks if self.hooks is None else self.hooks.merge(other.hooks)
        if other._hook_enforcement is not None:
            updates["_hook_enforcement"] = (
                other._hook_enforcement
                if self._hook_enforcement is None
                else self._hook_enforcement.merge(other._hook_enforcement)
            )
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
        if other.history is not None:
            updates["history"] = other.history
        if other.compaction is not None:
            updates["compaction"] = other.compaction
        if other.codemode is not None:
            updates["codemode"] = other.codemode
        if other.requirements:
            updates["requirements"] = _merge_requirements(
                self.requirements,
                other.requirements,
            )
            origins = dict(self.requirement_origins)
            for requirement in other.requirements:
                origins[requirement] = (
                    *origins.get(requirement, ()),
                    *other.requirement_origins.get(requirement, ()),
                )
            updates["requirement_origins"] = origins
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

    def with_hook_enforcement(self) -> AgentProfile:
        """Mark this profile's hooks as inherited runtime policy for child agents."""

        if self.hooks is None:
            return self
        return replace(self, _hook_enforcement=self.hooks)

    def frozen_copy(self) -> AgentProfile:
        return replace(
            self,
            settings=dict(self.settings or {}),
            tool_bindings=dict(self.tool_bindings),
            subagent_bindings=dict(self.subagent_bindings),
            requirement_origins={
                name: tuple(origins) for name, origins in self.requirement_origins.items()
            },
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
            or self.a2a_connection is not None
            or bool(self.mcp_servers)
            or bool(self.settings)
            or self.compaction is not None
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


def _merge_requirements(
    left: tuple[AdapterCapability, ...],
    right: tuple[AdapterCapability, ...],
) -> tuple[AdapterCapability, ...]:
    merged = list(left)
    for requirement in right:
        if requirement not in merged:
            merged.append(requirement)
    return tuple(merged)
