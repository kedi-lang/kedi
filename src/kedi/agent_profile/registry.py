from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping

from .profile import WorkflowMode

if TYPE_CHECKING:
    from kedi.agent_adapter.tool_override import ToolSpec
    from kedi.lang.ast import ArtifactsField, McpField, SettingsField, Span, TemplateLine, TypeExpr


@dataclass(frozen=True, kw_only=True)
class ProfileAgentMember:
    kind: Literal["agent"] = "agent"
    plain_value: str | None = None
    backtick_code: str | None = None
    command_plain_value: str | None = None
    command_backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileAdapterMember:
    kind: Literal["adapter"] = "adapter"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileModelMember:
    kind: Literal["model"] = "model"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileEffortMember:
    kind: Literal["effort"] = "effort"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileApprovalMember:
    kind: Literal["approval"] = "approval"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileHistoryMember:
    kind: Literal["history"] = "history"
    enabled: bool
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileSkillsMember:
    kind: Literal["skills"] = "skills"
    enabled: bool
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileCodeModeMember:
    kind: Literal["codemode"] = "codemode"
    enabled: bool
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileSystemMember:
    kind: Literal["system"] = "system"
    plain_value: str | None = None
    backtick_code: str | None = None
    template_lines: tuple["TemplateLine", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileToolMember:
    kind: Literal["tools"] = "tools"
    tool_names: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileUseMember:
    kind: Literal["use"] = "use"
    name: str


@dataclass(frozen=True, kw_only=True)
class ProfileMcpMember:
    kind: Literal["mcp"] = "mcp"
    fields: tuple["McpField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileSettingsMember:
    kind: Literal["settings"] = "settings"
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileArtifactsMember:
    kind: Literal["artifacts"] = "artifacts"
    fields: tuple["ArtifactsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileOutputMember:
    kind: Literal["output"] = "output"
    type_annotation: str
    type_expr: "TypeExpr"


@dataclass(frozen=True, kw_only=True)
class ProfileSubagentMember:
    kind: Literal["subagents"] = "subagents"
    name: str
    loc: "Span | None" = None


@dataclass(frozen=True, kw_only=True)
class ProfileMaxAgentsMember:
    kind: Literal["max_agents"] = "max_agents"
    value: int


@dataclass(frozen=True, kw_only=True)
class ProfileWorkflowMember:
    kind: Literal["workflow"] = "workflow"
    mode: WorkflowMode


@dataclass(frozen=True, kw_only=True)
class ProfileDefinition:
    name: str
    description: str | None = None
    # These bindings are compiler-owned implementation details for imported
    # profiles. Source-level profile syntax remains name based.
    tool_bindings: Mapping[str, ToolSpec] = field(default_factory=dict)
    profile_bindings: Mapping[str, ProfileDefinition] = field(default_factory=dict)
    subagent_bindings: Mapping[str, ProfileDefinition] = field(default_factory=dict)
    members: list[
        ProfileAgentMember
        | ProfileAdapterMember
        | ProfileModelMember
        | ProfileEffortMember
        | ProfileApprovalMember
        | ProfileHistoryMember
        | ProfileSkillsMember
        | ProfileCodeModeMember
        | ProfileSystemMember
        | ProfileToolMember
        | ProfileUseMember
        | ProfileMcpMember
        | ProfileSettingsMember
        | ProfileArtifactsMember
        | ProfileOutputMember
        | ProfileSubagentMember
        | ProfileMaxAgentsMember
        | ProfileWorkflowMember
    ] = field(default_factory=list)
