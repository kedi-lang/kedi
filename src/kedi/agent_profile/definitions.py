from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from .profile import WorkflowMode

if TYPE_CHECKING:
    from kedi.agent_adapter.tools.specs import ToolSpec
    from kedi.lang.ast import (
        AgentConnectionField,
        ArtifactsField,
        McpField,
        SettingsField,
        Span,
        TemplateLine,
        TypeExpr,
    )

    from .capability import Capability


@dataclass(frozen=True, kw_only=True)
class ProfileMember:
    loc: Span | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, kw_only=True)
class ProfileDeferMember(ProfileMember):
    kind: Literal["defer"] = "defer"
    enabled: bool = False
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileAgentMember(ProfileMember):
    kind: Literal["agent"] = "agent"
    plain_value: str | None = None
    backtick_code: str | None = None
    command_plain_value: str | None = None
    command_backtick_code: str | None = None
    connection_fields: tuple["AgentConnectionField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileAdapterMember(ProfileMember):
    kind: Literal["adapter"] = "adapter"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileModelMember(ProfileMember):
    kind: Literal["model"] = "model"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileEffortMember(ProfileMember):
    kind: Literal["effort"] = "effort"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileApprovalMember(ProfileMember):
    kind: Literal["approval"] = "approval"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileHooksMember(ProfileMember):
    kind: Literal["hooks"] = "hooks"
    enabled: bool | None = None
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileHistoryMember(ProfileMember):
    kind: Literal["history"] = "history"
    enabled: bool
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileSkillsMember(ProfileMember):
    kind: Literal["skills"] = "skills"
    enabled: bool
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileCodeModeMember(ProfileMember):
    kind: Literal["codemode"] = "codemode"
    enabled: bool
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileRequiresMember(ProfileMember):
    kind: Literal["requires"] = "requires"
    requirements: tuple[str, ...] = ()
    requirement_locs: Mapping[str, "Span"] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ProfileBudgetMember(ProfileMember):
    kind: Literal["budget"] = "budget"
    request_limit: int | None = None
    tool_attempt_limit: int | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileSystemMember(ProfileMember):
    kind: Literal["system"] = "system"
    plain_value: str | None = None
    backtick_code: str | None = None
    template_lines: tuple["TemplateLine", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileToolMember(ProfileMember):
    kind: Literal["tools"] = "tools"
    tool_names: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileUseMember(ProfileMember):
    kind: Literal["use"] = "use"
    name: str


@dataclass(frozen=True, kw_only=True)
class ProfileMcpMember(ProfileMember):
    kind: Literal["mcp"] = "mcp"
    fields: tuple["McpField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileSettingsMember(ProfileMember):
    kind: Literal["settings"] = "settings"
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileArtifactsMember(ProfileMember):
    kind: Literal["artifacts"] = "artifacts"
    fields: tuple["ArtifactsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileOutputMember(ProfileMember):
    kind: Literal["output"] = "output"
    type_annotation: str
    type_expr: "TypeExpr"


@dataclass(frozen=True, kw_only=True)
class ProfileSubagentMember(ProfileMember):
    kind: Literal["subagents"] = "subagents"
    name: str


@dataclass(frozen=True, kw_only=True)
class ProfileMaxAgentsMember(ProfileMember):
    kind: Literal["max_agents"] = "max_agents"
    value: int


@dataclass(frozen=True, kw_only=True)
class ProfileWorkflowMember(ProfileMember):
    kind: Literal["workflow"] = "workflow"
    mode: WorkflowMode


@dataclass(frozen=True, kw_only=True)
class ProfileDefinition:
    name: str
    loc: Span | None = field(default=None, repr=False, compare=False)
    kind: Literal["profile", "capability"] = "profile"
    identity: str | None = None
    description: str | None = None
    # These bindings are compiler-owned implementation details for imported
    # profiles. Source-level profile syntax remains name based.
    tool_bindings: Mapping[str, ToolSpec] = field(default_factory=dict)
    profile_bindings: Mapping[str, ProfileDefinition] = field(default_factory=dict)
    capability_bindings: Mapping[str, Capability] = field(default_factory=dict)
    subagent_bindings: Mapping[str, ProfileDefinition] = field(default_factory=dict)
    members: Sequence[
        ProfileAgentMember
        | ProfileAdapterMember
        | ProfileModelMember
        | ProfileEffortMember
        | ProfileApprovalMember
        | ProfileHooksMember
        | ProfileHistoryMember
        | ProfileSkillsMember
        | ProfileCodeModeMember
        | ProfileRequiresMember
        | ProfileBudgetMember
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
        | ProfileDeferMember
    ] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(deepcopy(self.members)))
        for name in (
            "tool_bindings",
            "profile_bindings",
            "capability_bindings",
            "subagent_bindings",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def __deepcopy__(self, memo: dict[int, Any]) -> ProfileDefinition:
        # Dynamic workflows snapshot the definition graph, including mutable AST
        # payloads. Mapping proxies need an explicit copy before being frozen again.
        result = object.__new__(type(self))
        memo[id(self)] = result
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "tool_bindings":
                value = {
                    name: replace(
                        spec,
                        json_schema=deepcopy(spec.json_schema, memo),
                        return_schema=deepcopy(spec.return_schema, memo),
                        metadata=deepcopy(spec.metadata, memo),
                    )
                    for name, spec in value.items()
                }
                object.__setattr__(result, item.name, value)
                continue
            if isinstance(value, Mapping):
                value = dict(value)
            object.__setattr__(result, item.name, deepcopy(value, memo))
        result.__post_init__()
        return result
