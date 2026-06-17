from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kedi.lang.ast import McpField, SettingsField, TemplateLine


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
class ProfileMcpMember:
    kind: Literal["mcp"] = "mcp"
    fields: tuple["McpField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileSettingsMember:
    kind: Literal["settings"] = "settings"
    fields: tuple["SettingsField", ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileDefinition:
    name: str
    members: list[
        ProfileModelMember
        | ProfileEffortMember
        | ProfileSystemMember
        | ProfileToolMember
        | ProfileMcpMember
        | ProfileSettingsMember
    ] = field(default_factory=list)
