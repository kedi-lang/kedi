from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from kedi.lang.ast import TemplateLine


@dataclass(frozen=True, kw_only=True)
class ProfileModelMember:
    kind: Literal["model"] = "model"
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
class ProfileDefinition:
    name: str
    members: list[ProfileModelMember | ProfileSystemMember | ProfileToolMember] = field(
        default_factory=list
    )
