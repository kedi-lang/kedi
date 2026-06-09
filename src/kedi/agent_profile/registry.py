from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class ProfileModelMember:
    kind: Literal["model"] = "model"
    plain_value: str | None = None
    backtick_code: str | None = None


@dataclass(frozen=True, kw_only=True)
class ProfileDefinition:
    name: str
    members: list[ProfileModelMember] = field(default_factory=list)
