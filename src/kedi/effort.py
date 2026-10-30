from __future__ import annotations

from typing import Final, Literal, cast

from .directive_values import DIRECTIVE_RULES

EffortValue = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
PydanticEffortValue = Literal["minimal", "low", "medium", "high", "xhigh"]

EFFORT_VALUES: Final[frozenset[str]] = frozenset(DIRECTIVE_RULES["effort"].options)


def is_valid_effort(value: str) -> bool:
    return value.strip() in EFFORT_VALUES


def validate_effort(value: str, *, directive_name: str = "effort") -> EffortValue:
    effort = value.strip()
    if effort not in EFFORT_VALUES:
        expected = ", ".join(sorted(EFFORT_VALUES))
        raise ValueError(f"{directive_name} directive must be one of: {expected}")
    return cast(EffortValue, effort)


def pydantic_effort_value(value: EffortValue | str) -> PydanticEffortValue:
    if value == "max":
        return "xhigh"
    return cast(PydanticEffortValue, value)
