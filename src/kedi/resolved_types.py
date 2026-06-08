"""Shared typing aliases for resolved Kedi/Python type annotations."""

from __future__ import annotations

from types import GenericAlias, UnionType
from typing import TypeAlias, cast

ResolvedKediType: TypeAlias = type | GenericAlias | UnionType


def is_resolved_kedi_type(value: object) -> bool:
    if isinstance(value, (type, GenericAlias, UnionType)):
        return True
    return hasattr(value, "__origin__") or str(type(value)).startswith("<class 'typing.")


def as_resolved_kedi_type(value: object) -> ResolvedKediType | None:
    if not is_resolved_kedi_type(value):
        return None
    return cast(ResolvedKediType, value)


def runtime_isinstance(value: object, expected: ResolvedKediType) -> bool:
    if isinstance(expected, type):
        return isinstance(value, expected)
    return True


def resolved_type_name(value: ResolvedKediType) -> str:
    if isinstance(value, type):
        return value.__name__
    name = getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return str(value)


__all__ = [
    "ResolvedKediType",
    "as_resolved_kedi_type",
    "is_resolved_kedi_type",
    "resolved_type_name",
    "runtime_isinstance",
]
